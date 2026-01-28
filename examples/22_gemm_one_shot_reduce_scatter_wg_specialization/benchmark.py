#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
# Copyright (c) 2025 Advanced Micro Devices, Inc. All rights reserved.

import torch
import torch.distributed as dist
import torch.multiprocessing as mp
import triton
import random
import argparse
import math

from examples.common.utils import JSONWriter, Timestamps, is_triton_interpret_set
from examples.common.validation import validate_reduce_scatter

import iris
from matmul_wrapper import MatMulReduceScatterWgSpecialized

torch.manual_seed(0)
random.seed(0)


def parse_args():
    parser = argparse.ArgumentParser(
        description="GEMM + ReduceScatter Benchmark with Workgroup Specialization",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("-m", type=int, default=8192, help="Number of rows in matrix A (M)")
    parser.add_argument("-n", type=int, default=4096, help="Number of columns in matrix B (N)")
    parser.add_argument("-k", type=int, default=12288, help="Common dimension (K), will be split across ranks")
    parser.add_argument("-d", "--debug", action="store_true", help="Enable debug mode")
    parser.add_argument("-v", "--validate", action="store_true", help="Enable validation mode")
    parser.add_argument("-t", "--trace_tiles", action="store_true", help="Enable tile-tracing mode")
    parser.add_argument("-b", "--benchmark", action="store_true", help="Enable benchmarking mode")
    parser.add_argument(
        "--datatype",
        type=str,
        default="fp16",
        choices=["fp16", "fp32", "bf16"],
        help="Datatype of computation",
    )
    parser.add_argument(
        "--output_file",
        type=str,
        default="log.json",
        help="Output file",
    )
    parser.add_argument("--BLK_M", type=int, default=128, help="Block size M")
    parser.add_argument("--BLK_N", type=int, default=256, help="Block size N")
    parser.add_argument("--BLK_K", type=int, default=32, help="Block size K")
    parser.add_argument("--gsize_m", type=int, default=1, help="L2-cache locality swizzle parameter")
    parser.add_argument("--heap_size", type=int, default=1 << 33, help="Iris heap size")
    parser.add_argument(
        "--num_sms",
        type=int,
        default=None,
        help="Number of total SMs (default: auto-detected)",
    )
    parser.add_argument(
        "--gemm_sms",
        type=int,
        default=None,
        help="Number of SMs for GEMM (default: auto-detected as power of 2)",
    )
    parser.add_argument("--num_stages", type=int, default=2, help="Number of stages")
    parser.add_argument("-r", "--num_ranks", type=int, default=2, help="Number of ranks/processes")

    return vars(parser.parse_args())


def _worker(local_rank: int, world_size: int, init_url: str, args: dict):
    """Worker function for PyTorch distributed execution."""
    backend = "nccl" if torch.cuda.is_available() else "gloo"
    dist.init_process_group(
        backend=backend,
        init_method=init_url,
        world_size=world_size,
        rank=local_rank,
        device_id=torch.device(f"cuda:{local_rank}"),
    )

    shmem = iris.iris(args["heap_size"])
    rank = shmem.get_rank()
    world_size = shmem.get_num_ranks()

    cu_count = torch.cuda.get_device_properties(rank).multi_processor_count
    if args["num_sms"] is None:
        args["num_sms"] = cu_count
    if args["gemm_sms"] is None:
        # Use next smaller power of 2 for GEMM SMs
        args["gemm_sms"] = 2 ** int(math.log2(cu_count)) if cu_count > 0 else 1

    datatype = torch.float16
    if args["datatype"] == "fp16":
        datatype = torch.float16
    elif args["datatype"] == "fp32":
        datatype = torch.float32
    elif args["datatype"] == "bf16":
        datatype = torch.bfloat16
    else:
        print("Unknown datatype.")
        exit(1)

    M, N, K = args["m"], args["n"], args["k"]

    assert M % world_size == 0, f"M ({M}) must be divisible by world size ({world_size})"
    assert K % world_size == 0, f"K ({K}) must be divisible by world size ({world_size})"
    assert (M // world_size) % args["BLK_M"] == 0, (
        f"M_per_rank ({M // world_size}) must be divisible by BLK_M ({args['BLK_M']})"
    )

    local_K = K // world_size
    M_per_rank = M // world_size

    A_full = shmem.randn(M, K, device="cuda", dtype=datatype)
    B_full = shmem.randn(K, N, device="cuda", dtype=datatype)

    # Each rank gets a portion of K dimension as input
    local_A = A_full[:, rank * local_K : (rank + 1) * local_K].clone()
    local_B = B_full[rank * local_K : (rank + 1) * local_K, :].clone()

    json_writer = JSONWriter(args["output_file"])
    json_writer.add_field("world_size", world_size)
    json_writer.add_field("M", M)
    json_writer.add_field("N", N)
    json_writer.add_field("K", K)
    json_writer.add_field("local_K", local_K)

    for key, value in args.items():
        json_writer.add_field(key, value)

    local_buf = shmem.zeros((M, N), device="cuda", dtype=datatype)

    output_buf = shmem.zeros((M_per_rank, N), device="cuda", dtype=datatype)

    total_blocks_M = triton.cdiv(M, args["BLK_M"])
    total_blocks_N = triton.cdiv(N, args["BLK_N"])
    total_tiles = total_blocks_M * total_blocks_N

    locks = shmem.zeros((total_tiles,), device="cuda", dtype=torch.int32)

    gemm_stream = torch.cuda.Stream()

    json_writer.add_field("num_sms", args["num_sms"])
    json_writer.add_field("gemm_sms", args["gemm_sms"])

    kernel_timing = {
        "gemm_rs": {
            "start_event": torch.cuda.Event(enable_timing=True),
            "end_event": torch.cuda.Event(enable_timing=True),
            "ms": 0,
            "experiments": 0,
        },
    }

    timestamps = Timestamps(num_tiles=total_tiles)

    def run_experiment():
        nonlocal local_buf, output_buf

        local_buf.zero_()
        output_buf.zero_()
        locks.zero_()
        shmem.barrier()

        if args["trace_tiles"]:
            timestamps.reset()
            shmem.barrier()

        torch.cuda.nvtx.range_push("GEMM + ReduceScatter")
        with torch.cuda.stream(gemm_stream):
            kernel_timing["gemm_rs"]["start_event"].record()
            MatMulReduceScatterWgSpecialized.apply(
                local_A,
                local_B,
                local_buf,
                output_buf,
                locks,
                rank,
                world_size,
                args["gemm_sms"],
                args["num_sms"],
                args["BLK_M"],
                args["BLK_N"],
                args["BLK_K"],
                args["gsize_m"],
                args["num_stages"],
                shmem.get_heap_bases(),
                torch.cuda.get_device_properties(rank).name,
                args["trace_tiles"],
                timestamps.mm_begin_timestamp,
                timestamps.mm_end_timestamp,
            )
            kernel_timing["gemm_rs"]["end_event"].record()
            kernel_timing["gemm_rs"]["experiments"] += 1

        torch.cuda.nvtx.range_pop()
        shmem.barrier()

        for k in ["gemm_rs"]:
            ms = kernel_timing[k]["start_event"].elapsed_time(kernel_timing[k]["end_event"])
            kernel_timing[k]["ms"] += ms

    shmem.barrier()

    # Warmup
    run_experiment()

    shmem.barrier()

    for k in ["gemm_rs"]:
        kernel_timing[k]["ms"] = 0
        kernel_timing[k]["experiments"] = 0

    if args["validate"]:
        shmem.info("Validating...")
        MatMulReduceScatterWgSpecialized.set_debug(True)

        local_gemm = local_buf.clone()
        local_output = output_buf.clone()

        # Allow larger tolerance for fp16 due to accumulated rounding errors in atomic operations
        atol = 1.0 if datatype == torch.float16 else 0.5

        tp_group = dist.new_group(ranks=list(range(world_size)))
        success = validate_reduce_scatter(local_gemm, local_output, shmem, tp_group, atol=atol)

        if success:
            shmem.info("✅ Triton and Torch match")
        else:
            shmem.info("❌ Triton and Torch differ")

        json_writer.add_field("success", success)

        if not is_triton_interpret_set():
            gemm_registers = MatMulReduceScatterWgSpecialized.get_matmul_registers()
            gemm_spills = MatMulReduceScatterWgSpecialized.get_matmul_spills()
            json_writer.add_field("gemm_registers", gemm_registers)
            json_writer.add_field("gemm_spills", gemm_spills)

        shmem.barrier()
        shmem.info("Validation completed")

    if args["benchmark"]:
        MatMulReduceScatterWgSpecialized.set_debug(False)
        shmem.info("Benchmarking...")

        perf = lambda ms: 2 * M * N * K * 1e-12 / (ms * 1e-3)

        triton_ms = iris.do_bench(run_experiment, shmem.barrier)
        triton_tflops = perf(triton_ms)

        shmem.info(f"GEMM + ReduceScatter (total_tiles={total_tiles}): {triton_ms:.3f} ms  {triton_tflops:.3f} tflops")

        json_writer.add_field("tflops", triton_tflops)
        json_writer.add_field("total_ms", triton_ms)

        for k in ["gemm_rs"]:
            json_writer.add_field(k + "_ms", kernel_timing[k]["ms"] / kernel_timing[k]["experiments"])
            json_writer.add_field(k + "_experiments", kernel_timing[k]["experiments"])

        shmem.barrier()

    if rank == 0:
        json_writer.flush()
        json_writer.display()

    if args["trace_tiles"] and rank == 0:
        gpu_freq = iris.hip.get_wall_clock_rate(rank) * 1e-3
        filename = f"gemm_tiles_reduce_scatter_trace_rank{rank}.json"
        timestamps.to_json(filename, gpu_freq)

    shmem.barrier()
    dist.destroy_process_group()


def main():
    args = parse_args()
    num_ranks = args["num_ranks"]

    init_url = "tcp://127.0.0.1:29500"
    mp.spawn(
        fn=_worker,
        args=(num_ranks, init_url, args),
        nprocs=num_ranks,
        join=True,
    )


if __name__ == "__main__":
    main()
