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

from examples.common.utils import (
    JSONWriter,
    Timestamps,
    is_triton_interpret_set,
)

import iris

from matmul_wrapper import matmul
from examples.common.validation import validate_gemm

torch.manual_seed(123)
random.seed(123)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Parse matrix dimensions and configuration.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("-m", type=int, default=8192, help="Number of rows in matrix A")
    parser.add_argument("-n", type=int, default=4608, help="Number of columns in matrix B")
    parser.add_argument("-k", type=int, default=36864, help="Common dimension between matrices A and B")
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
    # For All Scatter, use: 256x64x64
    # For One Shot, use: 256x256x64
    parser.add_argument("--BLK_M", type=int, default=256, help="Block size M")
    parser.add_argument("--BLK_N", type=int, default=64, help="Block size N")
    parser.add_argument("--BLK_K", type=int, default=64, help="Block size K")

    # Best to try 1, 6 or 8
    parser.add_argument("--gsize_m", type=int, default=6, help="Grid size M")
    parser.add_argument("--two_tiles", type=str, default="True", help="Use two tiles")
    parser.add_argument("--num_stages", type=int, default=1, help="Number of stages")
    parser.add_argument("--num_warps", type=int, default=8, help="Number of warps")
    parser.add_argument("--waves_per_eu", type=int, default=0, help="Waves per execution unit")
    parser.add_argument("--mfmaInstrSize", type=int, default=16, help="MFMA instruction size")
    parser.add_argument("--kpack", type=int, default=2, help="K packing size")
    parser.add_argument("--heap_size", type=int, default=1 << 33, help="Iris heap size")

    parser.add_argument("--gemm_sms", type=int, default=None, help="Number of SMs for GEMM (default: auto-detected)")
    parser.add_argument("--total_sms", type=int, default=None, help="Total number of SMs (default: auto-detected)")
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

    # Set default SM values if not provided
    cu_count = torch.cuda.get_device_properties(rank).multi_processor_count
    if args["total_sms"] is None:
        args["total_sms"] = cu_count
    if args["gemm_sms"] is None:
        # For all_reduce: use next smaller power of 2, rest for communication
        args["gemm_sms"] = 2 ** int(math.log2(cu_count)) if cu_count > 0 else 1

    # GEMM
    datatype = torch.float32
    if args["datatype"] == "fp16":
        datatype = torch.float16
    elif args["datatype"] == "fp32":
        datatype = torch.float32
    elif args["datatype"] == "bf16":
        datatype = torch.bfloat16
    else:
        print("Unknown datatype.")
        exit(1)

    assert args["n"] % world_size == 0, f"N ({args['n']}) must be divisible by world size ({world_size})."
    assert args["k"] % world_size == 0, f"K ({args['k']}) must be divisible by world size ({world_size})."

    A = shmem.randn(args["m"], args["k"], device="cuda", dtype=datatype)
    B = shmem.randn(args["n"], args["k"], device="cuda", dtype=datatype).T
    C = shmem.zeros((args["m"], args["n"]), device="cuda", dtype=A.dtype)

    args["M"] = args["m"]
    args["N"] = args["n"]
    args["K"] = args["k"]

    json_writer = JSONWriter(args["output_file"])
    json_writer.add_field("world_size", world_size)

    # Splitting
    rows_per_gpu = args["k"] // world_size
    args["k"] = rows_per_gpu
    start_row = rank * rows_per_gpu
    end_row = start_row + rows_per_gpu
    local_B = B[start_row:end_row, :]
    local_A = A[:, start_row:end_row]

    for key, value in args.items():
        json_writer.add_field(key, value)

    global_C = shmem.zeros((args["M"], args["N"]), device="cuda", dtype=A.dtype)
    local_C = shmem.zeros((args["m"], args["n"]), device="cuda", dtype=A.dtype)

    total_blocks_M = triton.cdiv(args["m"], args["BLK_M"])
    total_blocks_N = triton.cdiv(args["n"], args["BLK_N"])
    total_tiles = total_blocks_M * total_blocks_N

    if args["gemm_sms"] >= args["total_sms"]:
        print(f"Invalid number of GEMM SMs. {args['gemm_sms']} >= {args['total_sms']}")
        exit(1)

    tile_completed = shmem.zeros((total_tiles,), device="cuda", dtype=torch.int32)

    locks = shmem.zeros((args["gemm_sms"],), device="cuda", dtype=torch.int32)

    P = shmem.zeros(
        (args["gemm_sms"], args["BLK_M"] * args["BLK_N"]),
        device="cuda",
        dtype=torch.float32,
    )
    bias = None

    gemm_stream = torch.cuda.Stream()

    json_writer.add_field("gemm_sms", args["gemm_sms"])

    kernel_timing = {
        "gemm": {
            "start_event": torch.cuda.Event(enable_timing=True),
            "end_event": torch.cuda.Event(enable_timing=True),
            "ms": 0,
            "experiments": 0,
        }
    }

    # Timestamps
    timestamps = Timestamps(num_tiles=total_tiles)

    def preamble():
        shmem.barrier()
        tile_completed.zero_()
        shmem.barrier()

    def run_experiment():
        nonlocal local_C
        nonlocal global_C
        nonlocal kernel_timing

        shmem.barrier()

        if args["trace_tiles"]:
            timestamps.reset()
            shmem.barrier()

        torch.cuda.nvtx.range_push("GEMM + Communication")
        with torch.cuda.stream(gemm_stream):
            kernel_timing["gemm"]["start_event"].record()
            local_C = matmul.apply(
                local_A,
                local_B,
                local_C,
                global_C,
                bias,
                P,
                locks,
                tile_completed,
                rank,
                world_size,
                args["gemm_sms"],
                args["BLK_M"],
                args["BLK_N"],
                args["BLK_K"],
                args["gsize_m"],
                args["two_tiles"],
                args["num_stages"],
                args["num_warps"],
                args["waves_per_eu"],
                args["mfmaInstrSize"],
                args["kpack"],
                shmem.get_heap_bases(),
                cu_count,
                args["trace_tiles"],
                timestamps.mm_begin_timestamp,
                timestamps.mm_end_timestamp,
            )
            kernel_timing["gemm"]["end_event"].record()
            kernel_timing["gemm"]["experiments"] += 1

        torch.cuda.nvtx.range_pop()
        shmem.barrier()

        for k in ["gemm"]:
            ms = kernel_timing[k]["start_event"].elapsed_time(kernel_timing[k]["end_event"])
            kernel_timing[k]["ms"] += ms

    # Synchronize across all GPUs
    shmem.barrier()

    # Warmup
    run_experiment()

    shmem.barrier()
    preamble()
    shmem.barrier()

    for k in ["gemm"]:
        kernel_timing[k]["ms"] = 0
        kernel_timing[k]["experiments"] = 0

    if not is_triton_interpret_set():
        gemm_registers = matmul.streamk_registers
        gemm_spills = matmul.streamk_spills

        json_writer.add_field("gemm_registers", gemm_registers)
        json_writer.add_field("gemm_spills", gemm_spills)

    if args["validate"]:
        shmem.info("Validating...")

        matmul.set_debug(False)
        # Validate global result
        success = validate_gemm(A, B, global_C, shmem, atol=2)
        passed_str = "passed" if success else "failed"
        shmem.info(f"Final C validation {passed_str}.")

        # Wait for all to finish validation
        shmem.barrier()
        json_writer.add_field("success", success)
        shmem.info("Validation completed")

    if args["benchmark"]:
        shmem.info("Benchmarking...")
        perf = lambda ms: 2 * args["M"] * args["N"] * args["K"] * 1e-12 / (ms * 1e-3)
        triton_ms = iris.do_bench(run_experiment, shmem.barrier, preamble)
        triton_tflops = perf(triton_ms)
        shmem.info(f"tile matmul + all_reduce (grid={total_tiles}): {triton_ms:.3f} ms  {triton_tflops:.3f} tflops")

        json_writer.add_field("triton_tflops", triton_tflops)
        json_writer.add_field("triton_ms", triton_ms)

        for k in ["gemm"]:
            json_writer.add_field(k + "_ms", kernel_timing[k]["ms"] / kernel_timing[k]["experiments"])
            json_writer.add_field(k + "_experiments", kernel_timing[k]["experiments"])

        # Wait for all to finish benchmarking
        shmem.barrier()

    if rank == 0:
        json_writer.flush()
        json_writer.display()

    if args["trace_tiles"] and rank == 0:
        gpu_freq = iris.hip.get_wall_clock_rate(rank) * 1e-3
        filename = f"gemm_all_reduce_tiles_trace_rank{rank}.json"
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
