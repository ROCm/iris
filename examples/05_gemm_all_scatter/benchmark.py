#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
# Copyright (c) 2025 Advanced Micro Devices, Inc. All rights reserved.

import torch
import triton
import random
import sys
import os
import argparse
import json

from examples.common.utils import JSONWriter, Timestamps, is_triton_interpret_set
from examples.gemm.validation import validate_gemm

import iris

from matmul_wrapper import matmul

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
        choices=["fp16", "fp32", "int8", "bf16"],
        help="Datatype of computation",
    )
    parser.add_argument(
        "--algorithm",
        type=str,
        default="all_reduce",
        choices=["all_reduce", "all_scatter", "all_gather", "one_shot"],
        help="Algorithm to use",
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
    parser.add_argument(
        "--COMMUNICATION_TILE_M",
        type=int,
        default=128,
        help="M tile size for reduction, scatter or one-shot",
    )
    parser.add_argument(
        "--COMMUNICATION_TILE_N",
        type=int,
        default=64,
        help="N tile size for reduction, scatter or one-shot",
    )

    # Best to try 1, 6 or 8
    parser.add_argument("--gsize_m", type=int, default=6, help="Grid size M")
    parser.add_argument("--two_tiles", type=str, default="True", help="Use two tiles")
    parser.add_argument("--num_stages", type=int, default=1, help="Number of stages")
    parser.add_argument("--num_warps", type=int, default=8, help="Number of warps")
    parser.add_argument("--waves_per_eu", type=int, default=0, help="Waves per execution unit")
    parser.add_argument("--mfmaInstrSize", type=int, default=16, help="MFMA instruction size")
    parser.add_argument("--kpack", type=int, default=2, help="K packing size")
    parser.add_argument("--heap_size", type=int, default=1 << 33, help="Iris heap size")

    # For All Scatter, use: 288
    # For One Shot, use: 256
    parser.add_argument("--gemm_sms", type=int, default=288, help="Number of SMs for Stream-K")
    parser.add_argument("--total_sms", type=int, default=304, help="Total number of SMs")
    parser.add_argument(
        "--communication_block_size",
        type=int,
        default=256,
        help="Communication block size",
    )

    # For All Scatter, use: 10
    # For One Shot, use: 2
    parser.add_argument(
        "--communication_sms_multiplier",
        type=int,
        default=2,
        help="Communication SMS multiplier",
    )
    return vars(parser.parse_args())


def main():
    args = parse_args()

    shmem = iris.Iris(args["heap_size"])
    rank = shmem.get_rank()
    world_size = shmem.get_num_ranks()
    cu_count = shmem.get_cu_count()

    # GEMM
    datatype = torch.float32
    if args["datatype"] == "fp16":
        datatype = torch.float16
    elif args["datatype"] == "fp32":
        datatype = torch.float32
    elif args["datatype"] == "int8":
        datatype = torch.int8
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
    # if args["algorithm"] == "all_scatter" or args["algorithm"] == "all_gather":
    args["n"] = args["n"] // world_size
    local_B = B[:, rank * args["n"] : (rank + 1) * args["n"]].clone()
    local_A = A
    # elif args["algorithm"] == "all_reduce" or args["algorithm"] == "one_shot":
    #     rows_per_gpu = args["k"] // world_size
    #     args["k"] = rows_per_gpu
    #     start_row = rank * rows_per_gpu
    #     end_row = start_row + rows_per_gpu
    #     local_B = B[start_row:end_row, :]
    #     local_A = A[:, start_row:end_row]
    # else:
    #     print("Unknown algorithm.")
    #     exit(1)

    for key, value in args.items():
        json_writer.add_field(key, value)

    global_C = shmem.zeros((args["M"], args["N"]), device="cuda", dtype=A.dtype)
    local_C = shmem.zeros((args["m"], args["n"]), device="cuda", dtype=A.dtype)

    total_blocks_M = triton.cdiv(args["m"], args["BLK_M"])
    total_blocks_N = triton.cdiv(args["n"], args["BLK_N"])
    total_tiles = total_blocks_M * total_blocks_N

    # if args["gemm_sms"] >= args["total_sms"]:
    #     print(f"Invalid number of stream-K SMs. {args['gemm_sms']} >= {args['total_sms']}")
    #     exit(1)

    tile_completed = shmem.zeros((total_tiles,), device="cuda", dtype=torch.int32)

    locks = shmem.zeros((args["gemm_sms"],), device="cuda", dtype=torch.int32)

    P = shmem.zeros(
        (args["gemm_sms"], args["BLK_M"] * args["BLK_N"]),
        device="cuda",
        dtype=torch.float32,
    )
    bias = None

    gemm_stream = torch.cuda.Stream()

    # json_writer.add_field("communication_sms", communication_sms)
    json_writer.add_field("gemm_sms", args["gemm_sms"])

    kernel_timing = {
        "gemm": {
            "start_event": torch.cuda.Event(enable_timing=True),
            "end_event": torch.cuda.Event(enable_timing=True),
            "ms": 0,
            "experiments": 0,
        },
        "communication": {
            "start_event": torch.cuda.Event(enable_timing=True),
            "end_event": torch.cuda.Event(enable_timing=True),
            "ms": 0,
            "experiments": 0,
        },
    }

    # COMMUNICATION_ALGORITHM = None
    # if args["algorithm"] == "one_shot":
    #     COMMUNICATION_ALGORITHM = ONE_SHOT
    # elif args["algorithm"] == "all_reduce":
    #     COMMUNICATION_ALGORITHM = ALL_REDUCE
    # elif args["algorithm"] == "all_scatter":
    #     COMMUNICATION_ALGORITHM = ALL_SCATTER
    # elif args["algorithm"] == "all_gather":
    #     COMMUNICATION_ALGORITHM = ALL_GATHER

    # Timestamps
    timestamps = Timestamps(num_tiles=total_tiles)

    def preamble():
        shmem.barrier()
        iris.memset_tensor(tile_completed, 0)
        shmem.barrier()
        shmem.log("Preamble completed")

    def run_experiment():
        nonlocal local_C
        nonlocal global_C
        nonlocal kernel_timing

        shmem.barrier()

        if args["trace_tiles"]:
            timestamps.reset()
            shmem.barrier()

        torch.cuda.nvtx.range_push("GEMM + Communication")
        torch.cuda.nvtx.range_push("GEMM")
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

        torch.cuda.nvtx.range_pop()

    # Synchronize across all GPUs
    shmem.barrier()

    # Warmup
    run_experiment()

    shmem.barrier()
    preamble()
    shmem.barrier()

    for k in ["gemm", "communication"]:
        kernel_timing[k]["ms"] = 0
        kernel_timing[k]["experiments"] = 0

    if not is_triton_interpret_set():
        gemm_registers = matmul.streamk_registers
        gemm_spills = matmul.streamk_spills

        json_writer.add_field("gemm_registers", gemm_registers)
        json_writer.add_field("gemm_spills", gemm_spills)

    if args["validate"]:
        shmem.log("Validating...")

        matmul.set_debug(False)
        # Validate global result
        success = validate_gemm(A, B, global_C, shmem)
        passed_str = "passed" if success else "failed"
        shmem.log(f"Final C validation {passed_str}.")

        # Wait for all to finish validation
        shmem.barrier()
        shmem.log("Validating local C...")

        json_writer.add_field("success", success)

        # Validate partial result
        # success = validate_gemm(local_A, local_B, local_C, shmem)
        # json_writer.add_field("success_partial", success)
        # passed_str = "passed" if success else "failed"
        # shmem.log(f"Local C validation {passed_str}.")

        # # Wait for all to finish validation
        # shmem.barrier()
    shmem.log("Validation completed")

    if args["benchmark"]:
        shmem.log("Benchmarking...")
        perf = lambda ms: 2 * args["M"] * args["N"] * args["K"] * 1e-12 / (ms * 1e-3)
        triton_ms = iris.do_bench(run_experiment, shmem.barrier, preamble)
        triton_tflops = perf(triton_ms)
        algo_string = args["algorithm"]
        shmem.log_stats(
            f"tile matmul + {algo_string} (grid={total_tiles}): {triton_ms:.3f} ms  {triton_tflops:.3f} tflops"
        )

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
        gpu_freq = shmem.wall_clock_rate(rank) * 1e-3
        algo_string = args["algorithm"]
        filename = f"gemm_tiles_{algo_string}_trace_rank{rank}.json"
        timestamps.to_json(filename, gpu_freq)

    shmem.barrier()


if __name__ == "__main__":
    main()
