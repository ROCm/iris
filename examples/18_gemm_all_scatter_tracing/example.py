#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
# Copyright (c) 2025 Advanced Micro Devices, Inc. All rights reserved.
"""
Example: GEMM All-Scatter with Tile Tracing (N-split)

Demonstrates: Same GEMM all-scatter pattern as example 08 (each rank computes
full A x partial B, then scatters its portion of C to all other ranks), but
uses iris DeviceContext tracing to record per-tile put events for visualization.

Run with:
    torchrun --nproc_per_node=2 --standalone example.py --validate
    torchrun --nproc_per_node=2 --standalone example.py --trace_tiles
"""

import argparse
import os

import torch
import torch.distributed as dist

from matmul_wrapper import matmul

import iris


def parse_args():
    parser = argparse.ArgumentParser(
        description="GEMM All-Scatter with Tracing example.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("-m", type=int, default=8192, help="Rows of A")
    parser.add_argument("-n", type=int, default=4608, help="Columns of B")
    parser.add_argument("-k", type=int, default=36864, help="Common dimension")
    parser.add_argument("-v", "--validate", action="store_true", help="Validate output")
    parser.add_argument("-t", "--trace_tiles", action="store_true", help="Enable tile tracing")
    parser.add_argument(
        "--datatype",
        type=str,
        default="fp16",
        choices=["fp16", "fp32", "bf16"],
        help="Datatype of computation",
    )
    parser.add_argument("--BLK_M", type=int, default=256, help="Block size M")
    parser.add_argument("--BLK_N", type=int, default=64, help="Block size N")
    parser.add_argument("--BLK_K", type=int, default=64, help="Block size K")
    parser.add_argument("--gsize_m", type=int, default=6, help="L2-cache locality swizzle parameter")
    parser.add_argument("--num_stages", type=int, default=2, help="Number of stages")
    parser.add_argument("--heap_size", type=int, default=1 << 33, help="Iris heap size")
    return vars(parser.parse_args())


def main():
    args = parse_args()
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    torch.cuda.set_device(local_rank)
    dist.init_process_group(backend="gloo")

    ctx = iris.iris(heap_size=args["heap_size"])
    rank = ctx.get_rank()
    world_size = ctx.get_num_ranks()

    cu_count = torch.cuda.get_device_properties(local_rank).multi_processor_count

    dtype_map = {"fp16": torch.float16, "fp32": torch.float32, "bf16": torch.bfloat16}
    datatype = dtype_map[args["datatype"]]

    M, N, K = args["m"], args["n"], args["k"]
    assert N % world_size == 0, f"N ({N}) must be divisible by world size ({world_size})."

    # Allocate matrices
    A = ctx.randn(M, K, device="cuda", dtype=datatype)
    B = ctx.randn(N, K, device="cuda", dtype=datatype).T

    local_n = N // world_size
    local_B = B[:, rank * local_n : (rank + 1) * local_n].clone()

    global_C = ctx.zeros((M, N), device="cuda", dtype=datatype)
    local_C = ctx.zeros((M, local_n), device="cuda", dtype=datatype)

    bias = None

    # Warmup
    ctx.barrier()
    matmul._call(
        A,
        local_B,
        local_C,
        global_C,
        bias,
        rank,
        world_size,
        cu_count,
        args["BLK_M"],
        args["BLK_N"],
        args["BLK_K"],
        args["gsize_m"],
        args["num_stages"],
        ctx.get_device_context(),
        "gfx942",
        TRACING=args["trace_tiles"],
    )
    torch.cuda.synchronize()
    ctx.barrier()

    # Validate
    if args["validate"]:
        expected = A @ B
        if torch.allclose(global_C, expected, atol=2.0):
            if rank == 0:
                print("Validation PASSED")
        else:
            max_diff = (global_C - expected).abs().max().item()
            print(f"Rank {rank}: Validation FAILED (max diff: {max_diff})")

    # Report tracing status
    if args["trace_tiles"] and rank == 0:
        print("Tracing enabled -- tile traces captured by iris DeviceContext")

    ctx.barrier()
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
