#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
# Copyright (c) 2025 Advanced Micro Devices, Inc. All rights reserved.
"""
Example: GEMM All-Scatter with Workgroup Specialization (N-split)

Demonstrates: Each rank computes full A x partial B (N-split). The kernel uses
workgroup specialization -- GEMM WGs produce tiles into local C and signal via
locks, while scatter WGs consume completed tiles and write them to remote ranks
via iris.store.

Run with:
    torchrun --nproc_per_node=2 --standalone example.py --validate
"""
import argparse
import math
import os

import torch
import torch.distributed as dist
import triton

from matmul_wrapper import matmul

import iris


def parse_args():
    parser = argparse.ArgumentParser(
        description="GEMM All-Scatter WG Specialization example.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("-m", type=int, default=8192, help="Number of rows in matrix A")
    parser.add_argument("-n", type=int, default=4608, help="Number of columns in matrix B")
    parser.add_argument("-k", type=int, default=36864, help="Common dimension between matrices A and B")
    parser.add_argument("-v", "--validate", action="store_true", help="Enable validation mode")
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

    # Auto-detect SMs
    cu_count = torch.cuda.get_device_properties(local_rank).multi_processor_count
    num_sms = cu_count
    gemm_sms = 2 ** int(math.log2(cu_count)) if cu_count > 0 else 1

    # dtype
    dtype_map = {"fp16": torch.float16, "fp32": torch.float32, "bf16": torch.bfloat16}
    datatype = dtype_map[args["datatype"]]

    assert args["n"] % world_size == 0, f"N ({args['n']}) must be divisible by world size ({world_size})."

    # Allocate full matrices
    A = ctx.randn(args["m"], args["k"], device="cuda", dtype=datatype)
    B = ctx.randn(args["n"], args["k"], device="cuda", dtype=datatype).T

    M = args["m"]
    N = args["n"]
    K = args["k"]

    # N-split: each rank computes full A x partial B
    local_n = N // world_size
    local_B = B[:, rank * local_n : (rank + 1) * local_n].clone()
    local_A = A

    global_C = ctx.zeros((M, N), device="cuda", dtype=datatype)
    local_C = ctx.zeros((M, local_n), device="cuda", dtype=datatype)

    total_blocks_M = triton.cdiv(M, args["BLK_M"])
    total_blocks_N = triton.cdiv(local_n, args["BLK_N"])
    total_tiles = total_blocks_M * total_blocks_N

    locks = ctx.zeros((total_tiles,), device="cuda", dtype=torch.int32)

    bias = None

    def reset_locks():
        ctx.barrier()
        locks.zero_()
        ctx.barrier()

    # Warmup
    ctx.barrier()
    matmul._call(
        local_A, local_B, local_C, global_C, bias, locks,
        rank, world_size, gemm_sms, num_sms,
        args["BLK_M"], args["BLK_N"], args["BLK_K"],
        args["gsize_m"], args["num_stages"],
        ctx.get_heap_bases(), "gfx942",
    )
    torch.cuda.synchronize()
    ctx.barrier()

    # Run again with fresh locks for validation
    reset_locks()
    matmul._call(
        local_A, local_B, local_C, global_C, bias, locks,
        rank, world_size, gemm_sms, num_sms,
        args["BLK_M"], args["BLK_N"], args["BLK_K"],
        args["gsize_m"], args["num_stages"],
        ctx.get_heap_bases(), "gfx942",
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

    ctx.barrier()
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
