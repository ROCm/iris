#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
# Copyright (c) 2025 Advanced Micro Devices, Inc. All rights reserved.
"""
Example: GEMM All-Reduce Ring-Based (K-split)

Demonstrates: Each rank computes a partial-K GEMM (partial A x partial B),
then a separate ring-based all-reduce kernel sums partial results across all
ranks using a ring topology. GEMM and communication run on separate streams.

Run with:
    torchrun --nproc_per_node=2 --standalone example.py --validate
"""

import argparse
import os

import torch
import torch.distributed as dist
import triton

from matmul_wrapper import matmul
from gemm_all_reduce_ring_based import persistent_all_reduce

import iris


def parse_args():
    parser = argparse.ArgumentParser(
        description="GEMM All-Reduce Ring-Based example.",
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
    parser.add_argument("--BLK_M", type=int, default=128, help="Block size M")
    parser.add_argument("--BLK_N", type=int, default=128, help="Block size N")
    parser.add_argument("--BLK_K", type=int, default=64, help="Block size K")
    parser.add_argument("--gsize_m", type=int, default=6, help="L2-cache locality swizzle parameter")
    parser.add_argument("--num_stages", type=int, default=2, help="Number of stages")
    parser.add_argument("--gemm_sms", type=int, default=256, help="Number of SMs for GEMM")
    parser.add_argument("--comm_sms", type=int, default=48, help="Number of SMs for all-reduce kernel")
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
    num_xcds = iris.hip.get_num_xcc()

    gemm_sms = args["gemm_sms"]
    comm_sms = args["comm_sms"]

    # dtype
    dtype_map = {"fp16": torch.float16, "fp32": torch.float32, "bf16": torch.bfloat16}
    datatype = dtype_map[args["datatype"]]

    assert args["k"] % world_size == 0, f"K ({args['k']}) must be divisible by world size ({world_size})."

    # Allocate full matrices
    A = ctx.randn(args["m"], args["k"], device="cuda", dtype=datatype)
    B = ctx.randn(args["n"], args["k"], device="cuda", dtype=datatype).T

    M = args["m"]
    N = args["n"]
    K = args["k"]

    # K-split: each rank computes partial-A x partial-B
    rows_per_gpu = K // world_size
    start_row = rank * rows_per_gpu
    end_row = start_row + rows_per_gpu
    local_A = A[:, start_row:end_row]
    local_B = B[start_row:end_row, :]

    C = ctx.zeros((M, N), device="cuda", dtype=datatype)
    local_C = ctx.zeros((M, N), device="cuda", dtype=torch.float32)

    total_blocks_M = triton.cdiv(M, args["BLK_M"])
    total_blocks_N = triton.cdiv(N, args["BLK_N"])
    total_tiles = total_blocks_M * total_blocks_N

    locks = ctx.zeros((total_tiles,), device="cuda", dtype=torch.int32)
    flags = ctx.zeros((total_tiles,), device="cuda", dtype=torch.int32)
    ring_buffer = ctx.zeros_like(C, dtype=torch.float32)

    bias = None

    gemm_stream = torch.cuda.Stream()
    comm_stream = torch.cuda.Stream()

    def reset_buffers():
        ctx.barrier()
        locks.zero_()
        flags.zero_()
        ring_buffer.zero_()
        ctx.barrier()

    def run():
        ctx.barrier()
        with torch.cuda.stream(gemm_stream):
            matmul._call(
                local_A,
                local_B,
                local_C,
                bias,
                locks,
                rank,
                world_size,
                gemm_sms,
                args["BLK_M"],
                args["BLK_N"],
                args["BLK_K"],
                args["gsize_m"],
                args["num_stages"],
                ctx.get_heap_bases(),
                "gfx942",
            )
        with torch.cuda.stream(comm_stream):
            persistent_all_reduce[(comm_sms,)](
                C,
                local_C,
                ring_buffer,
                locks,
                flags,
                M,
                N,
                C.stride(0),
                C.stride(1),
                args["BLK_M"],
                args["BLK_N"],
                args["gsize_m"],
                comm_sms,
                num_xcds,
                ctx.get_heap_bases(),
                rank,
                world_size,
            )
        torch.cuda.synchronize()
        ctx.barrier()

    # Warmup
    run()
    reset_buffers()

    # Run for validation
    run()

    # Validate
    if args["validate"]:
        expected = A @ B
        if torch.allclose(C, expected, atol=2.0):
            if rank == 0:
                print("Validation PASSED")
        else:
            max_diff = (C - expected).abs().max().item()
            print(f"Rank {rank}: Validation FAILED (max diff: {max_diff})")

    ctx.barrier()
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
