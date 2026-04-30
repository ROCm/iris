#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
# Copyright (c) 2025 Advanced Micro Devices, Inc. All rights reserved.
"""
Example: GEMM + Reduce-Scatter with Workgroup Specialization (K-split)

Demonstrates: A fused kernel where GEMM and reduce-scatter run concurrently
via workgroup specialization. GEMM WGs compute tiles of C = A_local @ B_local,
signal completion via locks, and communication WGs consume completed tiles,
sending each M-chunk to the appropriate rank via iris.atomic_add.

Each rank starts with K/world_size columns of A and rows of B, computes
a partial (M, N) result, and reduce-scatters along M so each rank ends
up with (M/world_size, N).

Run with:
    torchrun --nproc_per_node=2 --standalone example.py --validate
"""
import argparse
import math
import os

import torch
import torch.distributed as dist
import triton

from matmul_wrapper import MatMulReduceScatterWgSpecialized as matmul

import iris


def parse_args():
    parser = argparse.ArgumentParser(
        description="GEMM + Reduce-Scatter WG Specialization example.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("-m", type=int, default=8192, help="Rows of A")
    parser.add_argument("-n", type=int, default=4608, help="Columns of B")
    parser.add_argument("-k", type=int, default=36864, help="Common dimension")
    parser.add_argument("-v", "--validate", action="store_true", help="Validate output")
    parser.add_argument(
        "--datatype",
        type=str,
        default="fp16",
        choices=["fp16", "fp32", "bf16"],
        help="Datatype of computation",
    )
    parser.add_argument("--BLK_M", type=int, default=256, help="Block size M")
    parser.add_argument("--BLK_N", type=int, default=128, help="Block size N")
    parser.add_argument("--BLK_K", type=int, default=64, help="Block size K")
    parser.add_argument("--gsize_m", type=int, default=6, help="L2-cache locality swizzle parameter")
    parser.add_argument("--num_stages", type=int, default=2, help="Number of stages")
    parser.add_argument("--gemm_sms", type=int, default=256, help="SMs dedicated to GEMM WGs")
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
    gemm_sms = args["gemm_sms"]
    num_sms = cu_count  # Total SMs: GEMM WGs + comm WGs

    assert K % world_size == 0, f"K ({K}) must be divisible by world size ({world_size})."
    assert M % world_size == 0, f"M ({M}) must be divisible by world size ({world_size})."

    # Allocate full matrices
    A = ctx.randn(M, K, device="cuda", dtype=datatype)
    B = ctx.randn(N, K, device="cuda", dtype=datatype).T

    # K-split: each rank gets a slice along K
    local_k = K // world_size
    local_A = A[:, rank * local_k : (rank + 1) * local_k].clone()
    local_B = B[rank * local_k : (rank + 1) * local_k, :].clone()

    m_per_rank = M // world_size

    # C: local GEMM output buffer (M, N)
    # C_global: reduce-scatter output (M/world_size, N) per rank
    C = ctx.zeros((M, N), device="cuda", dtype=datatype)
    C_global = ctx.zeros((m_per_rank, N), device="cuda", dtype=datatype)

    # Locks: one per GEMM tile, used for GEMM->comm WG synchronization
    total_blocks_M = triton.cdiv(M, args["BLK_M"])
    total_blocks_N = triton.cdiv(N, args["BLK_N"])
    total_tiles = total_blocks_M * total_blocks_N
    locks = ctx.zeros((total_tiles,), device="cuda", dtype=torch.int32)

    def reset_buffers():
        ctx.barrier()
        C.zero_()
        C_global.zero_()
        locks.zero_()
        ctx.barrier()

    # Warmup
    ctx.barrier()
    matmul._call(
        local_A, local_B, C, C_global, locks,
        rank, world_size, gemm_sms, num_sms,
        args["BLK_M"], args["BLK_N"], args["BLK_K"],
        args["gsize_m"], args["num_stages"],
        ctx.get_heap_bases(), "gfx942",
    )
    torch.cuda.synchronize()
    ctx.barrier()

    # Run again with fresh buffers for validation
    reset_buffers()
    matmul._call(
        local_A, local_B, C, C_global, locks,
        rank, world_size, gemm_sms, num_sms,
        args["BLK_M"], args["BLK_N"], args["BLK_K"],
        args["gsize_m"], args["num_stages"],
        ctx.get_heap_bases(), "gfx942",
    )
    torch.cuda.synchronize()
    ctx.barrier()

    # Validate
    if args["validate"]:
        expected_full = torch.matmul(A.float(), B.float()).to(datatype)
        expected_slice = expected_full[rank * m_per_rank : (rank + 1) * m_per_rank, :]
        if torch.allclose(C_global, expected_slice, atol=2.0):
            if rank == 0:
                print("Validation PASSED")
        else:
            max_diff = (C_global.float() - expected_slice.float()).abs().max().item()
            print(f"Rank {rank}: Validation FAILED (max diff: {max_diff})")

    ctx.barrier()
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
