#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
# Copyright (c) 2025 Advanced Micro Devices, Inc. All rights reserved.

"""
Example: Fused GEMM + iris.x all-reduce (atomic)

Demonstrates: iris.x.Tile, iris.x.make_tensor_view, iris.x.all_reduce_atomic, tl.dot

K-sharded GEMM with fused all-reduce: each rank holds A (M, K_local) and
B (K_local, N) and computes a partial C = A @ B. The partial products are
summed across all ranks via iris.x.all_reduce_atomic, producing the full
C = A_full @ B_full on every rank.

This replaces the old multi-file GEMM examples (08-18) with a single clean
iris.x implementation.

Run with:
    torchrun --nproc_per_node=<num_gpus> --standalone example.py [--validate]
"""

import argparse
import os

import torch
import torch.distributed as dist

import triton
import triton.language as tl

import iris
import iris.x
from iris.mem.triton.context import Context


def parse_args():
    parser = argparse.ArgumentParser(
        description="Fused GEMM + iris.x all-reduce example",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("-m", type=int, default=512, help="Rows of A / output")
    parser.add_argument("-n", type=int, default=256, help="Columns of B / output")
    parser.add_argument("--k_local", type=int, default=128, help="Inner dimension per rank (K_local)")
    parser.add_argument("--heap_size", type=int, default=1 << 31, help="Iris heap size")
    parser.add_argument("--datatype", type=str, default="fp16", choices=["fp16", "fp32", "bf16"], help="Data type")
    parser.add_argument("--block_size_m", type=int, default=128, help="Block size for M dimension")
    parser.add_argument("--block_size_n", type=int, default=128, help="Block size for N dimension")
    parser.add_argument("--block_size_k", type=int, default=64, help="Block size for K dimension")
    parser.add_argument("--num_sms", type=int, default=128, help="Number of persistent SMs")
    parser.add_argument("-v", "--validate", action="store_true", help="Validate output against reference")
    return vars(parser.parse_args())


@triton.jit
def gemm_all_reduce_kernel(
    A,
    B,
    output_ptr,
    device_ctx_tensor,
    M,
    N,
    K,
    stride_am,
    stride_ak,
    stride_bk,
    stride_bn,
    stride_out_m,
    stride_out_n,
    RANK: tl.constexpr,
    WORLD_SIZE: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
    NUM_SMS: tl.constexpr,
):
    dev_ctx = Context.initialize(device_ctx_tensor, RANK, WORLD_SIZE)

    num_pid_m = tl.cdiv(M, BLOCK_M)
    num_pid_n = tl.cdiv(N, BLOCK_N)
    total_tiles = num_pid_m * num_pid_n

    tl.assume(stride_am > 0)
    tl.assume(stride_ak > 0)
    tl.assume(stride_bk > 0)
    tl.assume(stride_bn > 0)

    pid = tl.program_id(0)
    for tile_id in range(pid, total_tiles, NUM_SMS):
        pid_m = tile_id // num_pid_n
        pid_n = tile_id % num_pid_n

        # Compute tile indices for A and B
        rm = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
        rn = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
        rm = tl.max_contiguous(tl.multiple_of(rm, BLOCK_M), BLOCK_M)
        rn = tl.max_contiguous(tl.multiple_of(rn, BLOCK_N), BLOCK_N)
        rk = tl.arange(0, BLOCK_K)

        # Pointers for A tile: (BLOCK_M, BLOCK_K)
        A_ptrs = A + rm[:, None] * stride_am + rk[None, :] * stride_ak
        # Pointers for B tile: (BLOCK_K, BLOCK_N)
        B_ptrs = B + rk[:, None] * stride_bk + rn[None, :] * stride_bn

        # Tiled GEMM accumulation
        acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
        for k_start in range(0, K, BLOCK_K):
            k_remaining = K - k_start
            a_mask = (rm[:, None] < M) & (rk[None, :] + k_start < K)
            b_mask = (rk[:, None] + k_start < K) & (rn[None, :] < N)

            a = tl.load(A_ptrs, mask=a_mask, other=0.0)
            b = tl.load(B_ptrs, mask=b_mask, other=0.0)
            acc += tl.dot(a, b)

            A_ptrs += BLOCK_K * stride_ak
            B_ptrs += BLOCK_K * stride_bk

        # Convert accumulator to output dtype
        c = acc.to(output_ptr.dtype.element_ty)

        # Fused all-reduce: atomically add partial GEMM result to all ranks
        tile = iris.x.Tile(pid_m, pid_n, BLOCK_M, BLOCK_N, c)
        dst_view = iris.x.make_tensor_view(output_ptr, M, N, stride_out_m, stride_out_n)
        iris.x.all_reduce_atomic(tile, dst_view, dev_ctx)


def main():
    args = parse_args()

    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    torch.cuda.set_device(local_rank)
    dist.init_process_group(backend="gloo")

    ctx = iris.iris(heap_size=args["heap_size"])
    rank = ctx.get_rank()
    world_size = ctx.get_num_ranks()
    device_ctx = ctx.get_device_context()

    dtype_map = {"fp16": torch.float16, "fp32": torch.float32, "bf16": torch.bfloat16}
    dtype = dtype_map[args["datatype"]]
    M, K_local, N = args["m"], args["k_local"], args["n"]

    # Each rank gets a different shard of A and B along K dimension
    torch.manual_seed(42 + rank)
    A = ctx.randn((M, K_local), dtype=dtype)
    B = ctx.randn((K_local, N), dtype=dtype)
    output = ctx.zeros((M, N), dtype=dtype)

    ctx.barrier()

    grid = (args["num_sms"],)
    gemm_all_reduce_kernel[grid](
        A,
        B,
        output,
        device_ctx,
        M,
        N,
        K_local,
        A.stride(0),
        A.stride(1),
        B.stride(0),
        B.stride(1),
        output.stride(0),
        output.stride(1),
        RANK=rank,
        WORLD_SIZE=world_size,
        BLOCK_M=args["block_size_m"],
        BLOCK_N=args["block_size_n"],
        BLOCK_K=args["block_size_k"],
        NUM_SMS=args["num_sms"],
    )
    torch.cuda.synchronize()

    if rank == 0:
        ctx.info(
            f"gemm_all_reduce: world_size={world_size}, "
            f"A=({M},{K_local}), B=({K_local},{N}), output=({M},{N}), dtype={dtype}"
        )

    if args["validate"]:
        # Gather full A and B across ranks, compute reference
        A_shards = [torch.zeros(M, K_local, dtype=dtype, device=A.device) for _ in range(world_size)]
        B_shards = [torch.zeros(K_local, N, dtype=dtype, device=B.device) for _ in range(world_size)]
        dist.all_gather(A_shards, A)
        dist.all_gather(B_shards, B)
        A_full = torch.cat(A_shards, dim=1)
        B_full = torch.cat(B_shards, dim=0)
        ref = torch.matmul(A_full.float(), B_full.float()).to(dtype)

        assert torch.allclose(output.float(), ref.float(), atol=1.0, rtol=0.05), (
            f"Rank {rank}: mismatch. Max diff: {(output.float() - ref.float()).abs().max().item():.4f}"
        )
        if rank == 0:
            ctx.info(f"Validation passed: output[0,0] = {output[0, 0].item():.4f}")

    ctx.barrier()
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
