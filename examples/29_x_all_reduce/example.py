#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
# Copyright (c) 2025 Advanced Micro Devices, Inc. All rights reserved.

"""
Example: iris.x all-reduce (atomic)

Demonstrates: iris.x.Tile, iris.x.make_tensor_view, iris.x.all_reduce_atomic

Each rank fills an (M, N) tensor with (rank + 1), then all-reduces into a shared
output tensor. The result on every rank is the element-wise sum:
    expected = world_size * (world_size + 1) / 2

Other all-reduce variants available in iris.x:
    all_reduce_ring, all_reduce_two_shot, all_reduce_one_shot, all_reduce_spinlock

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
        description="iris.x all-reduce (atomic) example",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("-m", type=int, default=1024, help="Number of rows")
    parser.add_argument("-n", type=int, default=512, help="Number of columns")
    parser.add_argument("--heap_size", type=int, default=1 << 31, help="Iris heap size")
    parser.add_argument("--datatype", type=str, default="fp32", choices=["fp16", "fp32", "bf16"], help="Data type")
    parser.add_argument("--block_size_m", type=int, default=64, help="Block size for M dimension")
    parser.add_argument("--block_size_n", type=int, default=64, help="Block size for N dimension")
    parser.add_argument("--num_sms", type=int, default=128, help="Number of persistent SMs")
    parser.add_argument("-v", "--validate", action="store_true", help="Validate output against reference")
    return vars(parser.parse_args())


@triton.jit
def all_reduce_kernel(
    input_ptr,
    output_ptr,
    device_ctx_tensor,
    M,
    N,
    stride_in_m,
    stride_in_n,
    stride_out_m,
    stride_out_n,
    RANK: tl.constexpr,
    WORLD_SIZE: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    NUM_SMS: tl.constexpr,
):
    dev_ctx = Context.initialize(device_ctx_tensor, RANK, WORLD_SIZE)

    num_pid_m = tl.cdiv(M, BLOCK_M)
    num_pid_n = tl.cdiv(N, BLOCK_N)
    total_tiles = num_pid_m * num_pid_n

    pid = tl.program_id(0)
    for tile_id in range(pid, total_tiles, NUM_SMS):
        pid_m = tile_id // num_pid_n
        pid_n = tile_id % num_pid_n

        # Load tile from local input
        rm = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
        rn = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
        mask = (rm[:, None] < M) & (rn[None, :] < N)
        in_offsets = rm[:, None] * stride_in_m + rn[None, :] * stride_in_n
        data = tl.load(input_ptr + in_offsets, mask=mask, other=0.0)

        # Wrap as iris.x Tile and all-reduce to output
        tile = iris.x.Tile(pid_m, pid_n, BLOCK_M, BLOCK_N, data)
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
    M, N = args["m"], args["n"]

    input_tensor = ctx.zeros((M, N), dtype=dtype)
    input_tensor.fill_(float(rank + 1))
    output_tensor = ctx.zeros((M, N), dtype=dtype)

    ctx.barrier()

    grid = (args["num_sms"],)
    all_reduce_kernel[grid](
        input_tensor,
        output_tensor,
        device_ctx,
        M,
        N,
        input_tensor.stride(0),
        input_tensor.stride(1),
        output_tensor.stride(0),
        output_tensor.stride(1),
        RANK=rank,
        WORLD_SIZE=world_size,
        BLOCK_M=args["block_size_m"],
        BLOCK_N=args["block_size_n"],
        NUM_SMS=args["num_sms"],
    )
    torch.cuda.synchronize()

    if rank == 0:
        ctx.info(f"all_reduce_atomic: world_size={world_size}, shape=({M},{N}), dtype={dtype}")

    if args["validate"]:
        expected = float(world_size * (world_size + 1) // 2)
        assert torch.allclose(output_tensor, torch.full_like(output_tensor, expected), atol=1.0), (
            f"Rank {rank}: mismatch. Got {output_tensor[0, 0].item():.1f}, expected {expected:.1f}"
        )
        if rank == 0:
            ctx.info(f"Validation passed: output[0,0] = {output_tensor[0, 0].item():.1f}")

    ctx.barrier()
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
