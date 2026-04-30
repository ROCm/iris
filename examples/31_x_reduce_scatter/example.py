#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
# Copyright (c) 2025 Advanced Micro Devices, Inc. All rights reserved.

"""
Example: iris.x reduce-scatter

Demonstrates: iris.x.Tile, iris.x.make_tensor_view, iris.x.reduce_scatter

Each rank fills an (M, N) input tensor with (rank + 1). The reduce-scatter
sums contributions from all ranks and distributes contiguous tile blocks
among ranks. Each rank writes only its assigned tiles to the output.

Requires a locks array for producer-consumer synchronization: each rank
stores its local data and signals readiness, then the responsible rank
gathers and reduces.

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
        description="iris.x reduce-scatter example",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("-m", type=int, default=1024, help="Number of rows")
    parser.add_argument("-n", type=int, default=512, help="Number of columns")
    parser.add_argument("--heap_size", type=int, default=1 << 31, help="Iris heap size")
    parser.add_argument("--datatype", type=str, default="fp16", choices=["fp16", "fp32", "bf16"], help="Data type")
    parser.add_argument("--block_size_m", type=int, default=64, help="Block size for M dimension")
    parser.add_argument("--block_size_n", type=int, default=64, help="Block size for N dimension")
    parser.add_argument("--num_sms", type=int, default=128, help="Number of persistent SMs")
    parser.add_argument("-v", "--validate", action="store_true", help="Validate output against reference")
    return vars(parser.parse_args())


@triton.jit
def reduce_scatter_kernel(
    input_ptr,
    temp_ptr,
    output_ptr,
    locks_ptr,
    device_ctx_tensor,
    M,
    N,
    stride_m,
    stride_n,
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
        offsets = rm[:, None] * stride_m + rn[None, :] * stride_n
        data = tl.load(input_ptr + offsets, mask=mask, other=0.0)

        # Store to temp buffer (shared across ranks) and signal readiness
        tl.store(temp_ptr + offsets, data, mask=mask)
        tl.debug_barrier()
        tl.atomic_xchg(locks_ptr + tile_id, 1, sem="release", scope="sys")

    # Synchronize to ensure all tiles are stored before reduce phase
    tl.debug_barrier()

    # Phase 2: Reduce assigned tiles from all ranks
    for tile_id in range(pid, total_tiles, NUM_SMS):
        pid_m = tile_id // num_pid_n
        pid_n = tile_id % num_pid_n

        rm = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
        rn = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
        mask = (rm[:, None] < M) & (rn[None, :] < N)
        offsets = rm[:, None] * stride_m + rn[None, :] * stride_n
        data = tl.load(temp_ptr + offsets, mask=mask, other=0.0)

        tile = iris.x.Tile(pid_m, pid_n, BLOCK_M, BLOCK_N, data)
        src_view = iris.x.make_tensor_view(temp_ptr, M, N, stride_m, stride_n)
        dst_view = iris.x.make_tensor_view(output_ptr, M, N, stride_m, stride_n)
        iris.x.reduce_scatter(tile, src_view, dst_view, locks_ptr, dev_ctx)


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
    BLOCK_M, BLOCK_N = args["block_size_m"], args["block_size_n"]

    input_tensor = ctx.zeros((M, N), dtype=dtype)
    input_tensor.fill_(float(rank + 1))
    temp_buffer = ctx.zeros((M, N), dtype=dtype)
    output_tensor = ctx.zeros((M, N), dtype=dtype)

    num_pid_m = (M + BLOCK_M - 1) // BLOCK_M
    num_pid_n = (N + BLOCK_N - 1) // BLOCK_N
    num_tiles = num_pid_m * num_pid_n
    locks = ctx.zeros((num_tiles,), dtype=torch.int32)

    ctx.barrier()

    grid = (args["num_sms"],)
    reduce_scatter_kernel[grid](
        input_tensor,
        temp_buffer,
        output_tensor,
        locks,
        device_ctx,
        M,
        N,
        input_tensor.stride(0),
        input_tensor.stride(1),
        RANK=rank,
        WORLD_SIZE=world_size,
        BLOCK_M=BLOCK_M,
        BLOCK_N=BLOCK_N,
        NUM_SMS=args["num_sms"],
    )
    torch.cuda.synchronize()

    if rank == 0:
        ctx.info(f"reduce_scatter: world_size={world_size}, shape=({M},{N}), dtype={dtype}")

    if args["validate"]:
        # Reference: gather all inputs, sum, then check assigned tiles
        ref_list = [torch.empty(M, N, dtype=dtype, device=input_tensor.device) for _ in range(world_size)]
        dist.all_gather(ref_list, input_tensor)
        full_reduced = sum(ref_list).float()

        total_tiles = num_pid_m * num_pid_n
        tiles_per_rank = total_tiles // world_size
        start_tile = rank * tiles_per_rank
        end_tile = total_tiles if rank == world_size - 1 else start_tile + tiles_per_rank

        pid_m_idx = torch.arange(M, device=output_tensor.device) // BLOCK_M
        pid_n_idx = torch.arange(N, device=output_tensor.device) // BLOCK_N
        tile_id = pid_m_idx[:, None] * num_pid_n + pid_n_idx[None, :]
        mask = (tile_id >= start_tile) & (tile_id < end_tile)

        out_float = output_tensor.float()
        assert torch.allclose(out_float[mask], full_reduced[mask], atol=0.6), (
            f"Rank {rank}: output mismatch on assigned tiles"
        )
        if rank == 0:
            ctx.info("Validation passed: output matches reference on assigned tiles")

    ctx.barrier()
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
