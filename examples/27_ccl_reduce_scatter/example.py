#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
# Copyright (c) 2025 Advanced Micro Devices, Inc. All rights reserved.

"""
Example: iris.ccl.reduce_scatter

Tile-based reduce-scatter: all ranks reduce their inputs; each rank stores only its assigned tiles.

Run with:
    torchrun --nproc_per_node=<num_gpus> --standalone example.py [--validate]
"""

import argparse
import os

import torch
import torch.distributed as dist

import iris
from iris.ccl import Config


def parse_args():
    parser = argparse.ArgumentParser(
        description="CCL reduce-scatter example",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("-m", type=int, default=1024, help="Number of rows")
    parser.add_argument("-n", type=int, default=512, help="Number of columns")
    parser.add_argument("--heap_size", type=int, default=1 << 31, help="Iris heap size")
    parser.add_argument("--datatype", type=str, default="fp16", choices=["fp16", "fp32", "bf16"], help="Data type")
    parser.add_argument("-v", "--validate", action="store_true", help="Validate output against reference")
    return vars(parser.parse_args())


def main():
    args = parse_args()

    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    torch.cuda.set_device(local_rank)
    dist.init_process_group(backend="nccl")

    ctx = iris.iris(heap_size=args["heap_size"])
    rank = ctx.get_rank()
    world_size = ctx.get_num_ranks()

    dtype_map = {"fp16": torch.float16, "fp32": torch.float32, "bf16": torch.bfloat16}
    dtype = dtype_map[args["datatype"]]
    M, N = args["m"], args["n"]

    # Each rank fills its input with (rank + 1)
    input_tensor = ctx.zeros((M, N), dtype=dtype)
    input_tensor.fill_(float(rank + 1))
    output_tensor = ctx.zeros((M, N), dtype=dtype)

    config = Config(block_size_m=32, block_size_n=64)

    ctx.barrier()
    ctx.ccl.reduce_scatter(output_tensor, input_tensor, config=config)
    torch.cuda.synchronize()

    if rank == 0:
        ctx.info(f"reduce_scatter: world_size={world_size}, shape=({M},{N}), dtype={dtype}")

    if args["validate"]:
        expected_sum = float(world_size * (world_size + 1) // 2)
        num_pid_m = (M + config.block_size_m - 1) // config.block_size_m
        num_pid_n = (N + config.block_size_n - 1) // config.block_size_n
        total_tiles = num_pid_m * num_pid_n
        tiles_per_rank = (total_tiles + world_size - 1) // world_size
        start_tile = rank * tiles_per_rank
        end_tile = min(start_tile + tiles_per_rank, total_tiles)
        for tile_id in range(start_tile, end_tile):
            pid_m, pid_n = tile_id // num_pid_n, tile_id % num_pid_n
            m0, m1 = pid_m * config.block_size_m, min((pid_m + 1) * config.block_size_m, M)
            n0, n1 = pid_n * config.block_size_n, min((pid_n + 1) * config.block_size_n, N)
            tile = output_tensor[m0:m1, n0:n1]
            assert torch.allclose(tile, torch.full_like(tile, expected_sum), atol=0.5), (
                f"Rank {rank}: tile ({pid_m},{pid_n}) mismatch. "
                f"Got {tile[0, 0].item():.1f}, expected {expected_sum:.1f}"
            )
        if rank == 0:
            ctx.info(f"Validation passed: expected tile sum = {expected_sum:.0f}")

    ctx.barrier()
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
