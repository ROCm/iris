# SPDX-License-Identifier: MIT
# Copyright (c) 2025 Advanced Micro Devices, Inc. All rights reserved.

"""
Example: iris.ccl.reduce_scatter

Reduces tensors across all ranks; each rank stores only its assigned tiles.
Input and output both have shape (M, N), but only the tiles assigned to each rank
are written in the output (tile-based assignment, not a contiguous scatter).

Run with:
    torchrun --nproc_per_node=<num_gpus> --standalone example.py
"""

import gc
import os

import torch
import torch.distributed as dist

import iris
from iris.ccl import Config


def main():
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    torch.cuda.set_device(local_rank)
    dist.init_process_group(backend="nccl")

    shmem = iris.iris(heap_size=2**31)
    rank = shmem.get_rank()
    world_size = shmem.get_num_ranks()

    M, N = 1024, 512
    dtype = torch.float16

    # Each rank fills its tensor with (rank + 1)
    input_tensor = shmem.zeros((M, N), dtype=dtype)
    input_tensor.fill_(float(rank + 1))

    output_tensor = shmem.zeros((M, N), dtype=dtype)

    config = Config(block_size_m=32, block_size_n=64)

    shmem.barrier()
    shmem.ccl.reduce_scatter(output_tensor, input_tensor, config=config)
    torch.cuda.synchronize()

    # Expected sum for any tile = 1 + 2 + ... + world_size
    expected_sum = float(world_size * (world_size + 1) // 2)

    # Determine tile assignment for this rank (block distribution)
    num_pid_m = (M + config.block_size_m - 1) // config.block_size_m
    num_pid_n = (N + config.block_size_n - 1) // config.block_size_n
    total_tiles = num_pid_m * num_pid_n
    tiles_per_rank = (total_tiles + world_size - 1) // world_size
    start_tile = rank * tiles_per_rank
    end_tile = min(start_tile + tiles_per_rank, total_tiles)

    for tile_id in range(start_tile, end_tile):
        pid_m = tile_id // num_pid_n
        pid_n = tile_id % num_pid_n
        m0 = pid_m * config.block_size_m
        m1 = min(m0 + config.block_size_m, M)
        n0 = pid_n * config.block_size_n
        n1 = min(n0 + config.block_size_n, N)
        tile = output_tensor[m0:m1, n0:n1]
        assert torch.allclose(tile, torch.full_like(tile, expected_sum), atol=0.5), (
            f"Rank {rank}: reduce_scatter tile ({pid_m},{pid_n}) mismatch. "
            f"Got {tile[0, 0].item():.1f}, expected {expected_sum:.1f}"
        )

    if rank == 0:
        print(f"iris.ccl.reduce_scatter: {world_size} ranks, shape ({M}, {N}), dtype {dtype}")
        print(f"  Expected tile sum = {expected_sum:.0f}")
        print(f"  Rank 0 assigned tiles {start_tile}..{end_tile - 1} ✓")

    shmem.barrier()
    del shmem
    gc.collect()
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
