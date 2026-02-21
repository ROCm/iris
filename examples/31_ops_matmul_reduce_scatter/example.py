# SPDX-License-Identifier: MIT
# Copyright (c) 2025 Advanced Micro Devices, Inc. All rights reserved.

"""
Example: iris.ops.matmul_reduce_scatter

Fused matrix multiplication and reduce-scatter.
Computes C = A @ B across all ranks and reduces, assigning each rank its tiles.
This is a tile-based reduce-scatter: every rank computes the full GEMM but only
the tiles assigned to it are reduced in the output.

Run with:
    torchrun --nproc_per_node=<num_gpus> --standalone example.py
"""

import os

import torch
import torch.distributed as dist

import iris
from iris.ops import FusedConfig


def run(shmem):
    rank = shmem.get_rank()
    world_size = shmem.get_num_ranks()

    M, K, N = 512, 256, 512
    dtype = torch.float16

    torch.manual_seed(42)
    A = shmem.randn((M, K), dtype=dtype)
    B = shmem.randn((K, N), dtype=dtype)
    output = shmem.zeros((M, N), dtype=dtype)

    config = FusedConfig(block_size_m=64, block_size_n=64, block_size_k=32)

    shmem.barrier()
    shmem.ops.matmul_reduce_scatter(output, A, B, config=config)
    torch.cuda.synchronize()

    # Reference: local GEMM + all_reduce (semantically equivalent for assigned tiles)
    C_ref = torch.matmul(A.clone().float(), B.clone().float()).to(dtype)
    dist.all_reduce(C_ref, op=dist.ReduceOp.SUM)
    torch.cuda.synchronize()

    # Verify only the tiles assigned to this rank
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
        iris_tile = output[m0:m1, n0:n1]
        ref_tile = C_ref[m0:m1, n0:n1]
        assert torch.allclose(iris_tile.float(), ref_tile.float(), atol=1.0, rtol=0.05), (
            f"Rank {rank}: matmul_reduce_scatter tile ({pid_m},{pid_n}) mismatch. "
            f"Max diff: {(iris_tile.float() - ref_tile.float()).abs().max().item():.4f}"
        )

    if rank == 0:
        print(f"iris.ops.matmul_reduce_scatter: {world_size} ranks, A ({M},{K}), B ({K},{N}), dtype {dtype}")
        print(f"  Rank 0 verified tiles {start_tile}..{end_tile - 1} ✓")


def main():
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    torch.cuda.set_device(local_rank)
    dist.init_process_group(backend="nccl")

    shmem = iris.iris(heap_size=2**31)
    try:
        run(shmem)
    finally:
        shmem.barrier()
        del shmem
        import gc

        gc.collect()
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
