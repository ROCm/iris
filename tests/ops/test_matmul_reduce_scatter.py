# SPDX-License-Identifier: MIT
# Copyright (c) 2025 Advanced Micro Devices, Inc. All rights reserved.

"""
Test suite for high-level matmul_reduce_scatter API.
"""

import pytest
import torch
import torch.distributed as dist
import iris
import iris.ops as ops


@pytest.mark.parametrize(
    "dtype, atol, rtol",
    [
        (torch.float16, 2e-1, 1e-2),
        (torch.float32, 1e-1, 1e-2),
    ],
)
@pytest.mark.parametrize("M, N, K", [(128, 128, 32)])
def test_matmul_reduce_scatter(dtype, atol, rtol, M, N, K):
    """
    Test matmul_reduce_scatter by comparing against torch matmul + all_reduce.

    Note: We use all_reduce for reference because our tile-based reduce_scatter
    is semantically equivalent to: matmul -> all_reduce -> each rank keeps assigned tiles.
    PyTorch's reduce_scatter operates on different semantics (scatter along dimensions).
    """
    if not dist.is_initialized():
        pytest.skip("torch.distributed not initialized")

    heap_size = 2**33
    shmem = iris.iris(heap_size)
    rank = shmem.get_rank()
    world_size = shmem.get_num_ranks()

    A = torch.randn(M, K, dtype=dtype, device=f"cuda:{rank}")
    B = torch.randn(K, N, dtype=dtype, device=f"cuda:{rank}")

    C_local = torch.matmul(A, B)
    C_reduced = C_local.clone()
    dist.all_reduce(C_reduced, op=dist.ReduceOp.SUM)
    torch.cuda.synchronize()

    config = ops.FusedConfig(block_size_m=64, block_size_n=64, block_size_k=32)
    num_pid_m = (M + config.block_size_m - 1) // config.block_size_m
    num_pid_n = (N + config.block_size_n - 1) // config.block_size_n
    total_tiles = num_pid_m * num_pid_n
    tiles_per_rank = total_tiles // world_size
    start_tile = rank * tiles_per_rank
    if rank == world_size - 1:
        tiles_per_rank = total_tiles - start_tile

    iris_A = shmem.zeros((M, K), dtype=dtype)
    iris_A.copy_(A)
    iris_B = shmem.zeros((K, N), dtype=dtype)
    iris_B.copy_(B)
    iris_C = shmem.zeros((M, N), dtype=dtype)

    shmem.barrier()

    ops.matmul_reduce_scatter(shmem, iris_C, iris_A, iris_B, config=config)

    torch.cuda.synchronize()
    shmem.barrier()

    # Adjust tolerance for 8 ranks due to accumulation error
    if world_size == 8 and dtype == torch.float32:
        atol = 2e-1

    for local_tile_idx in range(tiles_per_rank):
        tile_id = start_tile + local_tile_idx
        pid_m = tile_id // num_pid_n
        pid_n = tile_id % num_pid_n

        m_start = pid_m * config.block_size_m
        m_end = min(m_start + config.block_size_m, M)
        n_start = pid_n * config.block_size_n
        n_end = min(n_start + config.block_size_n, N)

        iris_tile = iris_C[m_start:m_end, n_start:n_end]
        ref_tile = C_reduced[m_start:m_end, n_start:n_end]

        max_diff = torch.abs(iris_tile - ref_tile).max().item()
        assert torch.allclose(iris_tile, ref_tile, atol=atol, rtol=rtol), (
            f"Rank {rank}, tile {tile_id} ({pid_m},{pid_n}): Max diff: {max_diff}, expected < {atol}"
        )

    if rank == 0:
        print(f"matmul_reduce_scatter: {dtype}, M={M}, N={N}, K={K}")

    shmem.barrier()
    del shmem
    import gc

    gc.collect()


@pytest.mark.parametrize(
    "dtype, atol, rtol",
    [
        (torch.float16, 2e-1, 1e-2),
        (torch.float32, 1e-1, 1e-2),
    ],
)
def test_matmul_reduce_scatter_semantics(dtype, atol, rtol):
    """
    Test that matmul_reduce_scatter is equivalent to:
    result = matmul(A, B)
    reduced = all_reduce(result)
    each rank keeps its assigned tile block
    """
    if not dist.is_initialized():
        pytest.skip("torch.distributed not initialized")

    heap_size = 2**33
    shmem = iris.iris(heap_size)
    rank = shmem.get_rank()
    world_size = shmem.get_num_ranks()

    M, N, K = 128, 128, 32

    A = shmem.randn((M, K), dtype=dtype)
    B = shmem.randn((K, N), dtype=dtype)
    output = shmem.zeros((M, N), dtype=dtype)

    A_ref = A.clone()
    B_ref = B.clone()
    C_ref = torch.matmul(A_ref, B_ref)
    dist.all_reduce(C_ref, op=dist.ReduceOp.SUM)
    torch.cuda.synchronize()

    config = ops.FusedConfig(block_size_m=64, block_size_n=64, block_size_k=32)
    from iris.ops.matmul_reduce_scatter import matmul_reduce_scatter

    matmul_reduce_scatter(shmem, output, A, B, config=config)

    torch.cuda.synchronize()
    shmem.barrier()

    num_pid_m = (M + config.block_size_m - 1) // config.block_size_m
    num_pid_n = (N + config.block_size_n - 1) // config.block_size_n
    total_tiles = num_pid_m * num_pid_n
    tiles_per_rank = total_tiles // world_size
    start_tile = rank * tiles_per_rank
    if rank == world_size - 1:
        tiles_per_rank = total_tiles - start_tile

    # Adjust tolerance for 8 ranks
    if world_size == 8 and dtype == torch.float32:
        atol = 2e-1

    for local_tile_idx in range(tiles_per_rank):
        tile_id = start_tile + local_tile_idx
        pid_m = tile_id // num_pid_n
        pid_n = tile_id % num_pid_n

        m_start = pid_m * config.block_size_m
        m_end = min(m_start + config.block_size_m, M)
        n_start = pid_n * config.block_size_n
        n_end = min(n_start + config.block_size_n, N)

        output_tile = output[m_start:m_end, n_start:n_end]
        ref_tile = C_ref[m_start:m_end, n_start:n_end]

        assert torch.allclose(output_tile, ref_tile, atol=atol, rtol=rtol), f"Rank {rank}, tile {tile_id}: mismatch"

    if rank == 0:
        print("matmul_reduce_scatter semantics verified")

    shmem.barrier()
    del shmem
    import gc

    gc.collect()
