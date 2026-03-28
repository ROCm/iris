# SPDX-License-Identifier: MIT
# Copyright (c) 2025 Advanced Micro Devices, Inc. All rights reserved.

"""
Test suite for reduce-scatter collective operation.
"""

import pytest
import torch
import torch.distributed as dist
import iris
from iris.ccl import Config


def _swizzle_tile_to_pid(tile_id, num_pid_m, num_pid_n, group_size_m):
    """Replicate the kernel's swizzled tile-to-(pid_m, pid_n) mapping."""
    num_pid_in_group = group_size_m * num_pid_n
    group_id = tile_id // num_pid_in_group
    first_pid_m = group_id * group_size_m
    actual_group_size_m = min(num_pid_m - first_pid_m, group_size_m)
    pid_m = first_pid_m + ((tile_id % num_pid_in_group) % actual_group_size_m)
    pid_n = (tile_id % num_pid_in_group) // actual_group_size_m
    return pid_m, pid_n


@pytest.mark.parametrize(
    "variant",
    [
        "two_shot",
        "ring_chunked",
    ],
)
@pytest.mark.parametrize(
    "dtype",
    [
        torch.float16,
        torch.float32,
        torch.bfloat16,
    ],
)
@pytest.mark.parametrize(
    "M, N, block_size_m, block_size_n",
    [
        (128, 64, 32, 64),  # Small
        (256, 128, 64, 64),  # Medium
        (1024, 256, 32, 64),  # Larger
        (8192, 8192, 32, 64),  # Large
    ],
)
def test_reduce_scatter(variant, dtype, M, N, block_size_m, block_size_n):
    """Test reduce-scatter by verifying each rank's assigned tiles equal the sum across all ranks."""
    if not dist.is_initialized():
        pytest.skip("torch.distributed not initialized")

    heap_size = 2**33  # 8GB
    ctx = iris.iris(heap_size)
    rank = ctx.get_rank()
    world_size = ctx.get_num_ranks()

    # Each rank fills its input with its rank+1 value
    pytorch_input_tensor = torch.full((M, N), float(rank + 1), dtype=dtype, device=f"cuda:{rank}")

    iris_input_tensor = ctx.zeros((M, N), dtype=dtype)
    iris_input_tensor.copy_(pytorch_input_tensor)
    iris_output_tensor = ctx.zeros((M, N), dtype=dtype)

    ctx.barrier()
    config = Config(
        reduce_scatter_variant=variant,
        block_size_m=block_size_m,
        block_size_n=block_size_n,
        all_reduce_distribution=1,
    )
    ctx.ccl.reduce_scatter(iris_output_tensor, iris_input_tensor, config=config)
    torch.cuda.synchronize()

    # Expected sum: 1 + 2 + ... + world_size
    expected_sum = sum(float(r + 1) for r in range(world_size))

    # Compute tile assignment for this rank (block distribution, DISTRIBUTION=1)
    num_pid_m = (M + block_size_m - 1) // block_size_m
    num_pid_n = (N + block_size_n - 1) // block_size_n
    total_tiles = num_pid_m * num_pid_n
    tiles_per_rank = (total_tiles + world_size - 1) // world_size
    start_tile = rank * tiles_per_rank
    remaining = max(total_tiles - start_tile, 0)
    num_assigned = min(tiles_per_rank, remaining)

    atol = 1e-3 if dtype in (torch.float16, torch.bfloat16) else 1e-5

    try:
        for offset in range(num_assigned):
            tile_id = start_tile + offset
            pid_m, pid_n = _swizzle_tile_to_pid(tile_id, num_pid_m, num_pid_n, config.swizzle_size)

            m_start = pid_m * block_size_m
            m_end = min(m_start + block_size_m, M)
            n_start = pid_n * block_size_n
            n_end = min(n_start + block_size_n, N)

            tile_data = iris_output_tensor[m_start:m_end, n_start:n_end]
            expected_tile = torch.full_like(tile_data, expected_sum)

            assert torch.allclose(tile_data, expected_tile, atol=atol, rtol=0), (
                f"Rank {rank}, tile {tile_id} ({pid_m},{pid_n}), variant={variant}: "
                f"Expected {expected_sum}, got max={tile_data.max().item()}, "
                f"min={tile_data.min().item()}"
            )
    finally:
        ctx.barrier()
        del ctx
        import gc

        gc.collect()


@pytest.mark.parametrize(
    "distribution",
    [
        0,  # striding
        1,  # block
    ],
)
def test_reduce_scatter_two_shot_distribution(distribution, dtype=torch.float32, M=1024, N=256):
    """Test two-shot reduce-scatter with different distribution modes."""
    if not dist.is_initialized():
        pytest.skip("torch.distributed not initialized")

    heap_size = 2**33
    ctx = iris.iris(heap_size)
    rank = ctx.get_rank()
    world_size = ctx.get_num_ranks()

    pytorch_input_tensor = torch.full((M, N), float(rank + 1), dtype=dtype, device=f"cuda:{rank}")

    iris_input_tensor = ctx.zeros((M, N), dtype=dtype)
    iris_input_tensor.copy_(pytorch_input_tensor)
    iris_output_tensor = ctx.zeros((M, N), dtype=dtype)

    ctx.barrier()
    config = Config(
        reduce_scatter_variant="two_shot",
        all_reduce_distribution=distribution,
    )
    ctx.ccl.reduce_scatter(iris_output_tensor, iris_input_tensor, config=config)
    torch.cuda.synchronize()

    expected_sum = sum(float(r + 1) for r in range(world_size))
    block_size_m = config.block_size_m
    block_size_n = config.block_size_n
    num_pid_m = (M + block_size_m - 1) // block_size_m
    num_pid_n = (N + block_size_n - 1) // block_size_n
    total_tiles = num_pid_m * num_pid_n
    tiles_per_rank = (total_tiles + world_size - 1) // world_size

    if distribution == 0:
        # Striding: rank gets tiles rank, rank+world_size, rank+2*world_size, ...
        assigned_tiles = list(range(rank, total_tiles, world_size))
    else:
        # Block: rank gets contiguous tiles
        start_tile = rank * tiles_per_rank
        remaining = max(total_tiles - start_tile, 0)
        num_assigned = min(tiles_per_rank, remaining)
        assigned_tiles = list(range(start_tile, start_tile + num_assigned))

    atol = 1e-5

    try:
        for tile_id in assigned_tiles:
            pid_m, pid_n = _swizzle_tile_to_pid(tile_id, num_pid_m, num_pid_n, config.swizzle_size)

            m_start = pid_m * block_size_m
            m_end = min(m_start + block_size_m, M)
            n_start = pid_n * block_size_n
            n_end = min(n_start + block_size_n, N)

            tile_data = iris_output_tensor[m_start:m_end, n_start:n_end]
            expected_tile = torch.full_like(tile_data, expected_sum)

            assert torch.allclose(tile_data, expected_tile, atol=atol, rtol=0), (
                f"Rank {rank}, tile {tile_id}, distribution={distribution}: "
                f"Expected {expected_sum}, got max={tile_data.max().item()}, "
                f"min={tile_data.min().item()}"
            )
    finally:
        ctx.barrier()
        del ctx
        import gc

        gc.collect()


def test_reduce_scatter_ring_flags_too_small():
    """Test that ValueError is raised when ring flags array is too small for current tile count.

    Scenario: workspace is prepared with larger block sizes (fewer tiles), then reduce_scatter
    is called with smaller block sizes (more tiles). The undersized flags array is detected.
    """
    if not dist.is_initialized():
        pytest.skip("torch.distributed not initialized")

    heap_size = 2**33
    ctx = iris.iris(heap_size)

    M, N = 512, 512

    iris_input = ctx.zeros((M, N), dtype=torch.float32)
    iris_output = ctx.zeros((M, N), dtype=torch.float32)

    ctx.barrier()

    from iris.ccl.reduce_scatter import ReduceScatterWorkspace, _prepare_ring_workspace

    # Step 1: prepare workspace with larger block sizes (fewer tiles)
    config_large = Config(reduce_scatter_variant="ring_chunked", block_size_m=128, block_size_n=128)
    num_pid_m_large = (M + 128 - 1) // 128
    num_pid_n_large = (N + 128 - 1) // 128
    total_tiles_large = num_pid_m_large * num_pid_n_large

    workspace = ReduceScatterWorkspace()
    _prepare_ring_workspace(ctx, M, N, torch.float32, total_tiles_large, workspace)

    # Step 2: call reduce_scatter with smaller block sizes (more tiles)
    config_small = Config(reduce_scatter_variant="ring_chunked", block_size_m=64, block_size_n=64)
    with pytest.raises(ValueError, match="Flags array too small"):
        ctx.ccl.reduce_scatter(iris_output, iris_input, config=config_small, workspace=workspace)

    ctx.barrier()
    del ctx
    import gc

    gc.collect()
