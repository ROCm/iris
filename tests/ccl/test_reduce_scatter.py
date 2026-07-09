# SPDX-License-Identifier: MIT
# Copyright (c) 2025 Advanced Micro Devices, Inc. All rights reserved.

"""
Test suite for reduce-scatter collective operation (CCL API).
"""

import pytest
import torch
import torch.distributed as dist
import iris
from iris.ccl import Config


@pytest.mark.parametrize(
    "variant",
    [
        "two_shot",
        "inreg",
        "twophase",
        "auto",
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
    "M, N",
    [
        (1, 1024),  # Small, 1D-like
        (1, 65536),  # Medium 1D
        (1, 524288),  # 1MB in bf16
        (128, 128),  # Small 2D
        (256, 256),  # Medium 2D
    ],
)
def test_reduce_scatter(variant, dtype, M, N):
    """Test reduce-scatter by comparing against torch.distributed."""
    if not dist.is_initialized():
        pytest.skip("torch.distributed not initialized")

    heap_size = 2**33  # 8GB
    shmem = iris.iris(heap_size)
    rank = shmem.get_rank()
    world_size = shmem.get_num_ranks()

    # Skip inreg variant if world_size != 8 (hardcoded for 8 peers)
    if variant == "inreg" and world_size != 8:
        shmem.barrier()
        del shmem
        import gc

        gc.collect()
        pytest.skip("inreg variant only supports world_size=8")

    # Each rank fills with deterministic values
    pytorch_input = torch.full((M, N), float(rank + 1), dtype=dtype, device=f"cuda:{rank}")

    # PyTorch reference: all-reduce gives sum, then each rank owns its chunk
    pytorch_ref = pytorch_input.clone()
    shmem.barrier()
    dist.all_reduce(pytorch_ref, op=dist.ReduceOp.SUM)
    torch.cuda.synchronize()

    # For reduce-scatter, each rank gets 1/W of the tiles
    # iris reduce_scatter stores the assigned chunk into the output at the rank's position
    iris_input = shmem.zeros((M, N), dtype=dtype)
    iris_input.copy_(pytorch_input)
    iris_output = shmem.zeros((M, N), dtype=dtype)

    shmem.barrier()

    config = Config(
        reduce_scatter_variant=variant,
        block_size_m=min(32, M),
        block_size_n=min(64, N),
        all_reduce_distribution=1,
    )

    shmem.ccl.reduce_scatter(iris_output, iris_input, config=config)
    torch.cuda.synchronize()

    # For two_shot variant: output has reduced tiles at rank's assigned positions
    # For inreg/twophase: output has reduced chunk at rank's offset
    # Both should match the all-reduced reference at the rank's assigned tile positions
    total_tiles_m = (M + config.block_size_m - 1) // config.block_size_m
    total_tiles_n = (N + config.block_size_n - 1) // config.block_size_n
    total_tiles = total_tiles_m * total_tiles_n
    tiles_per_rank = (total_tiles + world_size - 1) // world_size

    atol = 1e-3 if dtype in (torch.float16, torch.bfloat16) else 1e-5

    try:
        if variant in ("inreg", "twophase") or (variant == "auto" and world_size == 8):
            # These variants write to a contiguous chunk at rank's offset
            numel = M * N
            chunk_per_rank = numel // world_size
            chunk_offset = rank * chunk_per_rank
            iris_chunk = iris_output.view(-1)[chunk_offset : chunk_offset + chunk_per_rank]
            ref_chunk = pytorch_ref.view(-1)[chunk_offset : chunk_offset + chunk_per_rank]
            max_diff = torch.abs(iris_chunk - ref_chunk).max().item()
            assert torch.allclose(iris_chunk, ref_chunk, atol=atol), (
                f"Rank {rank}, variant={variant}: max diff={max_diff}, "
                f"expected < {atol}"
            )
        else:
            # two_shot variant: check assigned tiles based on distribution
            start_tile = rank * tiles_per_rank
            for t in range(tiles_per_rank):
                tile_id = start_tile + t
                if tile_id >= total_tiles:
                    break
                pid_m = tile_id // total_tiles_n
                pid_n = tile_id % total_tiles_n
                m_start = pid_m * config.block_size_m
                m_end = min(m_start + config.block_size_m, M)
                n_start = pid_n * config.block_size_n
                n_end = min(n_start + config.block_size_n, N)
                iris_tile = iris_output[m_start:m_end, n_start:n_end]
                ref_tile = pytorch_ref[m_start:m_end, n_start:n_end]
                max_diff = torch.abs(iris_tile - ref_tile).max().item()
                assert torch.allclose(iris_tile, ref_tile, atol=atol), (
                    f"Rank {rank}, tile {tile_id}: max diff={max_diff}, "
                    f"expected < {atol}"
                )
    finally:
        shmem.barrier()
        del shmem
        import gc

        gc.collect()
