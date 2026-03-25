# SPDX-License-Identifier: MIT
# Copyright (c) 2025 Advanced Micro Devices, Inc. All rights reserved.

import torch
import triton
import triton.language as tl
import pytest
import iris
from itertools import product


@triton.jit
def load_kernel(
    data,
    results,
    source_rank: tl.constexpr,
    num_ranks: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
    heap_bases: tl.tensor,
    cache_modifier: tl.constexpr,
    volatile: tl.constexpr,
):
    pid = tl.program_id(0)

    partner = int((source_rank + num_ranks // 2) % num_ranks)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < BLOCK_SIZE

    result = iris.load(
        data + offsets,
        source_rank,
        partner,
        heap_bases,
        mask=mask,
        cache_modifier=cache_modifier,
        volatile=volatile,
    )

    tl.store(results + offsets, result, mask=mask)


# Define cache modifiers and volatile options
CACHE_MODIFIERS = [None, "", ".ca", ".cg", ".cv"]
VOLATILE_OPTIONS = [False, True]


@pytest.mark.parametrize("cache_modifier,volatile", list(product(CACHE_MODIFIERS, VOLATILE_OPTIONS)))
def test_load_cache_modifiers(cache_modifier, volatile):
    """Test load with various cache modifiers and volatile settings."""
    shmem = iris.iris(1 << 20)
    num_ranks = shmem.get_num_ranks()
    heap_bases = shmem.get_heap_bases()
    source_rank = shmem.get_rank()
    partner = int((source_rank + num_ranks // 2) % num_ranks)

    BLOCK_SIZE = 16
    data = shmem.full((BLOCK_SIZE,), source_rank, dtype=torch.float32)
    results = shmem.zeros_like(data)

    shmem.barrier()

    grid = lambda meta: (1,)
    load_kernel[grid](data, results, source_rank, num_ranks, BLOCK_SIZE, heap_bases, cache_modifier, volatile)
    shmem.barrier()

    # Verify the result - should have loaded from partner rank
    expected = torch.ones(BLOCK_SIZE, dtype=torch.float32, device="cuda") * partner

    try:
        torch.testing.assert_close(results, expected, rtol=0, atol=0)
    except AssertionError as e:
        print(e)
        print("Expected:", expected)
        print("Actual:", results)
        raise
