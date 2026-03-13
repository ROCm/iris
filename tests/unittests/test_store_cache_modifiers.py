# SPDX-License-Identifier: MIT
# Copyright (c) 2025 Advanced Micro Devices, Inc. All rights reserved.

import torch
import triton
import triton.language as tl
import pytest
import iris


@triton.jit
def kernel(
    data,
    results,
    destination_rank: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
    heap_bases: tl.tensor,
    cache_modifier: tl.constexpr,
):
    pid = tl.program_id(0)

    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)

    mask = offsets < BLOCK_SIZE

    # Load the data from src for this block
    value = tl.load(data + offsets, mask=mask)

    # Store data locally (same rank) with the specified cache modifier.
    # Cache modifiers only apply to local stores; remote stores do not support them.
    if cache_modifier is None:
        iris.store(results + offsets, value, destination_rank, destination_rank, heap_bases, mask=mask)
    else:
        iris.store(
            results + offsets,
            value,
            destination_rank,
            destination_rank,
            heap_bases,
            mask=mask,
            cache_modifier=cache_modifier,
        )


# Define cache modifiers for store operations
CACHE_MODIFIERS = [None, "", ".wb", ".cg", ".cs", ".wt"]


@pytest.mark.parametrize("cache_modifier", CACHE_MODIFIERS)
def test_store_cache_modifiers(cache_modifier):
    """Test local store with various cache modifiers.

    Cache modifiers are only effective for local stores (from_rank == to_rank).
    Remote stores do not support cache modifiers.
    """
    shmem = iris.iris(1 << 20)
    heap_bases = shmem.get_heap_bases()
    destination_rank = shmem.get_rank()

    BLOCK_SIZE = 16
    src = shmem.ones(BLOCK_SIZE, dtype=torch.float32)
    results = shmem.zeros_like(src)

    shmem.barrier()

    grid = lambda meta: (1,)
    kernel[grid](src, results, destination_rank, BLOCK_SIZE, heap_bases, cache_modifier)
    shmem.barrier()

    # Verify the result
    expected = torch.ones(BLOCK_SIZE, dtype=torch.float32, device="cuda")

    try:
        torch.testing.assert_close(results, expected, rtol=0, atol=0)
    except AssertionError as e:
        print(e)
        print("Expected:", expected)
        print("Actual:", results)
        raise
