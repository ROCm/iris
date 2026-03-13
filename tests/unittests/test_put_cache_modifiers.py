# SPDX-License-Identifier: MIT
# Copyright (c) 2025 Advanced Micro Devices, Inc. All rights reserved.

import torch
import triton
import triton.language as tl
import pytest
import iris
from itertools import product


@triton.jit
def put_kernel(
    data,
    results,
    cur_rank: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
    heap_bases: tl.tensor,
    load_cache_modifier: tl.constexpr,
    store_cache_modifier: tl.constexpr,
):
    pid = tl.program_id(0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < BLOCK_SIZE

    # Put data locally (same rank) with cache modifiers.
    # store_cache_modifier only applies to local stores (from_rank == to_rank).
    # Remote stores do not support cache modifiers.
    if load_cache_modifier is None and store_cache_modifier is None:
        iris.put(data + offsets, results + offsets, cur_rank, cur_rank, heap_bases, mask=mask)
    elif load_cache_modifier is None:
        iris.put(
            data + offsets,
            results + offsets,
            cur_rank,
            cur_rank,
            heap_bases,
            mask=mask,
            store_cache_modifier=store_cache_modifier,
        )
    elif store_cache_modifier is None:
        iris.put(
            data + offsets,
            results + offsets,
            cur_rank,
            cur_rank,
            heap_bases,
            mask=mask,
            load_cache_modifier=load_cache_modifier,
        )
    else:
        iris.put(
            data + offsets,
            results + offsets,
            cur_rank,
            cur_rank,
            heap_bases,
            mask=mask,
            load_cache_modifier=load_cache_modifier,
            store_cache_modifier=store_cache_modifier,
        )


# Define cache modifiers for load and store operations
LOAD_CACHE_MODIFIERS = [None, "", ".ca", ".cg", ".cv"]
# store_cache_modifier is only effective for local stores (from_rank == to_rank)
STORE_CACHE_MODIFIERS = [None, "", ".wb", ".cg", ".cs", ".wt"]


@pytest.mark.parametrize(
    "load_cache_modifier,store_cache_modifier", list(product(LOAD_CACHE_MODIFIERS, STORE_CACHE_MODIFIERS))
)
def test_put_cache_modifiers(load_cache_modifier, store_cache_modifier):
    """Test put (local copy) with various cache modifiers.

    store_cache_modifier is only effective for local stores (from_rank == to_rank).
    Remote stores do not support cache modifiers.
    This test exercises only local puts to verify cache modifier behavior.
    """
    shmem = iris.iris(1 << 20)
    heap_bases = shmem.get_heap_bases()
    cur_rank = shmem.get_rank()

    BLOCK_SIZE = 16
    data = shmem.ones(BLOCK_SIZE, dtype=torch.float32)
    results = shmem.zeros_like(data)

    shmem.barrier()

    grid = lambda meta: (1,)
    put_kernel[grid](data, results, cur_rank, BLOCK_SIZE, heap_bases, load_cache_modifier, store_cache_modifier)
    shmem.barrier()

    # Verify the result - should have the data that was put (local copy)
    expected = torch.ones(BLOCK_SIZE, dtype=torch.float32, device="cuda")

    try:
        torch.testing.assert_close(results, expected, rtol=0, atol=0)
    except AssertionError as e:
        print(
            f"PUT test failed with load_cache_modifier={load_cache_modifier}, store_cache_modifier={store_cache_modifier}"
        )
        print(e)
        print("Expected:", expected)
        print("Actual:", results)
        raise
