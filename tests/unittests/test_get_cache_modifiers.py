# SPDX-License-Identifier: MIT
# Copyright (c) 2025 Advanced Micro Devices, Inc. All rights reserved.

import torch
import triton
import triton.language as tl
import pytest
import iris
from itertools import product


@triton.jit
def get_kernel(
    data,
    results,
    cur_rank: tl.constexpr,
    num_ranks: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
    heap_bases: tl.tensor,
    load_cache_modifier: tl.constexpr,
    store_cache_modifier: tl.constexpr,
):
    pid = tl.program_id(0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < BLOCK_SIZE

    acc = tl.zeros([BLOCK_SIZE], dtype=data.type.element_ty)

    # Loop over all ranks and get data with cache modifiers.
    # The load is remote when from_rank != cur_rank; the store to results is always local.
    for target_rank in range(num_ranks):
        iris.get(
            data + offsets,
            results + offsets,
            cur_rank,
            target_rank,
            heap_bases,
            mask=mask,
            load_cache_modifier=load_cache_modifier,
            store_cache_modifier=store_cache_modifier,
        )
        acc += tl.load(results + offsets, mask=mask)

    # Store the accumulated value back to the output
    tl.store(results + offsets, acc, mask=mask)


# Define cache modifiers for load and store operations
LOAD_CACHE_MODIFIERS = [None, "", ".ca", ".cg", ".cv"]
STORE_CACHE_MODIFIERS = [None, "", ".wb", ".cg", ".cs", ".wt"]


@pytest.mark.parametrize(
    "load_cache_modifier,store_cache_modifier", list(product(LOAD_CACHE_MODIFIERS, STORE_CACHE_MODIFIERS))
)
def test_get_cache_modifiers(load_cache_modifier, store_cache_modifier):
    """Test get (copy from other rank) with various cache modifiers.

    load_cache_modifier applies to the remote load when from_rank != to_rank.
    store_cache_modifier applies to the always-local store to to_ptr.
    """
    shmem = iris.iris(1 << 20)
    num_ranks = shmem.get_num_ranks()
    heap_bases = shmem.get_heap_bases()
    cur_rank = shmem.get_rank()

    BLOCK_SIZE = 16
    data = shmem.ones(BLOCK_SIZE, dtype=torch.float32)
    results = shmem.zeros_like(data)

    shmem.barrier()

    grid = lambda meta: (1,)
    get_kernel[grid](
        data, results, cur_rank, num_ranks, BLOCK_SIZE, heap_bases, load_cache_modifier, store_cache_modifier
    )
    shmem.barrier()

    # Verify the result - should get data from all ranks (including self)
    expected = torch.ones(BLOCK_SIZE, dtype=torch.float32, device="cuda") * num_ranks

    try:
        torch.testing.assert_close(results, expected, rtol=0, atol=0)
    except AssertionError as e:
        print(
            f"GET test failed with load_cache_modifier={load_cache_modifier}, store_cache_modifier={store_cache_modifier}"
        )
        print(e)
        print("Expected:", expected)
        print("Actual:", results)
        raise
