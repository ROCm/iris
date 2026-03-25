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
    from_rank: tl.constexpr,
    to_rank: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
    heap_bases: tl.tensor,
    load_cache_modifier: tl.constexpr,
    store_cache_modifier: tl.constexpr,
):
    pid = tl.program_id(0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < BLOCK_SIZE
    iris.put(
        data + offsets,
        results + offsets,
        from_rank,
        to_rank,
        heap_bases,
        mask=mask,
        load_cache_modifier=load_cache_modifier,
        store_cache_modifier=store_cache_modifier,
    )


# Define cache modifiers for load and store operations
LOAD_CACHE_MODIFIERS = [None, "", ".ca", ".cg", ".cv"]
STORE_CACHE_MODIFIERS = [None, "", ".wb", ".cg", ".cs", ".wt"]


@pytest.mark.parametrize(
    "load_cache_modifier,store_cache_modifier", list(product(LOAD_CACHE_MODIFIERS, STORE_CACHE_MODIFIERS))
)
def test_put_cache_modifiers_local(load_cache_modifier, store_cache_modifier):
    """Test local put (from_rank == to_rank) with various cache modifiers."""
    shmem = iris.iris(1 << 20)
    heap_bases = shmem.get_heap_bases()
    cur_rank = shmem.get_rank()

    BLOCK_SIZE = 16
    data = shmem.ones(BLOCK_SIZE, dtype=torch.float32)
    results = shmem.zeros_like(data)

    shmem.barrier()

    grid = lambda meta: (1,)
    put_kernel[grid](
        data, results, cur_rank, cur_rank, BLOCK_SIZE, heap_bases, load_cache_modifier, store_cache_modifier
    )
    shmem.barrier()

    expected = torch.ones(BLOCK_SIZE, dtype=torch.float32, device="cuda")
    try:
        torch.testing.assert_close(results, expected, rtol=0, atol=0)
    except AssertionError as e:
        print(
            f"LOCAL PUT test failed with load_cache_modifier={load_cache_modifier}, store_cache_modifier={store_cache_modifier}"
        )
        print(e)
        raise


@pytest.mark.parametrize(
    "load_cache_modifier,store_cache_modifier", list(product(LOAD_CACHE_MODIFIERS, STORE_CACHE_MODIFIERS))
)
def test_put_cache_modifiers_remote(load_cache_modifier, store_cache_modifier):
    """Test remote put (from_rank != to_rank) with various cache modifiers."""
    shmem = iris.iris(1 << 20)
    heap_bases = shmem.get_heap_bases()
    num_ranks = shmem.get_num_ranks()
    cur_rank = shmem.get_rank()

    if num_ranks < 2:
        pytest.skip("Remote put test requires at least 2 ranks")

    BLOCK_SIZE = 16
    data = shmem.ones(BLOCK_SIZE, dtype=torch.float32)
    results = shmem.zeros(BLOCK_SIZE, dtype=torch.float32)

    shmem.barrier()

    # rank 0 puts to rank 1
    remote_rank = (cur_rank + 1) % num_ranks
    grid = lambda meta: (1,)
    if cur_rank == 0:
        put_kernel[grid](
            data, results, cur_rank, remote_rank, BLOCK_SIZE, heap_bases, load_cache_modifier, store_cache_modifier
        )

    shmem.barrier()

    # rank 1 checks the data it received from rank 0
    if cur_rank == 1:
        expected = torch.ones(BLOCK_SIZE, dtype=torch.float32, device="cuda")
        try:
            torch.testing.assert_close(results, expected, rtol=0, atol=0)
        except AssertionError as e:
            print(
                f"REMOTE PUT test failed with load_cache_modifier={load_cache_modifier}, store_cache_modifier={store_cache_modifier}"
            )
            print(e)
            raise
