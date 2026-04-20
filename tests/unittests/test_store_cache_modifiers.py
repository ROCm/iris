# SPDX-License-Identifier: MIT
# Copyright (c) 2025 Advanced Micro Devices, Inc. All rights reserved.

import torch
import triton
import triton.language as tl
import pytest
import iris


@triton.jit
def local_store_kernel(
    data,
    results,
    cur_rank: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
    heap_bases: tl.tensor,
    cache_modifier: tl.constexpr,
):
    pid = tl.program_id(0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < BLOCK_SIZE
    value = tl.load(data + offsets, mask=mask)
    # Local store: from_rank == to_rank == cur_rank
    iris.store(results + offsets, value, cur_rank, cur_rank, heap_bases, mask=mask, cache_modifier=cache_modifier)


@triton.jit
def remote_store_kernel(
    data,
    results,
    from_rank: tl.constexpr,
    to_rank: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
    heap_bases: tl.tensor,
    cache_modifier: tl.constexpr,
):
    pid = tl.program_id(0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < BLOCK_SIZE
    value = tl.load(data + offsets, mask=mask)
    # Remote store: from_rank != to_rank
    iris.store(results + offsets, value, from_rank, to_rank, heap_bases, mask=mask, cache_modifier=cache_modifier)


# Define cache modifiers for store operations
CACHE_MODIFIERS = [None, "", ".wb", ".cg", ".cs", ".wt"]


@pytest.mark.parametrize("cache_modifier", CACHE_MODIFIERS)
def test_store_cache_modifiers_local(cache_modifier):
    """Test local store (from_rank == to_rank) with various cache modifiers."""
    shmem = iris.iris(1 << 20)
    heap_bases = shmem.get_heap_bases()
    cur_rank = shmem.get_rank()

    BLOCK_SIZE = 16
    src = shmem.ones(BLOCK_SIZE, dtype=torch.float32)
    results = shmem.zeros_like(src)

    shmem.barrier()

    grid = lambda meta: (1,)
    local_store_kernel[grid](src, results, cur_rank, BLOCK_SIZE, heap_bases, cache_modifier)
    shmem.barrier()

    expected = torch.ones(BLOCK_SIZE, dtype=torch.float32, device="cuda")
    try:
        torch.testing.assert_close(results, expected, rtol=0, atol=0)
    except AssertionError as e:
        print(f"LOCAL STORE test failed with cache_modifier={cache_modifier}")
        print(e)
        raise


@pytest.mark.parametrize("cache_modifier", CACHE_MODIFIERS)
def test_store_cache_modifiers_remote(cache_modifier):
    """Test remote store (from_rank != to_rank) with various cache modifiers."""
    shmem = iris.iris(1 << 20)
    heap_bases = shmem.get_heap_bases()
    num_ranks = shmem.get_num_ranks()
    cur_rank = shmem.get_rank()

    if num_ranks < 2:
        pytest.skip("Remote store test requires at least 2 ranks")

    BLOCK_SIZE = 16
    src = shmem.ones(BLOCK_SIZE, dtype=torch.float32)
    results = shmem.zeros(BLOCK_SIZE, dtype=torch.float32)

    shmem.barrier()

    # rank 0 stores to rank 1
    remote_rank = (cur_rank + 1) % num_ranks
    grid = lambda meta: (1,)
    if cur_rank == 0:
        remote_store_kernel[grid](src, results, cur_rank, remote_rank, BLOCK_SIZE, heap_bases, cache_modifier)

    shmem.barrier()

    # rank 1 checks the data it received from rank 0
    if cur_rank == 1:
        expected = torch.ones(BLOCK_SIZE, dtype=torch.float32, device="cuda")
        try:
            torch.testing.assert_close(results, expected, rtol=0, atol=0)
        except AssertionError as e:
            print(f"REMOTE STORE test failed with cache_modifier={cache_modifier}")
            print(e)
            raise
