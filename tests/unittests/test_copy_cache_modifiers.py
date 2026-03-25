# SPDX-License-Identifier: MIT
# Copyright (c) 2025 Advanced Micro Devices, Inc. All rights reserved.

import torch
import triton
import triton.language as tl
import pytest
import iris
from itertools import product


@triton.jit
def copy_kernel_local_read_remote_write(
    data,
    results,
    cur_rank: tl.constexpr,
    num_ranks: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
    heap_bases: tl.tensor,
    load_cache_modifier: tl.constexpr,
    store_cache_modifier: tl.constexpr,
):
    """Copy from local memory to remote memory (local read, remote write)"""
    pid = tl.program_id(0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < BLOCK_SIZE

    # Copy from current rank to other ranks.
    # Both load and store cache modifiers are supported on local and remote ops.
    for target_rank in range(num_ranks):
        src_data = data + BLOCK_SIZE * cur_rank
        dest_data = results + BLOCK_SIZE * cur_rank
        iris.copy(
            src_data + offsets,
            dest_data + offsets,
            cur_rank,
            target_rank,
            cur_rank,
            heap_bases,
            mask=mask,
            load_cache_modifier=load_cache_modifier,
            store_cache_modifier=store_cache_modifier,
        )


@triton.jit
def copy_kernel_remote_read_local_write(
    data,
    results,
    cur_rank: tl.constexpr,
    num_ranks: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
    heap_bases: tl.tensor,
    load_cache_modifier: tl.constexpr,
    store_cache_modifier: tl.constexpr,
):
    """Copy from remote memory to local memory (remote read, local write)"""
    pid = tl.program_id(0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < BLOCK_SIZE

    # Copy from other ranks to current rank.
    # Both load and store cache modifiers are supported on local and remote ops.
    for source_rank in range(num_ranks):
        src_data = data + BLOCK_SIZE * source_rank
        dest_data = results + BLOCK_SIZE * source_rank
        iris.copy(
            src_data + offsets,
            dest_data + offsets,
            source_rank,
            cur_rank,
            cur_rank,
            heap_bases,
            mask=mask,
            load_cache_modifier=load_cache_modifier,
            store_cache_modifier=store_cache_modifier,
        )


# Define cache modifiers for load and store operations.
LOAD_CACHE_MODIFIERS = [None, "", ".ca", ".cg", ".cv"]
STORE_CACHE_MODIFIERS = [None, "", ".wb", ".cg", ".cs", ".wt"]


@pytest.mark.parametrize(
    "load_cache_modifier,store_cache_modifier", list(product(LOAD_CACHE_MODIFIERS, STORE_CACHE_MODIFIERS))
)
def test_copy_local_read_remote_write(load_cache_modifier, store_cache_modifier):
    """Test copy: local read → remote write

    Direction: from_rank=cur_rank (local), to_rank=other (remote)
    - Load: from LOCAL memory
    - Store: to REMOTE memory
    """
    shmem = iris.iris(1 << 20)
    num_ranks = shmem.get_num_ranks()
    heap_bases = shmem.get_heap_bases()
    cur_rank = shmem.get_rank()

    BLOCK_SIZE = 16
    data = shmem.zeros((num_ranks, BLOCK_SIZE), dtype=torch.float32)
    base = cur_rank + num_ranks
    for i in range(num_ranks):
        data[i, :] = base * (i + 1)

    results = shmem.zeros((num_ranks, BLOCK_SIZE), dtype=torch.float32)

    # Barrier to ensure all ranks have initialized their data before any rank launches
    # the kernel (which reads remote data in the remote-read case).
    shmem.barrier()

    grid = lambda meta: (1,)
    copy_kernel_local_read_remote_write[grid](
        data, results, cur_rank, num_ranks, BLOCK_SIZE, heap_bases, load_cache_modifier, store_cache_modifier
    )

    shmem.barrier()

    # Verify results - each rank copies its data to all other ranks
    for rank_id in range(num_ranks):
        expected_value = (rank_id + num_ranks) * (rank_id + 1)
        assert torch.allclose(
            results[rank_id], torch.full((BLOCK_SIZE,), expected_value, dtype=torch.float32, device=results.device)
        ), (
            f"Mismatch at rank {cur_rank}, slot {rank_id} with load_cache_modifier={load_cache_modifier}, store_cache_modifier={store_cache_modifier}"
        )


@pytest.mark.parametrize(
    "load_cache_modifier,store_cache_modifier",
    list(product(LOAD_CACHE_MODIFIERS, STORE_CACHE_MODIFIERS)),
)
def test_copy_remote_read_local_write(load_cache_modifier, store_cache_modifier):
    """Test copy: remote read → local write

    Direction: from_rank=other (remote), to_rank=cur_rank (local)
    - Load: from REMOTE memory
    - Store: to LOCAL memory
    """
    shmem = iris.iris(1 << 20)
    num_ranks = shmem.get_num_ranks()
    heap_bases = shmem.get_heap_bases()
    cur_rank = shmem.get_rank()

    BLOCK_SIZE = 16
    data = shmem.zeros((num_ranks, BLOCK_SIZE), dtype=torch.float32)
    base = cur_rank + num_ranks
    for i in range(num_ranks):
        data[i, :] = base * (i + 1)

    results = shmem.zeros((num_ranks, BLOCK_SIZE), dtype=torch.float32)

    # Barrier to ensure all ranks have initialized their data before any rank launches
    # the kernel (which reads remote data in the remote-read case).
    shmem.barrier()

    grid = lambda meta: (1,)
    copy_kernel_remote_read_local_write[grid](
        data, results, cur_rank, num_ranks, BLOCK_SIZE, heap_bases, load_cache_modifier, store_cache_modifier
    )

    shmem.barrier()

    # Verify results - each rank pulls data from all ranks
    for rank_id in range(num_ranks):
        expected_value = (rank_id + num_ranks) * (rank_id + 1)
        assert torch.allclose(
            results[rank_id], torch.full((BLOCK_SIZE,), expected_value, dtype=torch.float32, device=results.device)
        ), (
            f"Mismatch at rank {cur_rank}, slot {rank_id} with load_cache_modifier={load_cache_modifier}, store_cache_modifier={store_cache_modifier}"
        )
