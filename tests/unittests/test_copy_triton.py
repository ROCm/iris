# SPDX-License-Identifier: MIT
# Copyright (c) 2025 Advanced Micro Devices, Inc. All rights reserved.

import torch
import triton
import triton.language as tl
import pytest
import iris


@triton.jit
def copy_get_kernel(
    data,
    results,
    cur_rank: tl.constexpr,
    num_ranks: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
    heap_bases: tl.tensor,
):
    """GET: cur_rank == to_rank (pull from remote)"""
    pid = tl.program_id(0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < BLOCK_SIZE

    for target_rank in range(num_ranks):
        src_data = data + BLOCK_SIZE * cur_rank
        dest_data = results + BLOCK_SIZE * target_rank
        iris.copy(src_data + offsets, dest_data + offsets, target_rank, cur_rank, cur_rank, heap_bases, mask)


@triton.jit
def copy_put_kernel(
    data,
    results,
    cur_rank: tl.constexpr,
    num_ranks: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
    heap_bases: tl.tensor,
):
    """PUT: cur_rank == from_rank (push to remote)"""
    pid = tl.program_id(0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < BLOCK_SIZE

    for target_rank in range(num_ranks):
        src_data = data + BLOCK_SIZE * cur_rank
        dest_data = results + BLOCK_SIZE * cur_rank
        iris.copy(src_data + offsets, dest_data + offsets, cur_rank, target_rank, cur_rank, heap_bases, mask)


@triton.jit
def copy_local_kernel(
    data,
    results,
    cur_rank: tl.constexpr,
    num_ranks: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
    heap_bases: tl.tensor,
):
    """LOCAL: from_rank == to_rank == cur_rank"""
    pid = tl.program_id(0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < BLOCK_SIZE

    for i in range(num_ranks):
        src_data = data + BLOCK_SIZE * i
        dest_data = results + BLOCK_SIZE * i
        iris.copy(src_data + offsets, dest_data + offsets, cur_rank, cur_rank, cur_rank, heap_bases, mask)


@pytest.mark.parametrize(
    "dtype",
    [
        torch.int8,
        torch.float16,
        torch.bfloat16,
        torch.float32,
    ],
)
@pytest.mark.parametrize(
    "BLOCK_SIZE",
    [
        1,
        8,
        16,
        32,
    ],
)
def test_copy_get(dtype, BLOCK_SIZE):
    """Test GET operation: cur_rank == to_rank"""
    shmem = iris.iris(1 << 20)
    num_ranks = shmem.get_num_ranks()
    heap_bases = shmem.get_heap_bases()
    cur_rank = shmem.get_rank()

    data = shmem.zeros((num_ranks, BLOCK_SIZE), dtype=dtype)
    base = cur_rank + num_ranks
    for i in range(num_ranks):
        data[i, :] = base * (i + 1)

    results = shmem.zeros((num_ranks, BLOCK_SIZE), dtype=dtype)
    grid = lambda meta: (1,)
    copy_get_kernel[grid](data, results, cur_rank, num_ranks, BLOCK_SIZE, heap_bases)
    shmem.barrier()

    expected = shmem.zeros((num_ranks, BLOCK_SIZE), dtype=dtype)
    for rank_id in range(num_ranks):
        expected[rank_id, :] = (rank_id + num_ranks) * (cur_rank + 1)

    try:
        torch.testing.assert_close(results, expected, rtol=0, atol=0)
    except AssertionError as e:
        print(e)
        print("Expected:", expected)
        print("Actual:", results)
        raise
    finally:
        # Final barrier to ensure all ranks complete before test cleanup
        # This helps with test isolation when running multiple tests
        # Note: shmem.barrier() already does cuda.synchronize()
        shmem.barrier()
        # Explicitly delete the shmem instance to trigger cleanup
        del shmem
        # Force garbage collection to ensure IPC handles are cleaned up
        import gc

        gc.collect()


@pytest.mark.parametrize(
    "dtype",
    [
        torch.int8,
        torch.float16,
        torch.bfloat16,
        torch.float32,
    ],
)
@pytest.mark.parametrize(
    "BLOCK_SIZE",
    [
        1,
        8,
        16,
        32,
    ],
)
def test_copy_put(dtype, BLOCK_SIZE):
    """Test PUT operation: cur_rank == from_rank"""
    shmem = iris.iris(1 << 20)
    num_ranks = shmem.get_num_ranks()
    heap_bases = shmem.get_heap_bases()
    cur_rank = shmem.get_rank()

    data = shmem.zeros((num_ranks, BLOCK_SIZE), dtype=dtype)
    base = cur_rank + num_ranks
    for i in range(num_ranks):
        data[i, :] = base * (i + 1)

    results = shmem.zeros((num_ranks, BLOCK_SIZE), dtype=dtype)
    grid = lambda meta: (1,)
    copy_put_kernel[grid](data, results, cur_rank, num_ranks, BLOCK_SIZE, heap_bases)
    shmem.barrier()

    # Each rank writes to results[cur_rank] on all targets
    # After barrier, results[rank_id] contains data from rank_id
    expected = shmem.zeros((num_ranks, BLOCK_SIZE), dtype=dtype)
    for rank_id in range(num_ranks):
        expected[rank_id, :] = (rank_id + num_ranks) * (rank_id + 1)

    try:
        torch.testing.assert_close(results, expected, rtol=0, atol=0)
    except AssertionError as e:
        print(e)
        print("Expected:", expected)
        print("Actual:", results)
        raise
    finally:
        # Final barrier to ensure all ranks complete before test cleanup
        # This helps with test isolation when running multiple tests
        # Note: shmem.barrier() already does cuda.synchronize()
        shmem.barrier()
        # Explicitly delete the shmem instance to trigger cleanup
        del shmem
        # Force garbage collection to ensure IPC handles are cleaned up
        import gc

        gc.collect()


@pytest.mark.parametrize(
    "dtype",
    [
        torch.int8,
        torch.float16,
        torch.bfloat16,
        torch.float32,
    ],
)
@pytest.mark.parametrize(
    "BLOCK_SIZE",
    [
        1,
        8,
        16,
        32,
    ],
)
def test_copy_local(dtype, BLOCK_SIZE):
    """Test LOCAL operation: from_rank == to_rank == cur_rank"""
    shmem = iris.iris(1 << 20)
    num_ranks = shmem.get_num_ranks()
    heap_bases = shmem.get_heap_bases()
    cur_rank = shmem.get_rank()

    data = shmem.zeros((num_ranks, BLOCK_SIZE), dtype=dtype)
    base = cur_rank + num_ranks
    for i in range(num_ranks):
        data[i, :] = base * (i + 1)

    results = shmem.zeros((num_ranks, BLOCK_SIZE), dtype=dtype)
    grid = lambda meta: (1,)
    copy_local_kernel[grid](data, results, cur_rank, num_ranks, BLOCK_SIZE, heap_bases)
    shmem.barrier()

    # Local copy: results should match data
    expected = data

    try:
        torch.testing.assert_close(results, expected, rtol=0, atol=0)
    except AssertionError as e:
        print(e)
        print("Expected:", expected)
        print("Actual:", results)
        raise
    finally:
        # Final barrier to ensure all ranks complete before test cleanup
        # This helps with test isolation when running multiple tests
        # Note: shmem.barrier() already does cuda.synchronize()
        shmem.barrier()
        # Explicitly delete the shmem instance to trigger cleanup
        del shmem
        # Force garbage collection to ensure IPC handles are cleaned up
        import gc

        gc.collect()
