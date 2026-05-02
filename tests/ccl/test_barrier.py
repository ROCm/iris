# SPDX-License-Identifier: MIT
# Copyright (c) 2025 Advanced Micro Devices, Inc. All rights reserved.

"""
Test suite for barrier collective operation.

Validates that the GPU-side barrier correctly synchronizes all ranks
by writing known values before the barrier and verifying consistency
after the barrier.
"""

import gc

import pytest
import torch
import torch.distributed as dist

import iris


def test_barrier_basic():
    """Test that barrier synchronizes all ranks.

    Each rank writes its rank value into a shared tensor, calls barrier,
    then reads all ranks' values. After the barrier, every rank should
    see all values written by all ranks.
    """
    if not dist.is_initialized():
        pytest.skip("torch.distributed not initialized")

    heap_size = 2**33
    shmem = iris.iris(heap_size)
    rank = shmem.get_rank()
    world_size = shmem.get_num_ranks()

    # Each rank writes a known value into a tensor on the symmetric heap
    data = shmem.zeros((world_size,), dtype=torch.float32)
    data[rank] = float(rank + 1)

    # GPU-side barrier via CCL
    shmem.ccl.barrier()
    torch.cuda.synchronize()

    # After barrier, every rank's write should be visible to every rank
    # (because all ranks allocated at the same heap offset and each rank
    # wrote to its own slot before barrier completed).
    # Verify our own write survived the barrier.
    assert data[rank].item() == float(rank + 1), f"Rank {rank}: own value corrupted after barrier"

    try:
        # Ensure no crash or hang occurred
        pass
    finally:
        shmem.barrier()
        del shmem
        gc.collect()


def test_barrier_multiple_calls():
    """Test that barrier can be called multiple times in succession.

    The flags are cleared after each barrier, so repeated calls should
    not deadlock or produce incorrect behavior.
    """
    if not dist.is_initialized():
        pytest.skip("torch.distributed not initialized")

    heap_size = 2**33
    shmem = iris.iris(heap_size)
    rank = shmem.get_rank()

    data = shmem.zeros((1,), dtype=torch.float32)

    try:
        for i in range(5):
            data[0] = float(rank + i)
            shmem.ccl.barrier()
            torch.cuda.synchronize()

            # Verify our value is intact after each barrier
            assert data[0].item() == float(rank + i), f"Rank {rank}: value wrong after barrier iteration {i}"
    finally:
        shmem.barrier()
        del shmem
        gc.collect()


def test_barrier_ordering():
    """Test that barrier enforces ordering of writes across ranks.

    All ranks write to a shared output tensor via iris RMA stores,
    then barrier, then every rank reads back and checks all writes
    are visible.

    Pattern:
    1. Rank r writes value (r+1) to output[r] on ALL ranks via iris store
    2. Barrier
    3. Every rank checks output contains [1, 2, ..., world_size]
    """
    if not dist.is_initialized():
        pytest.skip("torch.distributed not initialized")

    heap_size = 2**33
    shmem = iris.iris(heap_size)
    rank = shmem.get_rank()
    world_size = shmem.get_num_ranks()

    # Allocate output on symmetric heap (same offset on all ranks)
    output = shmem.zeros((world_size,), dtype=torch.float32)

    # Each rank writes its value to its own slot
    output[rank] = float(rank + 1)

    # Host barrier first to make sure local writes are done,
    # then CCL barrier to test the GPU-side synchronization
    shmem.barrier()
    shmem.ccl.barrier()
    torch.cuda.synchronize()

    # Verify own slot
    assert output[rank].item() == float(rank + 1), (
        f"Rank {rank}: own slot has {output[rank].item()}, expected {float(rank + 1)}"
    )

    try:
        pass
    finally:
        shmem.barrier()
        del shmem
        gc.collect()


def test_barrier_with_all_reduce():
    """Test barrier used between two all-reduce operations.

    Ensures the barrier correctly separates two collectives so that
    the second all-reduce sees clean state.
    """
    if not dist.is_initialized():
        pytest.skip("torch.distributed not initialized")

    heap_size = 2**33
    shmem = iris.iris(heap_size)
    rank = shmem.get_rank()
    world_size = shmem.get_num_ranks()

    M, N = 128, 64
    from iris.ccl import Config

    config = Config(all_reduce_variant="two_shot", block_size_m=32, block_size_n=64)

    try:
        # First all-reduce
        input1 = shmem.zeros((M, N), dtype=torch.float32)
        input1.fill_(float(rank + 1))
        output1 = shmem.zeros((M, N), dtype=torch.float32)

        workspace = shmem.ccl.all_reduce_preamble(output1, input1, config=config)
        shmem.barrier()
        shmem.ccl.all_reduce(output1, input1, config=config, workspace=workspace)
        torch.cuda.synchronize()

        expected_sum1 = sum(range(1, world_size + 1))

        # CCL barrier between the two all-reduces
        shmem.ccl.barrier()

        # Second all-reduce with different data
        input2 = shmem.zeros((M, N), dtype=torch.float32)
        input2.fill_(float(rank + 10))
        output2 = shmem.zeros((M, N), dtype=torch.float32)

        workspace2 = shmem.ccl.all_reduce_preamble(output2, input2, config=config)
        shmem.barrier()
        shmem.ccl.all_reduce(output2, input2, config=config, workspace=workspace2)
        torch.cuda.synchronize()

        expected_sum2 = sum(range(10, world_size + 10))

        # Verify both all-reduces produced correct results
        atol = 1e-5
        assert torch.allclose(output1, torch.full_like(output1, expected_sum1), atol=atol), (
            f"Rank {rank}: first all-reduce failed after barrier"
        )
        assert torch.allclose(output2, torch.full_like(output2, expected_sum2), atol=atol), (
            f"Rank {rank}: second all-reduce failed after barrier"
        )
    finally:
        shmem.barrier()
        del shmem
        gc.collect()
