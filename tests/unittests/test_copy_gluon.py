# SPDX-License-Identifier: MIT
# Copyright (c) 2025 Advanced Micro Devices, Inc. All rights reserved.

import torch
import pytest
from triton.experimental import gluon
from triton.experimental.gluon import language as gl
import iris.experimental.iris_gluon as iris_gl


@gluon.jit
def copy_get_kernel(
    IrisDeviceCtx: gl.constexpr,
    context_tensor,
    data,
    results,
    cur_rank: gl.constexpr,
    num_ranks: gl.constexpr,
    BLOCK_SIZE: gl.constexpr,
):
    """GET: cur_rank == to_rank (pull from remote)"""
    ctx = IrisDeviceCtx.initialize(context_tensor)
    pid = gl.program_id(0)
    block_start = pid * BLOCK_SIZE
    layout: gl.constexpr = gl.BlockedLayout([1], [64], [1], [0])
    offsets = block_start + gl.arange(0, BLOCK_SIZE, layout=layout)
    mask = offsets < BLOCK_SIZE

    for target_rank in range(num_ranks):
        src_data = data + BLOCK_SIZE * cur_rank
        dest_data = results + BLOCK_SIZE * target_rank
        ctx.copy(src_data + offsets, dest_data + offsets, target_rank, cur_rank, mask=mask)


@gluon.jit
def copy_put_kernel(
    IrisDeviceCtx: gl.constexpr,
    context_tensor,
    data,
    results,
    cur_rank: gl.constexpr,
    num_ranks: gl.constexpr,
    BLOCK_SIZE: gl.constexpr,
):
    """PUT: cur_rank == from_rank (push to remote)"""
    ctx = IrisDeviceCtx.initialize(context_tensor)
    pid = gl.program_id(0)
    block_start = pid * BLOCK_SIZE
    layout: gl.constexpr = gl.BlockedLayout([1], [64], [1], [0])
    offsets = block_start + gl.arange(0, BLOCK_SIZE, layout=layout)
    mask = offsets < BLOCK_SIZE

    for target_rank in range(num_ranks):
        src_data = data + BLOCK_SIZE * cur_rank
        dest_data = results + BLOCK_SIZE * cur_rank
        ctx.copy(src_data + offsets, dest_data + offsets, cur_rank, target_rank, mask=mask)


@gluon.jit
def copy_local_kernel(
    IrisDeviceCtx: gl.constexpr,
    context_tensor,
    data,
    results,
    cur_rank: gl.constexpr,
    num_ranks: gl.constexpr,
    BLOCK_SIZE: gl.constexpr,
):
    """LOCAL: from_rank == to_rank == cur_rank"""
    ctx = IrisDeviceCtx.initialize(context_tensor)
    pid = gl.program_id(0)
    block_start = pid * BLOCK_SIZE
    layout: gl.constexpr = gl.BlockedLayout([1], [64], [1], [0])
    offsets = block_start + gl.arange(0, BLOCK_SIZE, layout=layout)
    mask = offsets < BLOCK_SIZE

    for i in range(num_ranks):
        src_data = data + BLOCK_SIZE * i
        dest_data = results + BLOCK_SIZE * i
        ctx.copy(src_data + offsets, dest_data + offsets, cur_rank, cur_rank, mask=mask)


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
    shmem = iris_gl.iris(1 << 20)
    num_ranks = shmem.get_num_ranks()
    context_tensor = shmem.get_device_context()
    cur_rank = shmem.get_rank()

    data = shmem.zeros((num_ranks, BLOCK_SIZE), dtype=dtype)
    base = cur_rank + num_ranks
    for i in range(num_ranks):
        data[i, :] = base * (i + 1)

    results = shmem.zeros((num_ranks, BLOCK_SIZE), dtype=dtype)
    grid = (1,)
    copy_get_kernel[grid](
        iris_gl.IrisDeviceCtx,
        context_tensor,
        data,
        results,
        cur_rank,
        num_ranks,
        BLOCK_SIZE,
        num_warps=1,
    )
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
    shmem = iris_gl.iris(1 << 20)
    num_ranks = shmem.get_num_ranks()
    context_tensor = shmem.get_device_context()
    cur_rank = shmem.get_rank()

    data = shmem.zeros((num_ranks, BLOCK_SIZE), dtype=dtype)
    base = cur_rank + num_ranks
    for i in range(num_ranks):
        data[i, :] = base * (i + 1)

    results = shmem.zeros((num_ranks, BLOCK_SIZE), dtype=dtype)
    grid = (1,)
    copy_put_kernel[grid](
        iris_gl.IrisDeviceCtx,
        context_tensor,
        data,
        results,
        cur_rank,
        num_ranks,
        BLOCK_SIZE,
        num_warps=1,
    )
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
    shmem = iris_gl.iris(1 << 20)
    num_ranks = shmem.get_num_ranks()
    context_tensor = shmem.get_device_context()
    cur_rank = shmem.get_rank()

    data = shmem.zeros((num_ranks, BLOCK_SIZE), dtype=dtype)
    base = cur_rank + num_ranks
    for i in range(num_ranks):
        data[i, :] = base * (i + 1)

    results = shmem.zeros((num_ranks, BLOCK_SIZE), dtype=dtype)
    grid = (1,)
    copy_local_kernel[grid](
        iris_gl.IrisDeviceCtx,
        context_tensor,
        data,
        results,
        cur_rank,
        num_ranks,
        BLOCK_SIZE,
        num_warps=1,
    )
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
