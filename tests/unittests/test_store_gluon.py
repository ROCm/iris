# SPDX-License-Identifier: MIT
# Copyright (c) 2025 Advanced Micro Devices, Inc. All rights reserved.

import torch
import pytest
from triton.experimental import gluon
from triton.experimental.gluon import language as gl
import iris.experimental.iris_gluon as iris_gl


@gluon.jit
def store_kernel(
    IrisDeviceCtx: gl.constexpr,
    context_tensor,
    data,
    results,
    destination_rank: gl.constexpr,
    num_ranks: gl.constexpr,
    BLOCK_SIZE: gl.constexpr,
):
    ctx = IrisDeviceCtx.initialize(context_tensor)
    pid = gl.program_id(0)

    block_start = pid * BLOCK_SIZE
    layout: gl.constexpr = gl.BlockedLayout([1], [64], [1], [0])
    offsets = block_start + gl.arange(0, BLOCK_SIZE, layout=layout)

    mask = offsets < BLOCK_SIZE

    # Load the data from src for this block
    value = gl.load(data + offsets, mask=mask)

    # Store data to all ranks
    # Doesn't matter which rank stores at the end, the data should all be the same at the end.
    for dst_rank in range(num_ranks):
        ctx.store(results + offsets, value, dst_rank, mask=mask)


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
def test_store_api(dtype, BLOCK_SIZE):
    # TODO: Adjust heap size.
    shmem = iris_gl.iris(1 << 20)
    num_ranks = shmem.get_num_ranks()
    context_tensor = shmem.get_device_context()
    destination_rank = shmem.get_rank()

    src = shmem.ones(BLOCK_SIZE, dtype=dtype)
    results = shmem.zeros_like(src)

    shmem.barrier()

    grid = (1,)
    store_kernel[grid](
        iris_gl.IrisDeviceCtx,
        context_tensor,
        src,
        results,
        destination_rank,
        num_ranks,
        BLOCK_SIZE,
        num_warps=1,
    )
    shmem.barrier()

    # Verify the result
    expected = torch.ones(BLOCK_SIZE, dtype=dtype, device="cuda")

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
