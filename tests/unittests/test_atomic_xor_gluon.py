# SPDX-License-Identifier: MIT
# Copyright (c) 2025 Advanced Micro Devices, Inc. All rights reserved.

import torch
import pytest
from triton.experimental import gluon
from triton.experimental.gluon import language as gl
import iris.experimental.iris_gluon as iris_gl


@gluon.jit
def atomic_xor_kernel(
    IrisDeviceCtx: gl.constexpr,
    context_tensor,
    results,
    sem: gl.constexpr,
    scope: gl.constexpr,
    cur_rank: gl.constexpr,
    num_ranks: gl.constexpr,
    BLOCK_SIZE: gl.constexpr,
):
    ctx = IrisDeviceCtx.initialize(context_tensor)
    pid = gl.program_id(0)
    block_start = pid * BLOCK_SIZE
    layout: gl.constexpr = gl.BlockedLayout([1], [64], [1], [0])
    offsets = block_start + gl.arange(0, BLOCK_SIZE, layout=layout)
    mask = offsets < BLOCK_SIZE

    # Use 1 as the xor operand
    acc = gl.full([BLOCK_SIZE], 1, results.type.element_ty, layout)

    # Loop over all ranks and atomically xor acc into results.
    for target_rank in range(num_ranks):
        ctx.atomic_xor(results + offsets, acc, target_rank, mask=mask, sem=sem, scope=scope)


@pytest.mark.parametrize(
    "dtype",
    [
        torch.int32,
        torch.int64,
    ],
)
@pytest.mark.parametrize(
    "sem",
    [
        "acquire",
        "release",
        "acq_rel",
    ],
)
@pytest.mark.parametrize(
    "scope",
    [
        "cta",
        "gpu",
        "sys",
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
def test_atomic_xor_api(dtype, sem, scope, BLOCK_SIZE):
    # TODO: Adjust heap size.
    shmem = iris_gl.iris(1 << 20)
    num_ranks = shmem.get_num_ranks()
    context_tensor = shmem.get_device_context()
    cur_rank = shmem.get_rank()

    results = shmem.zeros(BLOCK_SIZE, dtype=dtype)

    shmem.barrier()

    grid = (1,)
    atomic_xor_kernel[grid](
        iris_gl.IrisDeviceCtx,
        context_tensor,
        results,
        sem,
        scope,
        cur_rank,
        num_ranks,
        BLOCK_SIZE,
        num_warps=1,
    )
    shmem.barrier()

    # If we xor '1' in num_ranks times:
    # - If num_ranks is odd  -> final = 1
    # - If num_ranks is even -> final = 0
    if (num_ranks % 2) == 1:
        expected = torch.ones(BLOCK_SIZE, dtype=dtype, device="cuda")
    else:
        expected = torch.zeros(BLOCK_SIZE, dtype=dtype, device="cuda")

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
