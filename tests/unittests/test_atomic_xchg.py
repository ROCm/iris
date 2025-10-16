# SPDX-License-Identifier: MIT
# Copyright (c) 2025 Advanced Micro Devices, Inc. All rights reserved.

import torch
import pytest
from triton.experimental import gluon
from triton.experimental.gluon import language as gl
import iris.experimental.iris_gluon as iris_gl


@gluon.jit
def atomic_xchg_kernel(
    IrisDeviceCtx: gl.constexpr,
    context_tensor,
    results,
    sem: gl.constexpr,
    scope: gl.constexpr,
    cur_rank: gl.constexpr,
    num_ranks: gl.constexpr,
):
    ctx = IrisDeviceCtx.initialize(context_tensor)
    # Cast constants to match results.dtype
    dtype = results.dtype.element_ty
    val = gl.full((), num_ranks, dtype=dtype)  # scalar num_ranks

    for target_rank in range(num_ranks):
        ctx.atomic_xchg(results, val, target_rank, mask=None, sem=sem, scope=scope)


@pytest.mark.parametrize(
    "dtype",
    [
        torch.int32,
        torch.int64,
        torch.float32,
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
def test_atomic_xchg_api(dtype, sem, scope):
    # TODO: Adjust heap size.
    shmem = iris_gl.iris(1 << 20)
    num_ranks = shmem.get_num_ranks()
    context_tensor = shmem.get_device_context()
    cur_rank = shmem.get_rank()

    results = shmem.zeros((1,), dtype=dtype)

    shmem.barrier()

    grid = (1,)
    atomic_xchg_kernel[grid](
        iris_gl.IrisDeviceCtx,
        context_tensor,
        results,
        sem,
        scope,
        cur_rank,
        num_ranks,
        num_warps=1,
    )
    shmem.barrier()

    # Verify the results
    expected = torch.full((1,), num_ranks, dtype=dtype, device="cuda")

    try:
        torch.testing.assert_close(results, expected, rtol=0, atol=0)
    except AssertionError as e:
        print(e)
        print("Expected:", expected)
        print("Actual:", results)
        raise
