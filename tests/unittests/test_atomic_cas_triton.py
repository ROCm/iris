# SPDX-License-Identifier: MIT
# Copyright (c) 2025 Advanced Micro Devices, Inc. All rights reserved.

import torch
import triton
import triton.language as tl
import pytest
import iris


@triton.jit
def atomic_cas_kernel(
    results,
    sem: tl.constexpr,
    scope: tl.constexpr,
    cur_rank: tl.constexpr,
    num_ranks: tl.constexpr,
    heap_bases: tl.tensor,
):
    # Cast constants to match results.dtype
    dtype = results.dtype.element_ty
    cmp = tl.full((), 0, dtype=dtype)  # scalar 0
    val = tl.full((), num_ranks, dtype=dtype)  # scalar num_ranks

    for target_rank in range(num_ranks):
        iris.atomic_cas(results, cmp, val, cur_rank, target_rank, heap_bases, sem=sem, scope=scope)


@pytest.mark.parametrize(
    "dtype",
    [
        torch.int16,
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
def test_atomic_cas_api(dtype, sem, scope):
    # TODO: Adjust heap size.
    shmem = iris.iris(1 << 20)
    num_ranks = shmem.get_num_ranks()
    heap_bases = shmem.get_heap_bases()
    cur_rank = shmem.get_rank()

    results = shmem.zeros((1,), dtype=dtype)

    shmem.barrier()

    grid = lambda meta: (1,)
    atomic_cas_kernel[grid](results, sem, scope, cur_rank, num_ranks, heap_bases)
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
