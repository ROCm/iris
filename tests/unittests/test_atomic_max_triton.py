# SPDX-License-Identifier: MIT
# Copyright (c) 2025 Advanced Micro Devices, Inc. All rights reserved.

import torch
import triton
import triton.language as tl
import pytest
import iris


@triton.jit
def atomic_max_kernel(
    results,
    sem: tl.constexpr,
    scope: tl.constexpr,
    cur_rank: tl.constexpr,
    num_ranks: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
    heap_bases: tl.tensor,
):
    pid = tl.program_id(0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < BLOCK_SIZE

    acc = tl.full([BLOCK_SIZE], cur_rank + 1, dtype=results.type.element_ty)

    for target_rank in range(num_ranks):
        iris.atomic_max(results + offsets, acc, cur_rank, target_rank, heap_bases, mask, sem=sem, scope=scope)


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
def test_atomic_max_api(dtype, sem, scope, BLOCK_SIZE):
    # TODO: Adjust heap size.
    shmem = iris.iris(1 << 20)
    num_ranks = shmem.get_num_ranks()
    heap_bases = shmem.get_heap_bases()
    cur_rank = shmem.get_rank()

    min_val = torch.iinfo(dtype).min
    results = shmem.full((BLOCK_SIZE,), min_val, dtype=dtype)

    shmem.barrier()

    grid = lambda meta: (1,)
    atomic_max_kernel[grid](results, sem, scope, cur_rank, num_ranks, BLOCK_SIZE, heap_bases)
    shmem.barrier()

    # All ranks participate in performing the max operation
    # Each rank performs the atomic operation: max(rank_id + 1)
    # The result equals the ID of the last rank + 1
    expected = torch.full((BLOCK_SIZE,), num_ranks, dtype=dtype, device="cuda")

    try:
        torch.testing.assert_close(results, expected, rtol=0, atol=0)
    except AssertionError as e:
        print(e)
        print("Expected:", expected)
        print("Actual  :", results)
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
