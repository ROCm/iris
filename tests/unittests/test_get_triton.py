# SPDX-License-Identifier: MIT
# Copyright (c) 2025 Advanced Micro Devices, Inc. All rights reserved.

import torch
import triton
import triton.language as tl
import pytest
import iris


# TODO: Separate this kernel out in the following categories:
# 1. for local get.
# 2. for remote get with one other rank.
# 3. for remote get with more than one rank (if num_ranks > 2).
@triton.jit
def get_kernel(
    data,
    results,
    cur_rank: tl.constexpr,
    num_ranks: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
    heap_bases: tl.tensor,
):
    pid = tl.program_id(0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < BLOCK_SIZE

    acc = tl.zeros([BLOCK_SIZE], dtype=data.type.element_ty)

    # Loop over all ranks, get the stored data.
    # load to local register, accumulate.
    for target_rank in range(num_ranks):
        iris.get(data + offsets, results + offsets, cur_rank, target_rank, heap_bases, mask=mask)
        acc += tl.load(results + offsets, mask=mask)

    # Store the accumulated value back to the output.
    tl.store(results + offsets, acc, mask=mask)


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
def test_get_api(dtype, BLOCK_SIZE):
    # TODO: Adjust heap size.
    shmem = iris.iris(1 << 20)
    num_ranks = shmem.get_num_ranks()
    heap_bases = shmem.get_heap_bases()
    cur_rank = shmem.get_rank()

    data = shmem.ones(BLOCK_SIZE, dtype=dtype)
    results = shmem.zeros_like(data)

    shmem.barrier()

    grid = lambda meta: (1,)
    get_kernel[grid](data, results, cur_rank, num_ranks, BLOCK_SIZE, heap_bases)
    shmem.barrier()

    # Verify the results
    expected = torch.ones(BLOCK_SIZE, dtype=dtype, device="cuda") * num_ranks

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
