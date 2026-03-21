# SPDX-License-Identifier: MIT
# Copyright (c) 2025 Advanced Micro Devices, Inc. All rights reserved.

import torch
import triton
import triton.language as tl
import pytest
import iris


@triton.jit
def put_with_other_kernel(
    data,
    results,
    cur_rank: tl.constexpr,
    num_ranks: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
    heap_bases: tl.tensor,
    other_value: tl.constexpr,
):
    pid = tl.program_id(0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)

    # Create a mask that is False for half the elements
    mask = offsets < BLOCK_SIZE // 2

    # Put data in all ranks with mask and other parameter
    # The "other" value will be used for masked-out elements during the load from data
    for target_rank in range(num_ranks):
        iris.put(data + offsets, results + offsets, cur_rank, target_rank, heap_bases, mask=mask, other=other_value)


@pytest.mark.parametrize(
    "dtype",
    [
        torch.float16,
        torch.bfloat16,
        torch.float32,
    ],
)
@pytest.mark.parametrize(
    "BLOCK_SIZE",
    [
        8,
        16,
        32,
    ],
)
def test_put_other_api(dtype, BLOCK_SIZE):
    # TODO: Adjust heap size.
    shmem = iris.iris(1 << 20)
    num_ranks = shmem.get_num_ranks()
    heap_bases = shmem.get_heap_bases()
    cur_rank = shmem.get_rank()

    # Fill data with ones
    data = shmem.ones(BLOCK_SIZE, dtype=dtype)
    results = shmem.zeros_like(data)

    # Use -1 as the "other" value for masked-out elements
    other_value = -1.0

    shmem.barrier()

    grid = lambda meta: (1,)
    put_with_other_kernel[grid](data, results, cur_rank, num_ranks, BLOCK_SIZE, heap_bases, other_value)
    shmem.barrier()

    # Verify the results
    # First half should contain the value 1.0 (from data, written via masked put)
    # Second half stays at 0.0 because iris.put stores with mask, so masked-out positions
    # in results are never written.
    expected = torch.zeros(BLOCK_SIZE, dtype=dtype, device="cuda")
    expected[: BLOCK_SIZE // 2] = 1.0
    expected[BLOCK_SIZE // 2 :] = 0.0

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
