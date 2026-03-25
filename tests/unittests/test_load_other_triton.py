# SPDX-License-Identifier: MIT
# Copyright (c) 2025 Advanced Micro Devices, Inc. All rights reserved.

import torch
import triton
import triton.language as tl
import pytest
import iris


@triton.jit
def load_with_other_kernel(
    data,
    results,
    source_rank: tl.constexpr,
    num_ranks: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
    heap_bases: tl.tensor,
    other_value: tl.constexpr,
):
    pid = tl.program_id(0)

    partner = int((source_rank + num_ranks // 2) % num_ranks)
    # Compute start index of this block
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)

    # Create a mask that is False for half the elements
    mask = offsets < BLOCK_SIZE // 2

    # Load with mask and other parameter
    result = iris.load(data + offsets, source_rank, partner, heap_bases, mask=mask, other=other_value)
    tl.store(results + offsets, result)


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
def test_load_other_api(dtype, BLOCK_SIZE):
    # TODO: Adjust heap size.
    shmem = iris.iris(1 << 20)
    num_ranks = shmem.get_num_ranks()
    heap_bases = shmem.get_heap_bases()
    source_rank = shmem.get_rank()
    partner = int((source_rank + num_ranks // 2) % num_ranks)

    # Fill data with source rank value so remote reads match expected values:
    # each rank's data[i] = source_rank, so loading from partner gives partner's rank value
    data = shmem.full((BLOCK_SIZE,), source_rank, dtype=dtype)
    results = shmem.zeros_like(data)

    # Use -1 as the "other" value for masked-out elements
    other_value = -1.0

    shmem.barrier()

    grid = lambda meta: (1,)
    load_with_other_kernel[grid](data, results, source_rank, num_ranks, BLOCK_SIZE, heap_bases, other_value)
    shmem.barrier()

    # Verify the result
    # First half should contain loaded values (partner rank)
    # Second half should contain the "other" value (-1.0)
    expected = torch.zeros(BLOCK_SIZE, dtype=dtype, device="cuda")
    expected[: BLOCK_SIZE // 2] = partner
    expected[BLOCK_SIZE // 2 :] = other_value

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
