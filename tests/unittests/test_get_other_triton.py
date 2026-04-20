# SPDX-License-Identifier: MIT
# Copyright (c) 2025 Advanced Micro Devices, Inc. All rights reserved.

import torch
import triton
import triton.language as tl
import pytest
import iris


@triton.jit
def get_with_other_kernel(
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

    acc = tl.zeros([BLOCK_SIZE], dtype=data.type.element_ty)

    # Loop over all ranks, get the stored data.
    # load to local register, accumulate.
    for target_rank in range(num_ranks):
        iris.get(data + offsets, results + offsets, cur_rank, target_rank, heap_bases, mask=mask, other=other_value)
        acc += tl.load(results + offsets)

    # Store the accumulated value back to the output.
    tl.store(results + offsets, acc)


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
def test_get_other_api(dtype, BLOCK_SIZE):
    # TODO: Adjust heap size.
    shmem = iris.iris(1 << 20)
    num_ranks = shmem.get_num_ranks()
    heap_bases = shmem.get_heap_bases()
    cur_rank = shmem.get_rank()

    data = shmem.ones(BLOCK_SIZE, dtype=dtype)
    results = shmem.zeros_like(data)

    # Use -1 as the "other" value for masked-out elements
    other_value = -1.0

    shmem.barrier()

    grid = lambda meta: (1,)
    get_with_other_kernel[grid](data, results, cur_rank, num_ranks, BLOCK_SIZE, heap_bases, other_value)
    shmem.barrier()

    # Verify the results
    # First half should contain loaded values accumulated from all ranks (num_ranks * 1.0)
    # Second half stays at 0.0 because iris.get stores with mask, so masked-out positions
    # in `results` are never written; tl.load(results + offsets) reads 0.0 from them.
    expected = torch.zeros(BLOCK_SIZE, dtype=dtype, device="cuda")
    expected[: BLOCK_SIZE // 2] = num_ranks * 1.0
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
