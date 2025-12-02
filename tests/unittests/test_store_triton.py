# SPDX-License-Identifier: MIT
# Copyright (c) 2025 Advanced Micro Devices, Inc. All rights reserved.

import torch
import triton
import triton.language as tl
import pytest
import iris


@triton.jit
def store_kernel(
    data,
    results,
    destination_rank: tl.constexpr,
    num_ranks: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
    heap_bases: tl.tensor,
):
    pid = tl.program_id(0)

    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)

    mask = offsets < BLOCK_SIZE

    # Load the data from src for this block
    value = tl.load(data + offsets, mask=mask)

    # Store data to all ranks
    # Doesn't matter which rank stores at the end, the data should all be the same at the end.
    for dst_rank in range(num_ranks):
        iris.store(results + offsets, value, destination_rank, dst_rank, heap_bases, mask=mask)


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
    shmem = iris.iris(1 << 20)
    num_ranks = shmem.get_num_ranks()
    heap_bases = shmem.get_heap_bases()
    destination_rank = shmem.get_rank()

    src = shmem.ones(BLOCK_SIZE, dtype=dtype)
    results = shmem.zeros_like(src)

    shmem.barrier()

    grid = lambda meta: (1,)
    store_kernel[grid](src, results, destination_rank, num_ranks, BLOCK_SIZE, heap_bases)
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
