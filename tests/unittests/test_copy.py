# SPDX-License-Identifier: MIT
# Copyright (c) 2025 Advanced Micro Devices, Inc. All rights reserved.

import torch
import triton
import triton.language as tl
import pytest
import iris


@triton.jit
def copy_kernel(
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

    for target_rank in range(num_ranks):
        src_data = data + BLOCK_SIZE * cur_rank
        dest_data = results + BLOCK_SIZE * target_rank
        iris.copy(src_data + offsets, dest_data + offsets, target_rank, cur_rank, cur_rank, heap_bases, mask)


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
def test_copy(dtype, BLOCK_SIZE):
    shmem = iris.iris(1 << 20)
    num_ranks = shmem.get_num_ranks()
    heap_bases = shmem.get_heap_bases()
    cur_rank = shmem.get_rank()

    data = shmem.zeros((num_ranks, BLOCK_SIZE), dtype=dtype)
    base = cur_rank + num_ranks
    for i in range(num_ranks):
        data[i, :] = base * (i + 1)

    results = shmem.zeros((num_ranks, BLOCK_SIZE), dtype=dtype)
    grid = lambda meta: (1,)
    copy_kernel[grid](data, results, cur_rank, num_ranks, BLOCK_SIZE, heap_bases)
    shmem.barrier()

    expected = shmem.zeros((num_ranks, BLOCK_SIZE), dtype=dtype)
    for rank_id in range(num_ranks):
        expected[rank_id, :] = (rank_id + num_ranks) * (cur_rank + 1)

    try:
        torch.testing.assert_close(results, expected, rtol=0, atol=0)
    except AssertionError as e:
        print(e)
        print("Expected:", expected)
        print("Actual:", results)
        raise
