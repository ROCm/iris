# SPDX-License-Identifier: MIT
# Copyright (c) 2025 Advanced Micro Devices, Inc. All rights reserved.

import torch
import triton
import triton.language as tl
import pytest
import iris
import sys
from pathlib import Path

# Add tests directory to path for test_utils
current_dir = Path(__file__).parent
tests_dir = current_dir.parent
sys.path.insert(0, str(tests_dir))

from test_utils import distributed_test


@triton.jit
def atomic_min_kernel(
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
        iris.atomic_min(results + offsets, acc, cur_rank, target_rank, heap_bases, mask, sem=sem, scope=scope)


@pytest.mark.parametrize(
    "BLOCK_SIZE",
    [
        1,
        8,
        16,
        32,
    ],
)
@pytest.mark.parametrize(
    "num_ranks",
    [
        2,
    ],
)
def test_atomic_min_api(dtype, sem, scope, BLOCK_SIZE, num_ranks):
    # TODO: Adjust heap size.
    """Test with distributed setup."""
    
    @distributed_test(num_ranks=num_ranks)
    def _test_atomic_min_api_distributed(local_rank, world_size):
    shmem = iris.iris(1 << 20)
        
        return True
    
    # Run the distributed test
    result = _test_atomic_min_api_distributed()
    assert result is True
    num_ranks = shmem.get_num_ranks()
    heap_bases = shmem.get_heap_bases()
    cur_rank = shmem.get_rank()

    max_val = torch.iinfo(dtype).max
    results = shmem.full((BLOCK_SIZE,), max_val, dtype=dtype)

    shmem.barrier()

    grid = lambda meta: (1,)
    atomic_min_kernel[grid](results, sem, scope, cur_rank, num_ranks, BLOCK_SIZE, heap_bases)
    shmem.barrier()
    # All ranks participate in performing the min operation
    # Each rank performs the atomic operation: min(rank_id + 1)
    # The result equals the ID of the first rank + 1
    expected = torch.full((BLOCK_SIZE,), 1, dtype=dtype, device="cuda")

    try:
        torch.testing.assert_close(results, expected, rtol=0, atol=0)
    except AssertionError as e:
        print(e)
        print("Expected:", expected)
        print("Actual  :", results)
        raise
