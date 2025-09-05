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
def atomic_xchg_kernel(
    results,
    sem: tl.constexpr,
    scope: tl.constexpr,
    cur_rank: tl.constexpr,
    num_ranks: tl.constexpr,
    heap_bases: tl.tensor,
):
    # Cast constants to match results.dtype
    dtype = results.dtype.element_ty
    val = tl.full((), num_ranks, dtype=dtype)  # scalar num_ranks

    for target_rank in range(num_ranks):
        iris.atomic_xchg(results, val, cur_rank, target_rank, heap_bases, mask=None, sem=sem, scope=scope)


@pytest.mark.parametrize(
    "scope",
    [
        "cta",
        "gpu",
        "sys",
    ],
)
@pytest.mark.parametrize(
    "num_ranks",
    [
        2,
    ],
)
def test_atomic_xchg_api(dtype, sem, scope, num_ranks):
    # TODO: Adjust heap size.
    """Test with distributed setup."""
    
    @distributed_test(num_ranks=num_ranks)
    def _test_atomic_xchg_api_distributed(local_rank, world_size):
    shmem = iris.iris(1 << 20)
        
        return True
    
    # Run the distributed test
    result = _test_atomic_xchg_api_distributed()
    assert result is True
    num_ranks = shmem.get_num_ranks()
    heap_bases = shmem.get_heap_bases()
    cur_rank = shmem.get_rank()

    results = shmem.zeros((1,), dtype=dtype)

    shmem.barrier()

    grid = lambda meta: (1,)
    atomic_xchg_kernel[grid](results, sem, scope, cur_rank, num_ranks, heap_bases)
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
