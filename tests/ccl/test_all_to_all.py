# SPDX-License-Identifier: MIT
# Copyright (c) 2025 Advanced Micro Devices, Inc. All rights reserved.

"""
Test suite for all-to-all collective operation.
"""

import pytest
import torch
import iris
from iris.ccl import all_to_all, Config


@pytest.mark.parametrize(
    "dtype",
    [
        torch.float16,
        torch.float32,
        torch.bfloat16,
    ],
)
@pytest.mark.parametrize(
    "M, N",
    [
        (128, 64),   # Small
        (512, 256),  # Medium
        (1024, 512), # Large
    ],
)
def test_all_to_all(dtype, M, N):
    """Test basic all-to-all functionality with various sizes and dtypes."""
    heap_size = 2**30  # 1GB
    shmem = iris.iris(heap_size)
    rank = shmem.get_rank()
    world_size = shmem.get_num_ranks()
    
    # Create concatenated input/output tensors
    input_concat = shmem.zeros((M, N * world_size), dtype=dtype)
    output_concat = shmem.zeros((M, N * world_size), dtype=dtype)
    expected_concat = shmem.zeros((M, N * world_size), dtype=dtype)
    
    # Initialize input: rank sends data at position (target_rank * N)
    for target_rank in range(world_size):
        val = float(rank * 1000 + target_rank)
        input_concat[:, target_rank * N : (target_rank + 1) * N] = val
    
    # Expected output: receive from target_rank at position (target_rank * N)
    for target_rank in range(world_size):
        expected_val = float(target_rank * 1000 + rank)
        expected_concat[:, target_rank * N : (target_rank + 1) * N] = expected_val
    
    # Perform all-to-all
    config = Config()
    shmem.barrier()
    all_to_all(output_concat, input_concat, shmem, config=config)
    torch.cuda.synchronize()
    shmem.barrier()
    
    # Validate results
    atol = 1e-3 if dtype == torch.float16 else 1e-5
    max_diff = torch.abs(output_concat - expected_concat).max().item()
    
    
    assert torch.allclose(output_concat, expected_concat, atol=atol), f"Max difference: {max_diff}, expected < {atol}"
