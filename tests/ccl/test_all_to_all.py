# SPDX-License-Identifier: MIT
# Copyright (c) 2025 Advanced Micro Devices, Inc. All rights reserved.

"""
Test suite for all-to-all collective operation.
"""

import pytest
import torch
import torch.distributed as dist
import iris
from iris.ccl import Config


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
        (128, 64),  # Small
        (1024, 256),  # Medium
        (8192, 8192),  # Large
    ],
)
def test_all_to_all(dtype, M, N):
    """Test all-to-all functionality by comparing against PyTorch's implementation."""
    # Ensure torch.distributed is initialized (should be done by test runner)
    if not dist.is_initialized():
        pytest.skip("torch.distributed not initialized")

    heap_size = 2**33  # 1GB
    shmem = iris.iris(heap_size)
    rank = shmem.get_rank()
    world_size = shmem.get_num_ranks()

    # PyTorch's all_to_all format: each rank has M x N data to send to all ranks
    # Create input data: each rank has its own M x N chunk
    # For rank r, the data it sends to all ranks is the same (M x N tensor)
    pytorch_input_tensor = torch.randn(M, N, dtype=dtype, device=f"cuda:{rank}")
    # Fill with deterministic values for easier debugging
    pytorch_input_tensor.fill_(float(rank))

    # PyTorch all_to_all expects list of tensors: input_list[i] is sent to rank i
    # Since we're sending the same data to all ranks, we replicate it
    pytorch_input_list = [pytorch_input_tensor.clone() for _ in range(world_size)]
    pytorch_output_list = [torch.zeros(M, N, dtype=dtype, device=f"cuda:{rank}") for _ in range(world_size)]

    # Run PyTorch's all_to_all to get reference output
    shmem.barrier()
    dist.all_to_all(pytorch_output_list, pytorch_input_list)
    torch.cuda.synchronize()

    # Convert PyTorch output to concatenated format for comparison
    # pytorch_output_list[i] contains data received from rank i
    pytorch_output_concat = torch.zeros(M, N * world_size, dtype=dtype, device=f"cuda:{rank}")
    for target_rank in range(world_size):
        pytorch_output_concat[:, target_rank * N : (target_rank + 1) * N] = pytorch_output_list[target_rank]

    # Now set up Iris all_to_all format
    # Iris format: concatenated tensor (M, N * world_size)
    # input[:, i*N:(i+1)*N] contains data to send to rank i
    # Since we're sending the same M x N data to all ranks, we replicate it
    iris_input_concat = shmem.zeros((M, N * world_size), dtype=dtype)
    for target_rank in range(world_size):
        iris_input_concat[:, target_rank * N : (target_rank + 1) * N] = pytorch_input_tensor

    iris_output_concat = shmem.zeros((M, N * world_size), dtype=dtype)

    # Run Iris all_to_all
    shmem.barrier()
    config = Config()
    shmem.ccl.all_to_all(iris_output_concat, iris_input_concat, config=config)
    torch.cuda.synchronize()

    # Compare results
    atol = 1e-3 if dtype == torch.float16 else 1e-5
    max_diff = torch.abs(iris_output_concat - pytorch_output_concat).max().item()

    try:
        assert torch.allclose(iris_output_concat, pytorch_output_concat, atol=atol), (
            f"Max difference: {max_diff}, expected < {atol}\nRank {rank}: Iris output doesn't match PyTorch's all_to_all"
        )
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
