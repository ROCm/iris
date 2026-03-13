# SPDX-License-Identifier: MIT
# Copyright (c) 2025 Advanced Micro Devices, Inc. All rights reserved.

"""
Test suite for all-gather collective operation.
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
    "M, N, block_size_m, block_size_n",
    [
        (128, 64, 32, 64),  # Small
        (128, 128, 32, 32),  # BLOCK_N < N/world_size (partial-width, multi-block per rank)
        (256, 128, 32, 16),  # Minimum BLOCK_N=16 (16-bit vectorization path)
        (1024, 256, 32, 64),  # Medium
        (8192, 8192, 32, 64),  # Large
    ],
)
def test_all_gather(dtype, M, N, block_size_m, block_size_n):
    """Test all-gather functionality by comparing against PyTorch's implementation."""
    # Ensure torch.distributed is initialized (should be done by test runner)
    if not dist.is_initialized():
        pytest.skip("torch.distributed not initialized")

    heap_size = 2**33  # 8GB
    shmem = iris.iris(heap_size)
    rank = shmem.get_rank()
    world_size = shmem.get_num_ranks()

    # PyTorch's all_gather_into_tensor format: each rank has M x N input
    # Output is (world_size * M, N) - concatenated along dimension 0
    pytorch_input_tensor = torch.randn(M, N, dtype=dtype, device=f"cuda:{rank}")
    # Fill with deterministic values for easier debugging
    pytorch_input_tensor.fill_(float(rank + 1))

    # Create output tensor for PyTorch: (world_size * M, N)
    pytorch_output_tensor = torch.zeros(world_size * M, N, dtype=dtype, device=f"cuda:{rank}")

    # Run PyTorch's all_gather_into_tensor to get reference output
    shmem.barrier()
    dist.all_gather_into_tensor(pytorch_output_tensor, pytorch_input_tensor)
    torch.cuda.synchronize()

    # Now set up Iris all_gather format
    # Iris format: same as PyTorch - input is (M, N), output is (world_size * M, N)
    iris_input_tensor = shmem.zeros((M, N), dtype=dtype)
    iris_input_tensor.copy_(pytorch_input_tensor)

    iris_output_tensor = shmem.zeros((world_size * M, N), dtype=dtype)

    # Run Iris all_gather
    shmem.barrier()
    config = Config(block_size_m=block_size_m, block_size_n=block_size_n)
    shmem.ccl.all_gather(iris_output_tensor, iris_input_tensor, config=config)
    torch.cuda.synchronize()

    # Compare results
    atol = 1e-3 if dtype == torch.float16 else 1e-5
    max_diff = torch.abs(iris_output_tensor - pytorch_output_tensor).max().item()

    try:
        assert torch.allclose(iris_output_tensor, pytorch_output_tensor, atol=atol), (
            f"Max difference: {max_diff}, expected < {atol}\n"
            f"Rank {rank}: Iris output doesn't match PyTorch's all_gather_into_tensor"
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


@pytest.mark.parametrize(
    "dtype",
    [
        torch.float16,
        torch.float32,
        torch.bfloat16,
    ],
)
@pytest.mark.parametrize(
    "M, N, block_size_m, block_size_n",
    [
        (128, 64, 32, 64),  # Small
        (128, 128, 32, 32),  # BLOCK_N < N/world_size (partial-width, multi-block per rank)
        (256, 128, 32, 16),  # Minimum BLOCK_N=16 (16-bit vectorization path)
        (1024, 256, 32, 64),  # Medium
        (8192, 8192, 32, 64),  # Large
    ],
)
def test_all_gather_partitioned(dtype, M, N, block_size_m, block_size_n):
    """Test all-gather with partitioned variant by comparing against PyTorch's implementation."""
    # Ensure torch.distributed is initialized (should be done by test runner)
    if not dist.is_initialized():
        pytest.skip("torch.distributed not initialized")

    heap_size = 2**33  # 8GB
    shmem = iris.iris(heap_size)
    rank = shmem.get_rank()
    world_size = shmem.get_num_ranks()

    # PyTorch's all_gather_into_tensor format: each rank has M x N input
    # Output is (world_size * M, N) - concatenated along dimension 0
    pytorch_input_tensor = torch.randn(M, N, dtype=dtype, device=f"cuda:{rank}")
    # Fill with deterministic values for easier debugging
    pytorch_input_tensor.fill_(float(rank + 1))

    # Create output tensor for PyTorch: (world_size * M, N)
    pytorch_output_tensor = torch.zeros(world_size * M, N, dtype=dtype, device=f"cuda:{rank}")

    # Run PyTorch's all_gather_into_tensor to get reference output
    shmem.barrier()
    dist.all_gather_into_tensor(pytorch_output_tensor, pytorch_input_tensor)
    torch.cuda.synchronize()

    # Now set up Iris all_gather format with partitioned variant
    # Iris format: same as PyTorch - input is (M, N), output is (world_size * M, N)
    iris_input_tensor = shmem.zeros((M, N), dtype=dtype)
    iris_input_tensor.copy_(pytorch_input_tensor)

    iris_output_tensor = shmem.zeros((world_size * M, N), dtype=dtype)

    # Run Iris all_gather with partitioned variant
    # COMM_SMS must be divisible by world_size for partitioned variant
    comm_sms = 64  # Assuming world_size divides 64 (e.g., 2, 4, 8)
    shmem.barrier()
    config = Config(
        block_size_m=block_size_m, block_size_n=block_size_n, all_gather_variant="partitioned", comm_sms=comm_sms
    )
    shmem.ccl.all_gather(iris_output_tensor, iris_input_tensor, config=config)
    torch.cuda.synchronize()

    # Compare results
    atol = 1e-3 if dtype == torch.float16 else 1e-5
    max_diff = torch.abs(iris_output_tensor - pytorch_output_tensor).max().item()

    try:
        assert torch.allclose(iris_output_tensor, pytorch_output_tensor, atol=atol), (
            f"Max difference: {max_diff}, expected < {atol}\n"
            f"Rank {rank}: Iris output (partitioned) doesn't match PyTorch's all_gather_into_tensor"
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
