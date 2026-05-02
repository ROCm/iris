# SPDX-License-Identifier: MIT
# Copyright (c) 2025 Advanced Micro Devices, Inc. All rights reserved.

"""
Test suite for scatter collective operation.
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
        (128, 128, 32, 32),  # BLOCK_N < N (partial-width, multi-block)
        (256, 128, 32, 16),  # Minimum BLOCK_N=16 (16-bit vectorization path)
        (1024, 256, 32, 64),  # Medium
        (8192, 8192, 32, 64),  # Large
    ],
)
def test_scatter(dtype, M, N, block_size_m, block_size_n):
    """Test scatter functionality by comparing against PyTorch's implementation."""
    if not dist.is_initialized():
        pytest.skip("torch.distributed not initialized")

    heap_size = 2**33  # 8GB
    shmem = iris.iris(heap_size)
    rank = shmem.get_rank()
    world_size = shmem.get_num_ranks()

    # PyTorch scatter: root has a list of tensors, each (M, N)
    # Build reference using torch.distributed.scatter
    pytorch_output = torch.zeros(M, N, dtype=dtype, device=f"cuda:{rank}")
    scatter_list = None
    if rank == 0:
        scatter_list = [torch.full((M, N), float(i + 1), dtype=dtype, device=f"cuda:{rank}") for i in range(world_size)]

    shmem.barrier()
    dist.scatter(pytorch_output, scatter_list, src=0)
    torch.cuda.synchronize()

    # Iris scatter: root has (world_size * M, N) input, output is (M, N)
    # All ranks must allocate input at the same shape to maintain symmetric heap offsets
    iris_input = shmem.zeros((world_size * M, N), dtype=dtype)
    if rank == 0:
        # Fill input: chunk i = (i+1) to match scatter_list above
        for i in range(world_size):
            iris_input[i * M : (i + 1) * M, :].fill_(float(i + 1))

    iris_output = shmem.zeros((M, N), dtype=dtype)

    shmem.barrier()
    config = Config(block_size_m=block_size_m, block_size_n=block_size_n)
    shmem.ccl.scatter(iris_output, iris_input, src=0, config=config)
    torch.cuda.synchronize()

    # Compare results
    atol = 1e-3 if dtype == torch.float16 else 1e-5
    max_diff = torch.abs(iris_output - pytorch_output).max().item()

    try:
        assert torch.allclose(iris_output, pytorch_output, atol=atol), (
            f"Max difference: {max_diff}, expected < {atol}\nRank {rank}: Iris output doesn't match PyTorch's scatter"
        )
    finally:
        shmem.barrier()
        del shmem
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
    "M, N",
    [
        (128, 64),
        (1024, 256),
    ],
)
def test_scatter_nonzero_root(dtype, M, N):
    """Test scatter with a non-zero root rank."""
    if not dist.is_initialized():
        pytest.skip("torch.distributed not initialized")

    heap_size = 2**33  # 8GB
    shmem = iris.iris(heap_size)
    rank = shmem.get_rank()
    world_size = shmem.get_num_ranks()

    if world_size < 2:
        pytest.skip("Need at least 2 ranks for non-zero root test")

    src = world_size - 1  # Use last rank as root

    # PyTorch scatter with non-zero root
    pytorch_output = torch.zeros(M, N, dtype=dtype, device=f"cuda:{rank}")
    scatter_list = None
    if rank == src:
        scatter_list = [torch.full((M, N), float(i + 1), dtype=dtype, device=f"cuda:{rank}") for i in range(world_size)]

    shmem.barrier()
    dist.scatter(pytorch_output, scatter_list, src=src)
    torch.cuda.synchronize()

    # Iris scatter with non-zero root
    # All ranks must allocate input at the same shape to maintain symmetric heap offsets
    iris_input = shmem.zeros((world_size * M, N), dtype=dtype)
    if rank == src:
        for i in range(world_size):
            iris_input[i * M : (i + 1) * M, :].fill_(float(i + 1))

    iris_output = shmem.zeros((M, N), dtype=dtype)

    shmem.barrier()
    config = Config(block_size_m=32, block_size_n=64)
    shmem.ccl.scatter(iris_output, iris_input, src=src, config=config)
    torch.cuda.synchronize()

    atol = 1e-3 if dtype == torch.float16 else 1e-5
    max_diff = torch.abs(iris_output - pytorch_output).max().item()

    try:
        assert torch.allclose(iris_output, pytorch_output, atol=atol), (
            f"Max difference: {max_diff}, expected < {atol}\n"
            f"Rank {rank}: Iris scatter (src={src}) doesn't match PyTorch"
        )
    finally:
        shmem.barrier()
        del shmem
        import gc

        gc.collect()
