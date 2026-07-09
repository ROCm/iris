# SPDX-License-Identifier: MIT
# Copyright (c) 2025 Advanced Micro Devices, Inc. All rights reserved.

"""
Test suite for broadcast collective operation.

Verifies correctness by comparing iris broadcast against
torch.distributed.broadcast (which uses RCCL on AMD GPUs).
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
        (32, 32, 32, 32),  # Tiny
        (128, 64, 32, 64),  # Small
        (128, 128, 32, 32),  # BLOCK_N < N (multi-block per row)
        (256, 128, 32, 16),  # Minimum BLOCK_N=16
        (1024, 256, 32, 64),  # Medium
        (8192, 8192, 32, 64),  # Large
    ],
)
def test_broadcast_src0(dtype, M, N, block_size_m, block_size_n):
    """Test broadcast from rank 0 by comparing against PyTorch's broadcast."""
    if not dist.is_initialized():
        pytest.skip("torch.distributed not initialized")

    heap_size = 2**33  # 8GB
    shmem = iris.iris(heap_size)
    rank = shmem.get_rank()
    world_size = shmem.get_num_ranks()

    src = 0

    # Create deterministic input: only src rank's data matters
    pytorch_tensor = torch.randn(M, N, dtype=dtype, device=f"cuda:{rank}")
    pytorch_tensor.fill_(float(rank + 1))  # Each rank has distinct values

    # PyTorch reference: broadcast from src
    pytorch_ref = pytorch_tensor.clone()
    shmem.barrier()
    dist.broadcast(pytorch_ref, src=src)
    torch.cuda.synchronize()

    # Iris broadcast
    iris_input = shmem.zeros((M, N), dtype=dtype)
    iris_input.copy_(pytorch_tensor)
    iris_output = shmem.zeros((M, N), dtype=dtype)

    shmem.barrier()
    config = Config(block_size_m=block_size_m, block_size_n=block_size_n)
    shmem.ccl.broadcast(iris_output, iris_input, src=src, config=config)
    torch.cuda.synchronize()

    # Compare results
    atol = 1e-3 if dtype == torch.float16 else 1e-5
    max_diff = torch.abs(iris_output - pytorch_ref).max().item()

    try:
        assert torch.allclose(iris_output, pytorch_ref, atol=atol), (
            f"Max difference: {max_diff}, expected < {atol}\n"
            f"Rank {rank}: Iris broadcast output doesn't match PyTorch's broadcast (src={src})\n"
            f"Expected all values to be {float(src + 1)}"
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
        (128, 64),  # Small
        (1024, 256),  # Medium
        (4096, 4096),  # Large
    ],
)
def test_broadcast_nonzero_src(dtype, M, N):
    """Test broadcast from non-zero source ranks."""
    if not dist.is_initialized():
        pytest.skip("torch.distributed not initialized")

    heap_size = 2**33
    shmem = iris.iris(heap_size)
    rank = shmem.get_rank()
    world_size = shmem.get_num_ranks()

    # Test with src = world_size - 1 (last rank)
    src = world_size - 1

    pytorch_tensor = torch.randn(M, N, dtype=dtype, device=f"cuda:{rank}")
    pytorch_tensor.fill_(float(rank + 1))

    # PyTorch reference
    pytorch_ref = pytorch_tensor.clone()
    shmem.barrier()
    dist.broadcast(pytorch_ref, src=src)
    torch.cuda.synchronize()

    # Iris broadcast
    iris_input = shmem.zeros((M, N), dtype=dtype)
    iris_input.copy_(pytorch_tensor)
    iris_output = shmem.zeros((M, N), dtype=dtype)

    shmem.barrier()
    config = Config(block_size_m=32, block_size_n=64)
    shmem.ccl.broadcast(iris_output, iris_input, src=src, config=config)
    torch.cuda.synchronize()

    atol = 1e-3 if dtype == torch.float16 else 1e-5
    max_diff = torch.abs(iris_output - pytorch_ref).max().item()

    try:
        assert torch.allclose(iris_output, pytorch_ref, atol=atol), (
            f"Max difference: {max_diff}, expected < {atol}\n"
            f"Rank {rank}: Iris broadcast output doesn't match PyTorch (src={src})\n"
            f"Expected all values to be {float(src + 1)}"
        )
    finally:
        shmem.barrier()
        del shmem
        import gc

        gc.collect()


@pytest.mark.parametrize("dtype", [torch.float16, torch.float32, torch.bfloat16])
def test_broadcast_random_data(dtype):
    """Test broadcast with random (non-fill) data to catch subtle indexing bugs."""
    if not dist.is_initialized():
        pytest.skip("torch.distributed not initialized")

    heap_size = 2**33
    shmem = iris.iris(heap_size)
    rank = shmem.get_rank()
    world_size = shmem.get_num_ranks()

    M, N = 512, 512
    src = 0

    # Use random data seeded by src rank for reproducibility
    torch.manual_seed(42)
    pytorch_tensor = torch.randn(M, N, dtype=dtype, device=f"cuda:{rank}")

    # All ranks need the same reference: broadcast from src
    pytorch_ref = pytorch_tensor.clone()
    shmem.barrier()
    dist.broadcast(pytorch_ref, src=src)
    torch.cuda.synchronize()

    # Iris broadcast
    iris_input = shmem.zeros((M, N), dtype=dtype)
    iris_input.copy_(pytorch_tensor)
    iris_output = shmem.zeros((M, N), dtype=dtype)

    shmem.barrier()
    config = Config(block_size_m=32, block_size_n=64)
    shmem.ccl.broadcast(iris_output, iris_input, src=src, config=config)
    torch.cuda.synchronize()

    atol = 1e-3 if dtype == torch.float16 else 1e-5
    max_diff = torch.abs(iris_output - pytorch_ref).max().item()

    try:
        assert torch.allclose(iris_output, pytorch_ref, atol=atol), (
            f"Max difference: {max_diff}, expected < {atol}\n"
            f"Rank {rank}: Iris broadcast with random data doesn't match PyTorch"
        )
    finally:
        shmem.barrier()
        del shmem
        import gc

        gc.collect()


@pytest.mark.parametrize("dtype", [torch.float16, torch.float32, torch.bfloat16])
def test_broadcast_inplace(dtype):
    """Test broadcast where input and output are the same tensor (in-place)."""
    if not dist.is_initialized():
        pytest.skip("torch.distributed not initialized")

    heap_size = 2**33
    shmem = iris.iris(heap_size)
    rank = shmem.get_rank()

    M, N = 256, 256
    src = 0

    pytorch_tensor = torch.randn(M, N, dtype=dtype, device=f"cuda:{rank}")
    pytorch_tensor.fill_(float(rank + 1))

    # PyTorch reference (in-place broadcast)
    pytorch_ref = pytorch_tensor.clone()
    shmem.barrier()
    dist.broadcast(pytorch_ref, src=src)
    torch.cuda.synchronize()

    # Iris broadcast in-place (same tensor for input and output)
    iris_tensor = shmem.zeros((M, N), dtype=dtype)
    iris_tensor.copy_(pytorch_tensor)

    shmem.barrier()
    config = Config(block_size_m=32, block_size_n=64)
    shmem.ccl.broadcast(iris_tensor, iris_tensor, src=src, config=config)
    torch.cuda.synchronize()

    atol = 1e-3 if dtype == torch.float16 else 1e-5

    try:
        assert torch.allclose(iris_tensor, pytorch_ref, atol=atol), (
            f"Rank {rank}: In-place broadcast doesn't match PyTorch"
        )
    finally:
        shmem.barrier()
        del shmem
        import gc

        gc.collect()
