# SPDX-License-Identifier: MIT
# Copyright (c) 2025 Advanced Micro Devices, Inc. All rights reserved.

"""
Test suite for broadcast collective operation.
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
        (128, 128, 32, 32),  # BLOCK_N < N
        (256, 128, 32, 16),  # Minimum BLOCK_N=16
        (1024, 256, 32, 64),  # Medium
        (8192, 8192, 32, 64),  # Large
    ],
)
@pytest.mark.parametrize("src", [0])
def test_broadcast(dtype, M, N, block_size_m, block_size_n, src):
    """Test broadcast functionality by comparing against PyTorch's implementation."""
    if not dist.is_initialized():
        pytest.skip("torch.distributed not initialized")

    heap_size = 2**33  # 8GB
    shmem = iris.iris(heap_size)
    rank = shmem.get_rank()
    world_size = shmem.get_num_ranks()

    if src >= world_size:
        del shmem
        pytest.skip(f"src={src} >= world_size={world_size}")

    # Create deterministic input per rank — root has meaningful data,
    # non-root has different data that should be overwritten.
    pytorch_tensor = torch.randn(M, N, dtype=dtype, device=f"cuda:{rank}")
    pytorch_tensor.fill_(float(rank + 1))

    # PyTorch reference: broadcast from src
    pytorch_ref = pytorch_tensor.clone()
    shmem.barrier()
    dist.broadcast(pytorch_ref, src=src)
    torch.cuda.synchronize()

    # Iris broadcast (in-place)
    iris_tensor = shmem.zeros((M, N), dtype=dtype)
    iris_tensor.copy_(pytorch_tensor)

    shmem.barrier()
    config = Config(block_size_m=block_size_m, block_size_n=block_size_n)
    shmem.ccl.broadcast(iris_tensor, src=src, config=config)
    torch.cuda.synchronize()

    # All ranks should match
    atol = 1e-3 if dtype == torch.float16 else 1e-5
    max_diff = torch.abs(iris_tensor - pytorch_ref).max().item()

    try:
        assert torch.allclose(iris_tensor, pytorch_ref, atol=atol), (
            f"Max difference: {max_diff}, expected < {atol}\n"
            f"Rank {rank}: Iris broadcast output doesn't match PyTorch's broadcast (src={src})"
        )
    finally:
        shmem.barrier()
        del shmem
        import gc

        gc.collect()


def test_broadcast_last_rank_src():
    """Test broadcast with the last rank as source."""
    if not dist.is_initialized():
        pytest.skip("torch.distributed not initialized")

    heap_size = 2**33
    shmem = iris.iris(heap_size)
    rank = shmem.get_rank()
    world_size = shmem.get_num_ranks()

    actual_src = world_size - 1

    M, N = 512, 256
    dtype = torch.float32

    pytorch_tensor = torch.randn(M, N, dtype=dtype, device=f"cuda:{rank}")
    pytorch_tensor.fill_(float(rank + 1))

    pytorch_ref = pytorch_tensor.clone()
    shmem.barrier()
    dist.broadcast(pytorch_ref, src=actual_src)
    torch.cuda.synchronize()

    iris_tensor = shmem.zeros((M, N), dtype=dtype)
    iris_tensor.copy_(pytorch_tensor)

    shmem.barrier()
    config = Config()
    shmem.ccl.broadcast(iris_tensor, src=actual_src, config=config)
    torch.cuda.synchronize()

    atol = 1e-5
    max_diff = torch.abs(iris_tensor - pytorch_ref).max().item()

    try:
        assert torch.allclose(iris_tensor, pytorch_ref, atol=atol), (
            f"Max difference: {max_diff}, expected < {atol}\n"
            f"Rank {rank}: Iris broadcast output doesn't match PyTorch's broadcast (src={actual_src})"
        )
    finally:
        shmem.barrier()
        del shmem
        import gc

        gc.collect()


def test_broadcast_invalid_src():
    """Test that broadcast raises ValueError for invalid src."""
    if not dist.is_initialized():
        pytest.skip("torch.distributed not initialized")

    heap_size = 2**33
    shmem = iris.iris(heap_size)
    world_size = shmem.get_num_ranks()

    M, N = 128, 64
    iris_tensor = shmem.zeros((M, N), dtype=torch.float32)

    shmem.barrier()

    with pytest.raises(ValueError, match="src must be in"):
        shmem.ccl.broadcast(iris_tensor, src=world_size)

    with pytest.raises(ValueError, match="src must be in"):
        shmem.ccl.broadcast(iris_tensor, src=-1)

    shmem.barrier()
    del shmem
    import gc

    gc.collect()
