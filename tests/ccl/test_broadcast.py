# SPDX-License-Identifier: MIT
# Copyright (c) 2025 Advanced Micro Devices, Inc. All rights reserved.

"""
Test suite for broadcast collective operation.
"""

import gc
import pytest
import torch
import torch.distributed as dist
import iris


@pytest.mark.parametrize(
    "dtype",
    [
        torch.float16,
        torch.float32,
        torch.bfloat16,
    ],
)
@pytest.mark.parametrize(
    "numel",
    [
        1,  # Single element
        128,  # Small
        1024,  # Medium
        65536,  # Large (64K)
        1048576,  # Very large (1M)
    ],
)
@pytest.mark.parametrize(
    "src",
    [
        0,  # First rank
        "mid",  # Middle rank
        "last",  # Last rank
    ],
)
def test_broadcast(dtype, numel, src):
    """Test broadcast by comparing against PyTorch's dist.broadcast."""
    if not dist.is_initialized():
        pytest.skip("torch.distributed not initialized")

    heap_size = 2**33  # 8GB
    shmem = iris.iris(heap_size)
    rank = shmem.get_rank()
    world_size = shmem.get_num_ranks()

    # Resolve symbolic src values
    if src == "mid":
        src = world_size // 2
    elif src == "last":
        src = world_size - 1

    # Create reference tensor: src fills with (rank+1), others with zeros
    ref_tensor = torch.zeros(numel, dtype=dtype, device=f"cuda:{rank}")
    if rank == src:
        ref_tensor.fill_(float(src + 1))

    # PyTorch reference
    pytorch_tensor = ref_tensor.clone()
    shmem.barrier()
    dist.broadcast(pytorch_tensor, src=src)
    torch.cuda.synchronize()

    # Iris broadcast
    iris_tensor = shmem.zeros((numel,), dtype=dtype)
    if rank == src:
        iris_tensor.fill_(float(src + 1))

    shmem.barrier()
    shmem.ccl.broadcast(iris_tensor, src=src)
    torch.cuda.synchronize()

    # Compare
    atol = 1e-3 if dtype == torch.float16 else 1e-5
    max_diff = torch.abs(iris_tensor - pytorch_tensor).max().item()

    try:
        assert torch.allclose(iris_tensor, pytorch_tensor, atol=atol), (
            f"Max difference: {max_diff}, expected < {atol}\n"
            f"Rank {rank}: Iris broadcast doesn't match PyTorch (src={src}, numel={numel}, dtype={dtype})"
        )
    finally:
        shmem.barrier()
        del shmem
        gc.collect()


@pytest.mark.parametrize(
    "shape",
    [
        (128, 64),  # 2D
        (16, 32, 8),  # 3D
    ],
)
def test_broadcast_multidim(shape):
    """Test broadcast works with multi-dimensional tensors."""
    if not dist.is_initialized():
        pytest.skip("torch.distributed not initialized")

    heap_size = 2**33
    shmem = iris.iris(heap_size)
    rank = shmem.get_rank()

    dtype = torch.float32
    src = 0

    # PyTorch reference
    pytorch_tensor = torch.zeros(shape, dtype=dtype, device=f"cuda:{rank}")
    if rank == src:
        pytorch_tensor.fill_(42.0)
    shmem.barrier()
    dist.broadcast(pytorch_tensor, src=src)
    torch.cuda.synchronize()

    # Iris
    iris_tensor = shmem.zeros(shape, dtype=dtype)
    if rank == src:
        iris_tensor.fill_(42.0)
    shmem.barrier()
    shmem.ccl.broadcast(iris_tensor, src=src)
    torch.cuda.synchronize()

    try:
        assert torch.allclose(iris_tensor, pytorch_tensor, atol=1e-5), (
            f"Rank {rank}: Iris broadcast doesn't match PyTorch for shape {shape}"
        )
    finally:
        shmem.barrier()
        del shmem
        gc.collect()
