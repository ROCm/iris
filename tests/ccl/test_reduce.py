# SPDX-License-Identifier: MIT
# Copyright (c) 2025 Advanced Micro Devices, Inc. All rights reserved.

"""
Test suite for reduce collective operation.
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
    "dst",
    [
        0,  # First rank
        "mid",  # Middle rank
        "last",  # Last rank
    ],
)
def test_reduce(dtype, numel, dst):
    """Test reduce by comparing against PyTorch's dist.reduce."""
    if not dist.is_initialized():
        pytest.skip("torch.distributed not initialized")

    heap_size = 2**33  # 8GB
    shmem = iris.iris(heap_size)
    rank = shmem.get_rank()
    world_size = shmem.get_num_ranks()

    # Resolve symbolic dst values
    if dst == "mid":
        dst = world_size // 2
    elif dst == "last":
        dst = world_size - 1

    # Each rank fills with (rank + 1)
    # After reduce with SUM on dst, result should be sum(1..world_size) = W*(W+1)/2
    pytorch_tensor = torch.full((numel,), float(rank + 1), dtype=dtype, device=f"cuda:{rank}")

    # PyTorch reference
    pytorch_ref = pytorch_tensor.clone()
    shmem.barrier()
    dist.reduce(pytorch_ref, dst=dst, op=dist.ReduceOp.SUM)
    torch.cuda.synchronize()

    # Iris reduce
    iris_tensor = shmem.zeros((numel,), dtype=dtype)
    iris_tensor.copy_(pytorch_tensor)

    shmem.barrier()
    shmem.ccl.reduce(iris_tensor, dst=dst)
    torch.cuda.synchronize()

    # Only dst rank has valid result
    if rank == dst:
        atol = 1e-2 if dtype == torch.float16 else 1e-4
        max_diff = torch.abs(iris_tensor - pytorch_ref).max().item()

        try:
            assert torch.allclose(iris_tensor, pytorch_ref, atol=atol), (
                f"Max difference: {max_diff}, expected < {atol}\n"
                f"Rank {rank}: Iris reduce doesn't match PyTorch (dst={dst}, numel={numel}, dtype={dtype})"
            )
        except AssertionError:
            # Print debug info
            expected_sum = world_size * (world_size + 1) / 2
            print(f"Rank {rank}: expected sum per element = {expected_sum}")
            print(f"Rank {rank}: iris[0] = {iris_tensor[0].item()}, pytorch[0] = {pytorch_ref[0].item()}")
            raise
    # Non-dst ranks: tensor should be unchanged (still rank+1)

    try:
        pass  # assertion above covers dst rank
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
def test_reduce_multidim(shape):
    """Test reduce works with multi-dimensional tensors."""
    if not dist.is_initialized():
        pytest.skip("torch.distributed not initialized")

    heap_size = 2**33
    shmem = iris.iris(heap_size)
    rank = shmem.get_rank()
    world_size = shmem.get_num_ranks()

    dtype = torch.float32
    dst = 0

    pytorch_tensor = torch.full(shape, float(rank + 1), dtype=dtype, device=f"cuda:{rank}")

    # PyTorch reference
    pytorch_ref = pytorch_tensor.clone()
    shmem.barrier()
    dist.reduce(pytorch_ref, dst=dst, op=dist.ReduceOp.SUM)
    torch.cuda.synchronize()

    # Iris
    iris_tensor = shmem.zeros(shape, dtype=dtype)
    iris_tensor.copy_(pytorch_tensor)
    shmem.barrier()
    shmem.ccl.reduce(iris_tensor, dst=dst)
    torch.cuda.synchronize()

    if rank == dst:
        try:
            assert torch.allclose(iris_tensor, pytorch_ref, atol=1e-4), (
                f"Rank {rank}: Iris reduce doesn't match PyTorch for shape {shape}"
            )
        finally:
            shmem.barrier()
            del shmem
            gc.collect()
    else:
        shmem.barrier()
        del shmem
        gc.collect()
