# SPDX-License-Identifier: MIT
# Copyright (c) 2025 Advanced Micro Devices, Inc. All rights reserved.

"""
Test suite for reduce collective operation.
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
        (128, 128, 32, 32),  # BLOCK_N < N/world_size
        (256, 128, 32, 16),  # Minimum BLOCK_N=16
        (1024, 256, 32, 64),  # Medium
        (8192, 8192, 32, 64),  # Large
    ],
)
@pytest.mark.parametrize("dst", [0])
def test_reduce(dtype, M, N, block_size_m, block_size_n, dst):
    """Test reduce functionality by comparing against PyTorch's implementation."""
    if not dist.is_initialized():
        pytest.skip("torch.distributed not initialized")

    heap_size = 2**33  # 8GB
    shmem = iris.iris(heap_size)
    rank = shmem.get_rank()
    world_size = shmem.get_num_ranks()

    if dst >= world_size:
        del shmem
        pytest.skip(f"dst={dst} >= world_size={world_size}")

    # Create deterministic input per rank
    pytorch_input_tensor = torch.randn(M, N, dtype=dtype, device=f"cuda:{rank}")
    pytorch_input_tensor.fill_(float(rank + 1))

    # PyTorch reference: reduce to dst
    pytorch_ref = pytorch_input_tensor.clone()
    shmem.barrier()
    dist.reduce(pytorch_ref, dst=dst, op=dist.ReduceOp.SUM)
    torch.cuda.synchronize()

    # Iris reduce
    iris_input_tensor = shmem.zeros((M, N), dtype=dtype)
    iris_input_tensor.copy_(pytorch_input_tensor)
    iris_output_tensor = shmem.zeros((M, N), dtype=dtype)

    shmem.barrier()
    config = Config(block_size_m=block_size_m, block_size_n=block_size_n)
    shmem.ccl.reduce(iris_output_tensor, iris_input_tensor, dst=dst, config=config)
    torch.cuda.synchronize()

    # Only root rank needs to match
    if rank == dst:
        atol = 1e-3 if dtype == torch.float16 else 1e-5
        max_diff = torch.abs(iris_output_tensor - pytorch_ref).max().item()

        try:
            assert torch.allclose(iris_output_tensor, pytorch_ref, atol=atol), (
                f"Max difference: {max_diff}, expected < {atol}\n"
                f"Rank {rank}: Iris reduce output doesn't match PyTorch's reduce (dst={dst})"
            )
        finally:
            shmem.barrier()
            del shmem
            import gc

            gc.collect()
    else:
        # Non-root: output is undefined, just clean up
        shmem.barrier()
        del shmem
        import gc

        gc.collect()


@pytest.mark.parametrize("dst", [0])
def test_reduce_non_root_dst(dst):
    """Test reduce with a non-zero destination rank."""
    if not dist.is_initialized():
        pytest.skip("torch.distributed not initialized")

    heap_size = 2**33
    shmem = iris.iris(heap_size)
    rank = shmem.get_rank()
    world_size = shmem.get_num_ranks()

    # Test with last rank as dst (if world_size > 1)
    actual_dst = world_size - 1
    if actual_dst == 0 and world_size > 1:
        actual_dst = 1

    M, N = 512, 256
    dtype = torch.float32

    pytorch_input_tensor = torch.randn(M, N, dtype=dtype, device=f"cuda:{rank}")
    pytorch_input_tensor.fill_(float(rank + 1))

    pytorch_ref = pytorch_input_tensor.clone()
    shmem.barrier()
    dist.reduce(pytorch_ref, dst=actual_dst, op=dist.ReduceOp.SUM)
    torch.cuda.synchronize()

    iris_input_tensor = shmem.zeros((M, N), dtype=dtype)
    iris_input_tensor.copy_(pytorch_input_tensor)
    iris_output_tensor = shmem.zeros((M, N), dtype=dtype)

    shmem.barrier()
    config = Config()
    shmem.ccl.reduce(iris_output_tensor, iris_input_tensor, dst=actual_dst, config=config)
    torch.cuda.synchronize()

    if rank == actual_dst:
        atol = 1e-5
        max_diff = torch.abs(iris_output_tensor - pytorch_ref).max().item()

        try:
            assert torch.allclose(iris_output_tensor, pytorch_ref, atol=atol), (
                f"Max difference: {max_diff}, expected < {atol}\n"
                f"Rank {rank}: Iris reduce output doesn't match PyTorch's reduce (dst={actual_dst})"
            )
        finally:
            shmem.barrier()
            del shmem
            import gc

            gc.collect()
    else:
        shmem.barrier()
        del shmem
        import gc

        gc.collect()


def test_reduce_invalid_dst():
    """Test that reduce raises ValueError for invalid dst."""
    if not dist.is_initialized():
        pytest.skip("torch.distributed not initialized")

    heap_size = 2**33
    shmem = iris.iris(heap_size)
    world_size = shmem.get_num_ranks()

    M, N = 128, 64
    iris_input = shmem.zeros((M, N), dtype=torch.float32)
    iris_output = shmem.zeros((M, N), dtype=torch.float32)

    shmem.barrier()

    with pytest.raises(ValueError, match="dst must be in"):
        shmem.ccl.reduce(iris_output, iris_input, dst=world_size)

    with pytest.raises(ValueError, match="dst must be in"):
        shmem.ccl.reduce(iris_output, iris_input, dst=-1)

    shmem.barrier()
    del shmem
    import gc

    gc.collect()
