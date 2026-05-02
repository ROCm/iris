# SPDX-License-Identifier: MIT
# Copyright (c) 2025 Advanced Micro Devices, Inc. All rights reserved.

"""
Test suite for gather collective operation.
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
def test_gather(dtype, M, N, block_size_m, block_size_n):
    """Test gather by comparing against torch.distributed.gather."""
    if not dist.is_initialized():
        pytest.skip("torch.distributed not initialized")

    heap_size = 2**33  # 8GB
    shmem = iris.iris(heap_size)
    rank = shmem.get_rank()
    world_size = shmem.get_num_ranks()
    dst = 0

    # Each rank fills its input with a deterministic value
    pytorch_input_tensor = torch.randn(M, N, dtype=dtype, device=f"cuda:{rank}")
    pytorch_input_tensor.fill_(float(rank + 1))

    # Build reference output on root using torch.distributed.gather
    if rank == dst:
        gather_list = [torch.zeros(M, N, dtype=dtype, device=f"cuda:{rank}") for _ in range(world_size)]
    else:
        gather_list = None

    shmem.barrier()
    dist.gather(pytorch_input_tensor, gather_list, dst=dst)
    torch.cuda.synchronize()

    # Concatenate on root to get (world_size * M, N) reference
    if rank == dst:
        pytorch_output = torch.cat(gather_list, dim=0)

    # Iris gather
    iris_input_tensor = shmem.zeros((M, N), dtype=dtype)
    iris_input_tensor.copy_(pytorch_input_tensor)
    iris_output_tensor = shmem.zeros((world_size * M, N), dtype=dtype)

    shmem.barrier()
    config = Config(block_size_m=block_size_m, block_size_n=block_size_n)
    shmem.ccl.gather(iris_output_tensor, iris_input_tensor, dst=dst, config=config)
    torch.cuda.synchronize()

    # Only verify output on root
    if rank == dst:
        atol = 1e-3 if dtype == torch.float16 else 1e-5
        max_diff = torch.abs(iris_output_tensor - pytorch_output).max().item()

        try:
            assert torch.allclose(iris_output_tensor, pytorch_output, atol=atol), (
                f"Max difference: {max_diff}, expected < {atol}\n"
                f"Rank {rank}: Iris output doesn't match torch.distributed.gather"
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
def test_gather_nonzero_root(dtype, M, N):
    """Test gather with a non-zero root rank."""
    if not dist.is_initialized():
        pytest.skip("torch.distributed not initialized")

    heap_size = 2**33  # 8GB
    shmem = iris.iris(heap_size)
    rank = shmem.get_rank()
    world_size = shmem.get_num_ranks()

    if world_size < 2:
        shmem.barrier()
        del shmem
        pytest.skip("Need at least 2 ranks for non-zero root test")

    dst = world_size - 1  # Last rank is root

    pytorch_input_tensor = torch.randn(M, N, dtype=dtype, device=f"cuda:{rank}")
    pytorch_input_tensor.fill_(float(rank + 1))

    if rank == dst:
        gather_list = [torch.zeros(M, N, dtype=dtype, device=f"cuda:{rank}") for _ in range(world_size)]
    else:
        gather_list = None

    shmem.barrier()
    dist.gather(pytorch_input_tensor, gather_list, dst=dst)
    torch.cuda.synchronize()

    if rank == dst:
        pytorch_output = torch.cat(gather_list, dim=0)

    iris_input_tensor = shmem.zeros((M, N), dtype=dtype)
    iris_input_tensor.copy_(pytorch_input_tensor)
    iris_output_tensor = shmem.zeros((world_size * M, N), dtype=dtype)

    shmem.barrier()
    config = Config(block_size_m=32, block_size_n=64)
    shmem.ccl.gather(iris_output_tensor, iris_input_tensor, dst=dst, config=config)
    torch.cuda.synchronize()

    if rank == dst:
        atol = 1e-3 if dtype == torch.float16 else 1e-5
        max_diff = torch.abs(iris_output_tensor - pytorch_output).max().item()

        try:
            assert torch.allclose(iris_output_tensor, pytorch_output, atol=atol), (
                f"Max difference: {max_diff}, expected < {atol}\n"
                f"Rank {rank}: Iris output doesn't match torch.distributed.gather (dst={dst})"
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
