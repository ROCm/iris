# SPDX-License-Identifier: MIT
# Copyright (c) 2025 Advanced Micro Devices, Inc. All rights reserved.

"""
Test suite for high-level all_gather_matmul API.

Note: This test requires tritonBLAS to be installed.
Install with: pip install git+https://github.com/ROCm/tritonBLAS.git
"""

import pytest
import torch
import torch.distributed as dist
import iris
import iris.ops as ops


@pytest.mark.parametrize(
    "dtype",
    [
        torch.float16,
        torch.float32,
        torch.bfloat16,
    ],
)
@pytest.mark.parametrize(
    "M, N, K",
    [
        (128, 64, 128),  # Small (K divisible by typical world_size)
        (1024, 256, 512),  # Medium
        (2048, 2048, 1024),  # Large
    ],
)
def test_all_gather_matmul(dtype, M, N, K):
    """Test all_gather_matmul by comparing against torch operations."""
    if not dist.is_initialized():
        pytest.skip("torch.distributed not initialized")

    heap_size = 2**33  # 8GB
    shmem = iris.iris(heap_size)
    rank = shmem.get_rank()
    world_size = shmem.get_num_ranks()

    # K must be divisible by world_size for column-wise sharding
    if K % world_size != 0:
        pytest.skip(f"K={K} not divisible by world_size={world_size}")

    K_local = K // world_size

    # Create sharded A (each rank has K_local columns)
    A_sharded = torch.randn(M, K_local, dtype=dtype, device=f"cuda:{rank}")
    B = torch.randn(K, N, dtype=dtype, device=f"cuda:{rank}")

    # Reference: gather A manually, then matmul
    A_gathered_list = [torch.zeros(M, K_local, dtype=dtype, device=f"cuda:{rank}") for _ in range(world_size)]
    dist.all_gather(A_gathered_list, A_sharded)
    A_gathered_ref = torch.cat(A_gathered_list, dim=1)  # Concatenate along K dimension
    pytorch_output = torch.matmul(A_gathered_ref, B)
    torch.cuda.synchronize()

    # Set up Iris tensors
    iris_A_sharded = shmem.zeros((M, K_local), dtype=dtype)
    iris_A_sharded.copy_(A_sharded)
    iris_B = shmem.zeros((K, N), dtype=dtype)
    iris_B.copy_(B)
    iris_C = shmem.zeros((M, N), dtype=dtype)

    shmem.barrier()

    # Use high-level API
    ops.all_gather_matmul(shmem, iris_C, iris_A_sharded, iris_B)

    torch.cuda.synchronize()
    shmem.barrier()

    # Compare results
    atol = 1e-2 if dtype == torch.float16 else 1e-3
    rtol = 1e-2 if dtype == torch.float16 else 1e-3
    max_diff = torch.abs(iris_C - pytorch_output).max().item()

    assert torch.allclose(iris_C, pytorch_output, atol=atol, rtol=rtol), (
        f"Max difference: {max_diff}, expected < {atol}\n"
        f"Rank {rank}: iris.ops.all_gather_matmul output doesn't match reference"
    )

    if rank == 0:
        print(f"✓ all_gather_matmul test passed: {dtype}, M={M}, N={N}, K={K}")

    shmem.barrier()
    del shmem
    import gc

    gc.collect()


def test_all_gather_matmul_via_shmem_ops():
    """Test accessing all_gather_matmul via shmem.ops namespace."""
    if not dist.is_initialized():
        pytest.skip("torch.distributed not initialized")

    heap_size = 2**33
    shmem = iris.iris(heap_size)
    rank = shmem.get_rank()
    world_size = shmem.get_num_ranks()

    M, N, K = 256, 128, 256
    dtype = torch.float16

    if K % world_size != 0:
        pytest.skip(f"K={K} not divisible by world_size={world_size}")

    K_local = K // world_size

    A_sharded = shmem.randn((M, K_local), dtype=dtype)
    B = shmem.randn((K, N), dtype=dtype)
    output = shmem.zeros((M, N), dtype=dtype)

    # Reference
    A_sharded_ref = A_sharded.clone()
    B_ref = B.clone()
    A_gathered_list = [torch.zeros(M, K_local, dtype=dtype, device=f"cuda:{rank}") for _ in range(world_size)]
    dist.all_gather(A_gathered_list, A_sharded_ref)
    A_gathered_ref = torch.cat(A_gathered_list, dim=1)
    pytorch_output = torch.matmul(A_gathered_ref, B_ref)
    torch.cuda.synchronize()

    # Use shmem.ops interface
    shmem.ops.all_gather_matmul(output, A_sharded, B)

    torch.cuda.synchronize()
    shmem.barrier()

    atol = 1e-2
    rtol = 1e-2
    assert torch.allclose(output, pytorch_output, atol=atol, rtol=rtol), (
        f"Rank {rank}: shmem.ops.all_gather_matmul doesn't match reference"
    )

    if rank == 0:
        print("✓ shmem.ops.all_gather_matmul test passed")

    shmem.barrier()
    del shmem
    import gc

    gc.collect()
