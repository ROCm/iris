# SPDX-License-Identifier: MIT
# Copyright (c) 2025 Advanced Micro Devices, Inc. All rights reserved.

"""
Test suite for high-level matmul_all_gather API.

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
        (64, 64, 32),  # Small (M divisible by typical world_size)
        (512, 256, 512),  # Medium
        (1024, 2048, 1024),  # Large
    ],
)
def test_matmul_all_gather(dtype, M, N, K):
    """Test matmul_all_gather by comparing against torch operations."""
    if not dist.is_initialized():
        pytest.skip("torch.distributed not initialized")

    heap_size = 2**33  # 8GB
    shmem = iris.iris(heap_size)
    rank = shmem.get_rank()
    world_size = shmem.get_num_ranks()

    # M must be divisible by world_size for row-wise sharding
    if M % world_size != 0:
        pytest.skip(f"M={M} not divisible by world_size={world_size}")

    M_local = M // world_size

    # Each rank computes local GEMM
    A_local = torch.randn(M_local, K, dtype=dtype, device=f"cuda:{rank}")
    B = torch.randn(K, N, dtype=dtype, device=f"cuda:{rank}")

    # Reference: compute local GEMM, then all-gather along M dimension
    C_local_ref = torch.matmul(A_local, B)
    C_gathered_list = [torch.zeros(M_local, N, dtype=dtype, device=f"cuda:{rank}") for _ in range(world_size)]
    dist.all_gather(C_gathered_list, C_local_ref)
    pytorch_output = torch.cat(C_gathered_list, dim=0)  # Concatenate along M dimension
    torch.cuda.synchronize()

    # Set up Iris tensors
    iris_A = shmem.zeros((M_local, K), dtype=dtype)
    iris_A.copy_(A_local)
    iris_B = shmem.zeros((K, N), dtype=dtype)
    iris_B.copy_(B)
    iris_C = shmem.zeros((M, N), dtype=dtype)  # Full output size

    shmem.barrier()

    # Use high-level API
    ops.matmul_all_gather(shmem, iris_C, iris_A, iris_B)

    torch.cuda.synchronize()
    shmem.barrier()

    # Compare results
    atol = 1e-2 if dtype == torch.float16 else 1e-3
    rtol = 1e-2 if dtype == torch.float16 else 1e-3
    max_diff = torch.abs(iris_C - pytorch_output).max().item()

    assert torch.allclose(iris_C, pytorch_output, atol=atol, rtol=rtol), (
        f"Max difference: {max_diff}, expected < {atol}\n"
        f"Rank {rank}: iris.ops.matmul_all_gather output doesn't match reference"
    )

    if rank == 0:
        print(f"✓ matmul_all_gather test passed: {dtype}, M={M}, N={N}, K={K}")

    shmem.barrier()
    del shmem
    import gc

    gc.collect()


def test_matmul_all_gather_via_shmem_ops():
    """Test accessing matmul_all_gather via shmem.ops namespace."""
    if not dist.is_initialized():
        pytest.skip("torch.distributed not initialized")

    heap_size = 2**33
    shmem = iris.iris(heap_size)
    rank = shmem.get_rank()
    world_size = shmem.get_num_ranks()

    M, N, K = 256, 128, 64
    dtype = torch.float16

    if M % world_size != 0:
        pytest.skip(f"M={M} not divisible by world_size={world_size}")

    M_local = M // world_size

    A_local = shmem.randn((M_local, K), dtype=dtype)
    B = shmem.randn((K, N), dtype=dtype)
    output = shmem.zeros((M, N), dtype=dtype)

    # Reference
    A_ref = A_local.clone()
    B_ref = B.clone()
    C_local_ref = torch.matmul(A_ref, B_ref)
    C_gathered_list = [torch.zeros(M_local, N, dtype=dtype, device=f"cuda:{rank}") for _ in range(world_size)]
    dist.all_gather(C_gathered_list, C_local_ref)
    pytorch_output = torch.cat(C_gathered_list, dim=0)
    torch.cuda.synchronize()

    # Use shmem.ops interface
    shmem.ops.matmul_all_gather(output, A_local, B)

    torch.cuda.synchronize()
    shmem.barrier()

    atol = 1e-2
    rtol = 1e-2
    assert torch.allclose(output, pytorch_output, atol=atol, rtol=rtol), (
        f"Rank {rank}: shmem.ops.matmul_all_gather doesn't match reference"
    )

    if rank == 0:
        print("✓ shmem.ops.matmul_all_gather test passed")

    shmem.barrier()
    del shmem
    import gc

    gc.collect()
