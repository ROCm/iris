# SPDX-License-Identifier: MIT
# Copyright (c) 2025 Advanced Micro Devices, Inc. All rights reserved.

"""
Test suite for high-level matmul_reduce_scatter API.

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
        (128, 128, 32),  # Small (N divisible by typical world_size)
        (1024, 512, 512),  # Medium
        (2048, 2048, 1024),  # Large
    ],
)
def test_matmul_reduce_scatter(dtype, M, N, K):
    """Test matmul_reduce_scatter by comparing against torch operations."""
    if not dist.is_initialized():
        pytest.skip("torch.distributed not initialized")

    heap_size = 2**33  # 8GB
    shmem = iris.iris(heap_size)
    rank = shmem.get_rank()
    world_size = shmem.get_num_ranks()

    # N must be divisible by world_size for column-wise scattering
    if N % world_size != 0:
        pytest.skip(f"N={N} not divisible by world_size={world_size}")

    N_local = N // world_size

    # Each rank has same inputs
    A = torch.randn(M, K, dtype=dtype, device=f"cuda:{rank}")
    B = torch.randn(K, N, dtype=dtype, device=f"cuda:{rank}")

    # Reference: compute full GEMM, reduce across ranks, then scatter
    C_full_local = torch.matmul(A, B)
    C_full_reduced = C_full_local.clone()
    dist.all_reduce(C_full_reduced, op=dist.ReduceOp.SUM)
    
    # Scatter: each rank keeps its portion of columns
    start_col = rank * N_local
    end_col = (rank + 1) * N_local
    pytorch_output = C_full_reduced[:, start_col:end_col].contiguous()
    torch.cuda.synchronize()

    # Set up Iris tensors
    iris_A = shmem.zeros((M, K), dtype=dtype)
    iris_A.copy_(A)
    iris_B = shmem.zeros((K, N), dtype=dtype)
    iris_B.copy_(B)
    iris_C = shmem.zeros((M, N_local), dtype=dtype)  # Local output size

    shmem.barrier()

    # Use high-level API
    ops.matmul_reduce_scatter(shmem, iris_C, iris_A, iris_B)

    torch.cuda.synchronize()
    shmem.barrier()

    # Compare results
    atol = 1e-2 if dtype == torch.float16 else 1e-3
    rtol = 1e-2 if dtype == torch.float16 else 1e-3
    max_diff = torch.abs(iris_C - pytorch_output).max().item()

    assert torch.allclose(iris_C, pytorch_output, atol=atol, rtol=rtol), (
        f"Max difference: {max_diff}, expected < {atol}\n"
        f"Rank {rank}: iris.ops.matmul_reduce_scatter output doesn't match reference"
    )

    if rank == 0:
        print(f"✓ matmul_reduce_scatter test passed: {dtype}, M={M}, N={N}, K={K}")

    shmem.barrier()
    del shmem
    import gc

    gc.collect()


def test_matmul_reduce_scatter_via_shmem_ops():
    """Test accessing matmul_reduce_scatter via shmem.ops namespace."""
    if not dist.is_initialized():
        pytest.skip("torch.distributed not initialized")

    heap_size = 2**33
    shmem = iris.iris(heap_size)
    rank = shmem.get_rank()
    world_size = shmem.get_num_ranks()

    M, N, K = 256, 256, 64
    dtype = torch.float16

    if N % world_size != 0:
        pytest.skip(f"N={N} not divisible by world_size={world_size}")

    N_local = N // world_size

    A = shmem.randn((M, K), dtype=dtype)
    B = shmem.randn((K, N), dtype=dtype)
    output = shmem.zeros((M, N_local), dtype=dtype)

    # Reference
    A_ref = A.clone()
    B_ref = B.clone()
    C_full = torch.matmul(A_ref, B_ref)
    C_full_reduced = C_full.clone()
    dist.all_reduce(C_full_reduced, op=dist.ReduceOp.SUM)
    start_col = rank * N_local
    end_col = (rank + 1) * N_local
    pytorch_output = C_full_reduced[:, start_col:end_col].contiguous()
    torch.cuda.synchronize()

    # Use shmem.ops interface
    shmem.ops.matmul_reduce_scatter(output, A, B)

    torch.cuda.synchronize()
    shmem.barrier()

    atol = 1e-2
    rtol = 1e-2
    assert torch.allclose(output, pytorch_output, atol=atol, rtol=rtol), (
        f"Rank {rank}: shmem.ops.matmul_reduce_scatter doesn't match reference"
    )

    if rank == 0:
        print("✓ shmem.ops.matmul_reduce_scatter test passed")

    shmem.barrier()
    del shmem
    import gc

    gc.collect()
