# SPDX-License-Identifier: MIT
# Copyright (c) 2025 Advanced Micro Devices, Inc. All rights reserved.

"""
Test suite for high-level matmul_all_reduce API.

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
        (128, 64, 32),  # Small
        (1024, 256, 512),  # Medium
        (2048, 2048, 1024),  # Large
    ],
)
def test_matmul_all_reduce(dtype, M, N, K):
    """Test matmul_all_reduce by comparing against torch.matmul + dist.all_reduce."""
    if not dist.is_initialized():
        pytest.skip("torch.distributed not initialized")

    heap_size = 2**33  # 8GB
    shmem = iris.iris(heap_size)
    rank = shmem.get_rank()
    world_size = shmem.get_num_ranks()

    # Create input matrices
    A_local = torch.randn(M, K, dtype=dtype, device=f"cuda:{rank}")
    B = torch.randn(K, N, dtype=dtype, device=f"cuda:{rank}")

    # Compute reference: torch.matmul + dist.all_reduce
    C_local_ref = torch.matmul(A_local, B)
    pytorch_output = C_local_ref.clone()
    shmem.barrier()
    dist.all_reduce(pytorch_output, op=dist.ReduceOp.SUM)
    torch.cuda.synchronize()

    # Set up Iris tensors
    iris_A = shmem.zeros((M, K), dtype=dtype)
    iris_A.copy_(A_local)
    iris_B = shmem.zeros((K, N), dtype=dtype)
    iris_B.copy_(B)
    iris_C = shmem.zeros((M, N), dtype=dtype)

    shmem.barrier()

    # Use high-level API
    ops.matmul_all_reduce(shmem, iris_C, iris_A, iris_B)

    torch.cuda.synchronize()
    shmem.barrier()

    # Compare results
    atol = 1e-2 if dtype == torch.float16 else 1e-3
    rtol = 1e-2 if dtype == torch.float16 else 1e-3
    max_diff = torch.abs(iris_C - pytorch_output).max().item()

    assert torch.allclose(iris_C, pytorch_output, atol=atol, rtol=rtol), (
        f"Max difference: {max_diff}, expected < {atol}\n"
        f"Rank {rank}: iris.ops.matmul_all_reduce output doesn't match reference"
    )

    if rank == 0:
        print(f"✓ matmul_all_reduce test passed: {dtype}, M={M}, N={N}, K={K}")

    shmem.barrier()
    del shmem
    import gc

    gc.collect()


def test_matmul_all_reduce_with_bias():
    """Test matmul_all_reduce with bias."""
    if not dist.is_initialized():
        pytest.skip("torch.distributed not initialized")

    heap_size = 2**33
    shmem = iris.iris(heap_size)
    rank = shmem.get_rank()

    M, N, K = 512, 256, 128
    dtype = torch.float16

    A_local = torch.randn(M, K, dtype=dtype, device=f"cuda:{rank}")
    B = torch.randn(K, N, dtype=dtype, device=f"cuda:{rank}")
    bias = torch.randn(M, dtype=dtype, device=f"cuda:{rank}")

    # Reference
    C_ref = torch.matmul(A_local, B) + bias.unsqueeze(1)
    pytorch_output = C_ref.clone()
    shmem.barrier()
    dist.all_reduce(pytorch_output, op=dist.ReduceOp.SUM)
    torch.cuda.synchronize()

    # Iris
    iris_A = shmem.zeros((M, K), dtype=dtype)
    iris_A.copy_(A_local)
    iris_B = shmem.zeros((K, N), dtype=dtype)
    iris_B.copy_(B)
    iris_bias = shmem.zeros((M,), dtype=dtype)
    iris_bias.copy_(bias)
    iris_C = shmem.zeros((M, N), dtype=dtype)

    shmem.barrier()

    ops.matmul_all_reduce(shmem, iris_C, iris_A, iris_B, bias=iris_bias)

    torch.cuda.synchronize()
    shmem.barrier()

    atol = 1e-2
    rtol = 1e-2
    assert torch.allclose(iris_C, pytorch_output, atol=atol, rtol=rtol), (
        f"Rank {rank}: matmul_all_reduce with bias doesn't match reference"
    )

    if rank == 0:
        print("✓ matmul_all_reduce with bias test passed")

    shmem.barrier()
    del shmem
    import gc

    gc.collect()


def test_matmul_all_reduce_via_shmem_ops():
    """Test accessing matmul_all_reduce via shmem.ops namespace."""
    if not dist.is_initialized():
        pytest.skip("torch.distributed not initialized")

    heap_size = 2**33
    shmem = iris.iris(heap_size)
    rank = shmem.get_rank()

    M, N, K = 256, 128, 64
    dtype = torch.float16

    A = shmem.randn((M, K), dtype=dtype)
    B = shmem.randn((K, N), dtype=dtype)
    output = shmem.zeros((M, N), dtype=dtype)

    # Reference using PyTorch
    A_ref = A.clone()
    B_ref = B.clone()
    C_ref = torch.matmul(A_ref, B_ref)
    pytorch_output = C_ref.clone()
    shmem.barrier()
    dist.all_reduce(pytorch_output, op=dist.ReduceOp.SUM)
    torch.cuda.synchronize()

    # Use shmem.ops interface
    shmem.ops.matmul_all_reduce(output, A, B)

    torch.cuda.synchronize()
    shmem.barrier()

    atol = 1e-2
    rtol = 1e-2
    assert torch.allclose(output, pytorch_output, atol=atol, rtol=rtol), (
        f"Rank {rank}: shmem.ops.matmul_all_reduce doesn't match reference"
    )

    if rank == 0:
        print("✓ shmem.ops.matmul_all_reduce test passed")

    shmem.barrier()
    del shmem
    import gc

    gc.collect()
