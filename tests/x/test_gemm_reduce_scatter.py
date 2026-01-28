# SPDX-License-Identifier: MIT
# Copyright (c) 2025 Advanced Micro Devices, Inc. All rights reserved.

"""
Test suite for GEMM + Reduce-Scatter primitive.

Note: This test requires tritonBLAS to be installed.
Install with: pip install git+https://github.com/ROCm/tritonBLAS.git
"""

import pytest
import torch
import torch.distributed as dist
import iris

try:
    from iris.x import gemm_reduce_scatter

    TRITONBLAS_AVAILABLE = True
except ImportError:
    TRITONBLAS_AVAILABLE = False


@pytest.mark.skipif(not TRITONBLAS_AVAILABLE, reason="tritonBLAS not available")
@pytest.mark.parametrize(
    "dtype",
    [
        torch.float16,
        torch.float32,
        torch.bfloat16,
    ],
)
@pytest.mark.parametrize(
    "M, N, K, BLOCK_SIZE_M, BLOCK_SIZE_N, BLOCK_SIZE_K",
    [
        (128, 64, 32, 64, 32, 16),  # Small
        (1024, 256, 512, 128, 128, 64),  # Medium
        (2048, 2048, 1024, 256, 256, 128),  # Large
    ],
)
def test_gemm_reduce_scatter(dtype, M, N, K, BLOCK_SIZE_M, BLOCK_SIZE_N, BLOCK_SIZE_K):
    """Test GEMM + Reduce-Scatter by comparing against manual GEMM + PyTorch reduce_scatter."""
    if not dist.is_initialized():
        pytest.skip("torch.distributed not initialized")

    heap_size = 2**33  # 8GB
    shmem = iris.iris(heap_size)
    rank = shmem.get_rank()
    world_size = shmem.get_num_ranks()

    # Create input matrices
    # A and B are replicated across ranks
    A = torch.randn(M, K, dtype=dtype, device=f"cuda:{rank}")
    B = torch.randn(K, N, dtype=dtype, device=f"cuda:{rank}")

    # Compute reference: manual GEMM + PyTorch reduce_scatter
    C_full_ref = torch.matmul(A, B)  # Full result
    C_local_ref = torch.empty(M, N // world_size, dtype=dtype, device=f"cuda:{rank}")
    shmem.barrier()
    dist.reduce_scatter(C_local_ref, [C_full_ref], op=dist.ReduceOp.SUM)
    torch.cuda.synchronize()

    # Set up Iris tensors
    iris_A = shmem.zeros((M, K), dtype=dtype)
    iris_A.copy_(A)
    iris_B = shmem.zeros((K, N), dtype=dtype)
    iris_B.copy_(B)
    iris_C_full = shmem.zeros((M, N), dtype=dtype)  # Temporary buffer for full result
    iris_C = shmem.zeros((M, N // world_size), dtype=dtype)  # Local output

    shmem.barrier()

    # Launch gemm_reduce_scatter kernel
    num_pid_m = (M + BLOCK_SIZE_M - 1) // BLOCK_SIZE_M
    num_pid_n = (N + BLOCK_SIZE_N - 1) // BLOCK_SIZE_N
    total_tiles = num_pid_m * num_pid_n
    NUM_SMS = 64
    NUM_XCDS = 1
    CHUNK_SIZE = 1
    GROUP_SIZE_M = 1

    grid = (NUM_SMS,)

    try:
        gemm_reduce_scatter[grid](
            iris_A,
            iris_B,
            iris_C_full,
            iris_C,
            None,  # bias_ptr
            M,
            N,
            K,
            iris_A.stride(0),
            iris_A.stride(1),
            iris_B.stride(0),
            iris_B.stride(1),
            iris_C_full.stride(0),
            iris_C_full.stride(1),
            iris_C.stride(0),
            iris_C.stride(1),
            0,  # stride_bias
            shmem.get_heap_bases(),
            rank,
            world_size,
            BLOCK_SIZE_M,
            BLOCK_SIZE_N,
            BLOCK_SIZE_K,
            GROUP_SIZE_M,
            NUM_SMS,
            NUM_XCDS,
            CHUNK_SIZE,
            0,  # BIAS: False
            1 if (K % BLOCK_SIZE_K == 0) else 0,  # EVEN_K
            ".ca",  # CACHE_MODIFIER_A
            ".ca",  # CACHE_MODIFIER_B
            torch.backends.cuda.matmul.allow_tf32,  # ALLOW_TF32
        )

        torch.cuda.synchronize()
        shmem.barrier()

        # Compare results
        atol = 1e-2 if dtype == torch.float16 else 1e-3  # GEMM has higher error tolerance
        rtol = 1e-2 if dtype == torch.float16 else 1e-3
        max_diff = torch.abs(iris_C - C_local_ref).max().item()

        assert torch.allclose(iris_C, C_local_ref, atol=atol, rtol=rtol), (
            f"Max difference: {max_diff}, expected < {atol}\n"
            f"Rank {rank}: Iris x.gemm_reduce_scatter output doesn't match reference"
        )
        
        if rank == 0:
            print(f"✓ GEMM+Reduce-Scatter test passed: {dtype}, M={M}, N={N}, K={K}, blocks=({BLOCK_SIZE_M},{BLOCK_SIZE_N},{BLOCK_SIZE_K})")
    except Exception as e:
        pytest.fail(f"gemm_reduce_scatter failed: {e}")
    finally:
        shmem.barrier()
        del shmem
        import gc

        gc.collect()

