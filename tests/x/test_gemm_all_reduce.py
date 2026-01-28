# SPDX-License-Identifier: MIT
# Copyright (c) 2025 Advanced Micro Devices, Inc. All rights reserved.

"""
Test suite for GEMM + All-Reduce primitive.

Note: This test requires tritonBLAS to be installed.
Install with: pip install git+https://github.com/ROCm/tritonBLAS.git
"""

import pytest
import torch
import torch.distributed as dist
import iris

try:
    from iris.x import gemm_all_reduce

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
def test_gemm_all_reduce(dtype, M, N, K, BLOCK_SIZE_M, BLOCK_SIZE_N, BLOCK_SIZE_K):
    """Test GEMM + All-Reduce by comparing against manual GEMM + PyTorch all_reduce."""
    if not dist.is_initialized():
        pytest.skip("torch.distributed not initialized")

    heap_size = 2**33  # 8GB
    shmem = iris.iris(heap_size)
    rank = shmem.get_rank()
    world_size = shmem.get_num_ranks()

    # Create input matrices
    # Each rank has its own A (data-parallel sharding)
    A_local = torch.randn(M, K, dtype=dtype, device=f"cuda:{rank}")
    # B is replicated across ranks
    B = torch.randn(K, N, dtype=dtype, device=f"cuda:{rank}")

    # Compute reference: manual GEMM + PyTorch all_reduce
    C_local_ref = torch.matmul(A_local, B)
    pytorch_output_tensor = C_local_ref.clone()
    shmem.barrier()
    dist.all_reduce(pytorch_output_tensor, op=dist.ReduceOp.SUM)
    torch.cuda.synchronize()

    # Set up Iris tensors
    iris_A = shmem.zeros((M, K), dtype=dtype)
    iris_A.copy_(A_local)
    iris_B = shmem.zeros((K, N), dtype=dtype)
    iris_B.copy_(B)
    iris_C = shmem.zeros((M, N), dtype=dtype)

    shmem.barrier()

    # Launch gemm_all_reduce kernel
    num_pid_m = (M + BLOCK_SIZE_M - 1) // BLOCK_SIZE_M
    num_pid_n = (N + BLOCK_SIZE_N - 1) // BLOCK_SIZE_N
    total_tiles = num_pid_m * num_pid_n
    NUM_SMS = 64
    NUM_XCDS = 1
    CHUNK_SIZE = 1
    GROUP_SIZE_M = 1

    grid = (NUM_SMS,)

    try:
        gemm_all_reduce[grid](
            iris_A,
            iris_B,
            iris_C,
            None,  # bias_ptr
            M,
            N,
            K,
            iris_A.stride(0),
            iris_A.stride(1),
            iris_B.stride(0),
            iris_B.stride(1),
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
        max_diff = torch.abs(iris_C - pytorch_output_tensor).max().item()

        assert torch.allclose(iris_C, pytorch_output_tensor, atol=atol, rtol=rtol), (
            f"Max difference: {max_diff}, expected < {atol}\n"
            f"Rank {rank}: Iris x.gemm_all_reduce output doesn't match reference"
        )
        
        if rank == 0:
            print(f"✓ GEMM+All-Reduce test passed: {dtype}, M={M}, N={N}, K={K}, blocks=({BLOCK_SIZE_M},{BLOCK_SIZE_N},{BLOCK_SIZE_K})")
    except Exception as e:
        pytest.fail(f"gemm_all_reduce failed: {e}")
    finally:
        shmem.barrier()
        del shmem
        import gc

        gc.collect()

