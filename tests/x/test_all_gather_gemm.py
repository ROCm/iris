# SPDX-License-Identifier: MIT
# Copyright (c) 2025 Advanced Micro Devices, Inc. All rights reserved.

"""
Test suite for All-Gather + GEMM primitive.

Note: This test requires tritonBLAS to be installed.
Install with: pip install git+https://github.com/ROCm/tritonBLAS.git
"""

import pytest
import torch
import torch.distributed as dist
import iris

try:
    from iris.x import all_gather_gemm

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
def test_all_gather_gemm(dtype, M, N, K, BLOCK_SIZE_M, BLOCK_SIZE_N, BLOCK_SIZE_K):
    """Test All-Gather + GEMM by comparing against PyTorch all_gather + matmul."""
    if not dist.is_initialized():
        pytest.skip("torch.distributed not initialized")

    heap_size = 2**33  # 8GB
    shmem = iris.iris(heap_size)
    rank = shmem.get_rank()
    world_size = shmem.get_num_ranks()

    # Create input matrices
    # A is sharded along K dimension: each rank has A_local of shape (M, K_local)
    K_total = K
    if K_total % world_size != 0:
        pytest.skip(f"K ({K_total}) must be divisible by world_size ({world_size})")
    K_local = K_total // world_size

    A_local = torch.randn(M, K_local, dtype=dtype, device=f"cuda:{rank}")
    # B is replicated across ranks
    B = torch.randn(K_total, N, dtype=dtype, device=f"cuda:{rank}")

    # Compute reference: PyTorch all_gather + matmul
    A_gathered_list = [torch.empty_like(A_local) for _ in range(world_size)]
    shmem.barrier()
    dist.all_gather(A_gathered_list, A_local)
    A_gathered_ref = torch.cat(A_gathered_list, dim=1)  # Concatenate along K dimension
    pytorch_output_tensor = torch.matmul(A_gathered_ref, B)
    torch.cuda.synchronize()

    # Set up Iris tensors
    iris_A_sharded = shmem.zeros((M, K_local), dtype=dtype)
    iris_A_sharded.copy_(A_local)
    iris_A_gathered = shmem.zeros((M, K_total), dtype=dtype)  # Will be populated by kernel
    iris_B = shmem.zeros((K_total, N), dtype=dtype)
    iris_B.copy_(B)
    iris_C = shmem.zeros((M, N), dtype=dtype)

    shmem.barrier()

    # Launch all_gather_gemm kernel
    num_pid_m = (M + BLOCK_SIZE_M - 1) // BLOCK_SIZE_M
    num_pid_n = (N + BLOCK_SIZE_N - 1) // BLOCK_SIZE_N
    total_tiles = num_pid_m * num_pid_n
    NUM_SMS = 64
    NUM_XCDS = 1
    CHUNK_SIZE = 1
    GROUP_SIZE_M = 1

    grid = (NUM_SMS,)

    try:
        all_gather_gemm[grid](
            iris_A_sharded,
            iris_B,
            iris_C,
            iris_A_gathered,
            None,  # bias_ptr
            M,
            N,
            K_total,
            iris_A_sharded.stride(0),
            iris_A_sharded.stride(1),
            iris_B.stride(0),
            iris_B.stride(1),
            iris_C.stride(0),
            iris_C.stride(1),
            iris_A_gathered.stride(0),
            iris_A_gathered.stride(1),  # stride_ag_n
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
            1 if (K_total % BLOCK_SIZE_K == 0) else 0,  # EVEN_K
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
            f"Rank {rank}: Iris x.all_gather_gemm output doesn't match reference"
        )
        
        if rank == 0:
            print(f"✓ All-Gather+GEMM test passed: {dtype}, M={M}, N={N}, K={K_total}, blocks=({BLOCK_SIZE_M},{BLOCK_SIZE_N},{BLOCK_SIZE_K})")
    except Exception as e:
        pytest.fail(f"all_gather_gemm failed: {e}")
    finally:
        shmem.barrier()
        del shmem
        import gc

        gc.collect()

