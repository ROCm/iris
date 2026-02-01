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


@pytest.mark.parametrize(
    "dtype, atol, rtol",
    [
        (torch.float16, 0.5, 0.01),
        (torch.float32, 0.5, 0.01),
        (torch.bfloat16, 0.5, 0.01),
    ],
)
@pytest.mark.parametrize(
    "M, N, K",
    [
        (64, 64, 32),
        (512, 256, 512),
        (1024, 2048, 1024),
    ],
)
def test_matmul_all_gather(dtype, atol, rtol, M, N, K):
    """Test matmul_all_gather using shmem.ops API with proper config."""
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

    # Skip if problem size is too small for world_size
    # With default or custom configs, we need at least one tile per rank
    min_block_size = 32  # Smallest block size we use
    if M_local < min_block_size:
        pytest.skip(f"M_local={M_local} too small for world_size={world_size} (need >= {min_block_size})")
    if K < min_block_size:
        pytest.skip(f"K={K} too small (need >= {min_block_size})")
    if N < min_block_size:
        pytest.skip(f"N={N} too small (need >= {min_block_size})")

    # Create shmem tensors directly
    A_local = shmem.randn((M_local, K), dtype=dtype)
    B = shmem.randn((K, N), dtype=dtype)
    output = shmem.zeros((M, N), dtype=dtype)

    # Reference: compute local GEMM, then all-gather along M dimension
    A_ref = A_local.clone()
    B_ref = B.clone()
    C_local_ref = torch.matmul(A_ref, B_ref)
    C_gathered_list = [torch.zeros(M_local, N, dtype=dtype, device=f"cuda:{rank}") for _ in range(world_size)]
    dist.all_gather(C_gathered_list, C_local_ref)
    pytorch_output = torch.cat(C_gathered_list, dim=0)  # Concatenate along M dimension
    torch.cuda.synchronize()

    shmem.barrier()

    # Use appropriate block sizes based on problem size
    from iris.ops.config import FusedConfig

    # Select config based on actual problem dimensions
    # Ensure block sizes don't exceed actual dimensions
    if M_local <= 64 or K <= 64 or N <= 64:
        # Small problems - use 32x32x32 blocks
        config = FusedConfig(block_size_m=32, block_size_n=32, block_size_k=32)
    elif M_local <= 128 or K <= 128 or N <= 128:
        # Medium problems - use 64x64x32 blocks
        config = FusedConfig(block_size_m=64, block_size_n=64, block_size_k=32)
    elif dtype == torch.float32:
        # Larger problems with fp32 - use 128x128x64 blocks
        config = FusedConfig(block_size_m=128, block_size_n=128, block_size_k=64)
    else:
        # Larger problems with fp16/bf16 - use 128x128x64 blocks
        config = FusedConfig(block_size_m=128, block_size_n=128, block_size_k=64)

    # Validate config against problem size
    if config is not None:
        assert M_local >= config.block_size_m, f"M_local ({M_local}) must be >= block_size_m ({config.block_size_m})"
        assert K >= config.block_size_k, f"K ({K}) must be >= block_size_k ({config.block_size_k})"
        assert N >= config.block_size_n, f"N ({N}) must be >= block_size_n ({config.block_size_n})"

    # Use shmem.ops API with proper config
    shmem.ops.matmul_all_gather(output, A_local, B, config=config)

    torch.cuda.synchronize()
    shmem.barrier()

    max_diff = torch.abs(output - pytorch_output).max().item()

    assert torch.allclose(output, pytorch_output, atol=atol, rtol=rtol), (
        f"Max difference: {max_diff}, expected < {atol}\n"
        f"Rank {rank}: shmem.ops.matmul_all_gather output doesn't match reference"
    )

    if rank == 0:
        print(f"✓ matmul_all_gather test passed: {dtype}, M={M}, N={N}, K={K}")

    shmem.barrier()
    del shmem
    import gc

    gc.collect()
