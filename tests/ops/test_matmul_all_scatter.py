# SPDX-License-Identifier: MIT
# Copyright (c) 2025 Advanced Micro Devices, Inc. All rights reserved.

"""
Test suite for high-level matmul_all_scatter API.

Note: This test requires tritonBLAS to be installed.
Install with: pip install git+https://github.com/ROCm/tritonBLAS.git
"""

import pytest
import torch
import torch.distributed as dist
import iris
import iris.ops as ops


@pytest.mark.parametrize(
    "dtype, atol, rtol",
    [
        (torch.float16, 0.5, 0.01),
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
def test_matmul_all_scatter(dtype, atol, rtol, M, N, K):
    """Test matmul_all_scatter using shmem.ops API with proper config.

    Validates against a PyTorch reference: local GEMM on each rank followed by
    all_gather to concatenate column shards along the N dimension.
    """
    if not dist.is_initialized():
        pytest.skip("torch.distributed not initialized")

    heap_size = 2**33  # 8GB
    shmem = iris.iris(heap_size)
    rank = shmem.get_rank()
    world_size = shmem.get_num_ranks()

    # N must be divisible by world_size for column-wise sharding
    if N % world_size != 0:
        pytest.skip(f"N={N} not divisible by world_size={world_size}")

    N_local = N // world_size

    # Skip if problem size is too small for world_size
    min_block_size = 32  # Smallest block size we use
    if N_local < min_block_size:
        pytest.skip(f"N_local={N_local} too small for world_size={world_size} (need >= {min_block_size})")
    if K < min_block_size:
        pytest.skip(f"K={K} too small (need >= {min_block_size})")
    if M < min_block_size:
        pytest.skip(f"M={M} too small (need >= {min_block_size})")

    # Create shmem tensors directly
    A = shmem.randn((M, K), dtype=dtype)
    B_shard = shmem.randn((K, N_local), dtype=dtype)
    output = shmem.zeros((M, N), dtype=dtype)

    # Reference: compute local GEMM, then all-gather along N dimension
    A_ref = A.clone()
    B_shard_ref = B_shard.clone()
    C_shard_ref = torch.matmul(A_ref, B_shard_ref)
    C_shards = [torch.zeros(M, N_local, dtype=dtype, device=f"cuda:{rank}") for _ in range(world_size)]
    dist.all_gather(C_shards, C_shard_ref)
    pytorch_output = torch.cat(C_shards, dim=1)  # Concatenate along N dimension
    torch.cuda.synchronize()

    shmem.barrier()

    # Use appropriate block sizes based on problem size
    from iris.ops.config import FusedConfig

    # Select config based on actual problem dimensions
    if M <= 64 or K <= 64 or N_local <= 64:
        config = FusedConfig(block_size_m=32, block_size_n=32, block_size_k=32)
    elif M <= 128 or K <= 128 or N_local <= 128:
        config = FusedConfig(block_size_m=64, block_size_n=64, block_size_k=32)
    else:
        config = FusedConfig(block_size_m=128, block_size_n=128, block_size_k=64)

    # Validate config against problem size
    assert M >= config.block_size_m, f"M ({M}) must be >= block_size_m ({config.block_size_m})"
    assert K >= config.block_size_k, f"K ({K}) must be >= block_size_k ({config.block_size_k})"
    assert N_local >= config.block_size_n, f"N_local ({N_local}) must be >= block_size_n ({config.block_size_n})"

    # Use shmem.ops API with proper config
    shmem.ops.matmul_all_scatter(output, A, B_shard, config=config)

    torch.cuda.synchronize()
    shmem.barrier()

    max_diff = torch.abs(output - pytorch_output).max().item()

    assert torch.allclose(output, pytorch_output, atol=atol, rtol=rtol), (
        f"Max difference: {max_diff}, expected < {atol}\n"
        f"Rank {rank}: shmem.ops.matmul_all_scatter output doesn't match reference"
    )

    if rank == 0:
        print(f"✓ matmul_all_scatter test passed: {dtype}, M={M}, N={N}, K={K}")

    shmem.barrier()
    del shmem
    import gc

    gc.collect()


@pytest.mark.parametrize(
    "dtype, atol, rtol",
    [
        (torch.float16, 0.5, 0.01),
        (torch.bfloat16, 0.5, 0.01),
    ],
)
def test_matmul_all_scatter_semantics(dtype, atol, rtol):
    """
    Test that matmul_all_scatter is equivalent to:
    C_local = A @ B_local  (on each rank)
    output = all_gather(C_local, dim=1)  (concatenate along N)
    """
    if not dist.is_initialized():
        pytest.skip("torch.distributed not initialized")

    heap_size = 2**33
    shmem = iris.iris(heap_size)
    rank = shmem.get_rank()
    world_size = shmem.get_num_ranks()

    M, N, K = 128, 128, 32

    if N % world_size != 0:
        pytest.skip(f"N={N} not divisible by world_size={world_size}")

    N_local = N // world_size

    if N_local < 32:
        pytest.skip(f"N_local={N_local} too small (need >= 32)")

    A = shmem.randn((M, K), dtype=dtype)
    B_shard = shmem.randn((K, N_local), dtype=dtype)
    output = shmem.zeros((M, N), dtype=dtype)

    # Reference
    C_shard_ref = torch.matmul(A.clone(), B_shard.clone())
    C_shards = [torch.zeros(M, N_local, dtype=dtype, device=f"cuda:{rank}") for _ in range(world_size)]
    dist.all_gather(C_shards, C_shard_ref)
    C_ref = torch.cat(C_shards, dim=1)
    torch.cuda.synchronize()

    config = ops.FusedConfig(block_size_m=64, block_size_n=64, block_size_k=32)

    if N_local < config.block_size_n:
        pytest.skip(f"N_local={N_local} < block_size_n={config.block_size_n}, skipping")

    from iris.ops.matmul_all_scatter import matmul_all_scatter

    matmul_all_scatter(shmem, output, A, B_shard, config=config)

    torch.cuda.synchronize()
    shmem.barrier()

    assert torch.allclose(output, C_ref, atol=atol, rtol=rtol), (
        f"Rank {rank}: matmul_all_scatter semantics mismatch. Max diff: {torch.abs(output - C_ref).max().item()}"
    )

    if rank == 0:
        print("matmul_all_scatter semantics verified")

    shmem.barrier()
    del shmem
    import gc

    gc.collect()
