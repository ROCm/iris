# SPDX-License-Identifier: MIT
# Copyright (c) 2025 Advanced Micro Devices, Inc. All rights reserved.

"""
Tests for fused all_gather + matmul operation.

Each rank has A_sharded (M x K_local), B is replicated.
The operation gathers A from all ranks and computes C = A_gathered @ B.
"""

import pytest
import torch
import torch.distributed as dist

import iris


@pytest.mark.parametrize(
    "dtype, atol, rtol",
    [
        (torch.float16, 1e-2, 1e-2),
    ],
)
@pytest.mark.parametrize(
    "M,K_local,N",
    [
        (128, 32, 64),
        (256, 64, 128),
    ],
)
def test_all_gather_matmul(dtype, atol, rtol, M, K_local, N):
    """Test all_gather_matmul against torch all_gather + matmul."""
    if not dist.is_initialized():
        pytest.skip("torch.distributed not initialized")

    heap_size = 2**33
    shmem = iris.iris(heap_size)
    rank = shmem.get_rank()
    world_size = shmem.get_num_ranks()

    K = K_local * world_size  # Full K dimension

    # Skip if problem size is too small for world_size or block sizes
    # With default or custom configs, we need at least one tile
    min_block_size = 32  # Smallest block size we use
    if M < min_block_size:
        pytest.skip(f"M={M} too small (need >= {min_block_size})")
    if K_local < min_block_size:
        pytest.skip(f"K_local={K_local} too small (need >= {min_block_size})")
    if N < min_block_size:
        pytest.skip(f"N={N} too small (need >= {min_block_size})")

    # Seed for reproducibility - different seed per rank for A_sharded
    torch.manual_seed(42 + rank)
    A_sharded = torch.randn(M, K_local, dtype=dtype, device=f"cuda:{rank}")

    # B must be identical on all ranks
    torch.manual_seed(123)
    B = torch.randn(K, N, dtype=dtype, device=f"cuda:{rank}")

    # Reference: torch all_gather + matmul
    A_gathered_list = [torch.zeros(M, K_local, dtype=dtype, device=f"cuda:{rank}") for _ in range(world_size)]
    dist.all_gather(A_gathered_list, A_sharded)
    A_gathered_ref = torch.cat(A_gathered_list, dim=1)  # (M, K)
    ref_output = torch.matmul(A_gathered_ref, B)
    torch.cuda.synchronize()

    # Create shmem tensors directly
    A_sharded_shmem = shmem.zeros((M, K_local), dtype=dtype)
    A_sharded_shmem.copy_(A_sharded)
    B_shmem = shmem.zeros((K, N), dtype=dtype)
    B_shmem.copy_(B)
    output = shmem.zeros((M, N), dtype=dtype)

    shmem.barrier()

    # Run fused all_gather + matmul using shmem.ops API
    from iris.ops.config import FusedConfig

    # Use appropriate block sizes based on problem size
    # For small problems, use smaller blocks
    if M <= 256 or K_local <= 64 or N <= 128:
        config = FusedConfig(block_size_m=64, block_size_n=64, block_size_k=32)
    else:
        config = FusedConfig()

    # Validate config against problem size
    assert M >= config.block_size_m, f"M ({M}) must be >= block_size_m ({config.block_size_m})"
    assert K_local >= config.block_size_k, f"K_local ({K_local}) must be >= block_size_k ({config.block_size_k})"
    assert N >= config.block_size_n, f"N ({N}) must be >= block_size_n ({config.block_size_n})"

    shmem.ops.all_gather_matmul(output, A_sharded_shmem, B_shmem, config=config)

    torch.cuda.synchronize()
    shmem.barrier()

    max_diff = (output - ref_output).abs().max().item()

    assert torch.allclose(output, ref_output, atol=atol, rtol=rtol), (
        f"Rank {rank}: Max diff {max_diff}, expected < {atol}"
    )


if __name__ == "__main__":
    # For quick debugging
    import sys

    if not dist.is_initialized():
        print("Run with: torchrun --nproc_per_node=2 tests/ops/test_all_gather_matmul.py")
        sys.exit(1)

    rank = dist.get_rank()
    torch.cuda.set_device(rank)

    print(f"[Rank {rank}] Testing all_gather_matmul...")
    test_all_gather_matmul(torch.float16, 128, 32, 64)
    print(f"[Rank {rank}] ✓ Test passed!")
