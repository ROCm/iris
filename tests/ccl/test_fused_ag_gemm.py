# SPDX-License-Identifier: MIT
# Copyright (c) 2025 Advanced Micro Devices, Inc. All rights reserved.

"""
Tests for fused all-gather + GEMM (prologue fusion) via ctx.ccl.all_gather_gemm().

Reference: dist.all_gather on the column-sharded activation, then torch.matmul.
"""

import gc

import pytest
import torch
import torch.distributed as dist

import iris
from iris.ccl import Config


# ── Correctness tests ────────────────────────────────────────────────────────


@pytest.mark.parametrize(
    "dtype, atol, rtol",
    [
        (torch.float16, 1e-1, 1e-1),
        (torch.bfloat16, 1e-1, 1e-1),
        (torch.float32, 5e-2, 5e-2),  # TF32 truncation in tl.dot
    ],
)
@pytest.mark.parametrize(
    "M, K_local, N",
    [
        (128, 64, 64),  # Small
        (1024, 1024, 1024),  # Medium
        (2048, 2048, 4096),  # Large / TP-relevant (H=8192 with world_size=4)
        (4096, 512, 512),  # Tall-skinny
    ],
)
def test_fused_ag_gemm_correctness(dtype, atol, rtol, M, K_local, N):
    """Test fused AG+GEMM against torch all_gather + matmul reference."""
    if not dist.is_initialized():
        pytest.skip("torch.distributed not initialized")

    heap_size = 2**33  # 8 GB
    shmem = iris.iris(heap_size)
    rank = shmem.get_rank()
    world_size = shmem.get_num_ranks()
    K = K_local * world_size

    # Deterministic per-rank A shard, replicated B
    torch.manual_seed(42 + rank)
    A_shard = torch.randn(M, K_local, dtype=dtype, device=f"cuda:{rank}")
    torch.manual_seed(123)
    weight = torch.randn(K, N, dtype=dtype, device=f"cuda:{rank}")

    # ── Reference: torch all_gather + matmul ─────────────────────────────
    gathered = [torch.zeros_like(A_shard) for _ in range(world_size)]
    dist.all_gather(gathered, A_shard)
    A_full = torch.cat(gathered, dim=1)  # (M, K)
    ref = torch.matmul(A_full, weight)
    torch.cuda.synchronize()

    # ── Iris fused AG+GEMM ───────────────────────────────────────────────
    A_shard_sym = shmem.zeros((M, K_local), dtype=dtype)
    A_shard_sym.copy_(A_shard)
    weight_sym = shmem.zeros((K, N), dtype=dtype)
    weight_sym.copy_(weight)
    output = shmem.zeros((M, N), dtype=dtype)

    shmem.barrier()

    # Pick block sizes that fit the problem dimensions
    bm = min(128, M)
    bn = min(128, N)
    bk = min(64, K_local)
    # Ensure power-of-two and >= 16
    for v in [16, 32, 64, 128]:
        if v <= M:
            bm = v
        if v <= N:
            bn = v
        if v <= K_local:
            bk = v

    config = Config(block_size_m=bm, block_size_n=bn, swizzle_size=4)
    shmem.ccl.all_gather_gemm(output, A_shard_sym, weight_sym, config=config, block_size_k=bk)
    torch.cuda.synchronize()

    max_diff = (output - ref).abs().max().item()

    try:
        assert torch.allclose(output, ref, atol=atol, rtol=rtol), (
            f"Rank {rank}: max diff {max_diff}, expected < {atol} (M={M}, K_local={K_local}, N={N}, dtype={dtype})"
        )
    finally:
        shmem.barrier()
        del shmem
        gc.collect()


# ── Edge-case tests ──────────────────────────────────────────────────────────


@pytest.mark.parametrize(
    "M, K_local, N",
    [
        (128, 48, 64),  # K_local not divisible by BLOCK_K=32
        (128, 80, 64),  # K_local not divisible by BLOCK_K=64, but by 16
    ],
)
def test_fused_ag_gemm_odd_k(M, K_local, N):
    """Test with K_local not evenly divisible by BLOCK_K."""
    if not dist.is_initialized():
        pytest.skip("torch.distributed not initialized")

    dtype = torch.float16
    atol, rtol = 1e-1, 1e-1
    heap_size = 2**33
    shmem = iris.iris(heap_size)
    rank = shmem.get_rank()
    world_size = shmem.get_num_ranks()
    K = K_local * world_size

    torch.manual_seed(42 + rank)
    A_shard = torch.randn(M, K_local, dtype=dtype, device=f"cuda:{rank}")
    torch.manual_seed(123)
    weight = torch.randn(K, N, dtype=dtype, device=f"cuda:{rank}")

    gathered = [torch.zeros_like(A_shard) for _ in range(world_size)]
    dist.all_gather(gathered, A_shard)
    A_full = torch.cat(gathered, dim=1)
    ref = torch.matmul(A_full, weight)
    torch.cuda.synchronize()

    A_sym = shmem.zeros((M, K_local), dtype=dtype)
    A_sym.copy_(A_shard)
    W_sym = shmem.zeros((K, N), dtype=dtype)
    W_sym.copy_(weight)
    output = shmem.zeros((M, N), dtype=dtype)

    shmem.barrier()

    bk = 32  # intentionally doesn't divide K_local
    config = Config(block_size_m=64, block_size_n=64)
    shmem.ccl.all_gather_gemm(output, A_sym, W_sym, config=config, block_size_k=bk)
    torch.cuda.synchronize()

    max_diff = (output - ref).abs().max().item()
    try:
        assert torch.allclose(output, ref, atol=atol, rtol=rtol), (
            f"Rank {rank}: max diff {max_diff} (odd K_local={K_local}, BLOCK_K={bk})"
        )
    finally:
        shmem.barrier()
        del shmem
        gc.collect()


@pytest.mark.parametrize(
    "comm_sms, block_size_m, block_size_n, block_size_k",
    [
        (32, 64, 64, 32),
        (128, 128, 128, 64),
    ],
)
def test_fused_ag_gemm_custom_config(comm_sms, block_size_m, block_size_n, block_size_k):
    """Test with various custom configurations."""
    if not dist.is_initialized():
        pytest.skip("torch.distributed not initialized")

    dtype = torch.float16
    atol, rtol = 1e-1, 1e-1
    M, K_local, N = 256, 128, 256
    heap_size = 2**33
    shmem = iris.iris(heap_size)
    rank = shmem.get_rank()
    world_size = shmem.get_num_ranks()
    K = K_local * world_size

    torch.manual_seed(42 + rank)
    A_shard = torch.randn(M, K_local, dtype=dtype, device=f"cuda:{rank}")
    torch.manual_seed(123)
    weight = torch.randn(K, N, dtype=dtype, device=f"cuda:{rank}")

    gathered = [torch.zeros_like(A_shard) for _ in range(world_size)]
    dist.all_gather(gathered, A_shard)
    A_full = torch.cat(gathered, dim=1)
    ref = torch.matmul(A_full, weight)
    torch.cuda.synchronize()

    A_sym = shmem.zeros((M, K_local), dtype=dtype)
    A_sym.copy_(A_shard)
    W_sym = shmem.zeros((K, N), dtype=dtype)
    W_sym.copy_(weight)
    output = shmem.zeros((M, N), dtype=dtype)

    shmem.barrier()

    config = Config(block_size_m=block_size_m, block_size_n=block_size_n, comm_sms=comm_sms)
    shmem.ccl.all_gather_gemm(output, A_sym, W_sym, config=config, block_size_k=block_size_k)
    torch.cuda.synchronize()

    max_diff = (output - ref).abs().max().item()
    try:
        assert torch.allclose(output, ref, atol=atol, rtol=rtol), (
            f"Rank {rank}: max diff {max_diff} "
            f"(comm_sms={comm_sms}, BM={block_size_m}, BN={block_size_n}, BK={block_size_k})"
        )
    finally:
        shmem.barrier()
        del shmem
        gc.collect()


if __name__ == "__main__":
    import sys

    if not dist.is_initialized():
        print("Run with: torchrun --nproc_per_node=<N> tests/ccl/test_fused_ag_gemm.py")
        sys.exit(1)

    rank = dist.get_rank()
    torch.cuda.set_device(rank)

    print(f"[Rank {rank}] Running fused AG+GEMM tests...")
    test_fused_ag_gemm_correctness(torch.float16, 1e-1, 1e-1, 128, 64, 64)
    print(f"[Rank {rank}] Correctness test passed (fp16, 128x64x64)")
    test_fused_ag_gemm_correctness(torch.float32, 1e-3, 1e-3, 256, 128, 256)
    print(f"[Rank {rank}] Correctness test passed (fp32, 256x128x256)")
    test_fused_ag_gemm_odd_k(128, 48, 64)
    print(f"[Rank {rank}] Odd-K test passed")
    print(f"[Rank {rank}] All tests passed!")
