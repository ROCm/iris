# SPDX-License-Identifier: MIT
# Copyright (c) 2025-2026 Advanced Micro Devices, Inc. All rights reserved.

"""
Tests for fused all_gather + matmul operations.

Each rank has A_sharded (M x K_local), B is replicated.
The operation gathers A from all ranks and computes C = A_gathered @ B.
Covers both the baseline pull kernel and the HBM-buffered kernel.
"""

import pytest
import torch
import torch.distributed as dist

import iris
from iris.ops.all_gather_matmul_hbm_buffer import (
    all_gather_matmul_hbm_buffer,
    all_gather_matmul_hbm_buffer_preamble,
)
from iris.ops.config import FusedConfig


def _make_reference(rank, world_size, M, K_local, N, dtype):
    """Build a torch reference output for all_gather + matmul."""
    device = f"cuda:{rank}"
    K = K_local * world_size

    torch.manual_seed(42 + rank)
    A_sharded = torch.randn(M, K_local, dtype=dtype, device=device)

    torch.manual_seed(123)
    B = torch.randn(K, N, dtype=dtype, device=device)

    A_gathered_list = [torch.zeros(M, K_local, dtype=dtype, device=device) for _ in range(world_size)]
    dist.all_gather(A_gathered_list, A_sharded)
    A_gathered_ref = torch.cat(A_gathered_list, dim=1)
    ref_output = torch.matmul(A_gathered_ref, B)
    torch.cuda.synchronize()
    return A_sharded, B, ref_output


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
def test_all_gather_matmul_baseline(dtype, atol, rtol, M, K_local, N):
    """Test baseline all_gather_matmul against torch all_gather + matmul."""
    if not dist.is_initialized():
        pytest.skip("torch.distributed not initialized")

    heap_size = 2**33
    ctx = iris.iris(heap_size)
    rank = ctx.get_rank()
    world_size = ctx.get_num_ranks()

    K = K_local * world_size

    min_block_size = 32
    if M < min_block_size or K_local < min_block_size or N < min_block_size:
        pytest.skip(f"Problem too small for min block size {min_block_size}")

    A_sharded, B, ref_output = _make_reference(rank, world_size, M, K_local, N, dtype)
    device = f"cuda:{rank}"

    A_sharded_shmem = ctx.zeros((M, K_local), dtype=dtype)
    A_sharded_shmem.copy_(A_sharded)
    B_shmem = ctx.zeros((K, N), dtype=dtype)
    B_shmem.copy_(B)
    output = ctx.zeros((M, N), dtype=dtype)

    ctx.barrier()

    config = (
        FusedConfig(block_size_m=64, block_size_n=64, block_size_k=32)
        if M <= 256 or K_local <= 64 or N <= 128
        else FusedConfig()
    )

    assert M >= config.block_size_m
    assert K_local >= config.block_size_k
    assert N >= config.block_size_n

    ctx.ops.all_gather_matmul(output, A_sharded_shmem, B_shmem, config=config)

    torch.cuda.synchronize()
    ctx.barrier()

    max_diff = (output - ref_output).abs().max().item()
    assert torch.allclose(output, ref_output, atol=atol, rtol=rtol), (
        f"Rank {rank}: Max diff {max_diff}, expected < {atol}"
    )


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
@pytest.mark.parametrize(
    "staged_a_layout",
    [
        "k_contiguous",
        "m_contiguous",
    ],
)
def test_all_gather_matmul_hbm_buffer(dtype, atol, rtol, M, K_local, N, staged_a_layout):
    """Test all_gather_matmul_hbm_buffer against torch all_gather + matmul."""
    if not dist.is_initialized():
        pytest.skip("torch.distributed not initialized")

    heap_size = 2**33
    ctx = iris.iris(heap_size)
    rank = ctx.get_rank()
    world_size = ctx.get_num_ranks()

    K = K_local * world_size

    A_sharded, B, ref_output = _make_reference(rank, world_size, M, K_local, N, dtype)

    A_sharded_shmem = ctx.zeros((M, K_local), dtype=dtype)
    A_sharded_shmem.copy_(A_sharded)
    B_shmem = ctx.zeros((K, N), dtype=dtype)
    B_shmem.copy_(B)
    output = ctx.zeros((M, N), dtype=dtype)

    ctx.barrier()

    config = FusedConfig(block_size_m=64, block_size_n=64, block_size_k=32)

    workspace = all_gather_matmul_hbm_buffer_preamble(
        ctx, A_sharded_shmem, B_shmem, config=config, staged_a_layout=staged_a_layout
    )

    all_gather_matmul_hbm_buffer(
        ctx,
        output,
        A_sharded_shmem,
        B_shmem,
        config=config,
        workspace=workspace,
        staged_a_layout=staged_a_layout,
        trace=False,
    )

    torch.cuda.synchronize()
    ctx.barrier()

    max_diff = (output - ref_output).abs().max().item()
    assert torch.allclose(output, ref_output, atol=atol, rtol=rtol), (
        f"Rank {rank}: Max diff {max_diff}, expected < {atol} "
        f"(staged_a_layout={staged_a_layout}, M={M}, K_local={K_local}, N={N})"
    )


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
    ],
)
def test_all_gather_matmul_hbm_buffer_with_bias(dtype, atol, rtol, M, K_local, N):
    """Test all_gather_matmul_hbm_buffer with a bias vector."""
    if not dist.is_initialized():
        pytest.skip("torch.distributed not initialized")

    heap_size = 2**33
    ctx = iris.iris(heap_size)
    rank = ctx.get_rank()
    world_size = ctx.get_num_ranks()

    K = K_local * world_size

    A_sharded, B, ref_output_no_bias = _make_reference(rank, world_size, M, K_local, N, dtype)
    device = f"cuda:{rank}"

    torch.manual_seed(77)
    bias = torch.randn(M, dtype=dtype, device=device)
    ref_output = ref_output_no_bias + bias[:, None]

    A_sharded_shmem = ctx.zeros((M, K_local), dtype=dtype)
    A_sharded_shmem.copy_(A_sharded)
    B_shmem = ctx.zeros((K, N), dtype=dtype)
    B_shmem.copy_(B)
    bias_shmem = ctx.zeros((M,), dtype=dtype)
    bias_shmem.copy_(bias)
    output = ctx.zeros((M, N), dtype=dtype)

    ctx.barrier()

    config = FusedConfig(block_size_m=64, block_size_n=64, block_size_k=32)

    all_gather_matmul_hbm_buffer(
        ctx,
        output,
        A_sharded_shmem,
        B_shmem,
        bias=bias_shmem,
        config=config,
        trace=False,
    )

    torch.cuda.synchronize()
    ctx.barrier()

    max_diff = (output - ref_output).abs().max().item()
    assert torch.allclose(output, ref_output, atol=atol, rtol=rtol), (
        f"Rank {rank}: Max diff {max_diff}, expected < {atol} (with bias)"
    )


if __name__ == "__main__":
    import sys

    if not dist.is_initialized():
        print("Run with: torchrun --nproc_per_node=2 tests/ops/test_all_gather_matmul.py")
        sys.exit(1)

    rank = dist.get_rank()
    torch.cuda.set_device(rank)
    print(f"[Rank {rank}] Tests in this file require pytest + torchrun. See tests/run_tests_distributed.py")
