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
    _auto_config,
    _CHAMPION_CONFIGS,
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
        (torch.bfloat16, 1e-2, 1e-2),
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
        (torch.bfloat16, 1e-2, 1e-2),
    ],
)
@pytest.mark.parametrize(
    "M,K_local,N",
    [
        (128, 32, 64),
        (256, 64, 128),
        (512, 64, 128),
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

    # k_per_flag must divide num_k_blocks = K // block_size_k; use 1 for small shapes
    num_k_blocks = K // config.block_size_k
    k_per_flag = 1
    while k_per_flag * 2 <= 8 and num_k_blocks % (k_per_flag * 2) == 0:
        k_per_flag *= 2

    workspace = all_gather_matmul_hbm_buffer_preamble(
        ctx, A_sharded_shmem, B_shmem, config=config, staged_a_layout=staged_a_layout, k_per_flag=k_per_flag
    )

    all_gather_matmul_hbm_buffer(
        ctx,
        output,
        A_sharded_shmem,
        B_shmem,
        config=config,
        workspace=workspace,
        k_per_flag=k_per_flag,
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
        (torch.bfloat16, 1e-2, 1e-2),
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

    # k_per_flag must divide num_k_blocks = K // block_size_k; use 1 for small shapes
    num_k_blocks = K // config.block_size_k
    k_per_flag = 1
    while k_per_flag * 2 <= 8 and num_k_blocks % (k_per_flag * 2) == 0:
        k_per_flag *= 2

    all_gather_matmul_hbm_buffer(
        ctx,
        output,
        A_sharded_shmem,
        B_shmem,
        bias=bias_shmem,
        config=config,
        k_per_flag=k_per_flag,
        trace=False,
    )

    torch.cuda.synchronize()
    ctx.barrier()

    max_diff = (output - ref_output).abs().max().item()
    assert torch.allclose(output, ref_output, atol=atol, rtol=rtol), (
        f"Rank {rank}: Max diff {max_diff}, expected < {atol} (with bias)"
    )


def test_all_gather_matmul_hbm_buffer_auto_workspace():
    """Test all_gather_matmul_hbm_buffer with workspace=None (auto preamble)."""
    if not dist.is_initialized():
        pytest.skip("torch.distributed not initialized")

    heap_size = 2**33
    ctx = iris.iris(heap_size)
    rank = ctx.get_rank()
    world_size = ctx.get_num_ranks()

    M, K_local, N = 128, 32, 64
    dtype = torch.float16
    atol, rtol = 1e-2, 1e-2

    K = K_local * world_size
    A_sharded, B, ref_output = _make_reference(rank, world_size, M, K_local, N, dtype)

    A_sharded_shmem = ctx.zeros((M, K_local), dtype=dtype)
    A_sharded_shmem.copy_(A_sharded)
    B_shmem = ctx.zeros((K, N), dtype=dtype)
    B_shmem.copy_(B)
    output = ctx.zeros((M, N), dtype=dtype)

    ctx.barrier()

    config = FusedConfig(block_size_m=64, block_size_n=64, block_size_k=32)
    k_per_flag = 1

    # workspace=None triggers automatic preamble inside the kernel function
    ws = all_gather_matmul_hbm_buffer(
        ctx,
        output,
        A_sharded_shmem,
        B_shmem,
        config=config,
        workspace=None,
        k_per_flag=k_per_flag,
        trace=False,
    )

    torch.cuda.synchronize()
    ctx.barrier()

    assert ws is not None, "all_gather_matmul_hbm_buffer should return workspace"
    assert ws.aux_buffer is not None, "Workspace aux_buffer should be allocated"
    assert ws.locks is not None, "Workspace locks should be allocated"

    max_diff = (output - ref_output).abs().max().item()
    assert torch.allclose(output, ref_output, atol=atol, rtol=rtol), (
        f"Rank {rank}: Max diff {max_diff}, expected < {atol} (auto workspace)"
    )


def test_all_gather_matmul_hbm_buffer_workspace_reuse():
    """Test that workspace can be reused across multiple kernel calls."""
    if not dist.is_initialized():
        pytest.skip("torch.distributed not initialized")

    heap_size = 2**33
    ctx = iris.iris(heap_size)
    rank = ctx.get_rank()
    world_size = ctx.get_num_ranks()

    M, K_local, N = 128, 32, 64
    dtype = torch.float16
    atol, rtol = 1e-2, 1e-2

    K = K_local * world_size
    A_sharded, B, ref_output = _make_reference(rank, world_size, M, K_local, N, dtype)

    A_sharded_shmem = ctx.zeros((M, K_local), dtype=dtype)
    A_sharded_shmem.copy_(A_sharded)
    B_shmem = ctx.zeros((K, N), dtype=dtype)
    B_shmem.copy_(B)
    output1 = ctx.zeros((M, N), dtype=dtype)
    output2 = ctx.zeros((M, N), dtype=dtype)

    ctx.barrier()

    config = FusedConfig(block_size_m=64, block_size_n=64, block_size_k=32)
    k_per_flag = 1

    workspace = all_gather_matmul_hbm_buffer_preamble(
        ctx, A_sharded_shmem, B_shmem, config=config, k_per_flag=k_per_flag
    )

    # First call
    all_gather_matmul_hbm_buffer(
        ctx, output1, A_sharded_shmem, B_shmem, config=config, workspace=workspace, k_per_flag=k_per_flag, trace=False
    )
    torch.cuda.synchronize()
    ctx.barrier()

    # Second call reusing workspace
    all_gather_matmul_hbm_buffer(
        ctx, output2, A_sharded_shmem, B_shmem, config=config, workspace=workspace, k_per_flag=k_per_flag, trace=False
    )
    torch.cuda.synchronize()
    ctx.barrier()

    max_diff1 = (output1 - ref_output).abs().max().item()
    max_diff2 = (output2 - ref_output).abs().max().item()
    assert torch.allclose(output1, ref_output, atol=atol, rtol=rtol), (
        f"Rank {rank}: First call max diff {max_diff1}, expected < {atol}"
    )
    assert torch.allclose(output2, ref_output, atol=atol, rtol=rtol), (
        f"Rank {rank}: Second call (workspace reuse) max diff {max_diff2}, expected < {atol}"
    )
    assert torch.allclose(output1, output2), "Both calls should produce identical results"


def test_all_gather_matmul_hbm_buffer_trace():
    """Test that trace_data is None when trace=False (default)."""
    if not dist.is_initialized():
        pytest.skip("torch.distributed not initialized")

    heap_size = 2**33
    ctx = iris.iris(heap_size)
    rank = ctx.get_rank()
    world_size = ctx.get_num_ranks()

    M, K_local, N = 128, 32, 64
    dtype = torch.float16

    K = K_local * world_size
    A_sharded, B, _ = _make_reference(rank, world_size, M, K_local, N, dtype)

    A_sharded_shmem = ctx.zeros((M, K_local), dtype=dtype)
    A_sharded_shmem.copy_(A_sharded)
    B_shmem = ctx.zeros((K, N), dtype=dtype)
    B_shmem.copy_(B)
    output = ctx.zeros((M, N), dtype=dtype)

    ctx.barrier()

    config = FusedConfig(block_size_m=64, block_size_n=64, block_size_k=32)
    k_per_flag = 1

    ws = all_gather_matmul_hbm_buffer_preamble(ctx, A_sharded_shmem, B_shmem, config=config, k_per_flag=k_per_flag)

    # With trace=False, trace_data should not be populated
    ws = all_gather_matmul_hbm_buffer(
        ctx,
        output,
        A_sharded_shmem,
        B_shmem,
        config=config,
        workspace=ws,
        k_per_flag=k_per_flag,
        trace=False,
    )

    torch.cuda.synchronize()
    ctx.barrier()

    assert not hasattr(ws, "trace_data") or ws.trace_data is None, (
        # FusedWorkspace is a dataclass; trace_data is set only when trace=True.
        # Both conditions handle the case where the attribute is absent (fresh workspace)
        # or explicitly set to None (workspace reused from a previous trace=False call).
        "trace_data should not be populated when trace=False"
    )


# ──────────────────────────────────────────────────────────────────────
# Unit tests for _auto_config (no distributed context required)
# ──────────────────────────────────────────────────────────────────────


@pytest.mark.parametrize(
    "M, N, K, world_size",
    [
        (1024, 256, 1024, 8),
        (4096, 3584, 8192, 8),
        (8192, 8192, 16384, 8),
        (16384, 3584, 8192, 4),
        (256, 256, 512, 2),
    ],
)
def test_auto_config_heuristic_validity(M, N, K, world_size):
    """Verify _auto_config returns valid configs where k_per_flag divides K//block_k."""
    config, kpf, fs, nfs, fsf = _auto_config(M, N, K, world_size)

    assert config.block_size_m > 0
    assert config.block_size_n > 0
    assert config.block_size_k > 0

    num_k_blocks = K // config.block_size_k
    assert num_k_blocks % kpf == 0, (
        f"k_per_flag={kpf} does not divide num_k_blocks={num_k_blocks} for M={M},N={N},K={K}"
    )
    assert fs > 0, "num_fetch_sms must be positive"
    assert nfs > 0, "num_fetch_stages must be positive"
    assert fsf > 0, "first_stage_fetch_sms must be positive"


def test_auto_config_champion_shapes():
    """Verify that champion shapes are returned directly from _CHAMPION_CONFIGS."""
    for key in _CHAMPION_CONFIGS:
        M, N, K = key
        config, kpf, fs, nfs, fsf = _auto_config(M, N, K, world_size=8)
        c = _CHAMPION_CONFIGS[key]

        assert config.block_size_m == c["bm"]
        assert config.block_size_n == c["bn"]
        assert config.block_size_k == c["bk"]
        assert config.group_size_m == c["gm"]

        # kpf may be adjusted down by _auto_config when champion["kpf"] doesn't divide
        # num_k_blocks (e.g. different world_size changes K and therefore num_k_blocks).
        num_k_blocks = K // c["bk"]
        assert num_k_blocks % kpf == 0, f"Champion kpf={kpf} does not divide num_k_blocks={num_k_blocks} for {key}"


def test_auto_config_large_m_uses_block_256():
    """Verify _auto_config picks block_m=256 for large M (M >= 8192, M divisible by 256)."""
    config, *_ = _auto_config(8192, 3584, 8192, world_size=8)
    assert config.block_size_m == 256, f"Expected block_m=256 for large M, got {config.block_size_m}"


def test_auto_config_small_m_uses_block_128():
    """Verify _auto_config picks block_m=128 for small M (M < 8192)."""
    config, *_ = _auto_config(1024, 3584, 8192, world_size=8)
    assert config.block_size_m == 128, f"Expected block_m=128 for small M, got {config.block_size_m}"


def test_auto_config_block_n_always_256():
    """Verify _auto_config always selects block_n=256 (from sweep data)."""
    for M in [1024, 4096, 16384]:
        config, *_ = _auto_config(M, 3584, 8192, world_size=8)
        assert config.block_size_n == 256, f"Expected block_n=256 for M={M}, got {config.block_size_n}"


def test_auto_config_block_k_always_64():
    """Verify _auto_config always selects block_k=64 (exceeding LDS on MI300X with 128)."""
    for M in [1024, 4096, 16384]:
        config, *_ = _auto_config(M, 3584, 8192, world_size=8)
        assert config.block_size_k == 64, f"Expected block_k=64 for M={M}, got {config.block_size_k}"


if __name__ == "__main__":
    import sys

    if not dist.is_initialized():
        print("Run with: torchrun --nproc_per_node=2 tests/ops/test_all_gather_matmul.py")
        sys.exit(1)

    rank = dist.get_rank()
    torch.cuda.set_device(rank)
    print(f"[Rank {rank}] Tests in this file require pytest + torchrun. See tests/run_tests_distributed.py")
