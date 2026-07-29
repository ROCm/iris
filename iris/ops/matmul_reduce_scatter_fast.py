# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.

"""
Fast GEMM + ReduceScatter: hipBLASLt GEMM + persistent one-shot pull RS.

1.23-2.06x faster than torch.mm + RCCL RS on GPT-OSS-120B MoE shapes.

Usage:
    >>> shmem = iris.iris(heap_size)
    >>> output = torch.zeros(M_local, N, dtype=dtype, device=device)
    >>> matmul_reduce_scatter_fast(shmem, output, A, B)
"""

from typing import Optional, Tuple
import torch
import triton
import triton.language as tl
import iris


# Per-TP optimal configs from exhaustive sweep on MI355X
_AUTO_CONFIG = {
    2: dict(block_m=128, block_n=64, num_sms=196, num_warps=4),
    4: dict(block_m=64, block_n=64, num_sms=32, num_warps=4),
    8: dict(block_m=32, block_n=64, num_sms=32, num_warps=4),
}

_DEFAULT_CONFIG = dict(block_m=64, block_n=64, num_sms=64, num_warps=4)


def _get_config(world_size: int, M_local: int) -> dict:
    cfg = _AUTO_CONFIG.get(world_size, _DEFAULT_CONFIG).copy()
    while cfg["block_m"] > M_local and cfg["block_m"] > 4:
        cfg["block_m"] //= 2
    return cfg


@triton.jit
def _fast_reduce_scatter_kernel(
    input_ptr,
    output_ptr,
    M,
    N,
    M_local,
    stride_in_m,
    stride_in_n,
    stride_out_m,
    stride_out_n,
    heap_bases: tl.tensor,
    cur_rank: tl.constexpr,
    world_size: tl.constexpr,
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
    NUM_SMS: tl.constexpr,
):
    """Persistent one-shot pull RS kernel."""
    pid = tl.program_id(0)
    acc_dtype = tl.float32
    num_m_tiles = M_local // BLOCK_SIZE_M
    num_n_tiles = tl.cdiv(N, BLOCK_SIZE_N)
    total_tiles = num_m_tiles * num_n_tiles
    m_offset = cur_rank * num_m_tiles

    for tile_id in range(pid, total_tiles, NUM_SMS):
        local_pid_m = tile_id // num_n_tiles
        pid_n = tile_id % num_n_tiles
        global_pid_m = m_offset + local_pid_m

        rm = global_pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
        rm = tl.max_contiguous(tl.multiple_of(rm, BLOCK_SIZE_M), BLOCK_SIZE_M)
        rn = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
        rn = tl.max_contiguous(tl.multiple_of(rn % N, BLOCK_SIZE_N), BLOCK_SIZE_N)

        in_offset = rm[:, None] * stride_in_m + rn[None, :] * stride_in_n
        base_ptr = input_ptr + in_offset
        is_full = (global_pid_m * BLOCK_SIZE_M + BLOCK_SIZE_M <= M) & (
            pid_n * BLOCK_SIZE_N + BLOCK_SIZE_N <= N
        )

        if is_full:
            start_rank = pid % world_size
            acc = iris.load(
                base_ptr, cur_rank, start_rank, heap_bases, hint=(1, BLOCK_SIZE_N)
            ).to(acc_dtype)
            for i in tl.static_range(1, world_size):
                r = (start_rank + i) % world_size
                acc += iris.load(
                    base_ptr, cur_rank, r, heap_bases, hint=(1, BLOCK_SIZE_N)
                ).to(acc_dtype)
            out_rm = local_pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
            out_rm = tl.max_contiguous(tl.multiple_of(out_rm, BLOCK_SIZE_M), BLOCK_SIZE_M)
            out_ptrs = output_ptr + out_rm[:, None] * stride_out_m + rn[None, :] * stride_out_n
            tl.store(out_ptrs, acc.to(output_ptr.type.element_ty))
        else:
            mask = (rm[:, None] < M) & (rn[None, :] < N)
            start_rank = pid % world_size
            acc = iris.load(
                base_ptr, cur_rank, start_rank, heap_bases, mask=mask, hint=(1, BLOCK_SIZE_N)
            ).to(acc_dtype)
            for i in tl.static_range(1, world_size):
                r = (start_rank + i) % world_size
                acc += iris.load(
                    base_ptr, cur_rank, r, heap_bases, mask=mask, hint=(1, BLOCK_SIZE_N)
                ).to(acc_dtype)
            out_rm = local_pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
            out_mask = (out_rm[:, None] < M_local) & (rn[None, :] < N)
            out_ptrs = output_ptr + out_rm[:, None] * stride_out_m + rn[None, :] * stride_out_n
            tl.store(out_ptrs, acc.to(output_ptr.type.element_ty), mask=out_mask)


def fast_reduce_scatter(
    ctx,
    output_tensor: torch.Tensor,
    input_tensor: torch.Tensor,
    block_m: Optional[int] = None,
    block_n: Optional[int] = None,
    num_sms: Optional[int] = None,
    num_warps: Optional[int] = None,
):
    """
    Fast one-shot pull reduce-scatter using iris.load.

    1.25-1.47x faster than RCCL RS at GPT-OSS-120B message sizes.

    Args:
        ctx: Iris context (symmetric heap must contain input_tensor)
        output_tensor: Output tensor (M_local, N) — this rank's reduced shard
        input_tensor: Input tensor (M, N) — full partial sum in symmetric heap
        block_m: Tile M dimension (auto-selected if None)
        block_n: Tile N dimension (auto-selected if None)
        num_sms: Number of persistent WGs (auto-selected if None)
        num_warps: Warps per WG (auto-selected if None)
    """
    M, N = input_tensor.shape
    rank = ctx.get_rank()
    world_size = ctx.get_num_ranks()
    M_local = M // world_size

    assert output_tensor.shape == (M_local, N)
    assert M % world_size == 0

    cfg = _get_config(world_size, M_local)
    bm = block_m or cfg["block_m"]
    bn = block_n or cfg["block_n"]
    sms = num_sms or cfg["num_sms"]
    warps = num_warps or cfg["num_warps"]

    assert M_local % bm == 0

    heap_bases = ctx.get_heap_bases()

    _fast_reduce_scatter_kernel[(sms,)](
        input_tensor,
        output_tensor,
        M,
        N,
        M_local,
        input_tensor.stride(0),
        input_tensor.stride(1),
        output_tensor.stride(0),
        output_tensor.stride(1),
        heap_bases,
        rank,
        world_size,
        bm,
        bn,
        sms,
        num_warps=warps,
    )


def matmul_reduce_scatter_fast(
    ctx,
    output_tensor: torch.Tensor,
    A: torch.Tensor,
    B: torch.Tensor,
    block_m: Optional[int] = None,
    block_n: Optional[int] = None,
    num_sms: Optional[int] = None,
    num_warps: Optional[int] = None,
) -> None:
    """
    Fast GEMM + ReduceScatter: hipBLASLt GEMM + one-shot pull RS.

    Computes C = reduce_scatter(A @ B) where each rank keeps M_local rows.
    Uses hipBLASLt for GEMM (via torch.mm) and a persistent Triton kernel
    for RS. No host barrier between the two (same-stream ordering).

    1.23-2.06x faster than torch.mm + RCCL RS on GPT-OSS-120B shapes.

    Args:
        ctx: Iris context
        output_tensor: Output (M_local, N) — this rank's reduced shard
        A: Input matrix (M, K_local) — this rank's K-shard
        B: Input matrix (K_local, N)
        block_m: RS tile M (auto-selected if None)
        block_n: RS tile N (auto-selected if None)
        num_sms: RS persistent WGs (auto-selected if None)
        num_warps: RS warps per WG (auto-selected if None)

    Example:
        >>> shmem = iris.iris(1 << 33)
        >>> A = shmem.zeros((M, K_local), dtype=torch.float16)
        >>> B = torch.randn(K_local, N, dtype=torch.float16, device="cuda")
        >>> output = torch.zeros(M_local, N, dtype=torch.float16, device="cuda")
        >>> matmul_reduce_scatter_fast(shmem, output, A, B)
    """
    M, K_local = A.shape
    _, N = B.shape
    world_size = ctx.get_num_ranks()
    M_local = M // world_size

    assert output_tensor.shape == (M_local, N)

    # GEMM: write partial C to symmetric heap buffer
    C_partial = ctx.zeros((M, N), dtype=A.dtype)
    torch.mm(A, B, out=C_partial)

    # RS: one-shot pull from all peers
    fast_reduce_scatter(
        ctx, output_tensor, C_partial,
        block_m=block_m, block_n=block_n,
        num_sms=num_sms, num_warps=num_warps,
    )
