# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.

"""
Fused GEMM + Staged ReduceScatter: all-local compute.

Architecture (mirrors AG+GEMM HBM buffer):
  - GEMM WGs: compute partial C → staged_c (symmetric heap, .wt)
  - Fetcher WGs: iris.get(peer staged_c → local_buf[peer]) per tile, set fetch flag
  - Reduce WGs: wait for fetch flags → sum local_buf[0..ws-1] → output
  - ALL reduce reads are from LOCAL HBM — no XGMI during reduce

Fetch sync is GPU-scope only (fetcher and reducer on same GPU).
Cross-rank XGMI happens only in the fetch phase, not during compute/reduce.
"""

from typing import Optional
import torch
import triton
import triton.language as tl
import iris

from .config import FusedConfig
from .workspace import FusedWorkspace


@triton.jit
def _staged_rs_kernel(
    staged_c,
    local_bufs,
    output_ptr,
    fetch_flags,
    M,
    N,
    M_local,
    stride_sc_m,
    stride_sc_n,
    stride_lb_peer,
    stride_lb_m,
    stride_lb_n,
    stride_out_m,
    stride_out_n,
    heap_bases: tl.tensor,
    cur_rank: tl.constexpr,
    world_size: tl.constexpr,
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
    NUM_FETCH_SMS: tl.constexpr,
    NUM_REDUCE_SMS: tl.constexpr,
    TOTAL_SMS: tl.constexpr,
    NUM_M_TILES_LOCAL: tl.constexpr,
    NUM_TILES_N: tl.constexpr,
    TOTAL_LOCAL_TILES: tl.constexpr,
):
    """
    Two-phase RS kernel:
    Phase 1 (fetcher WGs): iris.get each peer's staged_c → local_bufs[peer]
    Phase 2 (reduce WGs): sum local_bufs → output (all local HBM)
    """
    pid = tl.program_id(0)
    acc_dtype = tl.float32

    m_offset = cur_rank * NUM_M_TILES_LOCAL

    if pid < NUM_FETCH_SMS:
        # ==========================================================
        # FETCH PHASE — copy peer staged_c to local_bufs
        # ==========================================================
        for tile_id in range(pid, TOTAL_LOCAL_TILES, NUM_FETCH_SMS):
            local_pid_m = tile_id // NUM_TILES_N
            pid_n = tile_id % NUM_TILES_N
            global_pid_m = m_offset + local_pid_m

            rm = global_pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
            rm = tl.max_contiguous(tl.multiple_of(rm, BLOCK_SIZE_M), BLOCK_SIZE_M)
            rn = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
            rn = tl.max_contiguous(tl.multiple_of(rn % N, BLOCK_SIZE_N), BLOCK_SIZE_N)

            sc_offset = rm[:, None] * stride_sc_m + rn[None, :] * stride_sc_n
            is_full = (global_pid_m * BLOCK_SIZE_M + BLOCK_SIZE_M <= M) & (pid_n * BLOCK_SIZE_N + BLOCK_SIZE_N <= N)

            # Local tile indices for local_bufs
            out_rm = local_pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
            out_rm = tl.max_contiguous(tl.multiple_of(out_rm, BLOCK_SIZE_M), BLOCK_SIZE_M)
            lb_offset = out_rm[:, None] * stride_lb_m + rn[None, :] * stride_lb_n

            for peer in tl.static_range(world_size):
                # iris.get: load from peer's staged_c, store to local_bufs[peer]
                src_ptr = staged_c + sc_offset
                dst_ptr = local_bufs + peer * stride_lb_peer + lb_offset

                if is_full:
                    data = iris.load(src_ptr, cur_rank, peer, heap_bases, hint=(1, BLOCK_SIZE_N))
                    tl.store(dst_ptr, data)
                else:
                    mask = (rm[:, None] < M) & (rn[None, :] < N)
                    data = iris.load(src_ptr, cur_rank, peer, heap_bases, mask=mask, hint=(1, BLOCK_SIZE_N))
                    out_mask = (out_rm[:, None] < M_local) & (rn[None, :] < N)
                    tl.store(dst_ptr, data, mask=out_mask)

            # Signal: all peers staged for this tile
            tl.debug_barrier()
            tl.atomic_xchg(fetch_flags + tile_id, 1, sem="release", scope="gpu")

    else:
        # ==========================================================
        # REDUCE PHASE — all reads from local HBM, no XGMI
        # ==========================================================
        reduce_pid = pid - NUM_FETCH_SMS

        for tile_id in range(reduce_pid, TOTAL_LOCAL_TILES, NUM_REDUCE_SMS):
            local_pid_m = tile_id // NUM_TILES_N
            pid_n = tile_id % NUM_TILES_N

            # Wait for fetch to complete this tile
            while tl.atomic_add(fetch_flags + tile_id, 0, sem="acquire", scope="gpu") == 0:
                pass

            out_rm = local_pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
            out_rm = tl.max_contiguous(tl.multiple_of(out_rm, BLOCK_SIZE_M), BLOCK_SIZE_M)
            rn = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
            rn = tl.max_contiguous(tl.multiple_of(rn % N, BLOCK_SIZE_N), BLOCK_SIZE_N)

            lb_offset = out_rm[:, None] * stride_lb_m + rn[None, :] * stride_lb_n
            out_mask = (out_rm[:, None] < M_local) & (rn[None, :] < N)

            # Sum all ws local buffers — PURE LOCAL HBM READS
            acc = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=acc_dtype)
            for peer in tl.static_range(world_size):
                ptr = local_bufs + peer * stride_lb_peer + lb_offset
                tile = tl.load(ptr, mask=out_mask, other=0.0)
                acc += tile.to(acc_dtype)

            result = acc.to(output_ptr.type.element_ty)
            out_ptrs = output_ptr + out_rm[:, None] * stride_out_m + rn[None, :] * stride_out_n
            tl.store(out_ptrs, result, mask=out_mask)


def matmul_reduce_scatter_staged(
    ctx,
    output_tensor: torch.Tensor,
    A: torch.Tensor,
    B: torch.Tensor,
    block_m: int = 128,
    block_n: int = 64,
    num_fetch_sms: int = 64,
    num_reduce_sms: int = 64,
    num_warps: int = 4,
):
    """
    GEMM + Staged RS: hipBLASLt GEMM → fetcher WGs stage peers → local reduce.

    All reduce reads are from local HBM. XGMI only during fetch phase.

    Args:
        ctx: Iris context
        output_tensor: Output (M_local, N)
        A: Input (M, K_local) — in symmetric heap
        B: Input (K_local, N)
    """
    M, K_local = A.shape
    _, N = B.shape
    rank = ctx.get_rank()
    world_size = ctx.get_num_ranks()
    M_local = M // world_size

    assert output_tensor.shape == (M_local, N)
    assert M % world_size == 0
    assert M_local % block_m == 0

    # Phase 1: GEMM → staged_c (symmetric heap)
    staged_c = ctx.zeros((M, N), dtype=A.dtype)
    torch.mm(A, B, out=staged_c)

    # Allocate local staging buffers: [ws, M_local, N]
    local_bufs = torch.zeros(world_size, M_local, N, dtype=A.dtype, device=f"cuda:{rank}")

    num_m_tiles_local = M_local // block_m
    num_tiles_n = (N + block_n - 1) // block_n
    total_local_tiles = num_m_tiles_local * num_tiles_n

    fetch_flags = torch.zeros(total_local_tiles, dtype=torch.int32, device=f"cuda:{rank}")

    heap_bases = ctx.get_heap_bases()
    total_sms = num_fetch_sms + num_reduce_sms

    # Barrier: ensure all ranks' GEMM output is visible
    ctx.barrier()

    _staged_rs_kernel[(total_sms,)](
        staged_c,
        local_bufs,
        output_tensor,
        fetch_flags,
        M, N, M_local,
        staged_c.stride(0), staged_c.stride(1),
        local_bufs.stride(0), local_bufs.stride(1), local_bufs.stride(2),
        output_tensor.stride(0), output_tensor.stride(1),
        heap_bases, rank, world_size,
        block_m, block_n,
        num_fetch_sms, num_reduce_sms, total_sms,
        num_m_tiles_local, num_tiles_n, total_local_tiles,
        num_warps=num_warps,
    )

    ctx.barrier()
