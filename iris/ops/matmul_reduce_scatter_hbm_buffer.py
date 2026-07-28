# SPDX-License-Identifier: MIT
# Copyright (c) 2025-2026 Advanced Micro Devices, Inc. All rights reserved.

"""
Fused GEMM + Reduce-Scatter using a local HBM staging buffer with dedicated
GEMM and Scatter workgroups, launched data-parallel.

Architecture (mirror of all_gather_matmul_hbm_buffer):
  - GEMM WGs compute partial C = A_local @ B, store to staged_c in symmetric heap
  - Scatter WGs wait for GEMM flags, read peer staged_c tiles via iris.load, reduce, store to output

Data flow:
  A (M, K_local) x B (K_local, N) -> staged_c (M, N) -> reduce across peers -> C (M_local, N)

Each rank computes a partial sum (its K-shard). Scatter WGs pull the same tile
from all peers' staged_c, sum in fp32, and store to the local output C.
Only M_local = M/ws rows are assigned to each rank.
"""

from typing import Optional
import torch
import triton
import triton.language as tl
import iris

from .config import FusedConfig
from .workspace import FusedWorkspace


@triton.jit
def _hbm_buffer_matmul_reduce_scatter_kernel(
    A,
    B,
    C,
    staged_c,
    flags_ptr,
    M,
    N,
    K_local,
    M_local,
    stride_am,
    stride_ak,
    stride_bk,
    stride_bn,
    stride_cm,
    stride_cn,
    stride_sc_m,
    stride_sc_n,
    context_tensor: tl.tensor,
    cur_rank: tl.constexpr,
    world_size: tl.constexpr,
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr,
    GROUP_SIZE_M: tl.constexpr,
    NUM_SCATTER_SMS: tl.constexpr,
    NUM_M_TILES: tl.constexpr,
    NUM_TILES_N: tl.constexpr,
    NUM_M_TILES_LOCAL: tl.constexpr,
    GEMM_TILES_PER_STAGE: tl.constexpr,
    ALLOW_TF32: tl.constexpr,
):
    pid = tl.program_id(0)
    acc_dtype = tl.float32
    zero = tl.program_id(0) * 0

    ctx = iris.DeviceContext.initialize(context_tensor, cur_rank, world_size)

    # Interleaved layout: [GEMM WGs (G)] [Scatter WGs (S)]
    if pid < GEMM_TILES_PER_STAGE:
        # ==============================================================
        # GEMM PHASE — compute partial C = A_local @ B, store to staged_c
        # ==============================================================
        gemm_pid = pid

        # Tile assignment with GROUP_SIZE_M swizzle
        num_pid_in_group = GROUP_SIZE_M * NUM_TILES_N
        group_id = gemm_pid // num_pid_in_group
        first_pid_m = group_id * GROUP_SIZE_M
        first_pid_m = min(first_pid_m, NUM_M_TILES - 1)
        group_sz = min(NUM_M_TILES - first_pid_m, GROUP_SIZE_M)
        pid_m = first_pid_m + ((gemm_pid % num_pid_in_group) % group_sz)
        pid_n = (gemm_pid % num_pid_in_group) // group_sz
        pid_m = min(pid_m, NUM_M_TILES - 1)

        rm = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
        rm = tl.max_contiguous(tl.multiple_of(rm, BLOCK_SIZE_M), BLOCK_SIZE_M)
        rn = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
        rn = tl.max_contiguous(tl.multiple_of(rn % N, BLOCK_SIZE_N), BLOCK_SIZE_N)

        acc = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=acc_dtype)

        # K-loop over local K shard
        num_k_blocks = tl.cdiv(K_local, BLOCK_SIZE_K)
        for k_block in range(num_k_blocks):
            rk = k_block * BLOCK_SIZE_K + tl.arange(0, BLOCK_SIZE_K)
            rk = tl.max_contiguous(tl.multiple_of(rk, BLOCK_SIZE_K), BLOCK_SIZE_K)

            a_ptrs = A + rm[:, None] * stride_am + rk[None, :] * stride_ak
            a = tl.load(a_ptrs, mask=(rm[:, None] < M) & (rk[None, :] < K_local), other=0.0)

            b_ptrs = B + rk[:, None] * stride_bk + rn[None, :] * stride_bn
            b = tl.load(b_ptrs, mask=(rk[:, None] < K_local) & (rn[None, :] < N), other=0.0)

            if ALLOW_TF32:
                acc = tl.dot(a, b, acc, allow_tf32=True)
            else:
                acc += tl.dot(a, b, allow_tf32=False)

        # Store partial result to staged_c (in symmetric heap, peer-visible)
        c = acc.to(staged_c.type.element_ty)
        sc_ptrs = staged_c + rm.to(tl.int64)[:, None] * stride_sc_m + rn[None, :] * stride_sc_n
        sc_mask = (rm[:, None] < M) & (rn[None, :] < N)
        tl.store(sc_ptrs, c, mask=sc_mask, cache_modifier=".wt")

        # Signal tile is ready — scope="sys" for cross-GPU visibility
        tile_id = pid_m * NUM_TILES_N + pid_n
        tl.debug_barrier()
        tl.atomic_xchg(flags_ptr + tile_id, 1, sem="release", scope="sys")

    else:
        # ==============================================================
        # SCATTER PHASE — wait for GEMM, read all peers, reduce, store
        # ==============================================================
        scatter_pid = pid - GEMM_TILES_PER_STAGE

        # This rank owns M_local rows starting at m_offset
        m_offset = cur_rank * NUM_M_TILES_LOCAL

        # Scatter WGs loop over this rank's assigned tiles
        total_local_tiles = NUM_M_TILES_LOCAL * NUM_TILES_N

        for tile_offset in range(scatter_pid, total_local_tiles, NUM_SCATTER_SMS):
            local_pid_m = tile_offset // NUM_TILES_N
            pid_n = tile_offset % NUM_TILES_N

            # Global M-tile index (in the full M dimension)
            global_pid_m = m_offset + local_pid_m
            tile_id = global_pid_m * NUM_TILES_N + pid_n

            # Wait for ALL ranks' GEMM to finish this tile
            # Check each rank's flag via iris.load (flags are in symmetric heap)
            flag_offset = tile_id + tl.arange(0, 1) * 0  # 1-element offset
            for peer in tl.static_range(world_size):
                peer_done = zero
                while peer_done == 0:
                    peer_val = iris.load(
                        flags_ptr + flag_offset,
                        cur_rank, peer, ctx.heap_bases,
                        hint=(1, 1),
                    )
                    peer_done = tl.sum(peer_val)

            rm = global_pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
            rm = tl.max_contiguous(tl.multiple_of(rm, BLOCK_SIZE_M), BLOCK_SIZE_M)
            rn = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
            rn = tl.max_contiguous(tl.multiple_of(rn % N, BLOCK_SIZE_N), BLOCK_SIZE_N)

            sc_offset = rm[:, None] * stride_sc_m + rn[None, :] * stride_sc_n
            base_ptr = staged_c + sc_offset
            is_full = (global_pid_m * BLOCK_SIZE_M + BLOCK_SIZE_M <= M) & (pid_n * BLOCK_SIZE_N + BLOCK_SIZE_N <= N)

            if is_full:
                # Read from first rank and accumulate
                start_rank = scatter_pid % world_size
                acc = iris.load(base_ptr, cur_rank, start_rank, ctx.heap_bases, hint=(1, BLOCK_SIZE_N)).to(acc_dtype)
                for i in tl.static_range(1, world_size):
                    remote_rank = (start_rank + i) % world_size
                    acc += iris.load(base_ptr, cur_rank, remote_rank, ctx.heap_bases, hint=(1, BLOCK_SIZE_N)).to(acc_dtype)

                reduced = acc.to(C.type.element_ty)

                # Store to output — local M-tile index for output addressing
                out_rm = local_pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
                out_rm = tl.max_contiguous(tl.multiple_of(out_rm, BLOCK_SIZE_M), BLOCK_SIZE_M)
                out_ptrs = C + out_rm[:, None] * stride_cm + rn[None, :] * stride_cn
                tl.store(out_ptrs, reduced, cache_modifier=".wt")
            else:
                mask = (rm[:, None] < M) & (rn[None, :] < N)
                start_rank = scatter_pid % world_size
                acc = iris.load(base_ptr, cur_rank, start_rank, ctx.heap_bases, mask=mask, hint=(1, BLOCK_SIZE_N)).to(acc_dtype)
                for i in tl.static_range(1, world_size):
                    remote_rank = (start_rank + i) % world_size
                    acc += iris.load(base_ptr, cur_rank, remote_rank, ctx.heap_bases, mask=mask, hint=(1, BLOCK_SIZE_N)).to(acc_dtype)

                reduced = acc.to(C.type.element_ty)

                out_rm = local_pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
                out_mask = (out_rm[:, None] < M_local) & (rn[None, :] < N)
                out_ptrs = C + out_rm[:, None] * stride_cm + rn[None, :] * stride_cn
                tl.store(out_ptrs, reduced, mask=out_mask, cache_modifier=".wt")


def matmul_reduce_scatter_hbm_buffer_preamble(
    ctx,
    A: torch.Tensor,
    B: torch.Tensor,
    config: Optional[FusedConfig] = None,
) -> FusedWorkspace:
    """
    Allocate workspace for fused GEMM+ReduceScatter with HBM buffer staging.

    Args:
        ctx: Iris context
        A: Input matrix A (M, K_local) — each rank's K-shard
        B: Input matrix B (K_local, N)
        config: Optional FusedConfig
    """
    if config is None:
        config = FusedConfig(block_size_m=128, block_size_n=64, block_size_k=64, group_size_m=4)

    M, K_local = A.shape
    _, N = B.shape
    world_size = ctx.get_num_ranks()

    assert M % config.block_size_m == 0
    assert M % world_size == 0

    num_m_tiles = M // config.block_size_m
    num_tiles_n = (N + config.block_size_n - 1) // config.block_size_n
    total_tiles = num_m_tiles * num_tiles_n

    ws = FusedWorkspace(
        operation="matmul_reduce_scatter_hbm_buffer",
        shape=(M, N, K_local),
        dtype=A.dtype,
        world_size=world_size,
        variant="hbm_buffer",
        prepared=True,
    )

    # staged_c in symmetric heap — stores partial GEMM results, peer-visible
    ws.aux_buffer = ctx.zeros((M, N), dtype=A.dtype)

    # Flags in symmetric heap — one per tile, peer-visible for cross-rank synchronization
    ws.locks = ctx.zeros((total_tiles,), dtype=torch.int32)

    M_local = M // world_size
    buffer_mb = M * N * A.element_size() / (1024**2)
    ctx.info(
        f"HBM buffer RS: staged_c=({M},{N}) [{buffer_mb:.1f} MB], "
        f"flags={total_tiles}, M_local={M_local}"
    )

    ctx.barrier()
    return ws


def matmul_reduce_scatter_hbm_buffer(
    ctx,
    output_tensor: torch.Tensor,
    A: torch.Tensor,
    B: torch.Tensor,
    async_op: bool = False,
    config: Optional[FusedConfig] = None,
    workspace: Optional[FusedWorkspace] = None,
    num_scatter_sms: int = 32,
    num_warps: Optional[int] = 8,
    num_stages: Optional[int] = 2,
) -> FusedWorkspace:
    """
    Fused GEMM + Reduce-Scatter with HBM buffer staging.

    Each rank computes partial C = A_local @ B and stores to a peer-visible
    staging buffer. Dedicated scatter WGs then read all peers' partial results,
    reduce, and store to the output.

    Args:
        ctx: Iris context
        output_tensor: Output tensor (M_local, N) — this rank's reduced shard
        A: Input matrix A (M, K_local) — this rank's K-shard
        B: Input matrix B (K_local, N)
        async_op: If True, skip trailing barrier
        config: Optional FusedConfig
        workspace: Reusable workspace from preamble
        num_scatter_sms: Number of scatter workgroups
        num_warps: Warps per workgroup
        num_stages: Triton pipeline stages
    """
    if config is None:
        config = FusedConfig(block_size_m=128, block_size_n=64, block_size_k=64, group_size_m=4)

    M, K_local = A.shape
    _, N = B.shape
    rank = ctx.get_rank()
    world_size = ctx.get_num_ranks()
    M_local = M // world_size

    assert output_tensor.shape == (M_local, N)
    assert M % config.block_size_m == 0
    assert M % world_size == 0

    if workspace is None:
        workspace = matmul_reduce_scatter_hbm_buffer_preamble(ctx, A, B, config)

    workspace.locks.zero_()

    num_m_tiles = M // config.block_size_m
    num_tiles_n = (N + config.block_size_n - 1) // config.block_size_n
    gemm_tiles = num_m_tiles * num_tiles_n
    num_m_tiles_local = M_local // config.block_size_m

    grid_size = gemm_tiles + num_scatter_sms

    launch_kwargs = {}
    if getattr(torch.version, "hip", None):
        launch_kwargs["matrix_instr_nonkdim"] = 16
    if num_warps is not None:
        launch_kwargs["num_warps"] = num_warps
    if num_stages is not None:
        launch_kwargs["num_stages"] = num_stages

    _hbm_buffer_matmul_reduce_scatter_kernel[(grid_size,)](
        A,
        B,
        output_tensor,
        workspace.aux_buffer,
        workspace.locks,
        M,
        N,
        K_local,
        M_local,
        A.stride(0),
        A.stride(1),
        B.stride(0),
        B.stride(1),
        output_tensor.stride(0),
        output_tensor.stride(1),
        workspace.aux_buffer.stride(0),
        workspace.aux_buffer.stride(1),
        ctx.get_device_context(),
        rank,
        world_size,
        config.block_size_m,
        config.block_size_n,
        config.block_size_k,
        config.group_size_m,
        num_scatter_sms,
        num_m_tiles,
        num_tiles_n,
        num_m_tiles_local,
        gemm_tiles,
        config.allow_tf32,
        **launch_kwargs,
    )

    if not async_op:
        ctx.barrier()

    return workspace
