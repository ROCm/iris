# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.

"""
Fused GEMM + ReduceScatter via push: iris.store to owner + local reduce.

Each GEMM WG pushes its tile to the owning rank's local staging buffer
via iris.store (fire-and-forget XGMI write). Owner spins on LOCAL flags
(scope=gpu, cheap), then reduces from local HBM.

Key advantages:
- Only 1 XGMI store per tile (to owner), not ws reads
- All flag polling is GPU-local (scope=gpu)
- Reduce is pure local HBM (1.6 TB/s)
- No WG specialization — all WGs do both phases
"""

from typing import Optional
import torch
import triton
import triton.language as tl
import iris


@triton.jit
def _push_gemm_rs_kernel(
    A, B,
    local_bufs,
    output_ptr,
    arrival_flags,
    M, N, K, M_local,
    stride_am, stride_ak,
    stride_bk, stride_bn,
    stride_lb_peer, stride_lb_m, stride_lb_n,
    stride_out_m, stride_out_n,
    heap_bases: tl.tensor,
    cur_rank: tl.constexpr,
    world_size: tl.constexpr,
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr,
    GROUP_SIZE_M: tl.constexpr,
    NUM_SMS: tl.constexpr,
    EVEN_K: tl.constexpr,
    NUM_M_TILES: tl.constexpr,
    NUM_N_TILES: tl.constexpr,
    NUM_LOCAL_M_TILES: tl.constexpr,
    TOTAL_TILES: tl.constexpr,
    TOTAL_LOCAL_TILES: tl.constexpr,
):
    pid = tl.program_id(0)
    acc_dtype = tl.float32

    # ================================================================
    # PHASE 1: GEMM + push each tile to owner's local_bufs[my_rank]
    # ================================================================
    for tile_id in range(pid, TOTAL_TILES, NUM_SMS):
        num_pid_in_group = GROUP_SIZE_M * NUM_N_TILES
        group_id = tile_id // num_pid_in_group
        first_pid_m = group_id * GROUP_SIZE_M
        group_size_m = min(NUM_M_TILES - first_pid_m, GROUP_SIZE_M)
        pid_m = first_pid_m + ((tile_id % num_pid_in_group) % group_size_m)
        pid_n = (tile_id % num_pid_in_group) // group_size_m

        rm = (pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)) % M
        rn = (pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)) % N
        rm = tl.max_contiguous(tl.multiple_of(rm, BLOCK_SIZE_M), BLOCK_SIZE_M)
        rn = tl.max_contiguous(tl.multiple_of(rn, BLOCK_SIZE_N), BLOCK_SIZE_N)

        # GEMM
        rk = tl.arange(0, BLOCK_SIZE_K)
        A_BASE = A + rm[:, None] * stride_am + rk[None, :] * stride_ak
        B_BASE = B + rk[:, None] * stride_bk + rn[None, :] * stride_bn

        loop_k = tl.cdiv(K, BLOCK_SIZE_K)
        if not EVEN_K:
            loop_k -= 1

        acc = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=acc_dtype)
        for k in range(0, loop_k):
            a = tl.load(tl.multiple_of(A_BASE, (1, 16)))
            b = tl.load(tl.multiple_of(B_BASE, (16, 1)))
            acc += tl.dot(a, b)
            A_BASE += BLOCK_SIZE_K * stride_ak
            B_BASE += BLOCK_SIZE_K * stride_bk

        if not EVEN_K:
            rk2 = loop_k * BLOCK_SIZE_K + tl.arange(0, BLOCK_SIZE_K)
            A_LAST = A + rm[:, None] * stride_am + rk2[None, :] * stride_ak
            B_LAST = B + rk2[:, None] * stride_bk + rn[None, :] * stride_bn
            a = tl.load(A_LAST, mask=rk2[None, :] < K, other=0.0)
            b = tl.load(B_LAST, mask=rk2[:, None] < K, other=0.0)
            acc += tl.dot(a, b)

        c = acc.to(local_bufs.type.element_ty)

        # Determine owner rank for this tile
        tile_m_start = pid_m * BLOCK_SIZE_M
        owner_rank = tile_m_start // M_local

        # Local M-tile index within owner's partition
        local_m = tile_m_start - owner_rank * M_local
        local_pid_m = local_m // BLOCK_SIZE_M
        local_tile_id = local_pid_m * NUM_N_TILES + pid_n

        # Write to owner's local_bufs[cur_rank, local_m:local_m+bm, pid_n*bn:pid_n*bn+bn]
        out_rm = local_m + tl.arange(0, BLOCK_SIZE_M)
        out_rn = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
        lb_offset = cur_rank * stride_lb_peer + out_rm[:, None] * stride_lb_m + out_rn[None, :] * stride_lb_n

        if owner_rank == cur_rank:
            tl.store(local_bufs + lb_offset, c)
        else:
            iris.store(local_bufs + lb_offset, c, cur_rank, owner_rank, heap_bases)

        # Signal: tile arrived on owner
        tl.debug_barrier()
        flag_idx = cur_rank * TOTAL_LOCAL_TILES + local_tile_id
        if owner_rank == cur_rank:
            tl.atomic_xchg(arrival_flags + flag_idx, 1, sem="release", scope="gpu")
        else:
            iris.atomic_cas(
                arrival_flags + flag_idx, 0, 1,
                cur_rank, owner_rank, heap_bases,
                sem="release", scope="sys",
            )

    # ================================================================
    # PHASE 2: Local reduce (only for tiles this rank owns)
    # All reads from local HBM — no XGMI
    # ================================================================
    for tile_id in range(pid, TOTAL_LOCAL_TILES, NUM_SMS):
        local_pid_m = tile_id // NUM_N_TILES
        pid_n = tile_id % NUM_N_TILES

        # Wait for ALL peers to push their contribution (LOCAL flag spin)
        for peer in tl.static_range(world_size):
            flag_idx = peer * TOTAL_LOCAL_TILES + tile_id
            while tl.atomic_add(arrival_flags + flag_idx, 0, sem="acquire", scope="gpu") == 0:
                pass

        # Sum all ws slots from local HBM
        out_rm = local_pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
        out_rm = tl.max_contiguous(tl.multiple_of(out_rm, BLOCK_SIZE_M), BLOCK_SIZE_M)
        out_rn = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
        out_rn = tl.max_contiguous(tl.multiple_of(out_rn % N, BLOCK_SIZE_N), BLOCK_SIZE_N)

        out_mask = (out_rm[:, None] < M_local) & (out_rn[None, :] < N)
        lb_offset = out_rm[:, None] * stride_lb_m + out_rn[None, :] * stride_lb_n

        acc = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=acc_dtype)
        for peer in tl.static_range(world_size):
            ptr = local_bufs + peer * stride_lb_peer + lb_offset
            tile = tl.load(ptr, mask=out_mask, other=0.0)
            acc += tile.to(acc_dtype)

        result = acc.to(output_ptr.type.element_ty)
        out_ptrs = output_ptr + out_rm[:, None] * stride_out_m + out_rn[None, :] * stride_out_n
        tl.store(out_ptrs, result, mask=out_mask)


def matmul_reduce_scatter_push(
    ctx,
    output_tensor: torch.Tensor,
    A: torch.Tensor,
    B: torch.Tensor,
    block_m: int = 128,
    block_n: int = 64,
    block_k: int = 64,
    group_m: int = 4,
    num_sms: int = 304,
    num_warps: int = 8,
):
    """
    Fused GEMM+RS via push: GEMM WGs iris.store to owner + local reduce.

    Single kernel, no WG specialization. All WGs do GEMM+push, then reduce.
    """
    M, K = A.shape
    _, N = B.shape
    rank = ctx.get_rank()
    world_size = ctx.get_num_ranks()
    M_local = M // world_size

    assert output_tensor.shape == (M_local, N)
    assert M % (world_size * block_m) == 0

    num_m_tiles = M // block_m
    num_n_tiles = (N + block_n - 1) // block_n
    total_tiles = num_m_tiles * num_n_tiles
    num_local_m_tiles = M_local // block_m
    total_local_tiles = num_local_m_tiles * num_n_tiles

    # local_bufs[ws, M_local, N] in symmetric heap (peers write to our slots)
    local_bufs = ctx.zeros((world_size, M_local, N), dtype=A.dtype)
    # arrival_flags[ws * total_local_tiles] in symmetric heap
    arrival_flags = ctx.zeros((world_size * total_local_tiles,), dtype=torch.int32)

    heap_bases = ctx.get_heap_bases()
    ctx.barrier()

    launch_kwargs = {}
    if getattr(torch.version, "hip", None):
        launch_kwargs["matrix_instr_nonkdim"] = 16

    _push_gemm_rs_kernel[(num_sms,)](
        A, B,
        local_bufs, output_tensor, arrival_flags,
        M, N, K, M_local,
        A.stride(0), A.stride(1),
        B.stride(0), B.stride(1),
        local_bufs.stride(0), local_bufs.stride(1), local_bufs.stride(2),
        output_tensor.stride(0), output_tensor.stride(1),
        heap_bases, rank, world_size,
        block_m, block_n, block_k, group_m,
        num_sms, K % block_k == 0,
        num_m_tiles, num_n_tiles, num_local_m_tiles,
        total_tiles, total_local_tiles,
        num_warps=num_warps, num_stages=2,
        **launch_kwargs,
    )
