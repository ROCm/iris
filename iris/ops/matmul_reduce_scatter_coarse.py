# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.

"""
Fused GEMM+RS with COARSE flag granularity.

Instead of one flag per (m,n) tile, use one flag per M-tile-row.
At M=2048 bm=128 N=2880 bn=64: 16 flags instead of 720 (45x fewer atomics).

Each GEMM WG that completes the LAST tile in an M-row signals that row.
Comm WGs wait for the M-row flag, then process all N-tiles in that row.

Trade-off: coarser sync = less overlap granularity, but 45x fewer atomics.
At small shapes the atomic count dominates, so this should win.
"""

from typing import Optional
import torch
import triton
import triton.language as tl
import iris


@triton.jit
def _coarse_flag_gemm_rs_kernel(
    A, B,
    C_staged,
    C_out,
    row_flags,       # one counter per M-tile-row (not per tile!)
    row_progress,    # local counter: how many N-tiles done in this row
    M, N, K, M_local,
    stride_am, stride_ak,
    stride_bk, stride_bn,
    stride_sc_m, stride_sc_n,
    stride_out_m, stride_out_n,
    heap_bases: tl.tensor,
    cur_rank: tl.constexpr,
    world_size: tl.constexpr,
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr,
    GEMM_SMS: tl.constexpr,
    NUM_SMS: tl.constexpr,
    EVEN_K: tl.constexpr,
    NUM_M_TILES: tl.constexpr,
    NUM_N_TILES: tl.constexpr,
    NUM_LOCAL_M_TILES: tl.constexpr,
    TOTAL_TILES: tl.constexpr,
):
    pid = tl.program_id(0)
    acc_dtype = tl.float32

    if pid < GEMM_SMS:
        # ============ GEMM PHASE ============
        for tile_id in range(pid, TOTAL_TILES, GEMM_SMS):
            pid_m = tile_id // NUM_N_TILES
            pid_n = tile_id % NUM_N_TILES

            rm = (pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)) % M
            rn = (pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)) % N
            rm = tl.max_contiguous(tl.multiple_of(rm, BLOCK_SIZE_M), BLOCK_SIZE_M)
            rn = tl.max_contiguous(tl.multiple_of(rn, BLOCK_SIZE_N), BLOCK_SIZE_N)

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

            c = acc.to(C_staged.type.element_ty)
            sc_offset = rm[:, None] * stride_sc_m + rn[None, :] * stride_sc_n
            tl.store(C_staged + sc_offset, c, cache_modifier=".wt")

            # Track progress within this M-row (LOCAL counter, device scope)
            tl.debug_barrier()
            done_in_row = tl.atomic_add(row_progress + pid_m, 1, sem="release", scope="gpu")

            # Last tile in this M-row? Signal all ranks (COARSE — 1 flag per row)
            if done_in_row == NUM_N_TILES - 1:
                tl.atomic_add(row_flags + pid_m, 1, sem="release", scope="gpu")
                for peer in tl.static_range(world_size):
                    if peer != cur_rank:
                        iris.atomic_add(
                            row_flags + pid_m, 1,
                            cur_rank, peer, heap_bases,
                            sem="release", scope="sys",
                        )

    else:
        # ============ COMM PHASE ============
        comm_pid = pid - GEMM_SMS
        COMM_SMS: tl.constexpr = NUM_SMS - GEMM_SMS
        m_offset = cur_rank * NUM_LOCAL_M_TILES

        # Process one M-row at a time
        for local_m in range(NUM_LOCAL_M_TILES):
            global_pid_m = m_offset + local_m

            # Wait for ALL ranks to finish this M-row (1 flag poll per row!)
            while tl.atomic_add(row_flags + global_pid_m, 0, sem="acquire", scope="gpu") < world_size:
                pass

            # Now process all N-tiles in this row
            for pid_n in range(comm_pid, NUM_N_TILES, COMM_SMS):
                rm = global_pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
                rm = tl.max_contiguous(tl.multiple_of(rm, BLOCK_SIZE_M), BLOCK_SIZE_M)
                rn = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
                rn = tl.max_contiguous(tl.multiple_of(rn % N, BLOCK_SIZE_N), BLOCK_SIZE_N)

                sc_offset = rm[:, None] * stride_sc_m + rn[None, :] * stride_sc_n
                base_ptr = C_staged + sc_offset
                is_full = (global_pid_m * BLOCK_SIZE_M + BLOCK_SIZE_M <= M) & (
                    pid_n * BLOCK_SIZE_N + BLOCK_SIZE_N <= N
                )

                if is_full:
                    start_rank = comm_pid % world_size
                    acc = iris.load(base_ptr, cur_rank, start_rank, heap_bases, hint=(1, BLOCK_SIZE_N)).to(acc_dtype)
                    for i in tl.static_range(1, world_size):
                        r = (start_rank + i) % world_size
                        acc += iris.load(base_ptr, cur_rank, r, heap_bases, hint=(1, BLOCK_SIZE_N)).to(acc_dtype)

                    out_rm = local_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
                    out_rm = tl.max_contiguous(tl.multiple_of(out_rm, BLOCK_SIZE_M), BLOCK_SIZE_M)
                    tl.store(C_out + out_rm[:, None] * stride_out_m + rn[None, :] * stride_out_n,
                             acc.to(C_out.type.element_ty))
                else:
                    mask = (rm[:, None] < M) & (rn[None, :] < N)
                    start_rank = comm_pid % world_size
                    acc = iris.load(base_ptr, cur_rank, start_rank, heap_bases, mask=mask, hint=(1, BLOCK_SIZE_N)).to(acc_dtype)
                    for i in tl.static_range(1, world_size):
                        r = (start_rank + i) % world_size
                        acc += iris.load(base_ptr, cur_rank, r, heap_bases, mask=mask, hint=(1, BLOCK_SIZE_N)).to(acc_dtype)

                    out_rm = local_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
                    out_mask = (out_rm[:, None] < M_local) & (rn[None, :] < N)
                    tl.store(C_out + out_rm[:, None] * stride_out_m + rn[None, :] * stride_out_n,
                             acc.to(C_out.type.element_ty), mask=out_mask)


def matmul_reduce_scatter_coarse(
    ctx,
    output_tensor: torch.Tensor,
    A: torch.Tensor,
    B: torch.Tensor,
    block_m: int = 128,
    block_n: int = 64,
    block_k: int = 64,
    gemm_sms: int = 240,
    num_sms: int = 304,
    num_warps: int = 8,
):
    """
    Fused GEMM+RS with coarse (per-M-row) flag granularity.

    45x fewer atomics than per-tile flags at M=2048 N=2880.
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

    staged_c = ctx.zeros((M, N), dtype=A.dtype)
    # COARSE: one flag per M-tile-row (not per tile!)
    row_flags = ctx.zeros((num_m_tiles,), dtype=torch.int32)
    row_progress = torch.zeros(num_m_tiles, dtype=torch.int32, device=f"cuda:{rank}")

    heap_bases = ctx.get_heap_bases()

    launch_kwargs = {}
    if getattr(torch.version, "hip", None):
        launch_kwargs["matrix_instr_nonkdim"] = 16

    _coarse_flag_gemm_rs_kernel[(num_sms,)](
        A, B, staged_c, output_tensor, row_flags, row_progress,
        M, N, K, M_local,
        A.stride(0), A.stride(1),
        B.stride(0), B.stride(1),
        staged_c.stride(0), staged_c.stride(1),
        output_tensor.stride(0), output_tensor.stride(1),
        heap_bases, rank, world_size,
        block_m, block_n, block_k,
        gemm_sms, num_sms,
        K % block_k == 0,
        num_m_tiles, num_n_tiles, num_local_m_tiles, total_tiles,
        num_warps=num_warps, num_stages=2,
        **launch_kwargs,
    )
