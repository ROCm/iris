# SPDX-License-Identifier: MIT
# Copyright (c) 2025-2026 Advanced Micro Devices, Inc. All rights reserved.
"""
Fused expert matmul + EP->DP combine.

This module fuses:
  grouped_matmul(y_ep_local, w_ep_local, b_ep_local, ...)
  + convert_ep_to_dp(...)

into a single Triton kernel that:
  1) computes a tiled GEMM (BLOCK_M x BLOCK_N via tl.dot) for each expert
  2) scatters the output tile to token-owning ranks via per-rank 2D iris.store

Grid: (n_n_tiles * n_local_experts,)  -- same tiling as grouped_matmul.
Each program loops over M-tiles for one (expert, N-tile) pair, computes
the tile with tl.dot, then does per-rank masked 2D stores.
"""

import torch
import triton
import triton.language as tl
import iris

from ragged_metadata import RaggedTensorMetadata


@triton.jit
def _fused_exp_matmul_ep_to_dp_kernel(
    dst_ptr,
    dst_stride_m,
    x_ptr,
    x_stride_m,
    x_stride_k,
    w_ptr,
    w_stride_e,
    w_stride_k,
    w_stride_n,
    b_ptr,
    b_stride_e,
    b_stride_n,
    slice_offs_ptr,
    slice_sizes_ptr,
    expt_filter_ptr,
    expt_filter_stride_m,
    expt_indx_ptr,
    topk_indx_ptr,
    n_local_experts,
    n_slots_per_rank,
    K,
    N,
    heap_bases,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
    HAS_BIAS: tl.constexpr,
    SRC_RANK: tl.constexpr,
    N_RANKS: tl.constexpr,
):
    pid = tl.program_id(0)
    n_n_tiles = tl.cdiv(N, BLOCK_N)

    local_expert_id = pid // n_n_tiles
    pid_n = pid % n_n_tiles

    if local_expert_id >= n_local_experts:
        return

    local_expert_id_64 = local_expert_id.to(tl.int64)
    slice_off = tl.load(slice_offs_ptr + local_expert_id_64).to(tl.int64)
    slice_size = tl.load(slice_sizes_ptr + local_expert_id_64)
    if slice_size == 0:
        return

    n_m_tiles = tl.cdiv(slice_size, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_n = tl.max_contiguous(tl.multiple_of(offs_n, BLOCK_N), BLOCK_N)
    mask_n = offs_n < N

    for pid_m in range(0, n_m_tiles):
        offs_m_local = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
        offs_m = slice_off + offs_m_local
        mask_m = offs_m_local < slice_size

        # Pre-load scatter metadata for this M-tile.
        dst_indx_globals = tl.load(topk_indx_ptr + offs_m, mask=mask_m, other=-1)
        valid_dst = mask_m & (dst_indx_globals >= 0)

        safe_dst_indx = tl.where(valid_dst, dst_indx_globals, tl.zeros_like(dst_indx_globals))
        dst_expt_indxs = tl.load(expt_indx_ptr + safe_dst_indx, mask=valid_dst, other=0).to(tl.int32)

        expt_filter_ptr_local = expt_filter_ptr + SRC_RANK * expt_filter_stride_m
        has_dst_expts = (
            (tl.load(expt_filter_ptr_local + dst_expt_indxs // 32, mask=valid_dst, other=0) >> (dst_expt_indxs % 32))
            & 1
        ).to(tl.int1)

        row_valid = valid_dst & has_dst_expts
        dst_ranks = dst_indx_globals // n_slots_per_rank
        dst_indx_locals = dst_indx_globals - dst_ranks * n_slots_per_rank
        dst_indx_locals = tl.where(row_valid, dst_indx_locals, tl.zeros_like(dst_indx_locals))

        # Tiled GEMM.
        acc = tl.zeros([BLOCK_M, BLOCK_N], dtype=tl.float32)

        for start_k in range(0, K, BLOCK_K):
            offs_k = start_k + tl.arange(0, BLOCK_K)
            mask_k = offs_k < K

            x_ptrs = x_ptr + offs_m[:, None] * x_stride_m + offs_k[None, :] * x_stride_k
            x = tl.load(x_ptrs, mask=mask_m[:, None] & mask_k[None, :], other=0.0)

            w_ptrs = (
                w_ptr + local_expert_id_64 * w_stride_e + offs_k[:, None] * w_stride_k + offs_n[None, :] * w_stride_n
            )
            w = tl.load(w_ptrs, mask=mask_k[:, None] & mask_n[None, :], other=0.0)

            acc += tl.dot(x, w)

        if HAS_BIAS:
            b_ptrs = b_ptr + local_expert_id_64 * b_stride_e + offs_n * b_stride_n
            bias = tl.load(b_ptrs, mask=mask_n, other=0.0)
            acc += bias[None, :]

        out = acc.to(dst_ptr.dtype.element_ty)

        # Per-rank 2D masked scatter.
        dst_ptrs_2d = dst_ptr + dst_indx_locals[:, None] * dst_stride_m + offs_n[None, :]
        for r in tl.static_range(N_RANKS):
            rank_mask = row_valid & (dst_ranks == r)
            store_mask = rank_mask[:, None] & mask_n[None, :]
            if r == SRC_RANK:
                tl.store(dst_ptrs_2d, out, mask=store_mask)
            else:
                iris.store(dst_ptrs_2d, out, SRC_RANK, r, heap_bases, mask=store_mask, hint=(1, 16))


def fused_exp_matmul_ep_to_dp(
    x_ep_local: torch.Tensor,
    w_ep_local: torch.Tensor,
    b_ep_local: torch.Tensor | None,
    expt_assignment,
    expt_map_local: torch.Tensor,
    expt_indx_flat: torch.Tensor,
    combine_indx: torch.Tensor,
    shmem,
    ragged_metadata: RaggedTensorMetadata | None = None,
) -> torch.Tensor:
    """Compute expert matmul and scatter to DP-local output in one kernel.

    Uses tiled GEMM (tl.dot) with per-rank 2D masked scatter -- same
    compute throughput as grouped_matmul but fused with the EP->DP combine.

    Args:
        x_ep_local: (n_total_slots, d_model) dispatched activations.
        w_ep_local: (n_local_experts, d_model, d_model) local expert weights.
        b_ep_local: (n_local_experts, d_model) local expert biases or None.
        expt_assignment: ExptAssignment with bitmask for ownership check.
        expt_map_local: (n_expts_tot,) global expert -> local expert id or -1.
        expt_indx_flat: (n_total_slots,) flat global expert ids by token-slot.
        combine_indx: (n_total_slots,) col_sorted_indx.
        shmem: iris.Iris instance.
        ragged_metadata: local-expert-view ragged metadata (slice_offs, slice_sizes).

    Returns:
        (n_slots_per_rank, d_model) DP-local combined output.
    """
    expt_bitmask = expt_assignment.expt_bitmask
    n_total_slots, d_model = x_ep_local.shape
    n_local_experts = w_ep_local.shape[0]
    n_slots_per_rank = n_total_slots // shmem.get_num_ranks()
    K = d_model
    N = d_model

    dst_local = shmem.zeros((n_slots_per_rank, d_model), dtype=x_ep_local.dtype)
    shmem.barrier()

    BLOCK_M = 128
    BLOCK_N = min(triton.next_power_of_2(N), 128)
    BLOCK_K = min(triton.next_power_of_2(K), 64)

    n_n_tiles = triton.cdiv(N, BLOCK_N)
    grid = (n_n_tiles * n_local_experts,)

    _fused_exp_matmul_ep_to_dp_kernel[grid](
        dst_local,
        dst_local.stride(0),
        x_ep_local,
        x_ep_local.stride(0),
        x_ep_local.stride(1),
        w_ep_local,
        w_ep_local.stride(0),
        w_ep_local.stride(1),
        w_ep_local.stride(2),
        b_ep_local if b_ep_local is not None else x_ep_local,
        b_ep_local.stride(0) if b_ep_local is not None else 0,
        b_ep_local.stride(1) if b_ep_local is not None else 0,
        ragged_metadata.slice_offs,
        ragged_metadata.slice_sizes,
        expt_bitmask,
        expt_bitmask.stride(0),
        expt_indx_flat,
        combine_indx,
        n_local_experts,
        n_slots_per_rank,
        K,
        N,
        shmem.get_heap_bases(),
        BLOCK_M=BLOCK_M,
        BLOCK_N=BLOCK_N,
        BLOCK_K=BLOCK_K,
        HAS_BIAS=(b_ep_local is not None),
        SRC_RANK=shmem.get_rank(),
        N_RANKS=shmem.get_num_ranks(),
        num_warps=8,
        num_stages=2,
        matrix_instr_nonkdim=16,
        kpack=1,
    )

    shmem.barrier()
    return dst_local
