# SPDX-License-Identifier: MIT
# Copyright (c) 2025-2026 Advanced Micro Devices, Inc. All rights reserved.
"""
WG-specialized fused expert matmul + EP->DP combine.

Workgroup-specialized variant of fused_exp_matmul_ep_to_dp that splits CUs
into persistent GEMM and communication paths within a single kernel:

  GEMM CUs (pid < GEMM_SMS):
    Compute tiled GEMM per expert, write to intermediate buffer, signal lock.
  Comm CUs (pid >= GEMM_SMS):
    Spin-wait on lock, load GEMM output, scatter to token-owning ranks via
    per-rank masked iris.store.

This overlaps GEMM compute with cross-rank scatter communication.
Inspired by examples/10_gemm_all_scatter_wg_specialization.

Grid: (NUM_SMS,) -- one persistent program per CU.
Lock granularity: one lock per (expert, N-tile, M-tile) triple.
"""

import torch
import triton
import triton.language as tl
import iris

from ragged_metadata import RaggedTensorMetadata


@triton.jit
def _wg_fused_exp_matmul_ep_to_dp_kernel(
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
    y_buf_ptr,
    y_stride_m,
    y_stride_n,
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
    max_m_tiles,
    heap_bases,
    locks_ptr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
    HAS_BIAS: tl.constexpr,
    SRC_RANK: tl.constexpr,
    N_RANKS: tl.constexpr,
    GEMM_SMS: tl.constexpr,
    NUM_SMS: tl.constexpr,
):
    pid = tl.program_id(0)
    n_n_tiles = tl.cdiv(N, BLOCK_N)
    total_en_pairs = n_n_tiles * n_local_experts

    if pid < GEMM_SMS:
        # ====== GEMM PATH ======
        for en_pair in range(pid, total_en_pairs, GEMM_SMS):
            local_expert_id = en_pair // n_n_tiles
            pid_n = en_pair % n_n_tiles

            if local_expert_id < n_local_experts:
                local_expert_id_64 = local_expert_id.to(tl.int64)
                slice_off = tl.load(slice_offs_ptr + local_expert_id_64).to(tl.int64)
                slice_size = tl.load(slice_sizes_ptr + local_expert_id_64)

                if slice_size > 0:
                    n_m_tiles = tl.cdiv(slice_size, BLOCK_M)
                    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
                    offs_n = tl.max_contiguous(tl.multiple_of(offs_n, BLOCK_N), BLOCK_N)
                    mask_n = offs_n < N

                    for pid_m in range(0, n_m_tiles):
                        offs_m_local = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
                        offs_m = slice_off + offs_m_local
                        mask_m = offs_m_local < slice_size

                        acc = tl.zeros([BLOCK_M, BLOCK_N], dtype=tl.float32)

                        for start_k in range(0, K, BLOCK_K):
                            offs_k = start_k + tl.arange(0, BLOCK_K)
                            mask_k = offs_k < K

                            x_ptrs = x_ptr + offs_m[:, None] * x_stride_m + offs_k[None, :] * x_stride_k
                            x = tl.load(x_ptrs, mask=mask_m[:, None] & mask_k[None, :], other=0.0)

                            w_ptrs = (
                                w_ptr
                                + local_expert_id_64 * w_stride_e
                                + offs_k[:, None] * w_stride_k
                                + offs_n[None, :] * w_stride_n
                            )
                            w = tl.load(w_ptrs, mask=mask_k[:, None] & mask_n[None, :], other=0.0)

                            acc += tl.dot(x, w)

                        if HAS_BIAS:
                            b_ptrs = b_ptr + local_expert_id_64 * b_stride_e + offs_n * b_stride_n
                            bias = tl.load(b_ptrs, mask=mask_n, other=0.0)
                            acc += bias[None, :]

                        out = acc.to(y_buf_ptr.dtype.element_ty)

                        y_ptrs = y_buf_ptr + offs_m[:, None] * y_stride_m + offs_n[None, :] * y_stride_n
                        tl.store(y_ptrs, out, mask=mask_m[:, None] & mask_n[None, :], cache_modifier=".wt")

                        tl.debug_barrier()
                        lock_idx = en_pair * max_m_tiles + pid_m
                        tl.store(locks_ptr + lock_idx, 1, cache_modifier=".wt")

    else:
        # ====== COMMUNICATION PATH ======
        COMM_SMS = NUM_SMS - GEMM_SMS
        comm_pid = pid - GEMM_SMS

        for en_pair in range(comm_pid, total_en_pairs, COMM_SMS):
            local_expert_id = en_pair // n_n_tiles
            pid_n = en_pair % n_n_tiles

            if local_expert_id < n_local_experts:
                local_expert_id_64 = local_expert_id.to(tl.int64)
                slice_off = tl.load(slice_offs_ptr + local_expert_id_64).to(tl.int64)
                slice_size = tl.load(slice_sizes_ptr + local_expert_id_64)

                if slice_size > 0:
                    n_m_tiles = tl.cdiv(slice_size, BLOCK_M)
                    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
                    offs_n = tl.max_contiguous(tl.multiple_of(offs_n, BLOCK_N), BLOCK_N)
                    mask_n = offs_n < N

                    for pid_m in range(0, n_m_tiles):
                        lock_idx = en_pair * max_m_tiles + pid_m
                        while tl.load(locks_ptr + lock_idx, cache_modifier=".cv", volatile=True) != 1:
                            pass

                        offs_m_local = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
                        offs_m = slice_off + offs_m_local
                        mask_m = offs_m_local < slice_size

                        dst_indx_globals = tl.load(topk_indx_ptr + offs_m, mask=mask_m, other=-1)
                        valid_dst = mask_m & (dst_indx_globals >= 0)

                        safe_dst_indx = tl.where(valid_dst, dst_indx_globals, tl.zeros_like(dst_indx_globals))
                        dst_expt_indxs = tl.load(expt_indx_ptr + safe_dst_indx, mask=valid_dst, other=0).to(tl.int32)

                        expt_filter_ptr_local = expt_filter_ptr + SRC_RANK * expt_filter_stride_m
                        has_dst_expts = (
                            (
                                tl.load(expt_filter_ptr_local + dst_expt_indxs // 32, mask=valid_dst, other=0)
                                >> (dst_expt_indxs % 32)
                            )
                            & 1
                        ).to(tl.int1)

                        row_valid = valid_dst & has_dst_expts
                        dst_ranks = dst_indx_globals // n_slots_per_rank
                        dst_indx_locals = dst_indx_globals - dst_ranks * n_slots_per_rank
                        dst_indx_locals = tl.where(row_valid, dst_indx_locals, tl.zeros_like(dst_indx_locals))

                        y_ptrs = y_buf_ptr + offs_m[:, None] * y_stride_m + offs_n[None, :] * y_stride_n
                        out = tl.load(y_ptrs, mask=mask_m[:, None] & mask_n[None, :], other=0.0)

                        dst_ptrs_2d = dst_ptr + dst_indx_locals[:, None] * dst_stride_m + offs_n[None, :]
                        for r in tl.static_range(N_RANKS):
                            rank_mask = row_valid & (dst_ranks == r)
                            store_mask = rank_mask[:, None] & mask_n[None, :]
                            if r == SRC_RANK:
                                tl.store(dst_ptrs_2d, out, mask=store_mask)
                            else:
                                iris.store(dst_ptrs_2d, out, SRC_RANK, r, heap_bases, mask=store_mask, hint=(1, 16))


def _heuristic_wg_config(num_sms: int, avg_bpe: int) -> tuple[int, int]:
    """Select (gemm_sms, block_m) heuristically based on avg tokens-per-expert.

    Heuristic derived from a tune sweep on MI300X (304 CUs, 8 ranks) across
    bpe ∈ {64, 128, 256, 512, 1024} with block_m ∈ {32, 64, 128, 256} and
    gemm_sms ∈ {¼, ½, ¾} × num_sms:

      bpe ≤ 64  → gemm_sms = num_sms // 2,     block_m = 128
      bpe ≤ 128 → gemm_sms = 3 * num_sms // 4,  block_m = 128
      bpe > 128 → gemm_sms = 3 * num_sms // 4,  block_m = 256

    Args:
        num_sms: Total CU count on the device.
        avg_bpe: Average number of tokens routed per local expert
                 (n_slots_per_rank // n_local_experts).

    Returns:
        (gemm_sms, block_m) tuple.
    """
    if avg_bpe <= 64:
        return max(1, num_sms // 2), 128
    elif avg_bpe <= 128:
        return max(1, 3 * num_sms // 4), 128
    else:
        return max(1, 3 * num_sms // 4), 256


def wg_fused_exp_matmul_ep_to_dp(
    x_ep_local: torch.Tensor,
    w_ep_local: torch.Tensor,
    b_ep_local: torch.Tensor | None,
    expt_assignment,
    expt_map_local: torch.Tensor,
    expt_indx_flat: torch.Tensor,
    combine_indx: torch.Tensor,
    shmem,
    ragged_metadata: RaggedTensorMetadata | None = None,
    gemm_sms: int | None = None,
    block_m: int | None = None,
    block_n: int | None = None,
    block_k: int | None = None,
) -> torch.Tensor:
    """WG-specialized fused expert matmul + EP->DP scatter.

    Same semantics as fused_exp_matmul_ep_to_dp but uses persistent kernel
    with workgroup specialization to overlap GEMM with scatter communication.

    Args:
        x_ep_local: (n_total_slots, d_model) dispatched activations.
        w_ep_local: (n_local_experts, d_model, d_model) local expert weights.
        b_ep_local: (n_local_experts, d_model) local expert biases or None.
        expt_assignment: ExptAssignment with bitmask for ownership check.
        expt_map_local: (n_expts_tot,) global expert -> local expert id or -1.
        expt_indx_flat: (n_total_slots,) flat global expert ids by token-slot.
        combine_indx: (n_total_slots,) col_sorted_indx.
        shmem: iris.Iris instance.
        ragged_metadata: local-expert-view ragged metadata.
        gemm_sms: Number of CUs for GEMM path.
            Default: auto-selected by _heuristic_wg_config based on avg bpe.
        block_m: GEMM tile size along the M (token) dimension.
            Default: auto-selected by _heuristic_wg_config based on avg bpe.
        block_n: GEMM tile size along the N (output) dimension.
            Default: min(triton.next_power_of_2(N), 128).
        block_k: GEMM tile size along the K (reduction) dimension.
            Default: min(triton.next_power_of_2(K), 64).

    Returns:
        (n_slots_per_rank, d_model) DP-local combined output.
    """
    expt_bitmask = expt_assignment.expt_bitmask
    n_total_slots, d_model = x_ep_local.shape
    n_local_experts = w_ep_local.shape[0]
    n_slots_per_rank = n_total_slots // shmem.get_num_ranks()
    K = d_model
    N = d_model

    device = x_ep_local.device
    num_sms = torch.cuda.get_device_properties(device).multi_processor_count

    # Derive heuristic defaults for gemm_sms / block_m when not specified.
    if gemm_sms is None or block_m is None:
        avg_bpe = n_slots_per_rank // max(n_local_experts, 1)
        h_gemm_sms, h_block_m = _heuristic_wg_config(num_sms, avg_bpe)
        if gemm_sms is None:
            gemm_sms = h_gemm_sms
        if block_m is None:
            block_m = h_block_m

    BLOCK_M = block_m
    BLOCK_N = block_n if block_n is not None else min(triton.next_power_of_2(N), 128)
    BLOCK_K = block_k if block_k is not None else min(triton.next_power_of_2(K), 64)

    max_slice_size = int(ragged_metadata.slice_sizes.max().item())
    max_m_tiles = triton.cdiv(max_slice_size, BLOCK_M)
    n_n_tiles = triton.cdiv(N, BLOCK_N)

    if max_m_tiles == 0:
        dst_local = shmem.zeros((n_slots_per_rank, d_model), dtype=x_ep_local.dtype)
        shmem.barrier()
        shmem.barrier()
        return dst_local

    y_buf = torch.zeros((n_total_slots, N), dtype=x_ep_local.dtype, device=device)
    dst_local = shmem.zeros((n_slots_per_rank, d_model), dtype=x_ep_local.dtype)
    n_locks = n_n_tiles * n_local_experts * max_m_tiles
    locks = torch.zeros(n_locks, dtype=torch.int32, device=device)

    shmem.barrier()

    _wg_fused_exp_matmul_ep_to_dp_kernel[(num_sms,)](
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
        y_buf,
        y_buf.stride(0),
        y_buf.stride(1),
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
        max_m_tiles,
        shmem.get_heap_bases(),
        locks,
        BLOCK_M=BLOCK_M,
        BLOCK_N=BLOCK_N,
        BLOCK_K=BLOCK_K,
        HAS_BIAS=(b_ep_local is not None),
        SRC_RANK=shmem.get_rank(),
        N_RANKS=shmem.get_num_ranks(),
        GEMM_SMS=gemm_sms,
        NUM_SMS=num_sms,
        num_warps=8,
        num_stages=2,
        matrix_instr_nonkdim=16,
        kpack=1,
    )

    shmem.barrier()
    return dst_local
