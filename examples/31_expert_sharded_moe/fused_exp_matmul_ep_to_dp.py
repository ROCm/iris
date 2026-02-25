# SPDX-License-Identifier: MIT
# Copyright (c) 2025-2026 Advanced Micro Devices, Inc. All rights reserved.
"""
Fused expert matmul + EP->DP combine.

This module fuses:
  grouped_matmul(y_ep_local, w_ep_local, b_ep_local, ...)
  + convert_ep_to_dp(...)

into a single Triton kernel that:
  1) computes the expert output row for slots owned by this rank
  2) immediately scatters the result to token-owning rank via iris.store

The ownership check mirrors combine.py's bitmask approach exactly.
"""

import torch
import triton
import triton.language as tl
import iris


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
    expt_filter_ptr,
    expt_filter_stride_m,
    expt_indx_ptr,
    expt_map_ptr,
    topk_indx_ptr,
    n_slots_per_rank,
    d_model,
    heap_bases,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
    HAS_BIAS: tl.constexpr,
    SRC_RANK: tl.constexpr,
    N_RANKS: tl.constexpr,
):
    pid_m = tl.program_id(0)

    dst_indx_global = tl.load(topk_indx_ptr + pid_m)
    if dst_indx_global < 0:
        return

    dst_rank = dst_indx_global // n_slots_per_rank

    # Ownership check: mirrors combine.py exactly using bitmask.
    dst_expt_indx = tl.load(expt_indx_ptr + dst_indx_global).to(tl.int32)
    expt_filter_ptr_local = expt_filter_ptr + SRC_RANK * expt_filter_stride_m
    has_dst_expt = (
        tl.load(expt_filter_ptr_local + dst_expt_indx // 32) >> (dst_expt_indx % 32)
    ) & 1
    if not has_dst_expt.to(tl.int1):
        return

    # Look up local expert id for weight indexing.
    local_expt = tl.load(expt_map_ptr + dst_expt_indx).to(tl.int32)

    dst_indx_local = dst_indx_global - dst_rank * n_slots_per_rank

    offs_n = tl.arange(0, BLOCK_N)
    for start_n in range(0, d_model, BLOCK_N):
        cur_n = start_n + offs_n
        mask_n = cur_n < d_model

        acc = tl.zeros([BLOCK_N], dtype=tl.float32)
        for start_k in range(0, d_model, BLOCK_K):
            offs_k = start_k + tl.arange(0, BLOCK_K)
            mask_k = offs_k < d_model

            x_ptrs = x_ptr + pid_m * x_stride_m + offs_k * x_stride_k
            x = tl.load(x_ptrs, mask=mask_k, other=0.0).to(tl.float32)

            w_ptrs = (
                w_ptr
                + local_expt * w_stride_e
                + offs_k[:, None] * w_stride_k
                + cur_n[None, :] * w_stride_n
            )
            w = tl.load(
                w_ptrs, mask=mask_k[:, None] & mask_n[None, :], other=0.0
            ).to(tl.float32)
            acc += tl.sum(x[:, None] * w, axis=0)

        if HAS_BIAS:
            b_ptrs = b_ptr + local_expt * b_stride_e + cur_n * b_stride_n
            acc += tl.load(b_ptrs, mask=mask_n, other=0.0).to(tl.float32)

        dst_off = dst_indx_local * dst_stride_m + cur_n
        out = acc.to(dst_ptr.dtype.element_ty)
        for r in tl.static_range(N_RANKS):
            if dst_rank == r:
                if r == SRC_RANK:
                    tl.store(dst_ptr + dst_off, out, mask=mask_n)
                else:
                    iris.store(
                        dst_ptr + dst_off, out, SRC_RANK, r, heap_bases, mask=mask_n
                    )


def fused_exp_matmul_ep_to_dp(
    x_ep_local: torch.Tensor,
    w_ep_local: torch.Tensor,
    b_ep_local: torch.Tensor | None,
    expt_assignment,
    expt_map_local: torch.Tensor,
    expt_indx_flat: torch.Tensor,
    combine_indx: torch.Tensor,
    shmem,
) -> torch.Tensor:
    """Compute expert matmul and scatter to DP-local output in one kernel.

    Args:
        x_ep_local: (n_total_slots, d_model) dispatched activations.
        w_ep_local: (n_local_experts, d_model, d_model) local expert weights.
        b_ep_local: (n_local_experts, d_model) local expert biases or None.
        expt_assignment: ExptAssignment with bitmask for ownership check.
        expt_map_local: (n_expts_tot,) global expert -> local expert id or -1.
        expt_indx_flat: (n_total_slots,) flat global expert ids by token-slot.
        combine_indx: (n_total_slots,) col_sorted_indx.
        shmem: iris.Iris instance.

    Returns:
        (n_slots_per_rank, d_model) DP-local combined output.
    """
    expt_bitmask = expt_assignment.expt_bitmask
    n_total_slots, d_model = x_ep_local.shape
    n_slots_per_rank = n_total_slots // shmem.get_num_ranks()

    dst_local = shmem.zeros((n_slots_per_rank, d_model), dtype=x_ep_local.dtype)
    shmem.barrier()

    BLOCK_N = min(triton.next_power_of_2(d_model), 128)
    BLOCK_K = 64
    grid = (n_total_slots,)

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
        expt_bitmask,
        expt_bitmask.stride(0),
        expt_indx_flat,
        expt_map_local,
        combine_indx,
        n_slots_per_rank,
        d_model,
        shmem.get_heap_bases(),
        BLOCK_N=BLOCK_N,
        BLOCK_K=BLOCK_K,
        HAS_BIAS=(b_ep_local is not None),
        SRC_RANK=shmem.get_rank(),
        N_RANKS=shmem.get_num_ranks(),
    )

    torch.cuda.synchronize()
    shmem.barrier()
    return dst_local
