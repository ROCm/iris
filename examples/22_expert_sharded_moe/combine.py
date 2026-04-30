# SPDX-License-Identifier: MIT
# Copyright (c) 2025-2026 Advanced Micro Devices, Inc. All rights reserved.
"""
EP-to-DP result combine via iris symmetric heap.

Closely follows triton_kernels/distributed.py _convert_ep_to_dp:
  https://github.com/triton-lang/triton/blob/main/python/triton_kernels/triton_kernels/distributed.py

Each rank iterates over its expert-sorted output rows. For every row the
kernel looks up the global flat index via col_sorted_indx, determines
which rank owns the originating token, and writes the result into that
rank's per-rank destination buffer using iris.store.

Destination buffer shape per rank: (n_slots_per_rank, d_model) where
n_slots_per_rank = n_tokens_global // world_size.
"""

import triton
import triton.language as tl
import iris


@triton.jit
def _convert_ep_to_dp(
    dst_ptr,
    dst_stride_m,
    src_ptr,
    src_stride_m,
    src_shape_n,
    expt_filter_ptr,
    expt_filter_stride_m,
    expt_indx_ptr,
    dst_row_indx_ptr,
    n_slots_per_rank,
    heap_bases,
    BLOCK: tl.constexpr,
    SRC_RANK: tl.constexpr,
    N_RANKS: tl.constexpr,
):
    pid_m = tl.program_id(0)

    dst_indx_global = tl.load(dst_row_indx_ptr + pid_m)
    if dst_indx_global < 0:
        return

    dst_rank = dst_indx_global // n_slots_per_rank

    dst_expt_indx = tl.load(expt_indx_ptr + dst_indx_global).to(tl.int32)
    expt_filter_ptr_local = expt_filter_ptr + SRC_RANK * expt_filter_stride_m
    has_dst_expt = (tl.load(expt_filter_ptr_local + dst_expt_indx // 32) >> (dst_expt_indx % 32)) & 1
    if not has_dst_expt.to(tl.int1):
        return

    dst_indx_local = dst_indx_global - dst_rank * n_slots_per_rank

    offs_n = tl.arange(0, BLOCK)
    offs_n = tl.max_contiguous(tl.multiple_of(offs_n, BLOCK), BLOCK)
    for start_n in range(0, src_shape_n, BLOCK):
        mask_n = start_n + offs_n < src_shape_n
        src = tl.load(
            src_ptr + pid_m * src_stride_m + start_n + offs_n,
            mask=mask_n,
            other=0.0,
        )
        dst_off = dst_indx_local * dst_stride_m + start_n + offs_n
        for r in tl.static_range(N_RANKS):
            if dst_rank == r:
                iris.store(dst_ptr + dst_off, src, SRC_RANK, r, heap_bases, mask=mask_n, hint=16)


def convert_ep_to_dp(src, expt_assignment, expt_indx, topk_indx, shmem):
    """Scatter expert results back to token-owning ranks.

    Matches the upstream convert_ep_to_dp interface.

    Args:
        src: (n_total_slots, d_model) expert-sorted matmul output.
        expt_assignment: ExptAssignment with bitmask.
        expt_indx: (n_tokens_global * n_expts_act,) flat expert ids.
        topk_indx: (n_total_slots,) col_sorted_indx (combine order).
        shmem: iris.Iris instance.

    Returns:
        dst_local: (n_slots_per_rank, d_model) this rank's combine buffer.
    """
    expt_bitmask = expt_assignment.expt_bitmask
    n_tokens_global, d_model = src.shape
    n_tokens_local = n_tokens_global // shmem.get_num_ranks()

    dst_local = shmem.zeros((n_tokens_local, d_model), dtype=src.dtype)
    shmem.barrier()

    BLOCK = min(triton.next_power_of_2(d_model), 512)
    grid = (n_tokens_global,)

    _convert_ep_to_dp[grid](
        dst_local,
        dst_local.stride(0),
        src,
        src.stride(0),
        src.shape[1],
        expt_bitmask,
        expt_bitmask.stride(0),
        expt_indx,
        topk_indx,
        n_tokens_local,
        shmem.get_heap_bases(),
        BLOCK=BLOCK,
        SRC_RANK=shmem.get_rank(),
        N_RANKS=shmem.get_num_ranks(),
    )

    shmem.barrier()
    return dst_local
