# SPDX-License-Identifier: MIT
# Copyright (c) 2025-2026 Advanced Micro Devices, Inc. All rights reserved.
"""
DP-to-EP token dispatch via iris symmetric heap.

Closely follows triton_kernels/distributed.py _convert_dp_to_ep:
  https://github.com/triton-lang/triton/blob/main/python/triton_kernels/triton_kernels/distributed.py

One Triton program per local token.  For each of its k expert activations,
the kernel determines which rank owns the expert using a bitmask lookup and
scatters the token's activation row into that rank's destination buffer
via iris.store.
"""

import triton
import triton.language as tl
import iris


@triton.jit
def _convert_dp_to_ep(
    dst_ptr,
    dst_stride_m,
    src_ptr,
    src_stride_m,
    src_shape_n,
    expt_filter_ptr,
    expt_filter_stride_m,
    expt_indx_ptr,
    expt_indx_stride_m,
    dst_row_indx_ptr,
    dst_row_indx_stride_m,
    n_tokens_local,
    heap_bases,
    SRC_RANK: tl.constexpr,
    N_EXPT_ACT: tl.constexpr,
    N_RANKS: tl.constexpr,
    BLOCK: tl.constexpr,
):
    pid_m = tl.program_id(0)
    off_m_global = pid_m + n_tokens_local * SRC_RANK
    off_m_local = pid_m

    offs_n = tl.arange(0, BLOCK)
    offs_n = tl.max_contiguous(tl.multiple_of(offs_n, BLOCK), BLOCK)

    for act in tl.static_range(N_EXPT_ACT):
        dst_row = tl.load(dst_row_indx_ptr + off_m_global * dst_row_indx_stride_m + act)
        if dst_row >= 0:
            expt_id = tl.load(expt_indx_ptr + off_m_global * expt_indx_stride_m + act)

            dst_rank = 0
            for r in tl.static_range(N_RANKS):
                word = expt_id // 32
                bit = expt_id % 32
                filt = tl.load(expt_filter_ptr + r * expt_filter_stride_m + word)
                if (filt >> bit) & 1:
                    dst_rank = r

            for start_n in range(0, src_shape_n, BLOCK):
                mask_n = start_n + offs_n < src_shape_n
                src = tl.load(
                    src_ptr + off_m_local * src_stride_m + start_n + offs_n,
                    mask=mask_n,
                    other=0.0,
                )
                dst_off = dst_row * dst_stride_m + start_n + offs_n
                for r in tl.static_range(N_RANKS):
                    if dst_rank == r:
                        iris.store(dst_ptr + dst_off, src, SRC_RANK, r, heap_bases, mask=mask_n, hint=16)


def convert_dp_to_ep(src, expt_assignment, expt_indx, gate_indx, shmem):
    """Dispatch local tokens to expert-owning ranks.

    Args:
        src: (n_tokens_local, d_model) local activations.
        expt_assignment: ExptAssignment with bitmask.
        expt_indx: (n_tokens_global, n_expts_act) int16/int32 expert ids.
        gate_indx: (n_tokens_global * n_expts_act,) row_sorted_indx (dispatch order).
        shmem: iris.Iris instance.

    Returns:
        dst_local: (n_tokens_global * n_expts_act, d_model) dispatch buffer
                   on this rank's iris heap.
    """
    expt_bitmask = expt_assignment.expt_bitmask
    device = src.device
    n_tokens_local, d_model = src.shape
    n_tokens_global, n_expt_act = expt_indx.shape

    dst_local = shmem.zeros((n_tokens_global * n_expt_act, d_model), dtype=src.dtype)
    shmem.barrier()

    BLOCK = min(triton.next_power_of_2(d_model), 512)
    grid = (n_tokens_local,)

    _convert_dp_to_ep[grid](
        dst_local,
        dst_local.stride(0),
        src,
        src.stride(0),
        src.shape[1],
        expt_bitmask,
        expt_bitmask.stride(0),
        expt_indx,
        expt_indx.stride(0),
        gate_indx,
        n_expt_act,
        n_tokens_local,
        shmem.get_heap_bases(),
        SRC_RANK=shmem.get_rank(),
        N_EXPT_ACT=n_expt_act,
        N_RANKS=shmem.get_num_ranks(),
        BLOCK=BLOCK,
    )

    shmem.barrier()
    return dst_local
