# SPDX-License-Identifier: MIT
# Copyright (c) 2025 Advanced Micro Devices, Inc. All rights reserved.

"""
Gluon-based GEMM All-Scatter Example

This example demonstrates the Gluon port of the GEMM All-Scatter pattern,
which performs matrix multiplication with distributed computation and then
scatters results across all ranks.
"""

from triton.experimental import gluon
from triton.experimental.gluon import language as gl
import triton
import triton.language as tl
from examples.common.utils import read_realtime

import sys
import os

import iris.experimental.iris_gluon as iris_gl


@gluon.jit()
def persistent_gemm_all_scatter_gluon(
    IrisDeviceCtx: gl.constexpr,  # The aggregate class
    context_tensor,  # Encoded context
    A,
    B,
    C,
    c_global,
    bias_ptr,
    M,
    N,
    K,
    stride_am,
    stride_ak,
    stride_bk,
    stride_bn,
    stride_cm,
    stride_cn,
    stride_cm_global,
    stride_cn_global,
    stride_bias,
    BLOCK_SIZE_M: gl.constexpr,
    BLOCK_SIZE_N: gl.constexpr,
    BLOCK_SIZE_K: gl.constexpr,
    GROUP_SIZE_M: gl.constexpr,
    NUM_SMS: gl.constexpr,
    NUM_XCDS: gl.constexpr,
    BIAS: gl.constexpr,
    EVEN_K: gl.constexpr,
    world_size: gl.constexpr,
    COLLECT_TIMESTAMPS: gl.constexpr = False,
    mm_begin_timestamp_ptr: gl.tensor = None,
    mm_end_timestamp_ptr: gl.tensor = None,
):
    # Initialize device context from tensor
    ctx = IrisDeviceCtx.initialize(context_tensor)
    cur_rank = ctx.cur_rank
    
    pid = gl.program_id(0)

    if NUM_XCDS != 1:
        pid = (pid % NUM_XCDS) * (NUM_SMS // NUM_XCDS) + (pid // NUM_XCDS)
    num_pid_m = tl.cdiv(M, BLOCK_SIZE_M)
    num_pid_n = tl.cdiv(N, BLOCK_SIZE_N)
    total_tiles = num_pid_m * num_pid_n

    # Create layout for arange operations
    layout: gl.constexpr = gl.BlockedLayout([1], [64], [1], [0])

    # Assumptions for optimization
    tl.assume(stride_am > 0)
    tl.assume(stride_ak > 0)
    tl.assume(stride_bn > 0)
    tl.assume(stride_bk > 0)
    tl.assume(stride_cm > 0)
    tl.assume(stride_cn > 0)

    acc_dtype = gl.float32 if C.type.element_ty != gl.int8 else gl.int32

    for tile_id in range(pid, total_tiles, NUM_SMS):
        if COLLECT_TIMESTAMPS:
            timestamp = read_realtime()
            gl.atomic_min(mm_begin_timestamp_ptr + tile_id, timestamp)

        num_pid_in_group = GROUP_SIZE_M * num_pid_n
        group_id = tile_id // num_pid_in_group
        first_pid_m = group_id * GROUP_SIZE_M
        group_size_m = min(num_pid_m - first_pid_m, GROUP_SIZE_M)
        pid_m = first_pid_m + ((tile_id % num_pid_in_group) % group_size_m)
        pid_n = (tile_id % num_pid_in_group) // group_size_m

        rm = (pid_m * BLOCK_SIZE_M + gl.arange(0, BLOCK_SIZE_M, layout=layout)) % M
        rn = (pid_n * BLOCK_SIZE_N + gl.arange(0, BLOCK_SIZE_N, layout=layout)) % N

        rk = gl.arange(0, BLOCK_SIZE_K, layout=layout)
        rm = tl.max_contiguous(tl.multiple_of(rm, BLOCK_SIZE_M), BLOCK_SIZE_M)
        rn = tl.max_contiguous(tl.multiple_of(rn, BLOCK_SIZE_N), BLOCK_SIZE_N)
        A_BASE = A + rm[:, None] * stride_am + rk[None, :] * stride_ak
        B_BASE = B + rk[:, None] * stride_bk + rn[None, :] * stride_bn

        tl.assume(pid_m >= 0)
        tl.assume(pid_n >= 0)

        loop_k = tl.cdiv(K, BLOCK_SIZE_K)
        if not EVEN_K:
            loop_k -= 1

        acc = gl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=acc_dtype)
        for k in range(0, loop_k):
            a = gl.load(tl.multiple_of(A_BASE, (1, 16)))
            b = gl.load(tl.multiple_of(B_BASE, (16, 1)))
            acc += gl.dot(a, b)
            A_BASE += BLOCK_SIZE_K * stride_ak
            B_BASE += BLOCK_SIZE_K * stride_bk

        if not EVEN_K:
            k = loop_k
            rk = k * BLOCK_SIZE_K + gl.arange(0, BLOCK_SIZE_K, layout=layout)
            A_BASE = A + rm[:, None] * stride_am + rk[None, :] * stride_ak
            B_BASE = B + rk[:, None] * stride_bk + rn[None, :] * stride_bn
            A_BASE = tl.multiple_of(A_BASE, (1, 16))
            B_BASE = tl.multiple_of(B_BASE, (16, 1))
            a = gl.load(A_BASE, mask=rk[None, :] < K, other=0.0)
            b = gl.load(B_BASE, mask=rk[:, None] < K, other=0.0)
            acc += gl.dot(a, b)

        # Accumulator registers with C results
        c = tl.cast(acc, C.type.element_ty)

        rm = (pid_m * BLOCK_SIZE_M + gl.arange(0, BLOCK_SIZE_M, layout=layout)) % M
        rn = (pid_n * BLOCK_SIZE_N + gl.arange(0, BLOCK_SIZE_N, layout=layout)) % N

        # Add compiler hints
        rm = tl.max_contiguous(tl.multiple_of(rm, BLOCK_SIZE_M), BLOCK_SIZE_M)
        rn = tl.max_contiguous(tl.multiple_of(rn, BLOCK_SIZE_N), BLOCK_SIZE_N)

        # Define the C-mask (BLOCK_SIZE_M, 1) x (1, BLOCK_SIZE_N)
        sub_mask = (rm[:, None] < M) & (rn[None, :] < N)

        # Calculate the "global" offset of C based on the rank.
        # Note how the N-dimension is being multiplied by current rank.
        # This is because each rank is computing a portion of the N-dimension
        # locally and then scattering it to all other ranks to complete
        # the global N-dimension.
        global_offset = rm[:, None] * stride_cm_global + (rn[None, :] + cur_rank * N) * stride_cn_global

        # Timestamp for GEMM before store
        if COLLECT_TIMESTAMPS:
            timestamp = read_realtime()
            gl.atomic_max(mm_end_timestamp_ptr + tile_id, timestamp)

        # Store data to the global result using context methods
        for remote_rank in range(world_size):
            if remote_rank == cur_rank:
                # For the current rank, we can use store
                gl.store(c_global + global_offset, c, mask=sub_mask)
            else:
                ctx.store(
                    c_global + global_offset,
                    c,
                    remote_rank,
                    mask=sub_mask,
                )
