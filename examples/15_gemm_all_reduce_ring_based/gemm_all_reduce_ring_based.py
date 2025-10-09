# SPDX-License-Identifier: MIT
# Copyright (c) 2025 Advanced Micro Devices, Inc. All rights reserved.

import triton
import triton.language as tl
from examples.common.utils import read_realtime

import sys
import os

import iris


@triton.jit()
def persistent_gemm_all_reduce_ring_based(
    A,
    B,
    C,
    c_global,
    bias_ptr,
    ring_buffer,
    locks,
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
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr,
    GROUP_SIZE_M: tl.constexpr,
    NUM_SMS: tl.constexpr,
    NUM_XCDS: tl.constexpr,
    BIAS: tl.constexpr,
    EVEN_K: tl.constexpr,
    heap_bases: tl.tensor,
    cur_rank: tl.constexpr,
    world_size: tl.constexpr,
    COLLECT_TIMESTAMPS: tl.constexpr = False,
    mm_begin_timestamp_ptr: tl.tensor = None,
    mm_end_timestamp_ptr: tl.tensor = None,
):
    pid = tl.program_id(0)

    if NUM_XCDS != 1:
        pid = (pid % NUM_XCDS) * (NUM_SMS // NUM_XCDS) + (pid // NUM_XCDS)
    num_pid_m = tl.cdiv(M, BLOCK_SIZE_M)
    num_pid_n = tl.cdiv(N, BLOCK_SIZE_N)
    total_tiles = num_pid_m * num_pid_n

    tl.assume(stride_am > 0)
    tl.assume(stride_ak > 0)
    tl.assume(stride_bn > 0)
    tl.assume(stride_bk > 0)
    tl.assume(stride_cm > 0)
    tl.assume(stride_cn > 0)

    acc_dtype = tl.float32 if C.type.element_ty != tl.int8 else tl.int32

    # Ring topology
    next_rank = (cur_rank + 1) % world_size
    prev_rank = (cur_rank + world_size - 1) % world_size

    for tile_id in range(pid, total_tiles, NUM_SMS):
        if COLLECT_TIMESTAMPS:
            timestamp = read_realtime()
            tl.atomic_min(mm_begin_timestamp_ptr + tile_id, timestamp)

        num_pid_in_group = GROUP_SIZE_M * num_pid_n
        group_id = tile_id // num_pid_in_group
        first_pid_m = group_id * GROUP_SIZE_M
        group_size_m = min(num_pid_m - first_pid_m, GROUP_SIZE_M)
        pid_m = first_pid_m + ((tile_id % num_pid_in_group) % group_size_m)
        pid_n = (tile_id % num_pid_in_group) // group_size_m

        rm = (pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)) % M
        rn = (pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)) % N

        rk = tl.arange(0, BLOCK_SIZE_K)
        rm = tl.max_contiguous(tl.multiple_of(rm, BLOCK_SIZE_M), BLOCK_SIZE_M)
        rn = tl.max_contiguous(tl.multiple_of(rn, BLOCK_SIZE_N), BLOCK_SIZE_N)
        A_BASE = A + rm[:, None] * stride_am + rk[None, :] * stride_ak
        B_BASE = B + rk[:, None] * stride_bk + rn[None, :] * stride_bn

        tl.assume(pid_m > 0)
        tl.assume(pid_n > 0)

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
            k = loop_k
            rk = k * BLOCK_SIZE_K + tl.arange(0, BLOCK_SIZE_K)
            A_BASE = A + rm[:, None] * stride_am + rk[None, :] * stride_ak
            B_BASE = B + rk[:, None] * stride_bk + rn[None, :] * stride_bn
            A_BASE = tl.multiple_of(A_BASE, (1, 16))
            B_BASE = tl.multiple_of(B_BASE, (16, 1))
            a = tl.load(A_BASE, mask=rk[None, :] < K, other=0.0)
            b = tl.load(B_BASE, mask=rk[:, None] < K, other=0.0)
            acc += tl.dot(a, b)

        rm = (pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)) % M
        rn = (pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)) % N

        # Add compiler hints
        rm = tl.max_contiguous(tl.multiple_of(rm, BLOCK_SIZE_M), BLOCK_SIZE_M)
        rn = tl.max_contiguous(tl.multiple_of(rn, BLOCK_SIZE_N), BLOCK_SIZE_N)

        # Define the C-mask (BLOCK_SIZE_M, 1) x (1, BLOCK_SIZE_N)
        sub_mask = (rm[:, None] < M) & (rn[None, :] < N)

        # Calculate the "global" offset of C based on the rank.
        # Note how each GPU is producing the entire output but partial-K.
        global_offset = rm[:, None] * stride_cm_global + rn[None, :] * stride_cn_global

        # Timestamp for GEMM before store
        if COLLECT_TIMESTAMPS:
            timestamp = read_realtime()
            tl.atomic_max(mm_end_timestamp_ptr + tile_id, timestamp)

        # Ring All-Reduce (p-1 steps), lock synchronized
        # Time →
        # Rank 0:  ──send→────────wait(←flag)────────add───────
        # Rank 1:       ──send→────────wait──────────add───────
        # Rank 2:            ──send→────────wait────────add────
        # Rank 3:                 ──send→────────wait────────add

        # Step loop: send to next, wait/recv from prev, add.
        for _step in range(0, world_size - 1):
            # 1) Send our current accumulator tile to NEXT rank's ring buffer
            iris.store(
                ring_buffer + global_offset,
                acc,
                cur_rank,
                next_rank,
                heap_bases,
                mask=sub_mask,
            )
            tl.debug_barrier()

            # Signal "ready" by setting NEXT rank's flag for this tile to 1
            iris.atomic_xchg(locks + tile_id, 1, cur_rank, next_rank, heap_bases, sem="release", scope="sys")

            tl.debug_barrier()

            # 2) Wait for PREV rank to signal our local flag for this tile
            #    Spin; single-lane uniform load is fine here.
            while tl.atomic_cas(locks + tile_id, 0, 0, sem="acquire", scope="sys") != 1:
                pass

            # 3) Consume the received tile from our LOCAL ring buffer (prev wrote here)
            recv_tile = tl.load(ring_buffer + global_offset, mask=sub_mask, other=tl.zeros_like(acc))
            acc += recv_tile

            # 4) Reset our local flag to 0 (done consuming this step)
            tl.atomic_xchg(locks + tile_id, 0, sem="release", scope="sys")

        # Write fully-reduced tile to local result buffer (no remote writes)
        c = acc.to(C.type.element_ty)

        tl.store(c_global + global_offset, c, mask=sub_mask)

        if COLLECT_TIMESTAMPS:
            timestamp = read_realtime()
            tl.atomic_max(mm_end_timestamp_ptr + tile_id, timestamp)
