# SPDX-License-Identifier: MIT
# Copyright (c) 2025 Advanced Micro Devices, Inc. All rights reserved.

import triton
import triton.language as tl
from examples.common.utils import read_realtime

import sys
import os

import iris


@triton.jit()
def persistent_gemm(
    A,
    B,
    local_C,
    bias_ptr,
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

    acc_dtype = tl.float32 if local_C.type.element_ty != tl.int8 else tl.int32

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
        mask = (rm[:, None] < M) & (rn[None, :] < N)

        # Calculate the "global" offset of C based on the rank.
        # Note how each GPU is producing the entire output but partial-K.
        offset = rm[:, None] * stride_cm + rn[None, :] * stride_cn

        # Timestamp for GEMM before store
        if COLLECT_TIMESTAMPS:
            timestamp = read_realtime()
            tl.atomic_max(mm_end_timestamp_ptr + tile_id, timestamp)

        # Write fully-reduced tile to local result buffer (no remote writes)
        tl.store(local_C + offset, acc, mask=mask, cache_modifier=".wt")
        tl.debug_barrier()
        tl.store(locks + tile_id, 1, cache_modifier=".wt")

        if COLLECT_TIMESTAMPS:
            timestamp = read_realtime()
            tl.atomic_max(mm_end_timestamp_ptr + tile_id, timestamp)


@triton.jit()
def persistent_all_reduce(
    C,
    local_C,
    ring_buffer,
    locks,
    flags,
    M,
    N,
    stride_cm_global,
    stride_cn_global,
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
    GROUP_SIZE_M: tl.constexpr,
    COMM_SMS: tl.constexpr,
    NUM_XCDS: tl.constexpr,
    heap_bases: tl.tensor,
    cur_rank: tl.constexpr,
    world_size: tl.constexpr,
):
    pid = tl.program_id(0)

    if NUM_XCDS != 1:
        pid = (pid % NUM_XCDS) * (COMM_SMS // NUM_XCDS) + (pid // NUM_XCDS)
    num_pid_m = tl.cdiv(M, BLOCK_SIZE_M)
    num_pid_n = tl.cdiv(N, BLOCK_SIZE_N)
    total_tiles = num_pid_m * num_pid_n

    # Ring topology
    next_rank = (cur_rank + 1) % world_size
    prev_rank = (cur_rank + world_size - 1) % world_size

    acc_dtype = tl.float32 if C.type.element_ty != tl.int8 else tl.int32

    for tile_id in range(pid, total_tiles, COMM_SMS):
        num_pid_in_group = GROUP_SIZE_M * num_pid_n
        group_id = tile_id // num_pid_in_group
        first_pid_m = group_id * GROUP_SIZE_M
        group_size_m = min(num_pid_m - first_pid_m, GROUP_SIZE_M)
        pid_m = first_pid_m + ((tile_id % num_pid_in_group) % group_size_m)
        pid_n = (tile_id % num_pid_in_group) // group_size_m

        tl.assume(pid_m >= 0)
        tl.assume(pid_n >= 0)

        # Begin: masks/offset calculations
        rm = (pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)) % M
        rn = (pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)) % N
        rm = tl.max_contiguous(tl.multiple_of(rm, BLOCK_SIZE_M), BLOCK_SIZE_M)
        rn = tl.max_contiguous(tl.multiple_of(rn, BLOCK_SIZE_N), BLOCK_SIZE_N)
        sub_mask = (rm[:, None] < M) & (rn[None, :] < N)
        global_offset = rm[:, None] * stride_cm_global + rn[None, :] * stride_cn_global
        # End: masks/offset calculations.

        while tl.load(locks + tile_id, cache_modifier=".cv", volatile=True) != 1:
            pass

        # ============================================================
        # NEW ALGORITHM: Ring-based Reduce-Scatter + All-Gather
        # ============================================================
        # Ring all-reduce optimized for parallel tile reduction.
        #
        # Phase 1: Reduce-Scatter (world_size steps)
        # - Each rank starts with a unique tile from its local partial result
        # - In each step, ranks send accumulated data to next neighbor and receive from previous
        # - Each rank loads the next tile and accumulates received data
        # - After world_size steps, each rank has fully reduced one tile
        #
        # Phase 2: All-Gather
        # - Each rank broadcasts its fully-reduced tile to all other ranks
        #
        # Example for 3 GPUs processing the same tile_id:
        # Step 0: GPU 0 loads from rank 0 (a0), GPU 1 from rank 1 (b1), GPU 2 from rank 2 (c2)
        # Step 1: GPU 0 sends a0→GPU1, recv c2←GPU2, loads rank 2 data, acc=(a2+c2)
        #         GPU 1 sends b1→GPU2, recv a0←GPU0, loads rank 0 data, acc=(b0+a0)
        #         GPU 2 sends c2→GPU0, recv b1←GPU1, loads rank 1 data, acc=(c1+b1)
        # Step 2: GPU 0 sends (a2+c2)→GPU1, recv (c1+b1)←GPU2, loads rank 1 data, acc=(a1+b1+c1)
        #         GPU 1 sends (b0+a0)→GPU2, recv (a2+c2)←GPU0, loads rank 2 data, acc=(b2+a2+c2)
        #         GPU 2 sends (c1+b1)→GPU0, recv (b0+a0)←GPU1, loads rank 0 data, acc=(c0+a0+b0)

        acc = None

        # Reduce-scatter phase: world_size steps
        for step in range(0, world_size):
            # Determine which rank's data to load in this step
            # Step 0: rank r loads from rank r (its own initial data)
            # Step 1: rank r loads from rank (r - 1 + world_size) % world_size
            # Step 2: rank r loads from rank (r - 2 + world_size) % world_size
            # Pattern: rank r at step s loads from rank (r - s + world_size) % world_size
            # This is equivalent to: (r + world_size - s) % world_size
            source_rank = (cur_rank + world_size - step) % world_size

            if step == 0:
                # Initial load: load tile from our own local_C
                acc = tl.load(local_C + global_offset, mask=sub_mask).to(acc_dtype)
            else:
                # Subsequent steps: send, receive, load, accumulate

                # 1) Wait for next rank to be ready (its flag should be 0)
                while (
                    iris.atomic_cas(flags + tile_id, 0, 0, cur_rank, next_rank, heap_bases, sem="acquire", scope="sys")
                    != 0
                ):
                    pass

                # 2) Send current accumulator to next rank's ring buffer
                iris.store(ring_buffer + global_offset, acc, cur_rank, next_rank, heap_bases, mask=sub_mask)

                tl.debug_barrier()

                # 3) Signal next rank that data is ready
                iris.atomic_xchg(flags + tile_id, 1, cur_rank, next_rank, heap_bases, sem="release", scope="sys")

                # 4) Wait for prev rank to send us data (our flag should become 1)
                while tl.atomic_cas(flags + tile_id, 0, 0, sem="acquire", scope="sys") != 1:
                    pass

                # 5) Load tile from source_rank's local_C (cross-rank read)
                next_tile = iris.load(
                    local_C + global_offset, cur_rank, source_rank, heap_bases, mask=sub_mask, other=0.0
                )

                # 6) Load received data from our local ring_buffer (sent by prev rank)
                recv_tile = tl.load(ring_buffer + global_offset, mask=sub_mask, other=0.0)

                # 7) Accumulate: new_acc = next_tile + recv_tile
                acc = next_tile.to(acc_dtype) + recv_tile.to(acc_dtype)

                # 8) Reset our local flag to 0 (ready for next iteration)
                tl.atomic_xchg(flags + tile_id, 0, sem="release", scope="sys")

        # After reduce-scatter phase, all ranks have computed the fully reduced tile
        # In a traditional reduce-scatter, each rank would have a DIFFERENT tile,
        # but in this implementation, all ranks process the same tiles and end up
        # with the same results. The ring pattern with cross-rank loads ensures
        # efficient pipelining of the reduction across ranks.
        #
        # Write result to all ranks' C buffers to ensure visibility (all-gather phase)
        c = acc.to(C.type.element_ty)
        for remote_rank in range(world_size):
            if remote_rank != cur_rank:
                # Store our fully reduced tile to the remote rank's C buffer
                iris.store(C + global_offset, c, cur_rank, remote_rank, heap_bases, mask=sub_mask)

        # Store to our local C buffer
        tl.store(C + global_offset, c, mask=sub_mask)
