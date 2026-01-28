# SPDX-License-Identifier: MIT
# Copyright (c) 2025 Advanced Micro Devices, Inc. All rights reserved.

import triton
import triton.language as tl
from examples.common.utils import read_realtime


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

    # Precompute once
    num_pid_m = tl.cdiv(M, BLOCK_SIZE_M)
    num_pid_n = tl.cdiv(N, BLOCK_SIZE_N)
    total_tiles = num_pid_m * num_pid_n
    num_groups = tl.cdiv(total_tiles, world_size)

    next_rank = (cur_rank + 1) % world_size
    prev_rank = (cur_rank + world_size - 1) % world_size
    acc_dtype = tl.float32 if C.type.element_ty != tl.int8 else tl.int32

    # Persistent across *groups* now (not individual tiles):
    for g in range(pid, num_groups, COMM_SMS):
        group_base = g * world_size
        group_size = tl.minimum(world_size, total_tiles - group_base)  # tail-safe

        # ---- Reduce-Scatter over this group of up to 'group_size' tiles ----
        for s in range(0, group_size):
            # Tile index this rank handles at step s
            idx = group_base + ((cur_rank + group_size - s) % group_size)

            # Map linear tile idx -> (pid_m, pid_n) using existing swizzle
            num_pid_in_group = GROUP_SIZE_M * num_pid_n
            group_id = idx // num_pid_in_group
            first_pid_m = group_id * GROUP_SIZE_M
            group_size_m = tl.minimum(num_pid_m - first_pid_m, GROUP_SIZE_M)
            pid_m = first_pid_m + ((idx % num_pid_in_group) % group_size_m)
            pid_n = (idx % num_pid_in_group) // group_size_m

            # Offsets/masks for this tile
            rm = (pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)) % M
            rn = (pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)) % N
            rm = tl.max_contiguous(tl.multiple_of(rm, BLOCK_SIZE_M), BLOCK_SIZE_M)
            rn = tl.max_contiguous(tl.multiple_of(rn, BLOCK_SIZE_N), BLOCK_SIZE_N)
            sub_mask = (rm[:, None] < M) & (rn[None, :] < N)
            goff = rm[:, None] * stride_cm_global + rn[None, :] * stride_cn_global

            if s == 0:
                # First touch of this traveling tile on this rank:
                # wait for local GEMM, seed acc from local partial, and forward.
                while tl.atomic_cas(locks + idx, 0, 0, sem="acquire", scope="gpu") != 1:
                    pass
                acc = tl.load(local_C + goff, mask=sub_mask, other=0).to(acc_dtype)

                if group_size > 1:
                    # Wait for NEXT rank to be ready (its flag should be 0, meaning it finished previous step)
                    while (
                        iris.atomic_cas(flags + idx, 0, 0, cur_rank, next_rank, heap_bases, sem="acquire", scope="sys")
                        != 0
                    ):
                        pass
                    # Send to NEXT and signal that tile 'idx' is ready for neighbor
                    iris.store(ring_buffer + goff, acc, cur_rank, next_rank, heap_bases, mask=sub_mask)
                    tl.debug_barrier()  # Wait for all stores to complete before releasing the lock.
                    iris.atomic_xchg(flags + idx, 1, cur_rank, next_rank, heap_bases, sem="release", scope="sys")
            else:
                # Receive the traveling accumulator for this tile from PREV
                while tl.atomic_cas(flags + idx, 0, 0, sem="acquire", scope="sys") != 1:
                    pass
                recv = tl.load(ring_buffer + goff, mask=sub_mask, other=0).to(acc_dtype)

                # Wait for all to complete before releasing the lock.
                # This one can technically be moved lower (closer to recv = tl.load),
                # However, doing it much later allows for the two individual loads to issue and much-much
                # later reset the lock.
                tl.debug_barrier()
                tl.atomic_xchg(flags + idx, 0, sem="release", scope="sys")  # clear local flag

                # Fold in our local partial (wait if GEMM not done yet)
                while tl.atomic_cas(locks + idx, 0, 0, sem="acquire", scope="gpu") != 1:
                    pass

                part = tl.load(local_C + goff, mask=sub_mask, other=0).to(acc_dtype)
                acc = recv + part

                # Forward unless this is the last hop for this tile
                if s < group_size - 1:
                    # Wait for NEXT rank to be ready (its flag should be 0, meaning it finished previous step)
                    while (
                        iris.atomic_cas(flags + idx, 0, 0, cur_rank, next_rank, heap_bases, sem="acquire", scope="sys")
                        != 0
                    ):
                        pass
                    iris.store(ring_buffer + goff, acc, cur_rank, next_rank, heap_bases, mask=sub_mask)
                    tl.debug_barrier()
                    iris.atomic_xchg(flags + idx, 1, cur_rank, next_rank, heap_bases, sem="release", scope="sys")
                else:
                    # Last hop for tile idx on this rank: we own the fully reduced acc
                    c = acc.to(C.type.element_ty)

                    # All-scatter when the results are ready.
                    # TODO: Technically, we commonly use an all-gather operation at the end as a separate loop?
                    for rank in range(world_size):
                        if rank == cur_rank:
                            tl.store(C + goff, c, mask=sub_mask)
                        else:
                            iris.store(C + goff, c, cur_rank, rank, heap_bases, mask=sub_mask)
