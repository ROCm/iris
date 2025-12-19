# SPDX-License-Identifier: MIT
# Copyright (c) 2025 Advanced Micro Devices, Inc. All rights reserved.

import triton
import triton.language as tl
from examples.common.utils import read_realtime


import iris


@triton.jit
def tile_id_to_index_range(
    tile_id,
    M,
    N,
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
    GROUP_SIZE_M: tl.constexpr,
):
    num_pid_m = tl.cdiv(M, BLOCK_SIZE_M)
    num_pid_n = tl.cdiv(N, BLOCK_SIZE_N)
    num_pid_in_group = GROUP_SIZE_M * num_pid_n

    group_id = tile_id // num_pid_in_group
    first_pid_m = group_id * GROUP_SIZE_M
    group_size_m = min(num_pid_m - first_pid_m, GROUP_SIZE_M)

    tile_in_group = tile_id % num_pid_in_group
    pid_m = first_pid_m + (tile_in_group % group_size_m)
    pid_n = tile_in_group // group_size_m

    rm_start = pid_m * BLOCK_SIZE_M
    rn_start = pid_n * BLOCK_SIZE_N

    # clamp to the maximum valid index (M-1, N-1)
    max_m = M - 1
    max_n = N - 1

    # generate indices
    rm = rm_start + tl.arange(0, BLOCK_SIZE_M)
    rn = rn_start + tl.arange(0, BLOCK_SIZE_N)

    rm = tl.minimum(rm, max_m)
    rn = tl.minimum(rn, max_n)

    return rm, rn, rm_start, rn_start


@triton.jit
def offset_for_tile(local_tile_id, BLOCK_SIZE_M, BLOCK_SIZE_N, GROUP_SIZE_M, M_local, N_local):
    rm, rn, rm_start, rn_start = tile_id_to_index_range(
        local_tile_id, M_local, N_local, BLOCK_SIZE_M, BLOCK_SIZE_N, GROUP_SIZE_M
    )
    c_mask = (rm[:, None] < M_local) & (rn[None, :] < N_local)
    return rm, rn, c_mask, rm_start, rn_start


@triton.jit
def extract_submask_and_offset(
    rm,
    rn,
    mask,
    rm_start,
    rn_start,
    start_row,
    start_col,
    SUB_BLOCK_SIZE_M: tl.constexpr,
    SUB_BLOCK_SIZE_N: tl.constexpr,
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
    stride_cm_local: tl.constexpr,
    stride_cn_local: tl.constexpr,
):
    # Create indices for the sub-block
    sub_rm = tl.arange(0, SUB_BLOCK_SIZE_M) + start_row
    sub_rn = tl.arange(0, SUB_BLOCK_SIZE_N) + start_col

    # Create a 2D grid of indices for the sub-block
    sub_rm_2d = sub_rm[:, None]  # Shape: (SUB_BLOCK_SIZE_M, 1)
    sub_rn_2d = sub_rn[None, :]  # Shape: (1, SUB_BLOCK_SIZE_N)

    # Compute the sub-mask
    sub_mask = (sub_rm_2d < BLOCK_SIZE_M) & (sub_rn_2d < BLOCK_SIZE_N)

    # Compute the sub-offset relative to the start of the tile
    sub_offset = ((rm_start + sub_rm_2d) * stride_cm_local) + ((rn_start + sub_rn_2d) * stride_cn_local)

    return sub_mask, sub_offset


@triton.jit
def persistent_gemm_all_reduce(
    A,
    B,
    C,
    c_global,
    bias_ptr,
    P,
    locks,
    tile_completed,
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
    STREAMK_TILES: tl.constexpr,
    NUM_XCDS: tl.constexpr,
    BIAS: tl.constexpr,
    EVEN_K: tl.constexpr,
    heap_bases: tl.tensor,
    cur_rank: tl.constexpr,
    world_size: tl.constexpr,
    NOTIFY_REMOTES: tl.constexpr = False,
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

        # Store results so other ranks can see them
        c = acc.to(C.type.element_ty)
        rm = (pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)) % M
        rn = (pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)) % N
        rm = tl.max_contiguous(tl.multiple_of(rm, BLOCK_SIZE_M), BLOCK_SIZE_M)
        rn = tl.max_contiguous(tl.multiple_of(rn, BLOCK_SIZE_N), BLOCK_SIZE_N)
        c_mask = (rm[:, None] < M) & (rn[None, :] < N)
        C_ = C + rm[:, None] * stride_cm + rn[None, :] * stride_cn
        tl.store(C_, c, c_mask)

        # Signal to other ranks that we have completed this tile
        for remote in range(world_size):
            if remote != cur_rank:
                iris.atomic_add(
                    tile_completed + tile_id,
                    1,
                    cur_rank,
                    remote,
                    heap_bases,
                    sem="release",
                    scope="sys",
                )

        # Wait for all ranks to complete this tile
        result = 0
        while result < (world_size - 1):
            compare = world_size - 1
            value = 0
            result = iris.atomic_cas(
                tile_completed + tile_id,
                compare,
                value,
                cur_rank,
                cur_rank,
                heap_bases,
                sem="acquire",
                scope="sys",
            )

        # All ranks have completed this tile, so we can reduce
        rm, rn, mask, rm_start, rn_start = offset_for_tile(tile_id, BLOCK_SIZE_M, BLOCK_SIZE_N, GROUP_SIZE_M, M, N)

        # Calculate the number of sub-tiles in each dimension
        num_sub_tiles_m = tl.cdiv(BLOCK_SIZE_M, BLOCK_SIZE_M)
        num_sub_tiles_n = tl.cdiv(BLOCK_SIZE_N, BLOCK_SIZE_N)
        total_sub_tiles = num_sub_tiles_m * num_sub_tiles_n

        for sub_tile_idx in range(0, total_sub_tiles):
            # Calculate start_row and start_col for the current sub-tile
            start_row = (sub_tile_idx // num_sub_tiles_n) * BLOCK_SIZE_M
            start_col = (sub_tile_idx % num_sub_tiles_n) * BLOCK_SIZE_N

            # Translate to global
            sub_mask, sub_offset = extract_submask_and_offset(
                rm,
                rn,
                mask,
                rm_start,
                rn_start,
                start_row,
                start_col,
                BLOCK_SIZE_M,
                BLOCK_SIZE_N,
                BLOCK_SIZE_M,
                BLOCK_SIZE_N,
                stride_cm,
                stride_cn,
            )

            # Store data to the global result using puts
            acc = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=acc_dtype)
            for remote_rank in range(world_size):
                acc += iris.load(C + sub_offset, cur_rank, remote_rank, heap_bases, mask=sub_mask)

            tl.store(c_global + sub_offset, acc, mask=sub_mask, cache_modifier=".wt")

        if COLLECT_TIMESTAMPS:
            timestamp = read_realtime()
            tl.atomic_max(mm_end_timestamp_ptr + tile_id, timestamp)
