# SPDX-License-Identifier: MIT
# Copyright (c) 2025 Advanced Micro Devices, Inc. All rights reserved.

import triton
import triton.language as tl
from examples.common.utils import read_realtime

import sys
import os

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

    max_m = M - 1
    max_n = N - 1

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
    sub_rm = tl.arange(0, SUB_BLOCK_SIZE_M) + start_row
    sub_rn = tl.arange(0, SUB_BLOCK_SIZE_N) + start_col

    sub_rm_2d = sub_rm[:, None]
    sub_rn_2d = sub_rn[None, :]

    sub_mask = (sub_rm_2d < BLOCK_SIZE_M) & (sub_rn_2d < BLOCK_SIZE_N)

    sub_offset = ((rm_start + sub_rm_2d) * stride_cm_local) + ((rn_start + sub_rn_2d) * stride_cn_local)

    return sub_mask, sub_offset


@triton.jit
def compute_output_partition(cur_rank, world_size, M, N, BLOCK_SIZE_M, BLOCK_SIZE_N):
    """
    计算当前rank负责的输出分区
    ReduceScatter: 每个rank只负责最终结果的一部分
    """
    # 按行分区（也可以按列或其他方式）
    rows_per_rank = tl.cdiv(M, world_size)
    start_row = cur_rank * rows_per_rank
    end_row = min((cur_rank + 1) * rows_per_rank, M)
    
    return start_row, end_row


@triton.jit
def persistent_gemm_reduce_scatter(
    A,
    B,
    C,
    c_local,  # 修改：本地输出缓冲区，不是全局的
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
    stride_cm_local,  # 修改：本地输出的stride
    stride_cn_local,
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

    # 计算当前rank负责的输出分区
    start_row, end_row = compute_output_partition(cur_rank, world_size, M, N, BLOCK_SIZE_M, BLOCK_SIZE_N)
    
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

        # 存储中间结果
        c = acc.to(C.type.element_ty)
        rm = (pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)) % M
        rn = (pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)) % N
        rm = tl.max_contiguous(tl.multiple_of(rm, BLOCK_SIZE_M), BLOCK_SIZE_M)
        rn = tl.max_contiguous(tl.multiple_of(rn, BLOCK_SIZE_N), BLOCK_SIZE_N)
        c_mask = (rm[:, None] < M) & (rn[None, :] < N)
        # C和C_都是full size buffer
        C_ = C + rm[:, None] * stride_cm + rn[None, :] * stride_cn
        tl.store(C_, c, c_mask)

        # 信号通知其他rank
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

        # 等待所有rank完成这个tile
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

        # ReduceScatter关键修改：只收集和存储属于本rank分区的数据
        rm, rn, mask, rm_start, rn_start = offset_for_tile(tile_id, BLOCK_SIZE_M, BLOCK_SIZE_N, GROUP_SIZE_M, M, N)

        num_sub_tiles_m = tl.cdiv(BLOCK_SIZE_M, BLOCK_SIZE_M)
        num_sub_tiles_n = tl.cdiv(BLOCK_SIZE_N, BLOCK_SIZE_N)
        total_sub_tiles = num_sub_tiles_m * num_sub_tiles_n

        for sub_tile_idx in range(0, total_sub_tiles):
            start_row_sub = (sub_tile_idx // num_sub_tiles_n) * BLOCK_SIZE_M
            start_col_sub = (sub_tile_idx % num_sub_tiles_n) * BLOCK_SIZE_N

            sub_mask, sub_offset = extract_submask_and_offset(
                rm,
                rn,
                mask,
                rm_start,
                rn_start,
                start_row_sub,
                start_col_sub,
                BLOCK_SIZE_M,
                BLOCK_SIZE_N,
                BLOCK_SIZE_M,
                BLOCK_SIZE_N,
                stride_cm,
                stride_cn,
            )

            # 关键修改：检查这个sub-tile是否属于当前rank的负责区域
            global_row_start = rm_start + start_row_sub

            # 在存储部分使用block指针
            if global_row_start < end_row and (global_row_start + BLOCK_SIZE_M) > start_row:
                tile_start_row = max(0, start_row - global_row_start)
                tile_end_row = min(BLOCK_SIZE_M, end_row - global_row_start)
                local_start_row = max(global_row_start, start_row) - start_row
                
                # 归约数据
                acc = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=acc_dtype)
                for remote_rank in range(world_size):
                    remote_data = iris.load(C + sub_offset, cur_rank, remote_rank, heap_bases, mask=sub_mask)
                    acc += remote_data
                
                # 创建block指针
                row_idx = tl.arange(0, BLOCK_SIZE_M)
                col_idx = tl.arange(0, BLOCK_SIZE_N)
                
                # 计算每个元素的本地偏移
                local_offsets = (local_start_row + row_idx[:, None]) * stride_cm_local + \
                            (rn_start + start_col_sub + col_idx[None, :]) * stride_cn_local
                
                # 创建block指针
                local_ptr_block = c_local + local_offsets
                
                # 创建写入掩码
                valid_mask = (row_idx[:, None] >= tile_start_row) & \
                            (row_idx[:, None] < tile_end_row) & \
                            (col_idx[None, :] < BLOCK_SIZE_N) & \
                            sub_mask
                
                # 存储block数据
                tl.store(local_ptr_block, acc, mask=valid_mask, cache_modifier=".wt")                # for remote_rank in range(world_size):
                #     # C为full size buffer
                #     remote_data = iris.load(C + sub_offset, cur_rank, remote_rank, heap_bases, mask=sub_mask)
                #     acc += remote_data
                
                # # 计算本地偏移
                # local_offset = (local_start_row * stride_cm_local + 
                #             (rn_start + start_col_sub) * stride_cn_local)
                
                # tl.store(c_local + local_offset, acc, mask=write_mask, cache_modifier=".wt")

        if COLLECT_TIMESTAMPS:
            timestamp = read_realtime()
            tl.atomic_max(mm_end_timestamp_ptr + tile_id, timestamp)
