#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
# Copyright (c) 2025 Advanced Micro Devices, Inc. All rights reserved.

import triton
import triton.language as tl
import iris


@triton.jit
def push_shards_kernel(
    A_local,
    A_inbox,
    signal_flags,
    M,
    K_local,
    stride_al_m,
    stride_al_k,
    stride_ai_rank,
    stride_ai_m,
    stride_ai_k,
    stride_sf_d,
    stride_sf_s,
    stride_sf_m,
    stride_sf_k,
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr,
    cur_rank: tl.constexpr,
    world_size: tl.constexpr,
    heap_bases: tl.tensor,
):
    pid_m = tl.program_id(0)
    pid_k = tl.program_id(1)
    tl.assume(stride_al_m > 0)
    tl.assume(stride_al_k > 0)
    tl.assume(stride_ai_rank > 0)
    tl.assume(stride_ai_m > 0)
    tl.assume(stride_ai_k > 0)
    tl.assume(stride_sf_d > 0)
    tl.assume(stride_sf_s > 0)
    tl.assume(stride_sf_m > 0)
    tl.assume(stride_sf_k > 0)

    offsets_m = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    offsets_k = pid_k * BLOCK_SIZE_K + tl.arange(0, BLOCK_SIZE_K)

    offsets_m = tl.max_contiguous(tl.multiple_of(offsets_m, BLOCK_SIZE_M), BLOCK_SIZE_M)
    offsets_k = tl.max_contiguous(tl.multiple_of(offsets_k, BLOCK_SIZE_K), BLOCK_SIZE_K)
    mask = (offsets_m[:, None] < M) & (offsets_k[None, :] < K_local)

    A_ptr = A_local + offsets_m[:, None] * stride_al_m + offsets_k[None, :] * stride_al_k
    a_tile = tl.load(tl.multiple_of(A_ptr, (1, 16)), mask=mask, other=0.0)

    for dest_rank_id in range(world_size):
        dest_ptr = (
            A_inbox + cur_rank * stride_ai_rank + offsets_m[:, None] * stride_ai_m + offsets_k[None, :] * stride_ai_k
        )
        iris.store(dest_ptr, a_tile, cur_rank, dest_rank_id, heap_bases, mask=mask)

        flag_ptr = (
            signal_flags
            + dest_rank_id * stride_sf_d
            + cur_rank * stride_sf_s
            + pid_m * stride_sf_m
            + pid_k * stride_sf_k
        )
        iris.atomic_add(flag_ptr, 1, cur_rank, dest_rank_id, heap_bases, sem="release", scope="sys")


@triton.jit
def gemm_push_kernel(
    A_inbox,
    B,
    C,
    M,
    N,
    K,
    signal_flags,
    stride_ai_rank,
    stride_ai_m,
    stride_ai_k,
    stride_b_k,
    stride_b_n,
    stride_c_m,
    stride_c_n,
    stride_sf_d,
    stride_sf_s,
    stride_sf_m,
    stride_sf_k,
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr,
    GROUP_SIZE_M: tl.constexpr,
    NUM_SMS: tl.constexpr,
    NUM_XCDS: tl.constexpr,
    EVEN_K: tl.constexpr,
    cur_rank: tl.constexpr,
    world_size: tl.constexpr,
):
    pid = tl.program_id(0)
    if NUM_XCDS != 1:
        pid = (pid % NUM_XCDS) * (NUM_SMS // NUM_XCDS) + (pid // NUM_XCDS)

    num_pid_m = tl.cdiv(M, BLOCK_SIZE_M)
    num_pid_n = tl.cdiv(N, BLOCK_SIZE_N)
    total_tiles = num_pid_m * num_pid_n

    tl.assume(stride_ai_rank > 0)
    tl.assume(stride_ai_m > 0)
    tl.assume(stride_ai_k > 0)
    tl.assume(stride_b_k > 0)
    tl.assume(stride_b_n > 0)
    tl.assume(stride_c_m > 0)
    tl.assume(stride_c_n > 0)
    tl.assume(stride_sf_d > 0)
    tl.assume(stride_sf_s > 0)
    tl.assume(stride_sf_m > 0)
    tl.assume(stride_sf_k > 0)

    acc_dtype = tl.float32 if C.type.element_ty != tl.int8 else tl.int32

    for tile_id in range(pid, total_tiles, NUM_SMS):
        num_pid_in_group = GROUP_SIZE_M * num_pid_n
        group_id = tile_id // num_pid_in_group
        first_pid_m = group_id * GROUP_SIZE_M
        group_size_m = min(num_pid_m - first_pid_m, GROUP_SIZE_M)
        pid_m = first_pid_m + ((tile_id % num_pid_in_group) % group_size_m)
        pid_n = (tile_id % num_pid_in_group) // group_size_m

        tl.assume(pid_m >= 0)
        tl.assume(pid_n >= 0)

        rm = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
        rn = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
        rm = tl.max_contiguous(tl.multiple_of(rm, BLOCK_SIZE_M), BLOCK_SIZE_M)
        rn = tl.max_contiguous(tl.multiple_of(rn, BLOCK_SIZE_N), BLOCK_SIZE_N)

        acc = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=acc_dtype)
        K_local = K // world_size

        for source_rank_id in range(world_size):
            num_k_tiles = tl.cdiv(K_local, BLOCK_SIZE_K)
            loop_k_tiles = num_k_tiles
            if not EVEN_K:
                loop_k_tiles -= 1

            for k_tile_idx in range(loop_k_tiles):
                flag_ptr = (
                    signal_flags
                    + cur_rank * stride_sf_d
                    + source_rank_id * stride_sf_s
                    + pid_m * stride_sf_m
                    + k_tile_idx * stride_sf_k
                )
                while tl.load(flag_ptr, cache_modifier=".ca") == 0:
                    pass

                k_offset = k_tile_idx * BLOCK_SIZE_K
                rk_local = k_offset + tl.arange(0, BLOCK_SIZE_K)
                A_ptr = (
                    A_inbox
                    + source_rank_id * stride_ai_rank
                    + rm[:, None] * stride_ai_m
                    + rk_local[None, :] * stride_ai_k
                )
                a = tl.load(tl.multiple_of(A_ptr, (1, 16)))
                rk_global = (source_rank_id * K_local) + rk_local
                B_ptr = B + rk_global[:, None] * stride_b_k + rn[None, :] * stride_b_n
                b = tl.load(tl.multiple_of(B_ptr, (16, 1)))
                acc += tl.dot(a, b)

            if not EVEN_K:
                k_tile_idx = loop_k_tiles
                flag_ptr = (
                    signal_flags
                    + cur_rank * stride_sf_d
                    + source_rank_id * stride_sf_s
                    + pid_m * stride_sf_m
                    + k_tile_idx * stride_sf_k
                )
                while tl.load(flag_ptr, cache_modifier=".ca") == 0:
                    pass

                k_offset = k_tile_idx * BLOCK_SIZE_K
                rk_local = k_offset + tl.arange(0, BLOCK_SIZE_K)
                A_ptr = (
                    A_inbox
                    + source_rank_id * stride_ai_rank
                    + rm[:, None] * stride_ai_m
                    + rk_local[None, :] * stride_ai_k
                )
                a = tl.load(tl.multiple_of(A_ptr, (1, 16)), mask=(rk_local[None, :] < K_local), other=0.0)
                rk_global = (source_rank_id * K_local) + rk_local
                B_ptr = B + rk_global[:, None] * stride_b_k + rn[None, :] * stride_b_n
                b = tl.load(tl.multiple_of(B_ptr, (16, 1)), mask=(rk_global[:, None] < K), other=0.0)
                acc += tl.dot(a, b)

        rm_store = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
        rn_store = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
        C_BASE = C + rm_store[:, None] * stride_c_m + rn_store[None, :] * stride_c_n
        c = acc.to(C.type.element_ty)
        mask = (rm_store[:, None] < M) & (rn_store[None, :] < N)
        tl.store(C_BASE, c, mask=mask)
