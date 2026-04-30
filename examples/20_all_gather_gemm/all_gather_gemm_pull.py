#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
# Copyright (c) 2025 Advanced Micro Devices, Inc. All rights reserved.

import triton
import triton.language as tl
import iris


@triton.jit
def persistent_ag_gemm(
    A,
    B,
    C,
    M,
    N,
    K,
    stride_am,
    stride_ak,
    stride_bk,
    stride_bn,
    stride_cm,
    stride_cn,
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr,
    GROUP_SIZE_M: tl.constexpr,
    NUM_SMS: tl.constexpr,
    NUM_XCDS: tl.constexpr,
    EVEN_K: tl.constexpr,
    heap_bases: tl.tensor,
    cur_rank: tl.constexpr,
    world_size: tl.constexpr,
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
        num_pid_in_group = GROUP_SIZE_M * num_pid_n
        group_id = tile_id // num_pid_in_group
        first_pid_m = group_id * GROUP_SIZE_M
        group_size_m = min(num_pid_m - first_pid_m, GROUP_SIZE_M)
        pid_m = first_pid_m + ((tile_id % num_pid_in_group) % group_size_m)
        pid_n = (tile_id % num_pid_in_group) // group_size_m

        rm = (pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)) % M
        rn = (pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)) % N
        rm = tl.max_contiguous(tl.multiple_of(rm, BLOCK_SIZE_M), BLOCK_SIZE_M)
        rn = tl.max_contiguous(tl.multiple_of(rn, BLOCK_SIZE_N), BLOCK_SIZE_N)

        acc = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=acc_dtype)
        K_local = K // world_size

        for source_rank_id in range(world_size):
            loop_k_local = tl.cdiv(K_local, BLOCK_SIZE_K)
            if not EVEN_K:
                loop_k_local -= 1

            for k_block_idx in range(0, loop_k_local):
                k_offset = k_block_idx * BLOCK_SIZE_K
                rk_local = k_offset + tl.arange(0, BLOCK_SIZE_K)
                A_ptr = A + rm[:, None] * stride_am + rk_local[None, :] * stride_ak
                a = iris.load(tl.multiple_of(A_ptr, (1, 16)), cur_rank, source_rank_id, heap_bases)

                rk_global = (source_rank_id * K_local) + rk_local
                B_ptr = B + rk_global[:, None] * stride_bk + rn[None, :] * stride_bn
                b = tl.load(tl.multiple_of(B_ptr, (16, 1)))

                acc += tl.dot(a, b)

            if not EVEN_K:
                k_offset = loop_k_local * BLOCK_SIZE_K
                rk_local = k_offset + tl.arange(0, BLOCK_SIZE_K)
                rk_local_mask = rk_local < K_local
                A_ptr = A + rm[:, None] * stride_am + rk_local[None, :] * stride_ak
                a = iris.load(
                    tl.multiple_of(A_ptr, (1, 16)),
                    cur_rank,
                    source_rank_id,
                    heap_bases,
                    mask=rk_local_mask[None, :],
                    other=0.0,
                )

                rk_global = (source_rank_id * K_local) + rk_local
                rk_global_mask = rk_global < K
                B_ptr = B + rk_global[:, None] * stride_bk + rn[None, :] * stride_bn
                b = tl.load(tl.multiple_of(B_ptr, (16, 1)), mask=rk_global_mask[:, None], other=0.0)

                acc += tl.dot(a, b)

        c = acc.to(C.type.element_ty)
        C_BASE = (
            C
            + (pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M))[:, None] * stride_cm
            + (pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N))[None, :] * stride_cn
        )
        mask = ((pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M))[:, None] < M) & (
            (pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N))[None, :] < N
        )
        tl.store(C_BASE, c, mask=mask)
