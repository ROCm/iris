# SPDX-License-Identifier: MIT
# Copyright (c) 2025 Advanced Micro Devices, Inc. All rights reserved.

import triton
import triton.language as tl
from iris.device_utils import read_realtime

import iris


@triton.jit()
def persistent_gemm_reduce_scatter_wg_specialized(
    A,
    B,
    C,  # local buffer [M, N]
    C_global,  # global output buffer [M, N] on each rank
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
    stride_cg_m,
    stride_cg_n,
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr,
    GROUP_SIZE_M: tl.constexpr,
    GEMM_SMS: tl.constexpr,
    NUM_SMS: tl.constexpr,
    NUM_XCDS: tl.constexpr,
    EVEN_K: tl.constexpr,
    heap_bases: tl.tensor,
    cur_rank: tl.constexpr,
    world_size: tl.constexpr,
    COLLECT_TIMESTAMPS: tl.constexpr = False,
    mm_begin_timestamp_ptr: tl.tensor = None,
    mm_end_timestamp_ptr: tl.tensor = None,
):
    """
    GEMM + ReduceScatter with Workgroup Specialization

    Split SMs into two groups:
    - GEMM SMs: Perform matrix multiplication computation
    - Communication SMs: Handle data communication (scatter to target ranks)

    This approach enables overlapping computation and communication.

    Data partitioning (ReduceScatter):
    - A: [M, local_K] - Each rank has a portion of K dimension
    - B: [local_K, N] - Each rank has a portion of K dimension
    - Each rank computes partial C = A @ B of shape [M, N]
    - ReduceScatter: Split C along M dimension into world_size chunks,
      send chunk i to rank i, accumulate with atomic_add
    - Output: Each rank ends up with [M/world_size, N]
    """
    pid = tl.program_id(0)

    if NUM_XCDS != 1:
        pid = (pid % NUM_XCDS) * (NUM_SMS // NUM_XCDS) + (pid // NUM_XCDS)

    num_pid_m = tl.cdiv(M, BLOCK_SIZE_M)
    num_pid_n = tl.cdiv(N, BLOCK_SIZE_N)
    total_tiles = num_pid_m * num_pid_n
    M_per_rank = M // world_size

    tl.assume(stride_am > 0)
    tl.assume(stride_ak > 0)
    tl.assume(stride_bn > 0)
    tl.assume(stride_bk > 0)
    tl.assume(stride_cm > 0)
    tl.assume(stride_cn > 0)

    acc_dtype = tl.float32 if C.type.element_ty != tl.int8 else tl.int32

    if pid < GEMM_SMS:
        for tile_id in range(pid, total_tiles, GEMM_SMS):
            if COLLECT_TIMESTAMPS:
                timestamp = read_realtime()
                tl.atomic_min(mm_begin_timestamp_ptr + tile_id, timestamp)

            num_pid_in_group = GROUP_SIZE_M * num_pid_n
            group_id = tile_id // num_pid_in_group
            first_pid_m = group_id * GROUP_SIZE_M
            group_size_m = min(num_pid_m - first_pid_m, GROUP_SIZE_M)
            pid_m = first_pid_m + ((tile_id % num_pid_in_group) % group_size_m)
            pid_n = (tile_id % num_pid_in_group) // group_size_m

            tl.assume(pid_m >= 0)
            tl.assume(pid_n >= 0)

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

            c = acc.to(C.type.element_ty)

            sub_mask = (rm[:, None] < M) & (rn[None, :] < N)

            # Store to local buffer
            local_offset = rm[:, None] * stride_cm + rn[None, :] * stride_cn

            if COLLECT_TIMESTAMPS:
                timestamp = read_realtime()
                tl.atomic_max(mm_end_timestamp_ptr + tile_id, timestamp)

            tl.store(C + local_offset, c, mask=sub_mask, cache_modifier=".wt")
            iris.atomic_cas(locks + tile_id, 0, 1, cur_rank, cur_rank, heap_bases, sem="release", scope="sys")

    else:
        COMM_SMS = NUM_SMS - GEMM_SMS
        comm_pid = pid - GEMM_SMS

        for tile_id in range(comm_pid, total_tiles, COMM_SMS):
            num_pid_in_group = GROUP_SIZE_M * num_pid_n
            group_id = tile_id // num_pid_in_group
            first_pid_m = group_id * GROUP_SIZE_M
            group_size_m = min(num_pid_m - first_pid_m, GROUP_SIZE_M)
            pid_m = first_pid_m + ((tile_id % num_pid_in_group) % group_size_m)
            pid_n = (tile_id % num_pid_in_group) // group_size_m

            tl.assume(pid_m >= 0)
            tl.assume(pid_n >= 0)

            rm = (pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)) % M
            rn = (pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)) % N
            rm = tl.max_contiguous(tl.multiple_of(rm, BLOCK_SIZE_M), BLOCK_SIZE_M)
            rn = tl.max_contiguous(tl.multiple_of(rn, BLOCK_SIZE_N), BLOCK_SIZE_N)
            sub_mask = (rm[:, None] < M) & (rn[None, :] < N)

            local_offset = rm[:, None] * stride_cm + rn[None, :] * stride_cn

            done = 0
            while done == 0:
                done = iris.atomic_cas(
                    locks + tile_id, 1, 0, cur_rank, cur_rank, heap_bases, sem="acquire", scope="sys"
                )

            c = tl.load(C + local_offset, mask=sub_mask)

            # chunk i of M dimension goes to rank i
            tile_m_start = pid_m * BLOCK_SIZE_M
            target_rank = tile_m_start // M_per_rank

            # offset within target rank's output
            target_m = tile_m_start % M_per_rank
            offs_cm = target_m + tl.arange(0, BLOCK_SIZE_M)
            offs_cn = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
            global_offset = offs_cm[:, None] * stride_cg_m + offs_cn[None, :] * stride_cg_n

            global_mask = (offs_cm[:, None] < M_per_rank) & (offs_cn[None, :] < N)

            if target_rank == cur_rank:
                tl.atomic_add(C_global + global_offset, c, mask=global_mask)
            else:
                iris.atomic_add(
                    C_global + global_offset,
                    c,
                    cur_rank,
                    target_rank,
                    heap_bases,
                    mask=global_mask,
                    sem="relaxed",
                    scope="sys",
                )
