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
    tile_ready,
    M,
    N,
    K,
    stride_am,
    stride_ak,
    stride_bk,
    stride_bn,
    stride_cm_local,
    stride_cn_local,
    stride_bias,
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr,
    GROUP_SIZE_M: tl.constexpr,
    GEMM_SMS: tl.constexpr,
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
    """
    Producer kernel: Computes all tiles (each rank produces partial results).
    All ranks process all tiles and produce partials because K is split across ranks.
    """
    pid = tl.program_id(0)

    if NUM_XCDS != 1:
        pid = (pid % NUM_XCDS) * (GEMM_SMS // NUM_XCDS) + (pid // NUM_XCDS)

    num_pid_m = tl.cdiv(M, BLOCK_SIZE_M)
    num_pid_n = tl.cdiv(N, BLOCK_SIZE_N)
    total_tiles = num_pid_m * num_pid_n

    tl.assume(stride_am > 0)
    tl.assume(stride_ak > 0)
    tl.assume(stride_bn > 0)
    tl.assume(stride_bk > 0)
    tl.assume(stride_cm_local > 0)
    tl.assume(stride_cn_local > 0)

    acc_dtype = tl.float32 if local_C.type.element_ty != tl.int8 else tl.int32

    # All ranks process all tiles
    for tile_id in range(pid, total_tiles, GEMM_SMS):
        if COLLECT_TIMESTAMPS:
            timestamp = read_realtime()
            tl.atomic_min(mm_begin_timestamp_ptr + tile_id, timestamp)

        # Map tile_id to (pid_m, pid_n)
        num_pid_in_group = GROUP_SIZE_M * num_pid_n
        group_id = tile_id // num_pid_in_group
        first_pid_m = group_id * GROUP_SIZE_M
        group_size_m = min(num_pid_m - first_pid_m, GROUP_SIZE_M)
        pid_m = first_pid_m + ((tile_id % num_pid_in_group) % group_size_m)
        pid_n = (tile_id % num_pid_in_group) // group_size_m

        # Compute GEMM for this tile
        rm = (pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)) % M
        rn = (pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)) % N

        rk = tl.arange(0, BLOCK_SIZE_K)
        rm = tl.max_contiguous(tl.multiple_of(rm, BLOCK_SIZE_M), BLOCK_SIZE_M)
        rn = tl.max_contiguous(tl.multiple_of(rn, BLOCK_SIZE_N), BLOCK_SIZE_N)
        A_BASE = A + rm[:, None] * stride_am + rk[None, :] * stride_ak
        B_BASE = B + rk[:, None] * stride_bk + rn[None, :] * stride_bn

        tl.assume(pid_m >= 0)
        tl.assume(pid_n >= 0)

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

        # Store result locally
        rm = (pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)) % M
        rn = (pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)) % N
        rm = tl.max_contiguous(tl.multiple_of(rm, BLOCK_SIZE_M), BLOCK_SIZE_M)
        rn = tl.max_contiguous(tl.multiple_of(rn, BLOCK_SIZE_N), BLOCK_SIZE_N)
        mask = (rm[:, None] < M) & (rn[None, :] < N)
        local_offset = rm[:, None] * stride_cm_local + rn[None, :] * stride_cn_local

        # Write to local buffer
        tl.store(local_C + local_offset, acc, mask=mask, cache_modifier=".wt")

        # Signal that this tile is ready
        tl.debug_barrier()
        tl.store(locks + tile_id, 1, cache_modifier=".wt")

        # Signal to all remote ranks that this tile is ready
        for remote_rank in range(world_size):
            if remote_rank != cur_rank:
                iris.atomic_xchg(tile_ready + tile_id, 1, cur_rank, remote_rank, heap_bases, sem="release", scope="sys")

        if COLLECT_TIMESTAMPS:
            timestamp = read_realtime()
            tl.atomic_max(mm_end_timestamp_ptr + tile_id, timestamp)


@triton.jit()
def persistent_all_reduce(
    local_C,
    C_global,
    locks,
    tile_ready,
    M,
    N,
    stride_cm_local,
    stride_cn_local,
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
    DISTRIBUTION: tl.constexpr,  # 0 for striding, 1 for block
    COLLECT_TIMESTAMPS: tl.constexpr = False,
    mm_begin_timestamp_ptr: tl.tensor = None,
    mm_end_timestamp_ptr: tl.tensor = None,
):
    """
    Consumer kernel: Waits for tiles from all ranks, accumulates, and scatters results.
    Each rank only processes a subset of tiles for reduction based on DISTRIBUTION.
    """
    pid = tl.program_id(0)

    if NUM_XCDS != 1:
        pid = (pid % NUM_XCDS) * (COMM_SMS // NUM_XCDS) + (pid // NUM_XCDS)

    num_pid_m = tl.cdiv(M, BLOCK_SIZE_M)
    num_pid_n = tl.cdiv(N, BLOCK_SIZE_N)
    total_tiles = num_pid_m * num_pid_n

    acc_dtype = tl.float32 if C_global.type.element_ty != tl.int8 else tl.int32

    # Determine which tiles this rank is responsible for reducing
    if DISTRIBUTION == 0:
        # Striding: rank reduces tiles cur_rank, cur_rank + world_size, ...
        tiles_per_rank = tl.cdiv(total_tiles, world_size)
        start_tile = cur_rank
        stride = world_size
    else:
        # Block: rank reduces continuous block of tiles
        tiles_per_rank = tl.cdiv(total_tiles, world_size)
        start_tile = cur_rank * tiles_per_rank
        stride = 1

    # Each SM processes tiles assigned to this rank for reduction
    # Calculate max tile_offset to avoid boundary issues (no break allowed in Triton)
    max_tile_offset = tiles_per_rank
    if DISTRIBUTION == 0:  # striding
        # tile_id = start_tile + tile_offset * stride = cur_rank + tile_offset * world_size
        # tile_id < total_tiles  =>  tile_offset < (total_tiles - cur_rank) / world_size
        max_tile_offset = tl.minimum(tiles_per_rank, tl.cdiv(total_tiles - cur_rank, world_size))
    else:  # block
        # tile_id = start_tile + tile_offset = cur_rank * tiles_per_rank + tile_offset
        # tile_id < total_tiles  =>  tile_offset < total_tiles - cur_rank * tiles_per_rank
        max_tile_offset = tl.minimum(tiles_per_rank, total_tiles - cur_rank * tiles_per_rank)

    for tile_offset in range(pid, max_tile_offset, COMM_SMS):
        tile_id = start_tile + tile_offset * stride

        # Wait for all ranks to produce this tile (all ranks have partials)
        # Local tile
        while tl.load(locks + tile_id, cache_modifier=".cv", volatile=True) != 1:
            pass

        # Wait for remote ranks
        for remote_rank in range(world_size):
            if remote_rank != cur_rank:
                while (
                    iris.atomic_cas(
                        tile_ready + tile_id, 0, 0, cur_rank, remote_rank, heap_bases, sem="acquire", scope="sys"
                    )
                    != 1
                ):
                    pass

        # Map tile_id to (pid_m, pid_n)
        num_pid_in_group = GROUP_SIZE_M * num_pid_n
        group_id = tile_id // num_pid_in_group
        first_pid_m = group_id * GROUP_SIZE_M
        group_size_m = min(num_pid_m - first_pid_m, GROUP_SIZE_M)
        pid_m = first_pid_m + ((tile_id % num_pid_in_group) % group_size_m)
        pid_n = (tile_id % num_pid_in_group) // group_size_m

        # Compute offsets
        rm = (pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)) % M
        rn = (pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)) % N
        rm = tl.max_contiguous(tl.multiple_of(rm, BLOCK_SIZE_M), BLOCK_SIZE_M)
        rn = tl.max_contiguous(tl.multiple_of(rn, BLOCK_SIZE_N), BLOCK_SIZE_N)
        mask = (rm[:, None] < M) & (rn[None, :] < N)
        local_offset = rm[:, None] * stride_cm_local + rn[None, :] * stride_cn_local

        # Accumulate from all ranks
        acc = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=acc_dtype)
        for remote_rank in range(world_size):
            partial = iris.load(local_C + local_offset, cur_rank, remote_rank, heap_bases, mask=mask)
            acc += partial.to(acc_dtype)

        # Convert to output type
        c_out = acc.to(C_global.type.element_ty)

        # Scatter to all ranks
        global_offset = rm[:, None] * stride_cm_global + rn[None, :] * stride_cn_global
        for remote_rank in range(world_size):
            if remote_rank == cur_rank:
                tl.store(C_global + global_offset, c_out, mask=mask)
            else:
                iris.store(C_global + global_offset, c_out, cur_rank, remote_rank, heap_bases, mask=mask)
