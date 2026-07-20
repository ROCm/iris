# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.

"""
Device kernels for :mod:`iris.concurrent`.

Two overlap models for running an *independent* GEMM concurrently with a
collective on the same device:

* **Fused** (:func:`fused_ws_gemm_all_gather`) -- one persistent kernel with two
  work-stealing queues. Every workgroup has a *home* queue (GEMM or comm) chosen
  by ``GEMM_WGS``; once the home queue drains, the workgroup steals from the
  other queue. Dynamic rebalancing across the compute/comm boundary in a single
  launch.

* **Concurrent / two-kernel** (:func:`ws_gemm` + :func:`ws_all_gather`) -- two
  independent persistent work-stealing kernels launched on separate streams. Each
  owns one device-wide atomic counter and dynamically grabs its own tiles. CU
  occupancy is shared by the hardware scheduler; the WG grids set the split.

Both share the same per-tile primitives (:func:`_gemm_tile`, :func:`_all_gather_tile`)
so the two models are numerically identical tile-for-tile.
"""

import triton
import triton.language as tl

import iris


# ---------------------------------------------------------------------------
# Per-tile primitives
# ---------------------------------------------------------------------------
@triton.jit()
def _gemm_tile(
    tile_id,
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
    num_pid_n,
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr,
    GROUP_SIZE_M: tl.constexpr,
    EVEN_K: tl.constexpr,
):
    """Compute one output tile of ``C = A @ B`` and store it locally."""
    num_pid_m = tl.cdiv(M, BLOCK_SIZE_M)
    num_pid_in_group = GROUP_SIZE_M * num_pid_n
    group_id = tile_id // num_pid_in_group
    first_pid_m = group_id * GROUP_SIZE_M
    group_size_m = min(num_pid_m - first_pid_m, GROUP_SIZE_M)
    pid_m = first_pid_m + ((tile_id % num_pid_in_group) % group_size_m)
    pid_n = (tile_id % num_pid_in_group) // group_size_m
    tl.assume(pid_m >= 0)
    tl.assume(pid_n >= 0)

    acc_dtype = tl.float32 if C.type.element_ty != tl.int8 else tl.int32

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
    c_mask = (rm[:, None] < M) & (rn[None, :] < N)
    tl.store(C + rm[:, None] * stride_cm + rn[None, :] * stride_cn, c, mask=c_mask, cache_modifier=".wt")


@triton.jit()
def _all_gather_tile(
    tile_id,
    comm_src,
    comm_dst,
    Mc,
    Nc,
    stride_sm,
    stride_sn,
    stride_dm,
    stride_dn,
    num_pid_n,
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
    GROUP_SIZE_M: tl.constexpr,
    heap_bases: tl.tensor,
    cur_rank: tl.constexpr,
    world_size: tl.constexpr,
):
    """Push one local block into rows ``[cur_rank*Mc : ...]`` of every peer's
    symmetric ``comm_dst`` (all-gather along dim-0)."""
    num_pid_m = tl.cdiv(Mc, BLOCK_SIZE_M)
    num_pid_in_group = GROUP_SIZE_M * num_pid_n
    group_id = tile_id // num_pid_in_group
    first_pid_m = group_id * GROUP_SIZE_M
    group_size_m = min(num_pid_m - first_pid_m, GROUP_SIZE_M)
    pid_m = first_pid_m + ((tile_id % num_pid_in_group) % group_size_m)
    pid_n = (tile_id % num_pid_in_group) // group_size_m
    tl.assume(pid_m >= 0)
    tl.assume(pid_n >= 0)

    rm = (pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)) % Mc
    rn = (pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)) % Nc
    rm = tl.max_contiguous(tl.multiple_of(rm, BLOCK_SIZE_M), BLOCK_SIZE_M)
    rn = tl.max_contiguous(tl.multiple_of(rn, BLOCK_SIZE_N), BLOCK_SIZE_N)
    sub_mask = (rm[:, None] < Mc) & (rn[None, :] < Nc)

    src_off = rm[:, None] * stride_sm + rn[None, :] * stride_sn
    dst_off = (rm[:, None] + cur_rank * Mc) * stride_dm + rn[None, :] * stride_dn

    for remote_rank in range(world_size):
        if remote_rank == cur_rank:
            data = tl.load(comm_src + src_off, mask=sub_mask)
            tl.store(comm_dst + dst_off, data, mask=sub_mask)
        else:
            iris.put(
                comm_src + src_off,
                comm_dst + dst_off,
                cur_rank,
                remote_rank,
                heap_bases,
                mask=sub_mask,
            )


@triton.jit()
def _all_reduce_tile(
    local_tid,
    comm_src,
    comm_dst,
    Mc,
    Nc,
    stride_sm,
    stride_sn,
    stride_dm,
    stride_dn,
    AR_TOTAL_TILES,
    AR_TILES_PER_RANK,
    num_pid_n,
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
    GROUP_SIZE_M: tl.constexpr,
    heap_bases: tl.tensor,
    cur_rank: tl.constexpr,
    world_size: tl.constexpr,
):
    """One-shot all-reduce of one tile. This rank owns tiles
    ``[cur_rank*AR_TILES_PER_RANK : ...]`` of the (Mc, Nc) grid: read that tile
    from every peer, sum in fp32, and scatter the result to every peer's
    ``comm_dst`` (result replicated on all ranks)."""
    tile_id = cur_rank * AR_TILES_PER_RANK + local_tid
    valid = tile_id < AR_TOTAL_TILES
    tid = min(tile_id, AR_TOTAL_TILES - 1)

    num_pid_m = tl.cdiv(Mc, BLOCK_SIZE_M)
    num_pid_in_group = GROUP_SIZE_M * num_pid_n
    group_id = tid // num_pid_in_group
    first_pid_m = group_id * GROUP_SIZE_M
    group_size_m = min(num_pid_m - first_pid_m, GROUP_SIZE_M)
    pid_m = first_pid_m + ((tid % num_pid_in_group) % group_size_m)
    pid_n = (tid % num_pid_in_group) // group_size_m
    tl.assume(pid_m >= 0)
    tl.assume(pid_n >= 0)

    rm = (pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)) % Mc
    rn = (pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)) % Nc
    rm = tl.max_contiguous(tl.multiple_of(rm, BLOCK_SIZE_M), BLOCK_SIZE_M)
    rn = tl.max_contiguous(tl.multiple_of(rn, BLOCK_SIZE_N), BLOCK_SIZE_N)
    sub_mask = (rm[:, None] < Mc) & (rn[None, :] < Nc) & valid

    src_off = rm[:, None] * stride_sm + rn[None, :] * stride_sn
    dst_off = rm[:, None] * stride_dm + rn[None, :] * stride_dn

    acc = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)
    for remote_rank in range(world_size):
        partial = iris.load(comm_src + src_off, cur_rank, remote_rank, heap_bases, mask=sub_mask, other=0.0)
        acc += partial.to(tl.float32)
    result = acc.to(comm_dst.type.element_ty)

    for remote_rank in range(world_size):
        if remote_rank == cur_rank:
            tl.store(comm_dst + dst_off, result, mask=sub_mask)
        else:
            iris.store(comm_dst + dst_off, result, cur_rank, remote_rank, heap_bases, mask=sub_mask)


# ---------------------------------------------------------------------------
# Fused single-kernel, dual work-stealing queue
# ---------------------------------------------------------------------------
@triton.jit()
def fused_ws_gemm_all_gather(
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
    comm_src,
    comm_dst,
    Mc,
    Nc,
    stride_sm,
    stride_sn,
    stride_dm,
    stride_dn,
    gemm_counter,
    comm_counter,
    GEMM_TOTAL_TILES,
    COMM_TOTAL_TILES,
    GEMM_WGS,
    NUM_WGS,
    GEMM_BLOCK_M: tl.constexpr,
    GEMM_BLOCK_N: tl.constexpr,
    GEMM_BLOCK_K: tl.constexpr,
    GEMM_GROUP_M: tl.constexpr,
    COMM_BLOCK_M: tl.constexpr,
    COMM_BLOCK_N: tl.constexpr,
    COMM_GROUP_M: tl.constexpr,
    EVEN_K: tl.constexpr,
    heap_bases: tl.tensor,
    cur_rank: tl.constexpr,
    world_size: tl.constexpr,
):
    pid = tl.program_id(0)

    tl.assume(stride_am > 0)
    tl.assume(stride_ak > 0)
    tl.assume(stride_bn > 0)
    tl.assume(stride_bk > 0)
    tl.assume(stride_cm > 0)
    tl.assume(stride_cn > 0)

    gemm_num_pid_n = tl.cdiv(N, GEMM_BLOCK_N)
    comm_num_pid_n = tl.cdiv(Nc, COMM_BLOCK_N)

    if pid < GEMM_WGS:
        # Phase 1: drain home (GEMM) queue.
        idx = tl.atomic_add(gemm_counter, 1, scope="gpu")
        while idx < GEMM_TOTAL_TILES:
            _gemm_tile(
                idx,
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
                gemm_num_pid_n,
                GEMM_BLOCK_M,
                GEMM_BLOCK_N,
                GEMM_BLOCK_K,
                GEMM_GROUP_M,
                EVEN_K,
            )
            idx = tl.atomic_add(gemm_counter, 1, scope="gpu")

        # Phase 2: steal from the comm queue.
        idx = tl.atomic_add(comm_counter, 1, scope="gpu")
        while idx < COMM_TOTAL_TILES:
            _all_gather_tile(
                idx,
                comm_src,
                comm_dst,
                Mc,
                Nc,
                stride_sm,
                stride_sn,
                stride_dm,
                stride_dn,
                comm_num_pid_n,
                COMM_BLOCK_M,
                COMM_BLOCK_N,
                COMM_GROUP_M,
                heap_bases,
                cur_rank,
                world_size,
            )
            idx = tl.atomic_add(comm_counter, 1, scope="gpu")
    else:
        # Phase 1: drain home (comm) queue.
        idx = tl.atomic_add(comm_counter, 1, scope="gpu")
        while idx < COMM_TOTAL_TILES:
            _all_gather_tile(
                idx,
                comm_src,
                comm_dst,
                Mc,
                Nc,
                stride_sm,
                stride_sn,
                stride_dm,
                stride_dn,
                comm_num_pid_n,
                COMM_BLOCK_M,
                COMM_BLOCK_N,
                COMM_GROUP_M,
                heap_bases,
                cur_rank,
                world_size,
            )
            idx = tl.atomic_add(comm_counter, 1, scope="gpu")

        # Phase 2: steal from the GEMM queue.
        idx = tl.atomic_add(gemm_counter, 1, scope="gpu")
        while idx < GEMM_TOTAL_TILES:
            _gemm_tile(
                idx,
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
                gemm_num_pid_n,
                GEMM_BLOCK_M,
                GEMM_BLOCK_N,
                GEMM_BLOCK_K,
                GEMM_GROUP_M,
                EVEN_K,
            )
            idx = tl.atomic_add(gemm_counter, 1, scope="gpu")


# ---------------------------------------------------------------------------
# Standalone work-stealing persistent kernels (two-kernel concurrent model)
# ---------------------------------------------------------------------------
@triton.jit()
def ws_gemm(
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
    gemm_counter,
    GEMM_TOTAL_TILES,
    GEMM_BLOCK_M: tl.constexpr,
    GEMM_BLOCK_N: tl.constexpr,
    GEMM_BLOCK_K: tl.constexpr,
    GEMM_GROUP_M: tl.constexpr,
    EVEN_K: tl.constexpr,
):
    tl.assume(stride_am > 0)
    tl.assume(stride_ak > 0)
    tl.assume(stride_bn > 0)
    tl.assume(stride_bk > 0)
    tl.assume(stride_cm > 0)
    tl.assume(stride_cn > 0)

    gemm_num_pid_n = tl.cdiv(N, GEMM_BLOCK_N)
    idx = tl.atomic_add(gemm_counter, 1, scope="gpu")
    while idx < GEMM_TOTAL_TILES:
        _gemm_tile(
            idx,
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
            gemm_num_pid_n,
            GEMM_BLOCK_M,
            GEMM_BLOCK_N,
            GEMM_BLOCK_K,
            GEMM_GROUP_M,
            EVEN_K,
        )
        idx = tl.atomic_add(gemm_counter, 1, scope="gpu")


@triton.jit()
def ws_all_gather(
    comm_src,
    comm_dst,
    Mc,
    Nc,
    stride_sm,
    stride_sn,
    stride_dm,
    stride_dn,
    comm_counter,
    COMM_TOTAL_TILES,
    COMM_BLOCK_M: tl.constexpr,
    COMM_BLOCK_N: tl.constexpr,
    COMM_GROUP_M: tl.constexpr,
    heap_bases: tl.tensor,
    cur_rank: tl.constexpr,
    world_size: tl.constexpr,
):
    comm_num_pid_n = tl.cdiv(Nc, COMM_BLOCK_N)
    idx = tl.atomic_add(comm_counter, 1, scope="gpu")
    while idx < COMM_TOTAL_TILES:
        _all_gather_tile(
            idx,
            comm_src,
            comm_dst,
            Mc,
            Nc,
            stride_sm,
            stride_sn,
            stride_dm,
            stride_dn,
            comm_num_pid_n,
            COMM_BLOCK_M,
            COMM_BLOCK_N,
            COMM_GROUP_M,
            heap_bases,
            cur_rank,
            world_size,
        )
        idx = tl.atomic_add(comm_counter, 1, scope="gpu")


# ---------------------------------------------------------------------------
# All-reduce variants (one-shot): fused dual-queue + standalone comm kernel
# ---------------------------------------------------------------------------
@triton.jit()
def fused_ws_gemm_all_reduce(
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
    comm_src,
    comm_dst,
    Mc,
    Nc,
    stride_sm,
    stride_sn,
    stride_dm,
    stride_dn,
    gemm_counter,
    comm_counter,
    GEMM_TOTAL_TILES,
    COMM_TOTAL_TILES,
    AR_TOTAL_TILES,
    AR_TILES_PER_RANK,
    GEMM_WGS,
    NUM_WGS,
    GEMM_BLOCK_M: tl.constexpr,
    GEMM_BLOCK_N: tl.constexpr,
    GEMM_BLOCK_K: tl.constexpr,
    GEMM_GROUP_M: tl.constexpr,
    COMM_BLOCK_M: tl.constexpr,
    COMM_BLOCK_N: tl.constexpr,
    COMM_GROUP_M: tl.constexpr,
    EVEN_K: tl.constexpr,
    heap_bases: tl.tensor,
    cur_rank: tl.constexpr,
    world_size: tl.constexpr,
):
    pid = tl.program_id(0)
    tl.assume(stride_am > 0)
    tl.assume(stride_ak > 0)
    tl.assume(stride_bn > 0)
    tl.assume(stride_bk > 0)
    tl.assume(stride_cm > 0)
    tl.assume(stride_cn > 0)
    gemm_num_pid_n = tl.cdiv(N, GEMM_BLOCK_N)
    comm_num_pid_n = tl.cdiv(Nc, COMM_BLOCK_N)

    if pid < GEMM_WGS:
        idx = tl.atomic_add(gemm_counter, 1, scope="gpu")
        while idx < GEMM_TOTAL_TILES:
            _gemm_tile(
                idx,
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
                gemm_num_pid_n,
                GEMM_BLOCK_M,
                GEMM_BLOCK_N,
                GEMM_BLOCK_K,
                GEMM_GROUP_M,
                EVEN_K,
            )
            idx = tl.atomic_add(gemm_counter, 1, scope="gpu")
        idx = tl.atomic_add(comm_counter, 1, scope="gpu")
        while idx < COMM_TOTAL_TILES:
            _all_reduce_tile(
                idx,
                comm_src,
                comm_dst,
                Mc,
                Nc,
                stride_sm,
                stride_sn,
                stride_dm,
                stride_dn,
                AR_TOTAL_TILES,
                AR_TILES_PER_RANK,
                comm_num_pid_n,
                COMM_BLOCK_M,
                COMM_BLOCK_N,
                COMM_GROUP_M,
                heap_bases,
                cur_rank,
                world_size,
            )
            idx = tl.atomic_add(comm_counter, 1, scope="gpu")
    else:
        idx = tl.atomic_add(comm_counter, 1, scope="gpu")
        while idx < COMM_TOTAL_TILES:
            _all_reduce_tile(
                idx,
                comm_src,
                comm_dst,
                Mc,
                Nc,
                stride_sm,
                stride_sn,
                stride_dm,
                stride_dn,
                AR_TOTAL_TILES,
                AR_TILES_PER_RANK,
                comm_num_pid_n,
                COMM_BLOCK_M,
                COMM_BLOCK_N,
                COMM_GROUP_M,
                heap_bases,
                cur_rank,
                world_size,
            )
            idx = tl.atomic_add(comm_counter, 1, scope="gpu")
        idx = tl.atomic_add(gemm_counter, 1, scope="gpu")
        while idx < GEMM_TOTAL_TILES:
            _gemm_tile(
                idx,
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
                gemm_num_pid_n,
                GEMM_BLOCK_M,
                GEMM_BLOCK_N,
                GEMM_BLOCK_K,
                GEMM_GROUP_M,
                EVEN_K,
            )
            idx = tl.atomic_add(gemm_counter, 1, scope="gpu")


@triton.jit()
def ws_all_reduce(
    comm_src,
    comm_dst,
    Mc,
    Nc,
    stride_sm,
    stride_sn,
    stride_dm,
    stride_dn,
    comm_counter,
    COMM_TOTAL_TILES,
    AR_TOTAL_TILES,
    AR_TILES_PER_RANK,
    COMM_BLOCK_M: tl.constexpr,
    COMM_BLOCK_N: tl.constexpr,
    COMM_GROUP_M: tl.constexpr,
    heap_bases: tl.tensor,
    cur_rank: tl.constexpr,
    world_size: tl.constexpr,
):
    comm_num_pid_n = tl.cdiv(Nc, COMM_BLOCK_N)
    idx = tl.atomic_add(comm_counter, 1, scope="gpu")
    while idx < COMM_TOTAL_TILES:
        _all_reduce_tile(
            idx,
            comm_src,
            comm_dst,
            Mc,
            Nc,
            stride_sm,
            stride_sn,
            stride_dm,
            stride_dn,
            AR_TOTAL_TILES,
            AR_TILES_PER_RANK,
            comm_num_pid_n,
            COMM_BLOCK_M,
            COMM_BLOCK_N,
            COMM_GROUP_M,
            heap_bases,
            cur_rank,
            world_size,
        )
        idx = tl.atomic_add(comm_counter, 1, scope="gpu")


# ---------------------------------------------------------------------------
# Additional collectives: reduce-scatter, all-to-all, broadcast
# COMM_KIND: 2=reduce_scatter, 3=all_to_all, 4=broadcast
# ---------------------------------------------------------------------------
@triton.jit()
def _reduce_scatter_tile(
    tid,
    src,
    dst,
    Mc,
    Nc,
    Mout,
    stride_sm,
    stride_sn,
    stride_dm,
    stride_dn,
    num_pid_n,
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
    GROUP_SIZE_M: tl.constexpr,
    heap_bases: tl.tensor,
    cur_rank: tl.constexpr,
    world_size: tl.constexpr,
):
    """This rank's output slice (rows cur_rank*Mout..): sum that slice over all
    peers' src, store locally. Grid is over the (Mout, Nc) output."""
    num_pid_m = tl.cdiv(Mout, BLOCK_SIZE_M)
    num_pid_in_group = GROUP_SIZE_M * num_pid_n
    group_id = tid // num_pid_in_group
    first_pid_m = group_id * GROUP_SIZE_M
    group_size_m = min(num_pid_m - first_pid_m, GROUP_SIZE_M)
    pid_m = first_pid_m + ((tid % num_pid_in_group) % group_size_m)
    pid_n = (tid % num_pid_in_group) // group_size_m
    tl.assume(pid_m >= 0)
    tl.assume(pid_n >= 0)
    rm = (pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)) % Mout
    rn = (pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)) % Nc
    rm = tl.max_contiguous(tl.multiple_of(rm, BLOCK_SIZE_M), BLOCK_SIZE_M)
    rn = tl.max_contiguous(tl.multiple_of(rn, BLOCK_SIZE_N), BLOCK_SIZE_N)
    sub_mask = (rm[:, None] < Mout) & (rn[None, :] < Nc)
    grow = cur_rank * Mout + rm
    src_off = grow[:, None] * stride_sm + rn[None, :] * stride_sn
    dst_off = rm[:, None] * stride_dm + rn[None, :] * stride_dn
    acc = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)
    for p in range(world_size):
        acc += iris.load(src + src_off, cur_rank, p, heap_bases, mask=sub_mask, other=0.0).to(tl.float32)
    tl.store(dst + dst_off, acc.to(dst.type.element_ty), mask=sub_mask)


@triton.jit()
def _all_to_all_tile(
    tid,
    src,
    dst,
    Mc,
    Nc,
    CHUNK,
    stride_sm,
    stride_sn,
    stride_dm,
    stride_dn,
    num_pid_n,
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
    GROUP_SIZE_M: tl.constexpr,
    heap_bases: tl.tensor,
    cur_rank: tl.constexpr,
    world_size: tl.constexpr,
):
    """Send this rank's src chunk p (rows p*CHUNK..) to peer p's dst chunk
    cur_rank. Grid over src (Mc, Nc); CHUNK = Mc // world_size (assumed BM-aligned).

    NOTE: a rank-staggered write order was tried (rotate the m-grid by cur_rank
    chunks to spread concurrent writes across peers) but measured NO improvement
    on the comm-heavy A2A shapes -- one-shot A2A is aggregate-xGMI-bandwidth-bound
    regardless of write ordering, so a permutation can't help. A real speedup
    needs a different transport (DMA/SDMA or a staged multi-step). Selector should
    fall back to RCCL for comm-heavy A2A."""
    num_pid_m = tl.cdiv(Mc, BLOCK_SIZE_M)
    num_pid_in_group = GROUP_SIZE_M * num_pid_n
    group_id = tid // num_pid_in_group
    first_pid_m = group_id * GROUP_SIZE_M
    group_size_m = min(num_pid_m - first_pid_m, GROUP_SIZE_M)
    pid_m = first_pid_m + ((tid % num_pid_in_group) % group_size_m)
    pid_n = (tid % num_pid_in_group) // group_size_m
    tl.assume(pid_m >= 0)
    tl.assume(pid_n >= 0)
    peer = (pid_m * BLOCK_SIZE_M) // CHUNK
    rm = (pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)) % Mc
    rn = (pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)) % Nc
    rm = tl.max_contiguous(tl.multiple_of(rm, BLOCK_SIZE_M), BLOCK_SIZE_M)
    rn = tl.max_contiguous(tl.multiple_of(rn, BLOCK_SIZE_N), BLOCK_SIZE_N)
    sub_mask = (rm[:, None] < Mc) & (rn[None, :] < Nc)
    src_off = rm[:, None] * stride_sm + rn[None, :] * stride_sn
    dst_rows = cur_rank * CHUNK + (rm - peer * CHUNK)
    dst_off = dst_rows[:, None] * stride_dm + rn[None, :] * stride_dn
    if peer == cur_rank:
        data = tl.load(src + src_off, mask=sub_mask)
        tl.store(dst + dst_off, data, mask=sub_mask)
    else:
        iris.put(src + src_off, dst + dst_off, cur_rank, peer, heap_bases, mask=sub_mask)


@triton.jit()
def _broadcast_tile(
    tid,
    src,
    dst,
    Mc,
    Nc,
    stride_sm,
    stride_sn,
    stride_dm,
    stride_dn,
    num_pid_n,
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
    GROUP_SIZE_M: tl.constexpr,
    heap_bases: tl.tensor,
    cur_rank: tl.constexpr,
    world_size: tl.constexpr,
    ROOT,
):
    """ROOT pushes its src tile to every rank's dst (identical layout); non-root
    ranks do nothing for this tile."""
    if cur_rank == ROOT:
        num_pid_m = tl.cdiv(Mc, BLOCK_SIZE_M)
        num_pid_in_group = GROUP_SIZE_M * num_pid_n
        group_id = tid // num_pid_in_group
        first_pid_m = group_id * GROUP_SIZE_M
        group_size_m = min(num_pid_m - first_pid_m, GROUP_SIZE_M)
        pid_m = first_pid_m + ((tid % num_pid_in_group) % group_size_m)
        pid_n = (tid % num_pid_in_group) // group_size_m
        tl.assume(pid_m >= 0)
        tl.assume(pid_n >= 0)
        rm = (pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)) % Mc
        rn = (pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)) % Nc
        rm = tl.max_contiguous(tl.multiple_of(rm, BLOCK_SIZE_M), BLOCK_SIZE_M)
        rn = tl.max_contiguous(tl.multiple_of(rn, BLOCK_SIZE_N), BLOCK_SIZE_N)
        sub_mask = (rm[:, None] < Mc) & (rn[None, :] < Nc)
        src_off = rm[:, None] * stride_sm + rn[None, :] * stride_sn
        dst_off = rm[:, None] * stride_dm + rn[None, :] * stride_dn
        data = tl.load(src + src_off, mask=sub_mask)
        for p in range(world_size):
            if p == cur_rank:
                tl.store(dst + dst_off, data, mask=sub_mask)
            else:
                iris.put(src + src_off, dst + dst_off, cur_rank, p, heap_bases, mask=sub_mask)


@triton.jit()
def _comm_tile_ext(
    COMM_KIND: tl.constexpr,
    tid,
    src,
    dst,
    Mc,
    Nc,
    E0,
    stride_sm,
    stride_sn,
    stride_dm,
    stride_dn,
    num_pid_n,
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
    GROUP_SIZE_M: tl.constexpr,
    heap_bases: tl.tensor,
    cur_rank: tl.constexpr,
    world_size: tl.constexpr,
    ROOT,
):
    if COMM_KIND == 2:
        _reduce_scatter_tile(
            tid,
            src,
            dst,
            Mc,
            Nc,
            E0,
            stride_sm,
            stride_sn,
            stride_dm,
            stride_dn,
            num_pid_n,
            BLOCK_SIZE_M,
            BLOCK_SIZE_N,
            GROUP_SIZE_M,
            heap_bases,
            cur_rank,
            world_size,
        )
    elif COMM_KIND == 3:
        _all_to_all_tile(
            tid,
            src,
            dst,
            Mc,
            Nc,
            E0,
            stride_sm,
            stride_sn,
            stride_dm,
            stride_dn,
            num_pid_n,
            BLOCK_SIZE_M,
            BLOCK_SIZE_N,
            GROUP_SIZE_M,
            heap_bases,
            cur_rank,
            world_size,
        )
    else:
        _broadcast_tile(
            tid,
            src,
            dst,
            Mc,
            Nc,
            stride_sm,
            stride_sn,
            stride_dm,
            stride_dn,
            num_pid_n,
            BLOCK_SIZE_M,
            BLOCK_SIZE_N,
            GROUP_SIZE_M,
            heap_bases,
            cur_rank,
            world_size,
            ROOT,
        )


@triton.jit()
def fused_ws_gemm_comm_ext(
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
    comm_src,
    comm_dst,
    Mc,
    Nc,
    stride_sm,
    stride_sn,
    stride_dm,
    stride_dn,
    E0,
    gemm_counter,
    comm_counter,
    GEMM_TOTAL_TILES,
    COMM_TOTAL_TILES,
    GEMM_WGS,
    NUM_WGS,
    ROOT,
    COMM_KIND: tl.constexpr,
    GEMM_BLOCK_M: tl.constexpr,
    GEMM_BLOCK_N: tl.constexpr,
    GEMM_BLOCK_K: tl.constexpr,
    GEMM_GROUP_M: tl.constexpr,
    COMM_BLOCK_M: tl.constexpr,
    COMM_BLOCK_N: tl.constexpr,
    COMM_GROUP_M: tl.constexpr,
    EVEN_K: tl.constexpr,
    heap_bases: tl.tensor,
    cur_rank: tl.constexpr,
    world_size: tl.constexpr,
):
    pid = tl.program_id(0)
    tl.assume(stride_am > 0)
    tl.assume(stride_ak > 0)
    tl.assume(stride_bn > 0)
    tl.assume(stride_bk > 0)
    tl.assume(stride_cm > 0)
    tl.assume(stride_cn > 0)
    gemm_num_pid_n = tl.cdiv(N, GEMM_BLOCK_N)
    comm_num_pid_n = tl.cdiv(Nc, COMM_BLOCK_N)
    if pid < GEMM_WGS:
        idx = tl.atomic_add(gemm_counter, 1, scope="gpu")
        while idx < GEMM_TOTAL_TILES:
            _gemm_tile(
                idx,
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
                gemm_num_pid_n,
                GEMM_BLOCK_M,
                GEMM_BLOCK_N,
                GEMM_BLOCK_K,
                GEMM_GROUP_M,
                EVEN_K,
            )
            idx = tl.atomic_add(gemm_counter, 1, scope="gpu")
        idx = tl.atomic_add(comm_counter, 1, scope="gpu")
        while idx < COMM_TOTAL_TILES:
            _comm_tile_ext(
                COMM_KIND,
                idx,
                comm_src,
                comm_dst,
                Mc,
                Nc,
                E0,
                stride_sm,
                stride_sn,
                stride_dm,
                stride_dn,
                comm_num_pid_n,
                COMM_BLOCK_M,
                COMM_BLOCK_N,
                COMM_GROUP_M,
                heap_bases,
                cur_rank,
                world_size,
                ROOT,
            )
            idx = tl.atomic_add(comm_counter, 1, scope="gpu")
    else:
        idx = tl.atomic_add(comm_counter, 1, scope="gpu")
        while idx < COMM_TOTAL_TILES:
            _comm_tile_ext(
                COMM_KIND,
                idx,
                comm_src,
                comm_dst,
                Mc,
                Nc,
                E0,
                stride_sm,
                stride_sn,
                stride_dm,
                stride_dn,
                comm_num_pid_n,
                COMM_BLOCK_M,
                COMM_BLOCK_N,
                COMM_GROUP_M,
                heap_bases,
                cur_rank,
                world_size,
                ROOT,
            )
            idx = tl.atomic_add(comm_counter, 1, scope="gpu")
        idx = tl.atomic_add(gemm_counter, 1, scope="gpu")
        while idx < GEMM_TOTAL_TILES:
            _gemm_tile(
                idx,
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
                gemm_num_pid_n,
                GEMM_BLOCK_M,
                GEMM_BLOCK_N,
                GEMM_BLOCK_K,
                GEMM_GROUP_M,
                EVEN_K,
            )
            idx = tl.atomic_add(gemm_counter, 1, scope="gpu")


@triton.jit()
def ws_comm_ext(
    comm_src,
    comm_dst,
    Mc,
    Nc,
    stride_sm,
    stride_sn,
    stride_dm,
    stride_dn,
    E0,
    comm_counter,
    COMM_TOTAL_TILES,
    ROOT,
    COMM_KIND: tl.constexpr,
    COMM_BLOCK_M: tl.constexpr,
    COMM_BLOCK_N: tl.constexpr,
    COMM_GROUP_M: tl.constexpr,
    heap_bases: tl.tensor,
    cur_rank: tl.constexpr,
    world_size: tl.constexpr,
):
    comm_num_pid_n = tl.cdiv(Nc, COMM_BLOCK_N)
    idx = tl.atomic_add(comm_counter, 1, scope="gpu")
    while idx < COMM_TOTAL_TILES:
        _comm_tile_ext(
            COMM_KIND,
            idx,
            comm_src,
            comm_dst,
            Mc,
            Nc,
            E0,
            stride_sm,
            stride_sn,
            stride_dm,
            stride_dn,
            comm_num_pid_n,
            COMM_BLOCK_M,
            COMM_BLOCK_N,
            COMM_GROUP_M,
            heap_bases,
            cur_rank,
            world_size,
            ROOT,
        )
        idx = tl.atomic_add(comm_counter, 1, scope="gpu")
