# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.

"""
Launched work-stealing kernels for :mod:`iris.concurrent`.

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

These kernels only implement the two-queue work-stealing scheduling; the actual
per-tile work lives in :mod:`iris.concurrent._tiles` (``_gemm_tile``,
``_all_gather_tile``, ``_all_reduce_tile``, ``_comm_tile_ext``), so the fused and
two-kernel models are numerically identical tile-for-tile.
"""

import triton
import triton.language as tl

from ._tiles import _all_gather_tile, _all_reduce_tile, _comm_tile_ext, _gemm_tile


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
    context_tensor: tl.tensor,
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
                context_tensor,
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
                context_tensor,
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
    context_tensor: tl.tensor,
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
            context_tensor,
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
# Generic ext kernels: reduce-scatter / all-to-all / broadcast (COMM_KIND)
# ---------------------------------------------------------------------------
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
