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
from iris.mem.triton.types import tile_layout
from tritonblas.kernels.stages import (
    GemmContext,
    Tile as _TBTile,
    make_tensor_view as _tb_make_tensor_view,
)


@triton.jit()
def _grouped_coords(
    tid,
    M,
    N,
    num_pid_n,
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
    GROUP_SIZE_M: tl.constexpr,
):
    """Shared tile-coordinate helper: GROUP_M work-stealing swizzle of a flat
    ``tid`` into ``(pid_m, pid_n)`` plus the ``(rm, rn, mask)`` layout via the
    shared ``iris.mem.triton.types.tile_layout`` (with its vectorization hints).
    Replaces the copy-pasted preamble in every per-tile comm primitive."""
    num_pid_m = tl.cdiv(M, BLOCK_SIZE_M)
    num_pid_in_group = GROUP_SIZE_M * num_pid_n
    group_id = tid // num_pid_in_group
    first_pid_m = group_id * GROUP_SIZE_M
    group_size_m = min(num_pid_m - first_pid_m, GROUP_SIZE_M)
    pid_m = first_pid_m + ((tid % num_pid_in_group) % group_size_m)
    pid_n = (tid % num_pid_in_group) // group_size_m
    tl.assume(pid_m >= 0)
    tl.assume(pid_n >= 0)
    rm, rn, mask = tile_layout(pid_m, pid_n, M, N, BLOCK_SIZE_M, BLOCK_SIZE_N)
    return pid_m, pid_n, rm, rn, mask


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
    """Compute one output tile of ``C = A @ B`` and store it locally.

    The K-loop is delegated to tritonblas ``GemmContext.reduce_axis`` (shared with
    ``iris.ops``); we keep only the GROUP_M tile swizzle and the local store.
    ``cache_modifier=""`` is REQUIRED: GemmContext defaults to ``".cg"`` (bypass-L1
    streaming) which throws away GEMM operand reuse and is ~1.3-1.6x slower here;
    with default L1/L2 caching this matches/beats the old hand-rolled loop."""
    num_pid_m = tl.cdiv(M, BLOCK_SIZE_M)
    num_pid_in_group = GROUP_SIZE_M * num_pid_n
    group_id = tile_id // num_pid_in_group
    first_pid_m = group_id * GROUP_SIZE_M
    group_size_m = min(num_pid_m - first_pid_m, GROUP_SIZE_M)
    pid_m = first_pid_m + ((tile_id % num_pid_in_group) % group_size_m)
    pid_n = (tile_id % num_pid_in_group) // group_size_m
    tl.assume(pid_m >= 0)
    tl.assume(pid_n >= 0)

    tensorA = _tb_make_tensor_view(A, M, K, stride_am, stride_ak)
    tensorB = _tb_make_tensor_view(B, K, N, stride_bk, stride_bn)
    gemm_ctx = GemmContext(
        BLOCK_SIZE_M,
        BLOCK_SIZE_N,
        BLOCK_SIZE_K,
        num_sms=1,
        even_k=EVEN_K,
        cache_modifier_a="",
        cache_modifier_b="",
    )
    out_tile = _TBTile(pid_m, pid_n, BLOCK_SIZE_M, BLOCK_SIZE_N)
    acc = gemm_ctx.reduce_axis(tensorA, tensorB, out_tile)

    c = acc.to(C.type.element_ty)
    rm = (pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)) % M
    rn = (pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)) % N
    rm = tl.max_contiguous(tl.multiple_of(rm, BLOCK_SIZE_M), BLOCK_SIZE_M)
    rn = tl.max_contiguous(tl.multiple_of(rn, BLOCK_SIZE_N), BLOCK_SIZE_N)
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
    context_tensor: tl.tensor,
    cur_rank: tl.constexpr,
    world_size: tl.constexpr,
):
    """All-gather one local ``(Mc, Nc)`` block into rows ``[cur_rank*Mc : ...]``
    of every peer's symmetric ``comm_dst`` (dim-0), via the shared iris
    device-context tile collective ``ctx.all_gather`` (same primitive used by
    ``iris.ops``)."""
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
    data = tl.load(comm_src + rm[:, None] * stride_sm + rn[None, :] * stride_sn, mask=sub_mask)

    ctx = iris.DeviceContext.initialize(context_tensor, cur_rank, world_size)
    dst_view = iris.make_tensor_view(comm_dst, world_size * Mc, Nc, stride_dm, stride_dn)
    tile = iris.Tile(pid_m, pid_n, BLOCK_SIZE_M, BLOCK_SIZE_N, data)
    ctx.all_gather(tile, dst_view, dim=0)


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

    _, _, rm, rn, mask = _grouped_coords(tid, Mc, Nc, num_pid_n, BLOCK_SIZE_M, BLOCK_SIZE_N, GROUP_SIZE_M)
    sub_mask = mask & valid

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
    _, _, rm, rn, sub_mask = _grouped_coords(tid, Mout, Nc, num_pid_n, BLOCK_SIZE_M, BLOCK_SIZE_N, GROUP_SIZE_M)
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
    pid_m, _, rm, rn, sub_mask = _grouped_coords(tid, Mc, Nc, num_pid_n, BLOCK_SIZE_M, BLOCK_SIZE_N, GROUP_SIZE_M)
    peer = (pid_m * BLOCK_SIZE_M) // CHUNK
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
        _, _, rm, rn, sub_mask = _grouped_coords(tid, Mc, Nc, num_pid_n, BLOCK_SIZE_M, BLOCK_SIZE_N, GROUP_SIZE_M)
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
