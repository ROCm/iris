# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.

"""
Fast AllReduce — direct port of the one-shot pull kernel that beat RCCL
by 1.26-2.00x on ReduceScatter.

Applies every finding from the GEMM+RS study:
  - bulk iris.load (pull) — measured 3.2x faster than push, 7.6x faster than atomics
  - no iris.ccl wrapper — that cost 0.127ms of Python overhead on RS
  - no host barrier between GEMM and comm — barriers cost 0.07ms and are unnecessary
  - persistent grid, per-world-size tuned tile/SMS

Two variants:

  one_shot:  every rank reads the FULL [M, N] from every peer and sums.
             traffic = ws * M * N.  1 kernel, all reads in parallel.

  two_shot:  reduce-scatter then all-gather.
             traffic = 2 * (ws-1)/ws * M * N -- less data than one_shot, but
             requires a host barrier between phases (0.071ms at ws=8), which
             costs more than the traffic saving. Measured slower than one_shot
             at every M. Kept for reference; prefer one_shot.

At ws=2 one_shot moves 2x and two_shot moves 1x, so two_shot should win on
bytes; at small M one_shot should win on launch/step overhead. That crossover
is exactly what we measured for RS.
"""

from typing import Optional
import torch
import torch.distributed as dist
import triton
import triton.language as tl
import iris


# Per-world-size configs. Starting point is the RS-tuned table; AR moves ~2x
# the bytes so these will need their own sweep.
_AUTO_CONFIG = {
    2: dict(block_m=128, block_n=64, num_sms=196, num_warps=4),
    4: dict(block_m=64, block_n=64, num_sms=32, num_warps=4),
    8: dict(block_m=32, block_n=64, num_sms=32, num_warps=4),
}
_DEFAULT_CONFIG = dict(block_m=64, block_n=64, num_sms=64, num_warps=4)


def _get_config(world_size: int, M: int) -> dict:
    cfg = _AUTO_CONFIG.get(world_size, _DEFAULT_CONFIG).copy()
    while cfg["block_m"] > M and cfg["block_m"] > 4:
        cfg["block_m"] //= 2
    return cfg


@triton.jit
def _one_shot_ar_kernel(
    input_ptr,
    output_ptr,
    M,
    N,
    stride_in_m,
    stride_in_n,
    stride_out_m,
    stride_out_n,
    heap_bases: tl.tensor,
    cur_rank: tl.constexpr,
    world_size: tl.constexpr,
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
    NUM_SMS: tl.constexpr,
):
    """
    One-shot pull AllReduce: every rank reads every peer's full tensor,
    sums in fp32, writes its own copy of the result.

    Unlike RS, there is no M-partitioning — each rank produces the whole
    [M, N] output. Traffic is ws * M * N per rank.
    """
    pid = tl.program_id(0)
    acc_dtype = tl.float32
    num_m_tiles = tl.cdiv(M, BLOCK_SIZE_M)
    num_n_tiles = tl.cdiv(N, BLOCK_SIZE_N)
    total_tiles = num_m_tiles * num_n_tiles

    for tile_id in range(pid, total_tiles, NUM_SMS):
        pid_m = tile_id // num_n_tiles
        pid_n = tile_id % num_n_tiles

        rm = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
        rm = tl.max_contiguous(tl.multiple_of(rm, BLOCK_SIZE_M), BLOCK_SIZE_M)
        rn = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
        rn = tl.max_contiguous(tl.multiple_of(rn % N, BLOCK_SIZE_N), BLOCK_SIZE_N)

        in_off = rm[:, None] * stride_in_m + rn[None, :] * stride_in_n
        base_ptr = input_ptr + in_off
        out_off = rm[:, None] * stride_out_m + rn[None, :] * stride_out_n

        is_full = (pid_m * BLOCK_SIZE_M + BLOCK_SIZE_M <= M) & (
            pid_n * BLOCK_SIZE_N + BLOCK_SIZE_N <= N
        )

        if is_full:
            # Rotate the start rank per WG to spread XGMI load
            start_rank = pid % world_size
            acc = iris.load(base_ptr, cur_rank, start_rank, heap_bases,
                            hint=(1, BLOCK_SIZE_N)).to(acc_dtype)
            for i in tl.static_range(1, world_size):
                r = (start_rank + i) % world_size
                acc += iris.load(base_ptr, cur_rank, r, heap_bases,
                                 hint=(1, BLOCK_SIZE_N)).to(acc_dtype)
            tl.store(output_ptr + out_off, acc.to(output_ptr.type.element_ty))
        else:
            mask = (rm[:, None] < M) & (rn[None, :] < N)
            start_rank = pid % world_size
            acc = iris.load(base_ptr, cur_rank, start_rank, heap_bases,
                            mask=mask, hint=(1, BLOCK_SIZE_N)).to(acc_dtype)
            for i in tl.static_range(1, world_size):
                r = (start_rank + i) % world_size
                acc += iris.load(base_ptr, cur_rank, r, heap_bases,
                                 mask=mask, hint=(1, BLOCK_SIZE_N)).to(acc_dtype)
            tl.store(output_ptr + out_off, acc.to(output_ptr.type.element_ty), mask=mask)


@triton.jit
def _two_shot_reduce_kernel(
    input_ptr,
    scratch_ptr,
    M,
    N,
    M_local,
    stride_in_m,
    stride_in_n,
    heap_bases: tl.tensor,
    cur_rank: tl.constexpr,
    world_size: tl.constexpr,
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
    NUM_SMS: tl.constexpr,
):
    """
    Phase 1 of two-shot: reduce-scatter. Each rank reduces only its own
    M-shard by pulling that shard from every peer. Result stays in the
    symmetric heap (scratch) so phase 2 can gather it.
    """
    pid = tl.program_id(0)
    acc_dtype = tl.float32
    num_m_tiles = M_local // BLOCK_SIZE_M
    num_n_tiles = tl.cdiv(N, BLOCK_SIZE_N)
    total_tiles = num_m_tiles * num_n_tiles
    m_offset = cur_rank * num_m_tiles

    for tile_id in range(pid, total_tiles, NUM_SMS):
        local_pid_m = tile_id // num_n_tiles
        pid_n = tile_id % num_n_tiles
        global_pid_m = m_offset + local_pid_m

        rm = global_pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
        rm = tl.max_contiguous(tl.multiple_of(rm, BLOCK_SIZE_M), BLOCK_SIZE_M)
        rn = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
        rn = tl.max_contiguous(tl.multiple_of(rn % N, BLOCK_SIZE_N), BLOCK_SIZE_N)

        off = rm[:, None] * stride_in_m + rn[None, :] * stride_in_n
        base_ptr = input_ptr + off
        is_full = (global_pid_m * BLOCK_SIZE_M + BLOCK_SIZE_M <= M) & (
            pid_n * BLOCK_SIZE_N + BLOCK_SIZE_N <= N
        )

        if is_full:
            start_rank = pid % world_size
            acc = iris.load(base_ptr, cur_rank, start_rank, heap_bases,
                            hint=(1, BLOCK_SIZE_N)).to(acc_dtype)
            for i in tl.static_range(1, world_size):
                r = (start_rank + i) % world_size
                acc += iris.load(base_ptr, cur_rank, r, heap_bases,
                                 hint=(1, BLOCK_SIZE_N)).to(acc_dtype)
            # Write the reduced shard back in place (symmetric, peer-visible)
            tl.store(scratch_ptr + off, acc.to(scratch_ptr.type.element_ty),
                     cache_modifier=".wt")
        else:
            mask = (rm[:, None] < M) & (rn[None, :] < N)
            start_rank = pid % world_size
            acc = iris.load(base_ptr, cur_rank, start_rank, heap_bases,
                            mask=mask, hint=(1, BLOCK_SIZE_N)).to(acc_dtype)
            for i in tl.static_range(1, world_size):
                r = (start_rank + i) % world_size
                acc += iris.load(base_ptr, cur_rank, r, heap_bases,
                                 mask=mask, hint=(1, BLOCK_SIZE_N)).to(acc_dtype)
            tl.store(scratch_ptr + off, acc.to(scratch_ptr.type.element_ty),
                     mask=mask, cache_modifier=".wt")


@triton.jit
def _two_shot_gather_kernel(
    scratch_ptr,
    output_ptr,
    M,
    N,
    M_local,
    stride_s_m,
    stride_s_n,
    stride_out_m,
    stride_out_n,
    heap_bases: tl.tensor,
    cur_rank: tl.constexpr,
    world_size: tl.constexpr,
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
    NUM_SMS: tl.constexpr,
):
    """
    Phase 2 of two-shot: all-gather. Each rank pulls every peer's reduced
    shard into its own full [M, N] output. Pull direction throughout.
    """
    pid = tl.program_id(0)
    num_m_tiles = tl.cdiv(M, BLOCK_SIZE_M)
    num_n_tiles = tl.cdiv(N, BLOCK_SIZE_N)
    total_tiles = num_m_tiles * num_n_tiles
    m_tiles_per_rank = M_local // BLOCK_SIZE_M

    for tile_id in range(pid, total_tiles, NUM_SMS):
        pid_m = tile_id // num_n_tiles
        pid_n = tile_id % num_n_tiles
        # Which rank owns this M-tile after the reduce-scatter phase
        owner = pid_m // m_tiles_per_rank

        rm = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
        rm = tl.max_contiguous(tl.multiple_of(rm, BLOCK_SIZE_M), BLOCK_SIZE_M)
        rn = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
        rn = tl.max_contiguous(tl.multiple_of(rn % N, BLOCK_SIZE_N), BLOCK_SIZE_N)

        s_off = rm[:, None] * stride_s_m + rn[None, :] * stride_s_n
        o_off = rm[:, None] * stride_out_m + rn[None, :] * stride_out_n
        is_full = (pid_m * BLOCK_SIZE_M + BLOCK_SIZE_M <= M) & (
            pid_n * BLOCK_SIZE_N + BLOCK_SIZE_N <= N
        )

        if is_full:
            v = iris.load(scratch_ptr + s_off, cur_rank, owner, heap_bases,
                          hint=(1, BLOCK_SIZE_N))
            tl.store(output_ptr + o_off, v)
        else:
            mask = (rm[:, None] < M) & (rn[None, :] < N)
            v = iris.load(scratch_ptr + s_off, cur_rank, owner, heap_bases,
                          mask=mask, hint=(1, BLOCK_SIZE_N))
            tl.store(output_ptr + o_off, v, mask=mask)


def one_shot_all_reduce(ctx, output_tensor, input_tensor, **kw):
    """
    One-shot pull AllReduce. `input_tensor` must be in the symmetric heap.
    Traffic: ws * M * N per rank, one kernel launch.
    """
    M, N = input_tensor.shape
    rank = ctx.get_rank()
    world_size = ctx.get_num_ranks()
    assert output_tensor.shape == (M, N)

    cfg = _get_config(world_size, M)
    bm = kw.get("block_m") or cfg["block_m"]
    bn = kw.get("block_n") or cfg["block_n"]
    sms = kw.get("num_sms") or cfg["num_sms"]
    warps = kw.get("num_warps") or cfg["num_warps"]

    _one_shot_ar_kernel[(sms,)](
        input_tensor, output_tensor, M, N,
        input_tensor.stride(0), input_tensor.stride(1),
        output_tensor.stride(0), output_tensor.stride(1),
        ctx.get_heap_bases(), rank, world_size,
        bm, bn, sms, num_warps=warps,
    )


def two_shot_all_reduce(ctx, output_tensor, input_tensor, scratch=None,
                        use_barrier=True, **kw):
    """
    Two-shot AllReduce: reduce-scatter then all-gather, both pull-direction.
    Traffic: 2*(ws-1)/ws * M * N — less than one-shot, but two phases.

    Returns the scratch buffer for reuse.
    """
    M, N = input_tensor.shape
    rank = ctx.get_rank()
    world_size = ctx.get_num_ranks()
    M_local = M // world_size
    assert output_tensor.shape == (M, N)
    assert M % world_size == 0

    cfg = _get_config(world_size, M_local)
    bm = kw.get("block_m") or cfg["block_m"]
    bn = kw.get("block_n") or cfg["block_n"]
    sms = kw.get("num_sms") or cfg["num_sms"]
    warps = kw.get("num_warps") or cfg["num_warps"]
    assert M_local % bm == 0

    if scratch is None:
        scratch = ctx.zeros((M, N), dtype=input_tensor.dtype)

    hb = ctx.get_heap_bases()

    _two_shot_reduce_kernel[(sms,)](
        input_tensor, scratch, M, N, M_local,
        input_tensor.stride(0), input_tensor.stride(1),
        hb, rank, world_size, bm, bn, sms, num_warps=warps,
    )
    # The barrier is REQUIRED for correctness here. Do not remove it.
    #
    # Same-stream ordering only orders THIS rank's two kernels. The AG phase
    # reads a peer's `scratch`, which that peer writes in ITS OWN RS kernel --
    # nothing prevents us entering AG while a peer is still inside RS. That is
    # a genuine cross-rank read-after-write with no ordering primitive behind
    # it. Measured barrier-free: 6.5-8.6ms and max_diff=313.5 at M=2048.
    #
    # This differs from GEMM+RS, where dropping the barrier WAS safe: there the
    # RS kernel read peer `staged_c` written by torch.mm, and every rank's mm
    # starts at ~the same wall-clock time and runs for ~the same duration, so
    # skew is bounded and self-limiting (validated to n=1000). Here the
    # producer is a peer's *kernel 1*, with no such coupling.
    #
    # Cost: 0.071ms at ws=8, which exceeds the traffic saving two-shot buys
    # over one-shot -- so two-shot is not competitive for AR unless this sync
    # gets cheaper (e.g. one global counter + poll per rank, ~5-10us, instead
    # of a full host barrier).
    if use_barrier:
        ctx.barrier()
    _two_shot_gather_kernel[(sms,)](
        scratch, output_tensor, M, N, M_local,
        scratch.stride(0), scratch.stride(1),
        output_tensor.stride(0), output_tensor.stride(1),
        hb, rank, world_size, bm, bn, sms, num_warps=warps,
    )
    return scratch


def matmul_all_reduce_fast(ctx, output_tensor, A, B, variant="one_shot", **kw):
    """
    GEMM + fast AllReduce, two kernels.

    Computes output = all_reduce(A @ B) with A sharded on K.
    Uses hipBLASLt for the GEMM (measured optimal in the RS study) and the
    pull-direction AR kernel for the collective.

    Args:
        variant: "one_shot" or "two_shot"
    """
    M, K_local = A.shape
    _, N = B.shape
    assert output_tensor.shape == (M, N)

    C_partial = ctx.zeros((M, N), dtype=A.dtype)
    torch.mm(A, B, out=C_partial)

    if variant == "two_shot":
        return two_shot_all_reduce(ctx, output_tensor, C_partial, **kw)
    one_shot_all_reduce(ctx, output_tensor, C_partial, **kw)
    return None


# ==========================================================================
# DEPRECATED — single-kernel two-shot deadlocks by construction.
#
# Both phases share one WG pool with a grid-stride loop. A phase-2 WG waiting
# on tile T blocks until T's owner finishes its phase-1 work -- but that
# producer may be a later iteration of the SAME loop, queued behind the
# spinning consumer on the same CU. A spinning WG never yields, so the
# producer can never run. With 720 tiles and only ~200-300 resident WGs this
# is near-certain, not a rare race.
#
# Measured: 14.76ms (0.01x). Correctness passes only because every WG burns
# its full SPIN_LIMIT and the data has arrived by then -- accidentally correct
# via timeout, not via synchronization.
#
# WG specialization (the fix that worked for XCD GEMM+RS) does NOT apply here:
# there, GEMM WGs and comm WGs were disjoint pools both resident at once. In
# two-shot AR the AG phase needs the RS output of *every* tile, so no static
# split leaves the producer runnable while consumers spin.
#
# Use two_shot_all_reduce(use_barrier=False) instead: 2 launches, ~0.013ms of
# launch overhead, no deadlock. Kept below for reference.
# ==========================================================================

# ==========================================================================
# Fused two-shot: RS + AG in ONE kernel, monotonic-counter sync.
#
# The two-phase version pays a host ctx.barrier() between reduce-scatter and
# all-gather. Measured at ws=8 that barrier is 0.071ms — larger than the
# entire traffic saving two-shot buys over one-shot.
#
# This fuses both phases into a single launch and replaces the host barrier
# with per-tile monotonic counters:
#   - after its RS tile, a WG adds +1 to that tile's counter on EVERY rank
#     (local add is device-scope; remote adds are fire-and-forget sys-scope)
#   - before gathering a tile, a WG polls its LOCAL counter for >= target
#     (device-scope poll — cheap; peers pushed the increments to us)
#
# Counters are monotonic (never reset), so `target = iteration * world_size`
# and there is no flag-reset race between calls. Same pattern that fixed the
# XCD fused GEMM+RS kernel.
# ==========================================================================


@triton.jit
def _fused_two_shot_ar_kernel(
    input_ptr,
    scratch_ptr,
    output_ptr,
    flags_ptr,
    M,
    N,
    M_local,
    stride_in_m,
    stride_in_n,
    stride_s_m,
    stride_s_n,
    stride_out_m,
    stride_out_n,
    heap_bases: tl.tensor,
    flag_target,
    cur_rank: tl.constexpr,
    world_size: tl.constexpr,
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
    NUM_SMS: tl.constexpr,
    NUM_M_TILES_LOCAL: tl.constexpr,
    NUM_N_TILES: tl.constexpr,
    TOTAL_LOCAL_TILES: tl.constexpr,
    NUM_M_TILES: tl.constexpr,
    TOTAL_TILES: tl.constexpr,
    SPIN_LIMIT: tl.constexpr,
):
    pid = tl.program_id(0)
    acc_dtype = tl.float32
    m_offset = cur_rank * NUM_M_TILES_LOCAL

    # ---------------- Phase 1: reduce-scatter (pull) ----------------
    for tile_id in range(pid, TOTAL_LOCAL_TILES, NUM_SMS):
        local_pid_m = tile_id // NUM_N_TILES
        pid_n = tile_id % NUM_N_TILES
        global_pid_m = m_offset + local_pid_m

        rm = global_pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
        rm = tl.max_contiguous(tl.multiple_of(rm, BLOCK_SIZE_M), BLOCK_SIZE_M)
        rn = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
        rn = tl.max_contiguous(tl.multiple_of(rn % N, BLOCK_SIZE_N), BLOCK_SIZE_N)

        off = rm[:, None] * stride_in_m + rn[None, :] * stride_in_n
        base_ptr = input_ptr + off
        s_off = rm[:, None] * stride_s_m + rn[None, :] * stride_s_n
        is_full = (global_pid_m * BLOCK_SIZE_M + BLOCK_SIZE_M <= M) & (
            pid_n * BLOCK_SIZE_N + BLOCK_SIZE_N <= N
        )

        if is_full:
            start_rank = pid % world_size
            acc = iris.load(base_ptr, cur_rank, start_rank, heap_bases,
                            hint=(1, BLOCK_SIZE_N)).to(acc_dtype)
            for i in tl.static_range(1, world_size):
                r = (start_rank + i) % world_size
                acc += iris.load(base_ptr, cur_rank, r, heap_bases,
                                 hint=(1, BLOCK_SIZE_N)).to(acc_dtype)
            tl.store(scratch_ptr + s_off, acc.to(scratch_ptr.type.element_ty),
                     cache_modifier=".wt")
        else:
            mask = (rm[:, None] < M) & (rn[None, :] < N)
            start_rank = pid % world_size
            acc = iris.load(base_ptr, cur_rank, start_rank, heap_bases,
                            mask=mask, hint=(1, BLOCK_SIZE_N)).to(acc_dtype)
            for i in tl.static_range(1, world_size):
                r = (start_rank + i) % world_size
                acc += iris.load(base_ptr, cur_rank, r, heap_bases,
                                 mask=mask, hint=(1, BLOCK_SIZE_N)).to(acc_dtype)
            tl.store(scratch_ptr + s_off, acc.to(scratch_ptr.type.element_ty),
                     mask=mask, cache_modifier=".wt")

        # Signal this shard-tile is reduced. Index by GLOBAL tile position so
        # the gather phase on any rank can address it.
        global_tile_id = global_pid_m * NUM_N_TILES + pid_n
        tl.debug_barrier()
        tl.atomic_add(flags_ptr + global_tile_id, 1, sem="release", scope="gpu")
        for peer in tl.static_range(world_size):
            if peer != cur_rank:
                iris.atomic_add(flags_ptr + global_tile_id, 1,
                                cur_rank, peer, heap_bases,
                                sem="release", scope="sys")

    # ---------------- Phase 2: all-gather (pull) ----------------
    m_tiles_per_rank = NUM_M_TILES_LOCAL
    for tile_id in range(pid, TOTAL_TILES, NUM_SMS):
        pid_m = tile_id // NUM_N_TILES
        pid_n = tile_id % NUM_N_TILES
        owner = pid_m // m_tiles_per_rank

        # Wait until the owning rank has reduced this tile.
        # Each tile is reduced by exactly ONE rank (its owner), which pushes
        # +1 to every rank's copy of that tile's counter. So the counter
        # advances by exactly 1 per iteration -> target = iteration.
        # Poll is LOCAL (device-scope); the owner already pushed to us.
        spins = 0
        done = tl.atomic_add(flags_ptr + tile_id, 0, sem="acquire", scope="gpu")
        while (done < flag_target) and (spins < SPIN_LIMIT):
            done = tl.atomic_add(flags_ptr + tile_id, 0, sem="acquire", scope="gpu")
            spins += 1

        rm = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
        rm = tl.max_contiguous(tl.multiple_of(rm, BLOCK_SIZE_M), BLOCK_SIZE_M)
        rn = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
        rn = tl.max_contiguous(tl.multiple_of(rn % N, BLOCK_SIZE_N), BLOCK_SIZE_N)

        s_off = rm[:, None] * stride_s_m + rn[None, :] * stride_s_n
        o_off = rm[:, None] * stride_out_m + rn[None, :] * stride_out_n
        is_full = (pid_m * BLOCK_SIZE_M + BLOCK_SIZE_M <= M) & (
            pid_n * BLOCK_SIZE_N + BLOCK_SIZE_N <= N
        )

        if is_full:
            v = iris.load(scratch_ptr + s_off, cur_rank, owner, heap_bases,
                          hint=(1, BLOCK_SIZE_N))
            tl.store(output_ptr + o_off, v)
        else:
            mask = (rm[:, None] < M) & (rn[None, :] < N)
            v = iris.load(scratch_ptr + s_off, cur_rank, owner, heap_bases,
                          mask=mask, hint=(1, BLOCK_SIZE_N))
            tl.store(output_ptr + o_off, v, mask=mask)


def fused_two_shot_all_reduce(ctx, output_tensor, input_tensor,
                              workspace=None, spin_limit=100_000_000, **kw):
    """
    Two-shot AllReduce in ONE kernel launch, no host barrier.

    Traffic 2(ws-1)/ws * M * N — matches RCCL's ring — but pull-direction
    throughout, one launch, and device-scope polling for the phase handoff.

    Returns the workspace dict (reuse it across calls; it carries the
    monotonic iteration counter).
    """
    M, N = input_tensor.shape
    rank = ctx.get_rank()
    world_size = ctx.get_num_ranks()
    M_local = M // world_size
    assert output_tensor.shape == (M, N)
    assert M % world_size == 0

    cfg = _get_config(world_size, M_local)
    bm = kw.get("block_m") or cfg["block_m"]
    bn = kw.get("block_n") or cfg["block_n"]
    sms = kw.get("num_sms") or cfg["num_sms"]
    warps = kw.get("num_warps") or cfg["num_warps"]
    assert M_local % bm == 0

    num_m_tiles = M // bm
    num_n_tiles = (N + bn - 1) // bn
    total_tiles = num_m_tiles * num_n_tiles
    num_m_tiles_local = M_local // bm
    total_local_tiles = num_m_tiles_local * num_n_tiles

    if workspace is None:
        workspace = {}
    if workspace.get("shape") != (M, N, bm, bn):
        workspace["scratch"] = ctx.zeros((M, N), dtype=input_tensor.dtype)
        workspace["flags"] = ctx.zeros((total_tiles,), dtype=torch.int32)
        workspace["iteration"] = 0
        workspace["shape"] = (M, N, bm, bn)
        ctx.barrier()

    workspace["iteration"] += 1
    # Each tile's counter is incremented ONCE per iteration (by its owner,
    # broadcast to all ranks), so the target is the iteration number itself
    # -- NOT iteration * world_size.
    flag_target = workspace["iteration"]

    _fused_two_shot_ar_kernel[(sms,)](
        input_tensor, workspace["scratch"], output_tensor, workspace["flags"],
        M, N, M_local,
        input_tensor.stride(0), input_tensor.stride(1),
        workspace["scratch"].stride(0), workspace["scratch"].stride(1),
        output_tensor.stride(0), output_tensor.stride(1),
        ctx.get_heap_bases(), flag_target, rank, world_size,
        bm, bn, sms,
        num_m_tiles_local, num_n_tiles, total_local_tiles,
        num_m_tiles, total_tiles,
        spin_limit,
        num_warps=warps,
    )
    return workspace
