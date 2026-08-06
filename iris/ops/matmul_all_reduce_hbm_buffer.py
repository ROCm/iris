# SPDX-License-Identifier: MIT
# Copyright (c) 2025 Advanced Micro Devices, Inc. All rights reserved.
"""Fused GEMM + two-shot AllReduce with HBM staging and 3-way WG specialization.

WHY THIS EXISTS
---------------
Every GEMM+AR variant measured so far moves ``ws * M * N`` bytes per rank,
because they all fuse a *one-shot* AllReduce. That traffic is why the
two-kernel one-shot loses to torch once M >= 512: at ws=8 it moves 4.6x
RCCL's bytes, and no amount of overlap hides a 4.6x traffic penalty.

Two-shot AR moves ``2(ws-1)/ws * M * N`` -- 1.75 M*N at ws=8, a 4.6x
reduction. The reason nobody has fused it is that a naive single-kernel
two-shot **deadlocks**: the all-gather phase needs the reduce-scatter output
of every rank, so with one WG pool a phase-2 WG can block the phase-1 WG that
would unblock it (all-to-all dependency, see ``fused_two_shot_all_reduce``).

HBM staging breaks that cycle. With three *disjoint* WG pools and a staging
buffer between each stage, the dependency graph becomes a linear pipeline:

    GEMM pool  --gemm_flags-->  RS pool  --rs_flags-->  AG pool
      (never waits)             (waits on GEMM)         (waits on RS)

No pool can block its own producer, so there is no cycle and no deadlock.
This is the same producer->consumer shape that made GEMM+ReduceScatter work,
extended by one stage.

    staged_c [M,N] symmetric   GEMM partials, peer-visible
    scratch  [M,N] symmetric   RS output; only this rank's M-shard is valid
    output   [M,N] local       final all-reduced result

Flags are monotonic counters (never reset), so a stale flag from iteration
i-1 can never satisfy iteration i. ``gemm_target = iteration * world_size``
(every rank contributes to every tile); ``rs_target = iteration`` (each tile's
shard has exactly one owner).

TRAFFIC (per rank, ws=8)
    one-shot fused / two-kernel :  8.00 * M*N
    this kernel                 :  1.88 * M*N     (RS pulls M*N, AG pulls 7/8 M*N)

COST
    Three-way CU split. The GEMM gives up CUs to two comm pools instead of
    one, so this only pays off when the traffic saving exceeds the GEMM
    slowdown -- i.e. at large M, which is exactly where one-shot loses.
"""

from typing import Optional

import torch
import triton
import triton.language as tl

import iris

# iris.device.utils.read_realtime gates on importing BOTH memrealtime and smid,
# and smid is absent from triton 3.5.1+rocm7.2 -- so the whole module falls back
# to a static_assert stub and all iris tracing silently dies. We only need the
# timestamp, so import it directly.
try:
    from triton.language.extra.hip import memrealtime as read_realtime

    _HAS_TIMER = True
except ImportError:  # pragma: no cover - non-HIP builds
    from iris.device.utils import read_realtime

    _HAS_TIMER = False

# Flags are spaced this many int32s apart (32 * 4B = 128B, one cache line).
FLAG_STRIDE = 32


@triton.jit
def _fused_gemm_two_shot_ar_kernel(
    a_ptr,
    b_ptr,
    staged_c_ptr,
    scratch_ptr,
    output_ptr,
    gemm_flags_ptr,
    rs_flags_ptr,
    ag_next_ptr,
    own_next_ptr,
    rs_done_ptr,
    ts_gemm_beg,
    ts_gemm_end,
    ts_rs_beg,
    ts_rs_ready,
    ts_rs_end,
    ts_ag_beg,
    ts_ag_ready,
    ts_ag_end,
    M,
    N,
    K,
    M_local,
    stride_am,
    stride_ak,
    stride_bk,
    stride_bn,
    stride_cm,
    stride_cn,
    heap_bases: tl.tensor,
    gemm_target,
    rs_target,
    cur_rank: tl.constexpr,
    world_size: tl.constexpr,
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr,
    NUM_GEMM_SMS: tl.constexpr,
    NUM_RS_SMS: tl.constexpr,
    NUM_AG_SMS: tl.constexpr,
    TILES_PER_FLAG: tl.constexpr,
    FLAG_STRIDE: tl.constexpr,
    SKIP_GEMM: tl.constexpr,
    SPIN_LIMIT: tl.constexpr,
    TRACE: tl.constexpr,
):
    pid = tl.program_id(0)

    num_m_tiles = tl.cdiv(M, BLOCK_SIZE_M)
    num_n_tiles = tl.cdiv(N, BLOCK_SIZE_N)
    total_tiles = num_m_tiles * num_n_tiles
    m_tiles_per_rank = num_m_tiles // world_size
    total_groups = total_tiles // TILES_PER_FLAG
    acc_dtype = tl.float32

    # Pools are static during the contended phase and elastic in the TAIL.
    #
    # The kernel has two regimes, and treating it as one is what made the
    # first elastic attempt 1.8x slower:
    #   0 - 127us   GEMM + RS + AG all live, fabric saturated. Extra comm
    #               work-groups add contention and zero bandwidth here.
    #   127 - 214us RS has drained. Only AG is left, ~8 tiles in flight on 16
    #               WGs, and the fabric is IDLE. 41% of the kernel.
    #
    # Retired GEMM WGs previously joined at ~65us -- squarely inside the
    # contended phase, which is why it lost. Gate them on RS draining so they
    # arrive in the tail, where AG genuinely is work-starved (AG's whole
    # 322-tile workload is 30.4us standalone at 256 WGs, against a 214us span
    # in the fused kernel).
    #
    # Tail workers take from an atomic counter rather than a static stride: a
    # static partition would bind tiles to workers that have not arrived, and
    # stall the 16-WG AG pool waiting for them. The counter was measured slower
    # when 224 workers hammered it from t=0; here only 16 touch it until the
    # tail.
    #
    # The trace is suggestive: the GEMM pool finishes at 92us in a 277us kernel,
    # so 192 CUs -- 75% of the machine -- look idle for the back 185us. Retiring
    # them into the all-gather should have been free throughput. Measured, at
    # M=2048, it was 1.8x SLOWER (0.1958 static -> 0.3584 elastic), and the same
    # with an atomic work counter (0.3570).
    #
    # Why: RS is the gate, not AG. RS pulls from ws peers and AG from one, and
    # the fabric tops out around 85% of line with both pools active. Adding 192
    # more work-groups to AG does not raise that ceiling -- it takes bandwidth
    # from RS, which every AG tile is waiting on. The extra workers starve the
    # producer they depend on.
    #
    # So the idle CUs are not recoverable by handing them to AG. Whatever the 3x
    # over the fabric floor is, it is not simply unused compute.

    if pid < NUM_GEMM_SMS:
        # ------------------------------------------------------------------
        # POOL G -- GEMM. Never waits on anything, so the pipeline always
        # has a runnable producer.
        # ------------------------------------------------------------------
        # One WG owns a whole flag group, so it can signal once for the group
        # instead of once per tile. Flag traffic is the dominant cost here:
        # each signal is world_size remote system-scope atomics with a release
        # fence, and at TILES_PER_FLAG=1 that was ~3k fences per rank.
        for grp in range(pid, total_groups, NUM_GEMM_SMS):
            # Shard-interleaved group order. In row-major tile order a rank's
            # M-shard is a CONTIGUOUS block -- rank ws-1 owns the last 1/ws of
            # all tiles, so its RS pool cannot start until the GEMM has
            # emitted (ws-1)/ws of everything. Interleaving by shard means
            # after the first ws groups EVERY rank's RS pool has work.
            shard = grp % world_size
            gslot = grp // world_size
            base_seq = gslot * TILES_PER_FLAG

            for j in range(0, TILES_PER_FLAG):
                slot = base_seq + j
                pid_m = shard * m_tiles_per_rank + slot // num_n_tiles
                pid_n = slot % num_n_tiles
                tile_id = pid_m * num_n_tiles + pid_n
                if TRACE:
                    tl.atomic_min(ts_gemm_beg + tile_id, read_realtime())

                rm = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
                rm = tl.max_contiguous(tl.multiple_of(rm, BLOCK_SIZE_M), BLOCK_SIZE_M)
                rn = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
                rn = tl.max_contiguous(tl.multiple_of(rn % N, BLOCK_SIZE_N), BLOCK_SIZE_N)
                rk = tl.arange(0, BLOCK_SIZE_K)

                a_ptrs = a_ptr + rm[:, None] * stride_am + rk[None, :] * stride_ak
                b_ptrs = b_ptr + rk[:, None] * stride_bk + rn[None, :] * stride_bn

                acc = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=acc_dtype)
                if not SKIP_GEMM:
                    # SKIP_GEMM leaves the store and the flag publish intact and
                    # drops only the math, so a comm-only measurement keeps this
                    # kernel's own barriers, launch count and flag protocol.
                    # Scoring the fused row against a standalone kernel instead
                    # launders that difference into "overlap".
                    for k in range(0, tl.cdiv(K, BLOCK_SIZE_K)):
                        k_rem = K - k * BLOCK_SIZE_K
                        a = tl.load(a_ptrs, mask=rk[None, :] < k_rem, other=0.0)
                        b = tl.load(b_ptrs, mask=rk[:, None] < k_rem, other=0.0)
                        acc += tl.dot(a, b)
                        a_ptrs += BLOCK_SIZE_K * stride_ak
                        b_ptrs += BLOCK_SIZE_K * stride_bk

                c_off = rm[:, None] * stride_cm + rn[None, :] * stride_cn
                c_mask = (rm[:, None] < M) & (rn[None, :] < N)
                # .wt so the partial is visible to peers pulling it in the RS phase
                tl.store(staged_c_ptr + c_off, acc.to(staged_c_ptr.type.element_ty),
                         mask=c_mask, cache_modifier=".wt")
                if TRACE:
                    tl.atomic_max(ts_gemm_end + tile_id, read_realtime())

            # One release fence + world_size remote atomics for the WHOLE
            # group, not per tile. This is the knob that mattered most in the
            # all-gather HBM-buffer op (52% of its perf range).
            tl.debug_barrier()
            fslot = gemm_flags_ptr + grp * FLAG_STRIDE + tl.arange(0, 1)
            one = tl.zeros((1,), dtype=tl.int32) + 1
            for r in tl.static_range(world_size):
                iris.atomic_add(fslot, one, cur_rank, r, heap_bases,
                                sem="release", scope="sys")

    elif pid < NUM_GEMM_SMS + NUM_RS_SMS:
        # ------------------------------------------------------------------
        # POOL R -- reduce-scatter. Waits only on pool G (which never waits).
        # Reduces this rank's own M-shard by pulling that shard from everyone.
        # ------------------------------------------------------------------
        local_pid = pid - NUM_GEMM_SMS
        shard_tiles = m_tiles_per_rank * num_n_tiles
        m_offset = cur_rank * m_tiles_per_rank

        for t in range(local_pid, shard_tiles, NUM_RS_SMS):
            local_pid_m = t // num_n_tiles
            pid_n = t % num_n_tiles
            global_pid_m = m_offset + local_pid_m
            tile_id = global_pid_m * num_n_tiles + pid_n

            # Wait until every rank has produced the GROUP containing this
            # tile. Group index matches the GEMM pool's interleaved order:
            # this rank's shard slot t lives in group (t / TPF) * ws + rank.
            grp = (t // TILES_PER_FLAG) * world_size + cur_rank
            if TRACE:
                tl.atomic_min(ts_rs_beg + tile_id, read_realtime())
            spins = 0
            # MUST be an atomic RMW, not tl.load(volatile=True). volatile does
            # NOT block LICM on the AMD backend: the compiler hoists the load
            # out of this loop, every WG spins on a stale register, burns
            # SPIN_LIMIT and proceeds with garbage. It is silently WRONG, not
            # slow -- and it is faster precisely because it stops waiting.
            gslot = gemm_flags_ptr + grp * FLAG_STRIDE + tl.arange(0, 1)
            zero = tl.zeros((1,), dtype=tl.int32)
            done = tl.min(tl.atomic_add(gslot, zero, sem="acquire", scope="sys"))
            while (done < gemm_target) and (spins < SPIN_LIMIT):
                done = tl.min(tl.atomic_add(gslot, zero, sem="acquire", scope="sys"))
                spins += 1

            if TRACE:
                tl.atomic_max(ts_rs_ready + tile_id, read_realtime())

            rm = global_pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
            rm = tl.max_contiguous(tl.multiple_of(rm, BLOCK_SIZE_M), BLOCK_SIZE_M)
            rn = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
            rn = tl.max_contiguous(tl.multiple_of(rn % N, BLOCK_SIZE_N), BLOCK_SIZE_N)

            off = rm[:, None] * stride_cm + rn[None, :] * stride_cn
            mask = (rm[:, None] < M) & (rn[None, :] < N)
            base = staged_c_ptr + off

            # Rotate the starting peer per WG so we don't all hammer rank 0.
            start = local_pid % world_size
            acc = iris.load(base, cur_rank, start, heap_bases, mask=mask,
                            hint=(1, BLOCK_SIZE_N)).to(acc_dtype)
            for i in tl.static_range(1, world_size):
                r = (start + i) % world_size
                acc += iris.load(base, cur_rank, r, heap_bases, mask=mask,
                                 hint=(1, BLOCK_SIZE_N)).to(acc_dtype)

            tl.store(scratch_ptr + off, acc.to(scratch_ptr.type.element_ty),
                     mask=mask, cache_modifier=".wt")

            if TRACE:
                tl.atomic_max(ts_rs_end + tile_id, read_realtime())

            # Local drain counter: how many of this rank's RS tiles are done.
            # Device scope, one per tile (46 at M=2048) -- the tail gate reads
            # it, nothing remote does.
            tl.atomic_add(rs_done_ptr + tl.arange(0, 1),
                          tl.zeros((1,), dtype=tl.int32) + 1,
                          sem="release", scope="gpu")

            # Publish to every rank's AG pool. Per tile, not per group: an RS
            # WG grid-strides across groups so it never owns a whole one.
            tl.debug_barrier()
            fslot = rs_flags_ptr + tile_id * FLAG_STRIDE + tl.arange(0, 1)
            one = tl.zeros((1,), dtype=tl.int32) + 1
            for r in tl.static_range(world_size):
                iris.atomic_add(fslot, one, cur_rank, r, heap_bases,
                                sem="release", scope="sys")

    if not ((pid >= NUM_GEMM_SMS) and (pid < NUM_GEMM_SMS + NUM_RS_SMS)):
        # ------------------------------------------------------------------
        # POOL A -- all-gather, plus GEMM work-groups that have retired AND
        # waited for RS to drain.
        # ------------------------------------------------------------------
        if pid < NUM_GEMM_SMS:
            # Tail gate. Do not help until this rank's RS pool has finished;
            # before that the fabric is saturated and an extra puller is pure
            # contention.
            shard_tiles_g = m_tiles_per_rank * num_n_tiles
            tgate = 0
            zg = tl.zeros((1,), dtype=tl.int32)
            dn = tl.min(tl.atomic_add(rs_done_ptr + tl.arange(0, 1), zg,
                                      sem="acquire", scope="gpu"))
            while (dn < shard_tiles_g) and (tgate < SPIN_LIMIT):
                dn = tl.min(tl.atomic_add(rs_done_ptr + tl.arange(0, 1), zg,
                                          sem="acquire", scope="gpu"))
                tgate += 1

        # Peer tiles only, indexed so consecutive tiles fan across peers.
        #
        # Row-major order (owner = pid_m // m_tiles_per_rank) makes a cohort of
        # grid-strided WGs pull from the SAME peer -- one XGMI link hot, ws-1
        # idle. `owner = seq % ws` does not fix it either: NUM_AG_SMS is
        # typically a multiple of ws, so gcd(stride, ws) == ws and seq % ws is
        # CONSTANT per WG. It also wastes 1/ws of the iterations re-reading
        # this rank's own shard.
        #
        # The modulus has to be world_size-1: peers excluding self. At ws=8
        # that is 7, which is prime, so gcd(NUM_AG_SMS, 7) == 1 for any
        # workgroup count that is not a multiple of 7 -- every WG then walks
        # all 7 peers. Worth 106 -> 316 GB/s in a standalone all-gather.
        n_peers: tl.constexpr = world_size - 1
        peer_tiles = m_tiles_per_rank * num_n_tiles * n_peers

        one_i = tl.zeros((1,), dtype=tl.int32) + 1
        ctr = ag_next_ptr + tl.arange(0, 1)
        seq = tl.min(tl.atomic_add(ctr, one_i, sem="relaxed", scope="gpu"))
        while seq < peer_tiles:
            pk = seq % n_peers
            slot = seq // n_peers
            owner = (cur_rank + 1 + pk) % world_size
            pid_m = owner * m_tiles_per_rank + slot // num_n_tiles
            pid_n = slot % num_n_tiles
            tile_id = pid_m * num_n_tiles + pid_n

            if TRACE:
                tl.atomic_min(ts_ag_beg + tile_id, read_realtime())
            spins = 0
            rslot = rs_flags_ptr + tile_id * FLAG_STRIDE + tl.arange(0, 1)
            zero2 = tl.zeros((1,), dtype=tl.int32)
            done = tl.min(tl.atomic_add(rslot, zero2, sem="acquire", scope="sys"))
            while (done < rs_target) and (spins < SPIN_LIMIT):
                done = tl.min(tl.atomic_add(rslot, zero2, sem="acquire", scope="sys"))
                spins += 1
            if TRACE:
                tl.atomic_max(ts_ag_ready + tile_id, read_realtime())

            rm = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
            rm = tl.max_contiguous(tl.multiple_of(rm, BLOCK_SIZE_M), BLOCK_SIZE_M)
            rn = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
            rn = tl.max_contiguous(tl.multiple_of(rn % N, BLOCK_SIZE_N), BLOCK_SIZE_N)

            off = rm[:, None] * stride_cm + rn[None, :] * stride_cn
            mask = (rm[:, None] < M) & (rn[None, :] < N)

            v = iris.load(scratch_ptr + off, cur_rank, owner, heap_bases,
                          mask=mask, hint=(1, BLOCK_SIZE_N))
            tl.store(output_ptr + off, v, mask=mask)
            if TRACE:
                tl.atomic_max(ts_ag_end + tile_id, read_realtime())
            seq = tl.min(tl.atomic_add(ctr, one_i, sem="relaxed", scope="gpu"))

        # This rank's own shard: already reduced into scratch by our RS pool,
        # so it is a local copy with no peer read and no flag wait beyond the
        # RS ordering the pool already observed.
        own_tiles = m_tiles_per_rank * num_n_tiles
        if pid < NUM_GEMM_SMS:
            own_start = own_tiles
        else:
            own_start = pid - NUM_GEMM_SMS - NUM_RS_SMS
        for slot in range(own_start, own_tiles, NUM_AG_SMS):
            pid_m = cur_rank * m_tiles_per_rank + slot // num_n_tiles
            pid_n = slot % num_n_tiles
            tile_id = pid_m * num_n_tiles + pid_n

            spins = 0
            rslot = rs_flags_ptr + tile_id * FLAG_STRIDE + tl.arange(0, 1)
            zero3 = tl.zeros((1,), dtype=tl.int32)
            done = tl.min(tl.atomic_add(rslot, zero3, sem="acquire", scope="sys"))
            while (done < rs_target) and (spins < SPIN_LIMIT):
                done = tl.min(tl.atomic_add(rslot, zero3, sem="acquire", scope="sys"))
                spins += 1

            rm = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
            rm = tl.max_contiguous(tl.multiple_of(rm, BLOCK_SIZE_M), BLOCK_SIZE_M)
            rn = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
            rn = tl.max_contiguous(tl.multiple_of(rn % N, BLOCK_SIZE_N), BLOCK_SIZE_N)
            off = rm[:, None] * stride_cm + rn[None, :] * stride_cn
            mask = (rm[:, None] < M) & (rn[None, :] < N)
            v = tl.load(scratch_ptr + off, mask=mask)
            tl.store(output_ptr + off, v, mask=mask)


def matmul_all_reduce_hbm_buffer_preamble(
    ctx,
    M: int,
    N: int,
    dtype: torch.dtype,
    block_m: int = 128,
    block_n: int = 128,
):
    """Allocate the staging buffers and monotonic flag counters."""
    world_size = ctx.get_num_ranks()
    num_m_tiles = triton.cdiv(M, block_m)
    num_n_tiles = triton.cdiv(N, block_n)
    total_tiles = num_m_tiles * num_n_tiles

    if num_m_tiles % world_size != 0:
        raise ValueError(
            f"M-tiles ({num_m_tiles}) must divide evenly across {world_size} ranks; "
            f"pick a block_m that divides M/{world_size}"
        )

    return {
        "staged_c": ctx.zeros((M, N), device="cuda", dtype=dtype),
        "scratch": ctx.zeros((M, N), device="cuda", dtype=dtype),
        # Flags live in the symmetric heap so iris.atomic_add can reach peers.
        # FLAG_STRIDE int32s apart so each flag owns a cache line. Adjacent
        # flags meant 16 unrelated tiles shared one 64B line and their
        # spinners fought each other for it.
        "gemm_flags": ctx.zeros((total_tiles * FLAG_STRIDE,), device="cuda",
                                dtype=torch.int32),
        "rs_flags": ctx.zeros((total_tiles * FLAG_STRIDE,), device="cuda",
                              dtype=torch.int32),
        # Dynamic work counters. Unlike the flags these are NOT monotonic --
        # they index within a single launch, so they must be reset every call.
        "ag_next": ctx.zeros((1,), device="cuda", dtype=torch.int32),
        "own_next": ctx.zeros((1,), device="cuda", dtype=torch.int32),
        "rs_done": ctx.zeros((1,), device="cuda", dtype=torch.int32),
        "iteration": 0,
        "block_m": block_m,
        "block_n": block_n,
    }


def matmul_all_reduce_hbm_buffer(
    ctx,
    output_tensor,
    A,
    B,
    workspace: Optional[dict] = None,
    block_m: int = 128,
    block_n: int = 128,
    block_k: int = 64,
    num_gemm_sms: int = 128,
    num_rs_sms: int = 64,
    num_ag_sms: int = 64,
    num_warps: int = 8,
    mfma: int = 32,
    tiles_per_flag: int = 1,
    flag_stride: int = 32,
    skip_gemm: bool = False,
    spin_limit: int = 1_000_000,
    trace: bool = False,
):
    """Fused GEMM + two-shot AllReduce, three co-resident WG pools.

    ``A`` is [M, K_local] and ``B`` is [K_local, N] -- the row-parallel shard.
    ``A`` must live in the symmetric heap. Returns the workspace so the caller
    can reuse it (the monotonic counters live there).
    """
    M, _ = A.shape
    _, N = B.shape
    K = A.shape[1]
    world_size = ctx.get_num_ranks()
    rank = ctx.get_rank()

    if workspace is None:
        workspace = matmul_all_reduce_hbm_buffer_preamble(
            ctx, M, N, output_tensor.dtype, block_m, block_n
        )
        ctx.barrier()

    # Reset per launch: these count tiles within one call, not across calls.
    workspace["ag_next"].zero_()
    workspace["own_next"].zero_()
    workspace["rs_done"].zero_()

    workspace["iteration"] += 1
    it = workspace["iteration"]
    # Every rank increments each tile's gemm counter exactly once per
    # iteration; each tile's rs counter is incremented once by its owner.
    gemm_target = it * world_size
    rs_target = it

    staged_c = workspace["staged_c"]
    scratch = workspace["scratch"]

    # Per-tile timestamps. Six phase marks per tile let us separate "the pool
    # was waiting on its producer" from "the pool was doing work", which is
    # the difference between a serialization problem and a bandwidth problem.
    if trace:
        num_tiles = triton.cdiv(M, block_m) * triton.cdiv(N, block_n)
        dev = staged_c.device
        ts = workspace.get("trace")
        if ts is None or ts["gemm_beg"].numel() != num_tiles:
            ts = {k: torch.zeros(num_tiles, dtype=torch.int64, device=dev)
                  for k in ("gemm_beg", "gemm_end", "rs_beg", "rs_ready",
                            "rs_end", "ag_beg", "ag_ready", "ag_end")}
            workspace["trace"] = ts
        # atomic_min needs a high initial value; atomic_max needs a low one
        for k, v in ts.items():
            v.fill_(torch.iinfo(torch.int64).max if k.endswith("_beg") else 0)
        tb = [ts["gemm_beg"], ts["gemm_end"], ts["rs_beg"], ts["rs_ready"],
              ts["rs_end"], ts["ag_beg"], ts["ag_ready"], ts["ag_end"]]
    else:
        tb = [staged_c] * 8  # unused; TRACE=False compiles the stores away

    # Each GEMM WG owns a whole flag group, and a group lives entirely inside
    # one rank's shard, so the shard's tile count must divide by the group size.
    num_m_tiles = triton.cdiv(M, block_m)
    tiles_per_shard = (num_m_tiles // world_size) * triton.cdiv(N, block_n)
    if tiles_per_shard % tiles_per_flag != 0:
        raise ValueError(
            f"tiles_per_flag ({tiles_per_flag}) must divide the per-shard tile "
            f"count ({tiles_per_shard})"
        )

    total_sms = num_gemm_sms + num_rs_sms + num_ag_sms
    grid = (total_sms,)

    _fused_gemm_two_shot_ar_kernel[grid](
        A,
        B,
        staged_c,
        scratch,
        output_tensor,
        workspace["gemm_flags"],
        workspace["rs_flags"],
        workspace["ag_next"],
        workspace["own_next"],
        workspace["rs_done"],
        tb[0], tb[1], tb[2], tb[3], tb[4], tb[5], tb[6], tb[7],
        M,
        N,
        K,
        M // world_size,
        A.stride(0),
        A.stride(1),
        B.stride(0),
        B.stride(1),
        staged_c.stride(0),
        staged_c.stride(1),
        ctx.get_heap_bases(),
        gemm_target,
        rs_target,
        cur_rank=rank,
        world_size=world_size,
        BLOCK_SIZE_M=block_m,
        BLOCK_SIZE_N=block_n,
        BLOCK_SIZE_K=block_k,
        NUM_GEMM_SMS=num_gemm_sms,
        NUM_RS_SMS=num_rs_sms,
        NUM_AG_SMS=num_ag_sms,
        TILES_PER_FLAG=tiles_per_flag,
        FLAG_STRIDE=flag_stride,
        SKIP_GEMM=skip_gemm,
        SPIN_LIMIT=spin_limit,
        TRACE=trace,
        num_warps=num_warps,
        matrix_instr_nonkdim=mfma,
    )
    return workspace
