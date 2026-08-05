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


@triton.jit
def _fused_gemm_two_shot_ar_kernel(
    a_ptr,
    b_ptr,
    staged_c_ptr,
    scratch_ptr,
    output_ptr,
    gemm_flags_ptr,
    rs_flags_ptr,
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
    SPIN_LIMIT: tl.constexpr,
):
    pid = tl.program_id(0)

    num_m_tiles = tl.cdiv(M, BLOCK_SIZE_M)
    num_n_tiles = tl.cdiv(N, BLOCK_SIZE_N)
    total_tiles = num_m_tiles * num_n_tiles
    m_tiles_per_rank = num_m_tiles // world_size
    acc_dtype = tl.float32

    if pid < NUM_GEMM_SMS:
        # ------------------------------------------------------------------
        # POOL G -- GEMM. Never waits on anything, so the pipeline always
        # has a runnable producer.
        # ------------------------------------------------------------------
        for seq in range(pid, total_tiles, NUM_GEMM_SMS):
            # Shard-interleaved emission order. In row-major tile order a
            # rank's M-shard is a CONTIGUOUS block -- rank ws-1 owns the last
            # 1/ws of all tiles, so its RS pool cannot start until the GEMM
            # has emitted (ws-1)/ws of everything. That serialises the
            # pipeline for every rank but rank 0.
            #
            # Emitting shard (seq % ws) slot (seq // ws) instead means after
            # the first ws tiles EVERY rank's RS pool has work, so all ranks
            # overlap from the start.
            # The preamble requires num_m_tiles % ws == 0, so
            # total_tiles == ws * tiles_per_shard and this is a bijection.
            shard = seq % world_size
            slot = seq // world_size
            pid_m = shard * m_tiles_per_rank + slot // num_n_tiles
            pid_n = slot % num_n_tiles
            tile_id = pid_m * num_n_tiles + pid_n

            rm = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
            rm = tl.max_contiguous(tl.multiple_of(rm, BLOCK_SIZE_M), BLOCK_SIZE_M)
            rn = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
            rn = tl.max_contiguous(tl.multiple_of(rn % N, BLOCK_SIZE_N), BLOCK_SIZE_N)
            rk = tl.arange(0, BLOCK_SIZE_K)

            a_ptrs = a_ptr + rm[:, None] * stride_am + rk[None, :] * stride_ak
            b_ptrs = b_ptr + rk[:, None] * stride_bk + rn[None, :] * stride_bn

            acc = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=acc_dtype)
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

            # Publish: every rank's RS pool needs to know THIS rank produced
            # tile_id. One int per tile per peer -- negligible vs the data.
            tl.debug_barrier()
            fslot = gemm_flags_ptr + tile_id + tl.arange(0, 1)
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

            # Wait until every rank has produced this tile.
            spins = 0
            gslot = gemm_flags_ptr + tile_id + tl.arange(0, 1)
            zero = tl.zeros((1,), dtype=tl.int32)
            done = tl.min(tl.atomic_add(gslot, zero, sem="acquire", scope="sys"))
            while (done < gemm_target) and (spins < SPIN_LIMIT):
                done = tl.min(tl.atomic_add(gslot, zero, sem="acquire", scope="sys"))
                spins += 1

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

            # Publish to every rank's AG pool: owner's shard tile is reduced.
            tl.debug_barrier()
            fslot = rs_flags_ptr + tile_id + tl.arange(0, 1)
            one = tl.zeros((1,), dtype=tl.int32) + 1
            for r in tl.static_range(world_size):
                iris.atomic_add(fslot, one, cur_rank, r, heap_bases,
                                sem="release", scope="sys")

    else:
        # ------------------------------------------------------------------
        # POOL A -- all-gather. Waits only on pool R. Pulls each tile from
        # whichever rank owns it after the reduce-scatter.
        # ------------------------------------------------------------------
        local_pid = pid - NUM_GEMM_SMS - NUM_RS_SMS

        for tile_id in range(local_pid, total_tiles, NUM_AG_SMS):
            pid_m = tile_id // num_n_tiles
            pid_n = tile_id % num_n_tiles
            owner = pid_m // m_tiles_per_rank

            spins = 0
            rslot = rs_flags_ptr + tile_id + tl.arange(0, 1)
            zero = tl.zeros((1,), dtype=tl.int32)
            done = tl.min(tl.atomic_add(rslot, zero, sem="acquire", scope="sys"))
            while (done < rs_target) and (spins < SPIN_LIMIT):
                done = tl.min(tl.atomic_add(rslot, zero, sem="acquire", scope="sys"))
                spins += 1

            rm = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
            rm = tl.max_contiguous(tl.multiple_of(rm, BLOCK_SIZE_M), BLOCK_SIZE_M)
            rn = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
            rn = tl.max_contiguous(tl.multiple_of(rn % N, BLOCK_SIZE_N), BLOCK_SIZE_N)

            off = rm[:, None] * stride_cm + rn[None, :] * stride_cn
            mask = (rm[:, None] < M) & (rn[None, :] < N)

            v = iris.load(scratch_ptr + off, cur_rank, owner, heap_bases,
                          mask=mask, hint=(1, BLOCK_SIZE_N))
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
        "gemm_flags": ctx.zeros((total_tiles,), device="cuda", dtype=torch.int32),
        "rs_flags": ctx.zeros((total_tiles,), device="cuda", dtype=torch.int32),
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
    spin_limit: int = 1_000_000,
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

    workspace["iteration"] += 1
    it = workspace["iteration"]
    # Every rank increments each tile's gemm counter exactly once per
    # iteration; each tile's rs counter is incremented once by its owner.
    gemm_target = it * world_size
    rs_target = it

    staged_c = workspace["staged_c"]
    scratch = workspace["scratch"]

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
        SPIN_LIMIT=spin_limit,
        num_warps=num_warps,
        matrix_instr_nonkdim=mfma,
    )
    return workspace
