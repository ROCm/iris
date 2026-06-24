# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
"""Persistent per-rank kernels for the multi-GPU GPT-OSS decode.

Each rank launches ONE kernel per token that loops all L layers internally and
rendezvouses attention<->MoE with device-side iris flags (sys-scope atomics),
instead of a host shmem.barrier() between every phase. This removes the
108-host-syncs-per-token (3 barriers x 36 layers) overhead of the correctness-first
driver; the 4 experts still run fully in parallel across the MoE ranks.

Flag protocol (host zeroes all flags once before the token launch):
  in_flag[L]  (one per MoE rank): the attention rank, after delivering layer L's
              expert input to that MoE rank, does a remote atomic_xchg(in_flag[L]=1,
              release, sys). The MoE rank spins on its LOCAL in_flag[L] (acquire).
  out_flag[L*TOPK] (on the attention rank): each MoE rank, after delivering its
              result for layer L, does a remote atomic_xchg(out_flag[L*TOPK+slot]=1,
              release, sys). The attention rank spins on its LOCAL out_flag (acquire).

Within a rank, the NWG programs synchronize with the local grid barrier (_barrier);
only program 0 touches the cross-rank flags, gated by local barriers so the data
stores are complete/visible before a flag is raised and before peers read results."""

import triton
import triton.language as tl
import iris

from common.barrier import _barrier
from common.gemv_bf16 import _gemv_bf16_tiled, _gemv_bf16_rmsnorm, _gemv_bf16_resid_rmsnorm
from common.gemv_fp8 import _gemv_fp8_tiled, _gemv_fp8_rmsnorm, _gemv_fp8_resid_rmsnorm
from common.gemv_fp4 import _gemv_fp4_scaled
from common.quant import _quant_norm_fp8
from common.swiglu import _swiglu_quant_fp8
from common.attention import _rope_kv_append, _flash_decode_head
from common.router import _topk_softmax


@triton.jit
def attn_persistent_kernel(
    norm_attn_p, norm_moe_p,
    wq_p, bq_p, wk_p, bk_p, wv_p, bv_p, wo_p, bo_p,
    sinks_p, router_w_p, router_b_p,
    wq_s_p, wk_s_p, wv_s_p, wo_s_p, router_w_s_p,
    final_norm_p, lm_head_p,
    # runtime (local heap)
    x_p, q_p, k_p, v_p, kcache_p, vcache_p, attn_p, o_p,
    logits_p, ids_p, gw_p, nfp8_p, nfp8_scl_p, res_p,
    amax_v_p, amax_i_p, next_tok_p,
    # remote inboxes (same symmetric offsets on MoE ranks)
    r_nfp8_p, r_nfp8_scl_p, r_meta_p, r_gw_p, r_in_flag_p,
    out_flag_p,  # local: raised by MoE ranks
    cos_p, sin_p, bar_p,
    pos, scale, eps,
    heap_bases,
    NWG: tl.constexpr, L: tl.constexpr,
    H: tl.constexpr, q_dim: tl.constexpr, kv_dim: tl.constexpr,
    NH: tl.constexpr, NKV: tl.constexpr, DH: tl.constexpr,
    E: tl.constexpr, TOPK: tl.constexpr, V: tl.constexpr, SLIDING: tl.constexpr,
    GU_NB: tl.constexpr, max_seq: tl.constexpr,
    BLOCK_K: tl.constexpr, BLOCK_M: tl.constexpr, BLOCK_M_LM: tl.constexpr, NORMK: tl.constexpr,
    BLOCK_T: tl.constexpr, NSTAGES: tl.constexpr,
    FP8_QKV: tl.constexpr, FP8_O: tl.constexpr, FP8_ROUTER: tl.constexpr, MXFP8_BLK: tl.constexpr,
    ATTN_RANK: tl.constexpr, BLOCK: tl.constexpr,
):
    pid = tl.program_id(0)
    HALF: tl.constexpr = DH // 2
    GROUP: tl.constexpr = NH // NKV
    NSB_H: tl.constexpr = (H + MXFP8_BLK - 1) // MXFP8_BLK if MXFP8_BLK < H else 1
    NSB_Q: tl.constexpr = (q_dim + MXFP8_BLK - 1) // MXFP8_BLK if MXFP8_BLK < q_dim else 1
    nblk: tl.constexpr = (H + BLOCK - 1) // BLOCK  # blocks to cover H in the scatter/accumulate
    bc = 0
    for layer in range(L):
        na = norm_attn_p + layer * H
        nm = norm_moe_p + layer * H
        wq = wq_p + layer * q_dim * H
        wk = wk_p + layer * kv_dim * H
        wv = wv_p + layer * kv_dim * H
        wo = wo_p + layer * H * q_dim
        bq = bq_p + layer * q_dim
        bk = bk_p + layer * kv_dim
        bv = bv_p + layer * kv_dim
        bo = bo_p + layer * H
        sinks = sinks_p + layer * NH
        rw = router_w_p + layer * E * H
        rb = router_b_p + layer * E
        wq_s = wq_s_p + layer * q_dim * NSB_H
        wk_s = wk_s_p + layer * kv_dim * NSB_H
        wv_s = wv_s_p + layer * kv_dim * NSB_H
        wo_s = wo_s_p + layer * H * NSB_Q
        rw_s = router_w_s_p + layer * E * NSB_H
        kcache = kcache_p + layer * max_seq * kv_dim
        vcache = vcache_p + layer * max_seq * kv_dim

        # ---- QKV (fused RMSNorm) ----
        if FP8_QKV:
            _gemv_fp8_rmsnorm(wq, wq_s, x_p, na, q_p, True, bq, q_dim, H, pid, eps, BLOCK_M, BLOCK_K, NORMK, NSTAGES, MXFP8_BLK)
            _gemv_fp8_rmsnorm(wk, wk_s, x_p, na, k_p, True, bk, kv_dim, H, pid, eps, BLOCK_M, BLOCK_K, NORMK, NSTAGES, MXFP8_BLK)
            _gemv_fp8_rmsnorm(wv, wv_s, x_p, na, v_p, True, bv, kv_dim, H, pid, eps, BLOCK_M, BLOCK_K, NORMK, NSTAGES, MXFP8_BLK)
        else:
            _gemv_bf16_rmsnorm(wq, x_p, na, q_p, True, bq, q_dim, H, pid, eps, BLOCK_M, BLOCK_K, NORMK)
            _gemv_bf16_rmsnorm(wk, x_p, na, k_p, True, bk, kv_dim, H, pid, eps, BLOCK_M, BLOCK_K, NORMK)
            _gemv_bf16_rmsnorm(wv, x_p, na, v_p, True, bv, kv_dim, H, pid, eps, BLOCK_M, BLOCK_K, NORMK)
        bc += 1
        _barrier(bar_p, bc * NWG)

        # ---- RoPE + KV append + flash decode ----
        if pid < NKV:
            _rope_kv_append(k_p, v_p, cos_p, sin_p, kcache, vcache, pos, pid, kv_dim, DH, HALF)
        if pid < NH:
            _flash_decode_head(q_p, k_p, v_p, cos_p, sin_p, kcache, vcache, sinks, attn_p,
                               pos, scale, pid, kv_dim, DH, HALF, GROUP, SLIDING, BLOCK_T)
        bc += 1
        _barrier(bar_p, bc * NWG)

        # ---- O-proj ----
        if FP8_O:
            _gemv_fp8_tiled(wo, wo_s, attn_p, o_p, True, bo, H, q_dim, pid, BLOCK_M, BLOCK_K, NSTAGES, MXFP8_BLK)
        else:
            _gemv_bf16_tiled(wo, attn_p, o_p, True, bo, H, q_dim, pid, BLOCK_M, BLOCK_K)
        bc += 1
        _barrier(bar_p, bc * NWG)

        # ---- router (fused resid+RMSNorm) + FP8 quant of the expert input ----
        if FP8_ROUTER:
            _gemv_fp8_resid_rmsnorm(rw, rw_s, x_p, o_p, nm, logits_p, True, rb, E, H, pid, eps, BLOCK_M, BLOCK_K, NORMK, NSTAGES, MXFP8_BLK)
        else:
            _gemv_bf16_resid_rmsnorm(rw, x_p, o_p, nm, logits_p, True, rb, E, H, pid, eps, BLOCK_M, BLOCK_K, NORMK)
        _quant_norm_fp8(x_p, o_p, nm, nfp8_p, nfp8_scl_p, H, GU_NB, pid, eps, NORMK)
        bc += 1
        _barrier(bar_p, bc * NWG)
        _topk_softmax(logits_p, ids_p, gw_p, E, TOPK)

        # ---- scatter the FP8 expert input + meta to each MoE rank, strided ----
        for slot in tl.static_range(TOPK):
            dst = 1 + slot
            b = pid
            while b < nblk:
                off = b * BLOCK + tl.arange(0, BLOCK)
                m = off < H
                vv = iris.load(nfp8_p + off, ATTN_RANK, ATTN_RANK, heap_bases, mask=m)
                iris.store(r_nfp8_p + off, vv, ATTN_RANK, dst, heap_bases, mask=m)
                ms = off < GU_NB
                vs = iris.load(nfp8_scl_p + off, ATTN_RANK, ATTN_RANK, heap_bases, mask=ms)
                iris.store(r_nfp8_scl_p + off, vs, ATTN_RANK, dst, heap_bases, mask=ms)
                b += NWG
        bc += 1
        _barrier(bar_p, bc * NWG)  # all scatter stores issued

        # ---- raise per-MoE input-ready flags (pid 0 only), then everyone waits for
        # results. No barrier between: barrier(scatter) already flushed the inputs to
        # L2; only pid0 raises input flags; the result wait below is gated by its own
        # out_flag acquires, so non-pid0 programs simply spin until results land. ----
        if pid == 0:
            for slot in tl.static_range(TOPK):
                dst = 1 + slot
                iris.store(r_meta_p, tl.load(ids_p + slot), ATTN_RANK, dst, heap_bases)
                iris.store(r_gw_p, tl.load(gw_p + slot), ATTN_RANK, dst, heap_bases)
                iris.atomic_xchg(r_in_flag_p + layer, layer + 1, ATTN_RANK, dst, heap_bases, sem="release", scope="sys")
        # ---- wait for all TOPK results. EVERY program acquires each out_flag so the
        # remotely-delivered res[] is coherent for the program that will read it. ----
        for slot in tl.static_range(TOPK):
            done = 0
            while done == 0:
                fv = iris.atomic_add(out_flag_p + layer * TOPK + slot, 0, ATTN_RANK, ATTN_RANK, heap_bases, sem="acquire", scope="sys")
                if fv >= layer + 1:
                    done = 1
        bc += 1
        _barrier(bar_p, bc * NWG)  # all programs past the acquires before accumulate

        # ---- accumulate: x += o + sum_slot res[slot] (strided) ----
        b = pid
        while b < nblk:
            off = b * BLOCK + tl.arange(0, BLOCK)
            m = off < H
            acc = tl.load(x_p + off, mask=m, other=0.0).to(tl.float32) + tl.load(o_p + off, mask=m, other=0.0).to(tl.float32)
            for s in range(0, TOPK):
                acc += tl.load(res_p + s * H + off, mask=m, other=0.0).to(tl.float32)
            tl.store(x_p + off, acc, mask=m)
            b += NWG
        bc += 1
        _barrier(bar_p, bc * NWG)

    # ===== final norm (fused) + lm_head + argmax =====
    fnoff = tl.arange(0, NORMK)
    fnmask = fnoff < H
    fxall = tl.load(x_p + fnoff, mask=fnmask, other=0.0).to(tl.float32)
    fss = tl.sum(fxall * fxall, axis=0)
    frms = 1.0 / tl.sqrt(fss / H + eps)
    mo = tl.arange(0, BLOCK_M_LM)
    ko = tl.max_contiguous(tl.multiple_of(tl.arange(0, BLOCK_K), BLOCK_K), BLOCK_K)
    n_tiles = (V + BLOCK_M_LM - 1) // BLOCK_M_LM
    NK_LM: tl.constexpr = (H + BLOCK_K - 1) // BLOCK_K
    best_v = -1e30
    best_i = 0
    tile = pid
    while tile < n_tiles:
        rows = tile * BLOCK_M_LM + mo
        rmask = rows < V
        acc = tl.zeros((BLOCK_M_LM, BLOCK_K), dtype=tl.float32)
        for ki in tl.range(0, NK_LM, num_stages=NSTAGES):
            kk = ki * BLOCK_K + ko
            kmask = kk < H
            w = tl.load(lm_head_p + rows[:, None] * H + kk[None, :], mask=rmask[:, None] & kmask[None, :], other=0.0).to(tl.float32)
            xk = tl.load(x_p + kk, mask=kmask, other=0.0).to(tl.float32)
            gk = tl.load(final_norm_p + kk, mask=kmask, other=0.0).to(tl.float32)
            acc += w * (xk * frms * gk)[None, :]
        logit = tl.sum(acc, axis=1)
        logit = tl.where(rmask, logit, -1e30)
        tmax = tl.max(logit, axis=0)
        if tmax > best_v:
            ismax = logit == tmax
            best_i = tl.min(tl.where(ismax, rows, V), axis=0)
            best_v = tmax
        tile += NWG
    tl.store(amax_v_p + pid, best_v)
    tl.store(amax_i_p + pid, best_i)
    bc += 1
    _barrier(bar_p, bc * NWG)
    if pid == 0:
        bv = -1e30
        bi = 0
        j = 0
        while j < NWG:
            vvv = tl.load(amax_v_p + j)
            if vvv > bv:
                bv = vvv
                bi = tl.load(amax_i_p + j)
            j += 1
        tl.store(next_tok_p, bi)


@triton.jit
def moe_persistent_kernel(
    gu_blk_p, gu_scl_p, gu_b_p, dn_blk_p, dn_scl_p, dn_b_p,
    nfp8_p, nfp8_scl_p, meta_p, gw_p,  # local inbox (written by attn)
    gu_p, afp8_p, afp8_scl_p, out_p,   # local scratch + result
    in_flag_p,                          # local: raised by attn rank
    r_res_p, r_out_flag_p,              # remote (attn rank) result inbox + flag
    bar_p,
    alpha, limit,
    heap_bases,
    NWG: tl.constexpr, L: tl.constexpr,
    E: tl.constexpr, H: tl.constexpr, I: tl.constexpr,
    GU_NB: tl.constexpr, DN_NB: tl.constexpr,
    BLOCK_NQ: tl.constexpr, BLOCK_ND: tl.constexpr, BLOCK_KQ: tl.constexpr,
    MTILE: tl.constexpr, NSTAGES: tl.constexpr,
    TOPK: tl.constexpr, SLOT: tl.constexpr, MOE_RANK: tl.constexpr, ATTN_RANK: tl.constexpr,
    BLOCK: tl.constexpr,
):
    pid = tl.program_id(0)
    nblk: tl.constexpr = (H + BLOCK - 1) // BLOCK
    bc = 0
    for layer in range(L):
        # ---- wait for this layer's expert input. EVERY program spins on the local
        # in_flag with a sys-scope acquire, so each program's caches are invalidated
        # before it reads the remotely-delivered nfp8/meta/gw (a single-program acquire
        # would only make the delivered data coherent for that one program). ----
        done = 0
        while done == 0:
            fv = iris.atomic_add(in_flag_p + layer, 0, MOE_RANK, MOE_RANK, heap_bases, sem="acquire", scope="sys")
            if fv >= layer + 1:
                done = 1
        bc += 1
        _barrier(bar_p, bc * NWG)  # all programs past the acquire before compute

        e_id = tl.load(meta_p)
        gwv = tl.load(gw_p).to(tl.float32)
        eidx = (layer * E + e_id).to(tl.int64)

        # ---- gate-up ----
        gu_blk = gu_blk_p + eidx * (2 * I) * (H // 2)
        gu_scl = gu_scl_p + eidx * (2 * I) * GU_NB
        gu_b = gu_b_p + eidx * (2 * I)
        _gemv_fp4_scaled(gu_blk, gu_scl, nfp8_p, nfp8_scl_p, gu_p, gu_b, True, 2 * I, H, GU_NB, pid, 1.0, False,
                         BLOCK_NQ, BLOCK_KQ, MTILE)
        bc += 1
        _barrier(bar_p, bc * NWG)

        # ---- SwiGLU -> FP8 ----
        _swiglu_quant_fp8(gu_p, afp8_p, afp8_scl_p, DN_NB, pid, alpha, limit)
        bc += 1
        _barrier(bar_p, bc * NWG)

        # ---- down (gate-weighted) -> out_p ----
        dn_blk = dn_blk_p + eidx * H * (I // 2)
        dn_scl = dn_scl_p + eidx * H * DN_NB
        dn_b = dn_b_p + eidx * H
        _gemv_fp4_scaled(dn_blk, dn_scl, afp8_p, afp8_scl_p, out_p, dn_b, True, H, I, DN_NB, pid, gwv, False,
                         BLOCK_ND, BLOCK_KQ, MTILE, NSTAGES)
        bc += 1
        _barrier(bar_p, bc * NWG)

        # ---- ship result back to attn rank's res[SLOT], strided ----
        b = pid
        while b < nblk:
            off = b * BLOCK + tl.arange(0, BLOCK)
            m = off < H
            vv = iris.load(out_p + off, MOE_RANK, MOE_RANK, heap_bases, mask=m)
            iris.store(r_res_p + SLOT * H + off, vv, MOE_RANK, ATTN_RANK, heap_bases, mask=m)
            b += NWG
        bc += 1
        _barrier(bar_p, bc * NWG)  # all result stores issued
        if pid == 0:
            iris.atomic_xchg(r_out_flag_p + layer * TOPK + SLOT, layer + 1, MOE_RANK, ATTN_RANK, heap_bases, sem="release", scope="sys")
