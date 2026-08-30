# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
"""
Single-GPU SPLIT decode kernels for GPT-OSS-120B.

This is the phase-split counterpart to the persistent megakernel in
gpt_oss_120b_quantized_megakernel.py, generated from it so the arithmetic is
identical. THERE IS NO GRID-WIDE BARRIER HERE. One phase runs per launch and the
kernel boundary provides the ordering the barrier used to provide, so the caller
drives 7*L + 2 launches per token (254 at L=36) instead of one.

`PHASE` is a constexpr, so Triton dead-code-eliminates the other branches and each
phase compiles as its own kernel with its own resource allocation.

Note on vocabulary: the per-phase comments below are inherited from the fused kernel and
several still read "no barrier needed" / "without a separate barrier". Those describe why
a barrier was not required at that point and remain accurate, but in THIS file there is no
barrier anywhere -- every ordering guarantee comes from the kernel boundary between launches.

Per layer the kernel computes:

    RMSNorm -> QKV + bias -> RoPE -> KV-cache append
            -> grouped-query attention with attention sinks and a per-layer
               sliding or full window
            -> output projection + residual
            -> RMSNorm -> router (top-k with softmax over the selected experts)
            -> top-k SwiGLU experts -> gated sum + residual

followed once by the final RMSNorm, the language-model head, and an argmax.

The attention weights, router, embedding and LM head are stored in BF16 by
default. The experts are stored in MXFP4 (4-bit weights with per-32 block scales)
and only the selected experts are read each step. Two expert compute paths are
available:

    default     dequantize the FP4 weights to BF16 and multiply in BF16
    quantized   quantize the activations to FP8 and multiply FP4 x FP8 with the
                scaled matrix-multiply instruction (enable with quant=True)

The attention/router weights can optionally be stored in a smaller dtype with
fp8_attn=True (or --fp8-attn): the weights become FP8-e4m3 (a per-row or, with
fp8_scale_blk=32, a 1x32-block scale) while the activations stay in fp32. This
halves the attention weight traffic for a small accuracy cost; BF16 (the default)
is more accurate. See acc_eval.py for the measured accuracy tradeoff.
"""

from __future__ import annotations

import torch
import triton
import triton.language as tl

from reference import GptOssConfig, build_yarn_rope
from load_hf import load_hf_weights
from tokenizer_util import load_tokenizer

# Device ops live in common/ (one module per op+dtype) so the single-GPU megakernel
# below and the planned per-GPU multi-GPU kernels share the same kernels.
from common import *  # noqa: F401,F403  (re-exports every @triton.jit device op)

# Grid width. 180 is the expert-GEMV tile count (gate_up 2*I/BLOCK_NQ = 5760/32 and
# down_proj H/BLOCK_ND = 2880/16 both give 180), so every program owns exactly one tile.
#
# INHERITED AND UNREVISITED FOR THE SPLIT DESIGN. In the fused kernel this width was also
# forced from above: every program must be resident for a grid-wide barrier to complete,
# and 257+ hangs. That constraint does not exist here -- there is no barrier -- so the
# optimum for these kernels has not been re-derived. A barrier-free ablation preferred a
# considerably wider grid, and a pure streaming read at this width reaches only ~44% of
# the device's bandwidth. Treat 180 as a starting point, not a tuned value.
NUM_WG = 180
_NWG = tl.constexpr(NUM_WG)



# ══════════ SPLIT KERNEL — one phase per launch, no grid barrier ══════════
# Generated from gpt_oss_120b_quantized_megakernel.py. Same device functions, same
# arithmetic; the grid barrier is replaced by the kernel boundary. PHASE is constexpr, so
# Triton dead-code-eliminates the other branches and each phase compiles independently.
# Measured on this build (gfx950, 9 specializations, uninstrumented n_regs):
#   registers per phase: 3, 18, 62, 104, 114, 115, 122, 128, 150   -> max 150
#   the fused kernel allocates 306 (the max over ALL phases, since it inlines all of them)
#   occupancy: 8 of 9 phases reach 2-99; one is at 1, LDS-bound at 104,384 B (the FP4 GEMV)
#   PHASE 0..6  the 7 barrier-separated regions of one layer (arg `layer`)
#   PHASE 7     final norm + lm_head + per-WG argmax
#   PHASE 8     pid==0 cross-WG argmax reduction -> next_tok
# 7*L + 2 launches per token (254 at L=36).

@triton.jit
def gpt_oss_megakernel(
    # weights (per-layer contiguous)
    norm_attn_p,
    norm_moe_p,
    wq_p,
    bq_p,
    wk_p,
    bk_p,
    wv_p,
    bv_p,
    wo_p,
    bo_p,
    sinks_p,
    router_w_p,
    router_b_p,
    # FP8 attn per-row weight scales (unused when FP8_ATTN is False)
    wq_s_p,
    wk_s_p,
    wv_s_p,
    wo_s_p,
    router_w_s_p,
    gu_blk_p,
    gu_scl_p,
    gu_b_p,
    dn_blk_p,
    dn_scl_p,
    dn_b_p,
    final_norm_p,
    lm_head_p,
    # runtime buffers
    x_p,
    normed_p,
    q_p,
    k_p,
    v_p,
    kcache_p,
    vcache_p,
    attn_p,
    o_p,
    logits_p,
    ids_p,
    gw_p,
    gu_p,
    act_p,
    moe_p,
    nfp8_p,
    nfp8_scl_p,
    afp8_p,
    afp8_scl_p,
    vlogits_p,
    amax_v_p,
    amax_i_p,
    next_tok_p,
    cos_p,
    sin_p,
    bar_p,
    # scalars
    pos,
    scale,
    eps,
    alpha,
    limit,
    # constexpr dims
    L: tl.constexpr,
    H: tl.constexpr,
    q_dim: tl.constexpr,
    kv_dim: tl.constexpr,
    NH: tl.constexpr,
    NKV: tl.constexpr,
    DH: tl.constexpr,
    E: tl.constexpr,
    TOPK: tl.constexpr,
    I: tl.constexpr,
    V: tl.constexpr,
    SLIDING: tl.constexpr,
    GU_NB: tl.constexpr,
    DN_NB: tl.constexpr,
    max_seq: tl.constexpr,
    BLOCK_K: tl.constexpr,
    BLOCK_KI: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_M_LM: tl.constexpr,
    NORMK: tl.constexpr,
    QUANT: tl.constexpr,
    BLOCK_NQ: tl.constexpr,
    BLOCK_ND: tl.constexpr,
    BLOCK_KQ: tl.constexpr,
    MTILE: tl.constexpr,
    BLOCK_T: tl.constexpr,
    NSTAGES: tl.constexpr,
    FP8_QKV: tl.constexpr,
    FP8_O: tl.constexpr,
    FP8_ROUTER: tl.constexpr,
    MXFP8_BLK: tl.constexpr,
    DUMP_LOGITS: tl.constexpr,
    layer,
    PHASE: tl.constexpr,
):
    pid = tl.program_id(0)
    HALF: tl.constexpr = DH // 2
    GROUP: tl.constexpr = NH // NKV
    if PHASE < 7:
        # weight bases for this layer
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
        # scale stride per layer = N_rows * (#K-blocks). #K-blocks is 1 for per-row,
        # or K/MXFP8_BLK for block scales (qkv/router contract over H, o-proj over q_dim).
        NSB_H: tl.constexpr = (H + MXFP8_BLK - 1) // MXFP8_BLK if MXFP8_BLK < H else 1
        NSB_Q: tl.constexpr = (q_dim + MXFP8_BLK - 1) // MXFP8_BLK if MXFP8_BLK < q_dim else 1
        wq_s = wq_s_p + layer * q_dim * NSB_H
        wk_s = wk_s_p + layer * kv_dim * NSB_H
        wv_s = wv_s_p + layer * kv_dim * NSB_H
        wo_s = wo_s_p + layer * H * NSB_Q
        rw_s = router_w_s_p + layer * E * NSB_H
        kcache = kcache_p + layer * max_seq * kv_dim
        vcache = vcache_p + layer * max_seq * kv_dim
        rstep = _NWG * BLOCK_M   # hoisted: defined in P2, used in P3 (cross-phase local)

        if PHASE == 0:
            # ---- P0+P1 FUSED: RMSNorm(attn) folded into QKV GEMV (no separate norm
            # phase). Each WG recomputes the norm scale from x_p, which is stable because
            # the previous layer's last phase completed at a KERNEL BOUNDARY before this
            # launch began -- that is what provides the ordering here, not a barrier. ----
            if FP8_QKV:
                _gemv_fp8_rmsnorm(wq, wq_s, x_p, na, q_p, True, bq, q_dim, H, pid, eps, BLOCK_M, BLOCK_K, NORMK, NSTAGES, MXFP8_BLK)
                _gemv_fp8_rmsnorm(wk, wk_s, x_p, na, k_p, True, bk, kv_dim, H, pid, eps, BLOCK_M, BLOCK_K, NORMK, NSTAGES, MXFP8_BLK)
                _gemv_fp8_rmsnorm(wv, wv_s, x_p, na, v_p, True, bv, kv_dim, H, pid, eps, BLOCK_M, BLOCK_K, NORMK, NSTAGES, MXFP8_BLK)
            else:
                _gemv_bf16_rmsnorm(wq, x_p, na, q_p, True, bq, q_dim, H, pid, eps, BLOCK_M, BLOCK_K, NORMK)
                _gemv_bf16_rmsnorm(wk, x_p, na, k_p, True, bk, kv_dim, H, pid, eps, BLOCK_M, BLOCK_K, NORMK)
                _gemv_bf16_rmsnorm(wv, x_p, na, v_p, True, bv, kv_dim, H, pid, eps, BLOCK_M, BLOCK_K, NORMK)
        if PHASE == 1:

            # ---- P2+P3 FUSED: RoPE, KV-cache append, and attention in one phase (no
            # separate KV-append barrier). The current position's key/value are RoPE'd and
            # used in-register for its own attention term, and also written to the cache
            # for future tokens; the history [lo, pos-1] is read from the cache. Since no
            # attention head reads the just-written cache[pos], the append and the
            # attention can share a phase. ----
            # the KV-head owners append the current k,v (RoPE'd) to the cache for next time
            if pid < NKV:
                _rope_kv_append(k_p, v_p, cos_p, sin_p, kcache, vcache, pos, pid, kv_dim, DH, HALF)
            if pid < NH:
                _flash_decode_head(
                    q_p, k_p, v_p, cos_p, sin_p, kcache, vcache, sinks, attn_p,
                    pos, scale, pid, kv_dim, DH, HALF, GROUP, SLIDING, BLOCK_T,
                )
        if PHASE == 2:

            # ---- P4: O-proj -> o_p (the post-attention output). The residual add is
            # deferred to the layer's final accumulation, so there is no separate
            # residual phase or barrier. ----
            if FP8_O:
                _gemv_fp8_tiled(wo, wo_s, attn_p, o_p, True, bo, H, q_dim, pid, BLOCK_M, BLOCK_K, NSTAGES, MXFP8_BLK)
            else:
                _gemv_bf16_tiled(wo, attn_p, o_p, True, bo, H, q_dim, pid, BLOCK_M, BLOCK_K)
            rstep = _NWG * BLOCK_M
        if PHASE == 3:

            # ---- P5: router GEMV, expert-input quant, and accumulator reset all run in
            # one phase -- they only depend on x + o, complete at the O-proj kernel boundary.
            # The router writes logits; the quant writes the FP8 expert input; the moe
            # accumulator is zeroed. ----
            if FP8_ROUTER:
                _gemv_fp8_resid_rmsnorm(
                    rw, rw_s, x_p, o_p, nm, logits_p, True, rb, E, H, pid, eps, BLOCK_M, BLOCK_K, NORMK, NSTAGES, MXFP8_BLK
                )
            else:
                _gemv_bf16_resid_rmsnorm(rw, x_p, o_p, nm, logits_p, True, rb, E, H, pid, eps, BLOCK_M, BLOCK_K, NORMK)
            base_z = pid * BLOCK_M
            while base_z < H:
                zoff = base_z + tl.arange(0, BLOCK_M)
                tl.store(moe_p + zoff, tl.zeros((BLOCK_M,), dtype=tl.float32), mask=zoff < H)
                base_z += rstep
            if QUANT:
                _quant_norm_fp8(x_p, o_p, nm, nfp8_p, nfp8_scl_p, H, GU_NB, pid, eps, NORMK)
            else:
                _store_resid_rmsnorm(x_p, o_p, nm, normed_p, H, pid, eps, BLOCK_M, NORMK)
        if PHASE == 4:

            # Every program does the top-k + softmax over the E router logits redundantly
            # in registers (E is tiny) and writes ids_p/gw_p. The writes are identical
            # across programs and each program reads back only what it wrote, so the
            # experts proceed without a separate top-k barrier.
            _topk_softmax(logits_p, ids_p, gw_p, E, TOPK)

            # ---- P6: experts. The top-k experts are independent until the final
            # accumulation, so each expert phase (gate-up, SwiGLU, down) runs over all
            # experts before the phase ends -- 3 phase boundaries per layer instead of 3
            # per expert. QUANT uses the native FP4xFP8 scaled multiply; otherwise the FP4
            # weights are dequantized to BF16 in the GEMV. ----
            # --- phase A: gate-up for every expert -> gu_p[slot] ---
            for slot in range(0, TOPK):
                e_id = tl.load(ids_p + slot)
                eidx = (layer * E + e_id).to(tl.int64)  # int64: expert blobs overflow int32
                gu_blk = gu_blk_p + eidx * (2 * I) * (H // 2)
                gu_scl = gu_scl_p + eidx * (2 * I) * GU_NB
                gu_b = gu_b_p + eidx * (2 * I)
                gu_out = gu_p + slot * (2 * I)
                if QUANT:
                    _gemv_fp4_scaled(
                        gu_blk,
                        gu_scl,
                        nfp8_p,
                        nfp8_scl_p,
                        gu_out,
                        gu_b,
                        True,
                        2 * I,
                        H,
                        GU_NB,
                        pid,
                        1.0,
                        False,
                        BLOCK_NQ,
                        BLOCK_KQ,
                        MTILE,
                    )
                else:
                    _gemv_fp4(gu_blk, gu_scl, normed_p, gu_out, gu_b, True, 2 * I, H, GU_NB, pid, 1.0, ACCUM=False)
        if PHASE == 5:
            # --- phase B: SwiGLU for every expert. QUANT quantizes each 32-element block
            # in place (producer == consumer for that block, no barrier needed). ---
            for slot in range(0, TOPK):
                gu_out = gu_p + slot * (2 * I)
                if QUANT:
                    _swiglu_quant_fp8(gu_out, afp8_p + slot * I, afp8_scl_p + slot * DN_NB, DN_NB, pid, alpha, limit)
                else:
                    _swiglu_bf16(gu_out, act_p + slot * I, I, pid, alpha, limit)
        if PHASE == 6:
            # --- phase C: down for every expert, accumulating gate-weighted sum into moe_p.
            # Each program owns the same output rows across all experts, so the running
            # accumulation is program-local and needs no barrier between experts. ---
            for slot in range(0, TOPK):
                e_id = tl.load(ids_p + slot)
                gwv = tl.load(gw_p + slot).to(tl.float32)
                eidx = (layer * E + e_id).to(tl.int64)
                dn_blk = dn_blk_p + eidx * H * (I // 2)
                dn_scl = dn_scl_p + eidx * H * DN_NB
                dn_b = dn_b_p + eidx * H
                # The last expert's down GEMV finalizes the residual x[n] += o[n] + moe[n]
                # for the rows it owns (QUANT path), folding the final residual into this
                # phase and removing a phase boundary.
                finalize = QUANT and (slot == TOPK - 1)
                if QUANT:
                    afp8_out = afp8_p + slot * I
                    afp8_scl_out = afp8_scl_p + slot * DN_NB
                    _gemv_fp4_scaled(
                        dn_blk,
                        dn_scl,
                        afp8_out,
                        afp8_scl_out,
                        moe_p,
                        dn_b,
                        True,
                        H,
                        I,
                        DN_NB,
                        pid,
                        gwv,
                        (slot > 0),
                        BLOCK_ND,
                        BLOCK_KQ,
                        MTILE,
                        NSTAGES,
                        FINALIZE=finalize,
                        x_ptr=x_p,
                        o_ptr=o_p,
                    )
                else:
                    act_out = act_p + slot * I
                    _gemv_fp4(dn_blk, dn_scl, act_out, moe_p, dn_b, True, H, I, DN_NB, pid, gwv, ACCUM=(slot > 0))
    elif PHASE == 7:
        # ===== final norm (fused) + lm_head + argmax =====
        # The final RMSNorm is folded into the LM-head GEMV: every program computes the
        # rms scalar from x once and applies norm*gamma inline, so there is no separate
        # final-norm phase or barrier. Each program computes the logits for its row tiles
        # and tracks a running (max, index) instead of writing all V logits to HBM.
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
                w = tl.load(
                    lm_head_p + rows[:, None] * H + kk[None, :], mask=rmask[:, None] & kmask[None, :], other=0.0
                ).to(tl.float32)
                xk = tl.load(x_p + kk, mask=kmask, other=0.0).to(tl.float32)
                gk = tl.load(final_norm_p + kk, mask=kmask, other=0.0).to(tl.float32)
                acc += w * (xk * frms * gk)[None, :]
            logit = tl.sum(acc, axis=1)  # [BLOCK_M]
            logit = tl.where(rmask, logit, -1e30)
            if DUMP_LOGITS:
                tl.store(vlogits_p + rows, logit, mask=rmask)
            tmax = tl.max(logit, axis=0)
            if tmax > best_v:
                ismax = logit == tmax
                best_i = tl.min(tl.where(ismax, rows, V), axis=0)
                best_v = tmax
            tile += _NWG
        tl.store(amax_v_p + pid, best_v)
        tl.store(amax_i_p + pid, best_i)
    else:
        if pid == 0:
            bv = -1e30
            bi = 0
            j = 0
            while j < _NWG:
                vv = tl.load(amax_v_p + j)
                if vv > bv:
                    bv = vv
                    bi = tl.load(amax_i_p + j)
                j += 1
            tl.store(next_tok_p, bi)

