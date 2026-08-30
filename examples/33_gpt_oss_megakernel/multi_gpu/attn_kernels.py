# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
"""Attention-rank kernels for the multi-GPU GPT-OSS decode.

The attention rank runs the per-layer pipeline up through the router + FP8 quant of
the expert input (attn_prologue_kernel), then scatters that input + per-expert meta
to the MoE ranks (scatter_to_moe_kernel), and after the MoE ranks have produced
their results, sums them into the residual (accumulate_kernel). The lm_head / argmax
tail reuses the single-GPU path on this rank.

Cross-rank sync is host-orchestrated with shmem.barrier() between stages (one
all-rank barrier per exchange), so the kernels themselves only do plain iris.store /
iris.load. This is the correctness-first version; device-flag pipelining is a later
optimization. Compute reuses the single-GPU device ops in common/."""

import triton
import triton.language as tl
import iris

from common.barrier import _barrier
from common.gemv_bf16 import _gemv_bf16_tiled, _gemv_bf16_rmsnorm, _gemv_bf16_resid_rmsnorm
from common.gemv_fp8 import _gemv_fp8_tiled, _gemv_fp8_rmsnorm, _gemv_fp8_resid_rmsnorm
from common.quant import _quant_norm_fp8
from common.attention import _rope_kv_append, _flash_decode_head
from common.router import _topk_softmax


@triton.jit
def attn_prologue_kernel(
    norm_attn_p,
    norm_moe_p,
    wq_p, bq_p, wk_p, bk_p, wv_p, bv_p, wo_p, bo_p,
    sinks_p,
    router_w_p, router_b_p,
    wq_s_p, wk_s_p, wv_s_p, wo_s_p, router_w_s_p,
    x_p, q_p, k_p, v_p, kcache_p, vcache_p, attn_p, o_p,
    logits_p, ids_p, gw_p,
    nfp8_p, nfp8_scl_p,
    cos_p, sin_p, bar_p,
    pos, scale, eps,
    layer,
    bar_base,
    NWG: tl.constexpr,
    H: tl.constexpr, q_dim: tl.constexpr, kv_dim: tl.constexpr,
    NH: tl.constexpr, NKV: tl.constexpr, DH: tl.constexpr,
    E: tl.constexpr, TOPK: tl.constexpr, SLIDING: tl.constexpr,
    GU_NB: tl.constexpr, max_seq: tl.constexpr,
    BLOCK_K: tl.constexpr, BLOCK_M: tl.constexpr, NORMK: tl.constexpr,
    BLOCK_T: tl.constexpr, NSTAGES: tl.constexpr,
    FP8_QKV: tl.constexpr, FP8_O: tl.constexpr, FP8_ROUTER: tl.constexpr, MXFP8_BLK: tl.constexpr,
):
    pid = tl.program_id(0)
    HALF: tl.constexpr = DH // 2
    GROUP: tl.constexpr = NH // NKV

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
    NSB_H: tl.constexpr = (H + MXFP8_BLK - 1) // MXFP8_BLK if MXFP8_BLK < H else 1
    NSB_Q: tl.constexpr = (q_dim + MXFP8_BLK - 1) // MXFP8_BLK if MXFP8_BLK < q_dim else 1
    wq_s = wq_s_p + layer * q_dim * NSB_H
    wk_s = wk_s_p + layer * kv_dim * NSB_H
    wv_s = wv_s_p + layer * kv_dim * NSB_H
    wo_s = wo_s_p + layer * H * NSB_Q
    rw_s = router_w_s_p + layer * E * NSB_H
    kcache = kcache_p + layer * max_seq * kv_dim
    vcache = vcache_p + layer * max_seq * kv_dim

    if FP8_QKV:
        _gemv_fp8_rmsnorm(wq, wq_s, x_p, na, q_p, True, bq, q_dim, H, pid, eps, BLOCK_M, BLOCK_K, NORMK, NSTAGES, MXFP8_BLK)
        _gemv_fp8_rmsnorm(wk, wk_s, x_p, na, k_p, True, bk, kv_dim, H, pid, eps, BLOCK_M, BLOCK_K, NORMK, NSTAGES, MXFP8_BLK)
        _gemv_fp8_rmsnorm(wv, wv_s, x_p, na, v_p, True, bv, kv_dim, H, pid, eps, BLOCK_M, BLOCK_K, NORMK, NSTAGES, MXFP8_BLK)
    else:
        _gemv_bf16_rmsnorm(wq, x_p, na, q_p, True, bq, q_dim, H, pid, eps, BLOCK_M, BLOCK_K, NORMK)
        _gemv_bf16_rmsnorm(wk, x_p, na, k_p, True, bk, kv_dim, H, pid, eps, BLOCK_M, BLOCK_K, NORMK)
        _gemv_bf16_rmsnorm(wv, x_p, na, v_p, True, bv, kv_dim, H, pid, eps, BLOCK_M, BLOCK_K, NORMK)
    _barrier(bar_p, (bar_base + 1) * NWG)

    if pid < NKV:
        _rope_kv_append(k_p, v_p, cos_p, sin_p, kcache, vcache, pos, pid, kv_dim, DH, HALF)
    if pid < NH:
        _flash_decode_head(
            q_p, k_p, v_p, cos_p, sin_p, kcache, vcache, sinks, attn_p,
            pos, scale, pid, kv_dim, DH, HALF, GROUP, SLIDING, BLOCK_T,
        )
    _barrier(bar_p, (bar_base + 2) * NWG)

    if FP8_O:
        _gemv_fp8_tiled(wo, wo_s, attn_p, o_p, True, bo, H, q_dim, pid, BLOCK_M, BLOCK_K, NSTAGES, MXFP8_BLK)
    else:
        _gemv_bf16_tiled(wo, attn_p, o_p, True, bo, H, q_dim, pid, BLOCK_M, BLOCK_K)
    _barrier(bar_p, (bar_base + 3) * NWG)

    if FP8_ROUTER:
        _gemv_fp8_resid_rmsnorm(rw, rw_s, x_p, o_p, nm, logits_p, True, rb, E, H, pid, eps, BLOCK_M, BLOCK_K, NORMK, NSTAGES, MXFP8_BLK)
    else:
        _gemv_bf16_resid_rmsnorm(rw, x_p, o_p, nm, logits_p, True, rb, E, H, pid, eps, BLOCK_M, BLOCK_K, NORMK)
    _quant_norm_fp8(x_p, o_p, nm, nfp8_p, nfp8_scl_p, H, GU_NB, pid, eps, NORMK)
    _barrier(bar_p, (bar_base + 4) * NWG)

    _topk_softmax(logits_p, ids_p, gw_p, E, TOPK)


@triton.jit
def scatter_to_moe_kernel(
    nfp8_p, nfp8_scl_p, ids_p, gw_p,
    r_nfp8_p, r_nfp8_scl_p, r_meta_p, r_gw_p,
    H: tl.constexpr, GU_NB: tl.constexpr,
    slot: tl.constexpr, attn_rank: tl.constexpr, dst_rank: tl.constexpr,
    BLOCK: tl.constexpr,
    heap_bases,
):
    """Copy this token's FP8 expert input + (expert id, gate weight) to dst_rank's
    inbox. An all-rank shmem.barrier() after this launch makes the writes visible."""
    pid = tl.program_id(0)
    off = pid * BLOCK + tl.arange(0, BLOCK)
    m = off < H
    v = iris.load(nfp8_p + off, attn_rank, attn_rank, heap_bases, mask=m)
    iris.store(r_nfp8_p + off, v, attn_rank, dst_rank, heap_bases, mask=m)
    ms = off < GU_NB
    vs = iris.load(nfp8_scl_p + off, attn_rank, attn_rank, heap_bases, mask=ms)
    iris.store(r_nfp8_scl_p + off, vs, attn_rank, dst_rank, heap_bases, mask=ms)
    if pid == 0:
        iris.store(r_meta_p, tl.load(ids_p + slot), attn_rank, dst_rank, heap_bases)
        iris.store(r_gw_p, tl.load(gw_p + slot), attn_rank, dst_rank, heap_bases)


@triton.jit
def accumulate_kernel(
    res_p,  # [TOPK, H] fp32 expert outputs (already gathered locally)
    x_p, o_p,
    H: tl.constexpr, TOPK: tl.constexpr,
    BLOCK: tl.constexpr,
):
    """x += o + sum_slot res[slot]; striped across the grid."""
    pid = tl.program_id(0)
    off = pid * BLOCK + tl.arange(0, BLOCK)
    m = off < H
    acc = tl.load(x_p + off, mask=m, other=0.0).to(tl.float32) + tl.load(o_p + off, mask=m, other=0.0).to(tl.float32)
    for s in range(0, TOPK):
        acc += tl.load(res_p + s * H + off, mask=m, other=0.0).to(tl.float32)
    tl.store(x_p + off, acc, mask=m)


@triton.jit
def lm_head_kernel(
    x_p, final_norm_p, lm_head_p,
    amax_v_p, amax_i_p, next_tok_p, bar_p,
    eps, bar_base,
    NWG: tl.constexpr,
    H: tl.constexpr, V: tl.constexpr,
    BLOCK_K: tl.constexpr, BLOCK_M_LM: tl.constexpr, NORMK: tl.constexpr, NSTAGES: tl.constexpr,
):
    """Final RMSNorm (fused) + LM head + grid argmax, identical to the single-GPU
    tail. Run once after all layers on the attention rank."""
    pid = tl.program_id(0)
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
    _barrier(bar_p, (bar_base + 1) * NWG)
    if pid == 0:
        bv = -1e30
        bi = 0
        j = 0
        while j < NWG:
            vv = tl.load(amax_v_p + j)
            if vv > bv:
                bv = vv
                bi = tl.load(amax_i_p + j)
            j += 1
        tl.store(next_tok_p, bi)
