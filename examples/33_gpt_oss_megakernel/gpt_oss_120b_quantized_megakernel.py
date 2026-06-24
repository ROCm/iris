# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
"""
Single-GPU persistent megakernel for GPT-OSS-120B decode.

One Triton kernel runs both attention and the mixture-of-experts for all layers
of GPT-OSS-120B on a single GPU. It is launched once per token and loops over the
layers internally; its persistent programs synchronize at each phase with a
grid-wide barrier, so attention and MoE are phases of the same resident kernel
rather than separate launches.

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

# Number of persistent programs. Fewer than the 256 CUs on MI355X: with a grid-wide
# barrier between phases, a smaller grid makes each barrier cheaper while still
# keeping every phase's GEMV well filled. 180 measured fastest for this model.
NUM_WG = 180
_NWG = tl.constexpr(NUM_WG)


# ───────────────────────── device helpers ─────────────────────────
@triton.jit
def _barrier(bar_ptr, target):
    # Arrive: release so this program's phase writes are flushed to the shared L2
    # before the counter increment becomes visible to peers.
    tl.debug_barrier()
    tl.atomic_add(bar_ptr, 1, sem="release", scope="gpu")
    # Spin on a relaxed read: polling with acquire emits a full L1 invalidate
    # (buffer_inv sc1) every iteration, which dominates the barrier cost. A relaxed
    # poll just reads the counter cheaply.
    done = 0
    while done == 0:
        cur = tl.atomic_add(bar_ptr, 0, sem="relaxed", scope="gpu")
        if cur >= target:
            done = 1
    # One acquire after the count is reached invalidates L1 a single time, so the
    # next phase reads every peer's writes fresh from L2.
    _ = tl.atomic_add(bar_ptr, 0, sem="acquire", scope="gpu")


@triton.jit
def _fp4_lut(mag_idx):
    return tl.where(
        mag_idx == 0,
        0.0,
        tl.where(
            mag_idx == 1,
            0.5,
            tl.where(
                mag_idx == 2,
                1.0,
                tl.where(
                    mag_idx == 3,
                    1.5,
                    tl.where(mag_idx == 4, 2.0, tl.where(mag_idx == 5, 3.0, tl.where(mag_idx == 6, 4.0, 6.0))),
                ),
            ),
        ),
    )


@triton.jit
def _gemv_bf16_tiled(
    w_base,
    x_ptr,
    y_ptr,
    has_bias,
    b_base,
    M,
    K: tl.constexpr,
    pid,
    BLOCK_M: tl.constexpr,
    BLOCK_K: tl.constexpr,
    NSTAGES: tl.constexpr = 3,
):
    """y[r] = sum_k w[r,k]*x[k] (+b[r]) for a batch-1 GEMV.

    Each program owns a tile of BLOCK_M contiguous output rows and walks K in
    BLOCK_K-wide steps, loading a 2D [BLOCK_M, BLOCK_K] weight tile so the inner
    (contiguous-K) dimension vectorizes to dwordx4. max_contiguous/multiple_of tell
    the compiler the K index is contiguous and aligned, enabling wide loads. Tiles
    are strided across the grid by num programs."""
    npid = tl.num_programs(0)
    mo = tl.arange(0, BLOCK_M)
    ko = tl.max_contiguous(tl.multiple_of(tl.arange(0, BLOCK_K), BLOCK_K), BLOCK_K)
    n_tiles = (M + BLOCK_M - 1) // BLOCK_M
    NK: tl.constexpr = (K + BLOCK_K - 1) // BLOCK_K
    tile = pid
    while tile < n_tiles:
        rows = tile * BLOCK_M + mo
        rmask = rows < M
        acc = tl.zeros((BLOCK_M, BLOCK_K), dtype=tl.float32)
        for ki in tl.range(0, NK, num_stages=NSTAGES):
            kk = ki * BLOCK_K + ko
            kmask = kk < K
            w = tl.load(w_base + rows[:, None] * K + kk[None, :], mask=rmask[:, None] & kmask[None, :], other=0.0).to(
                tl.float32
            )
            x = tl.load(x_ptr + kk, mask=kmask, other=0.0).to(tl.float32)
            acc += w * x[None, :]
        s = tl.sum(acc, axis=1)  # [BLOCK_M]
        if has_bias:
            s += tl.load(b_base + rows, mask=rmask, other=0.0).to(tl.float32)
        tl.store(y_ptr + rows, s, mask=rmask)
        tile += npid


@triton.jit
def _store_resid_rmsnorm(
    x_ptr, o_ptr, g_ptr, out_ptr, H: tl.constexpr, pid, eps, BLOCK_M: tl.constexpr, NORMK: tl.constexpr
):
    """Materialize normed = rmsnorm(x + o)*g into out_ptr (bf16), striped across WGs.
    Each program computes the rms scalar from the full (x + o), then writes its row
    slices. Used for the non-quantized expert input; needs a barrier after."""
    noff = tl.arange(0, NORMK)
    nmask = noff < H
    xall = tl.load(x_ptr + noff, mask=nmask, other=0.0).to(tl.float32) + tl.load(
        o_ptr + noff, mask=nmask, other=0.0
    ).to(tl.float32)
    ss = tl.sum(xall * xall, axis=0)
    rms = 1.0 / tl.sqrt(ss / H + eps)
    base = pid * BLOCK_M
    step = tl.num_programs(0) * BLOCK_M
    while base < H:
        off = base + tl.arange(0, BLOCK_M)
        m = off < H
        xv = tl.load(x_ptr + off, mask=m, other=0.0).to(tl.float32) + tl.load(o_ptr + off, mask=m, other=0.0).to(
            tl.float32
        )
        g = tl.load(g_ptr + off, mask=m, other=0.0).to(tl.float32)
        tl.store(out_ptr + off, (xv * rms * g).to(tl.bfloat16), mask=m)
        base += step


@triton.jit
def _gemv_bf16_resid_rmsnorm(
    w_base,
    x_ptr,
    o_ptr,
    g_ptr,
    y_ptr,
    has_bias,
    b_base,
    M,
    H: tl.constexpr,
    pid,
    eps,
    BLOCK_M: tl.constexpr,
    BLOCK_K: tl.constexpr,
    NORMK: tl.constexpr,
    NSTAGES: tl.constexpr = 3,
):
    """Fused residual + RMSNorm + GEMV: y = GEMV(rmsnorm(x + o) * g).

    Reads the attention output o and the residual x and uses (x + o) as the norm
    input, without writing it back. This folds the post-attention residual add and
    its RMSNorm (and the grid barrier between them) into the router GEMV. The single
    residual write x += o happens later in the layer's final accumulation. Each
    program derives the rms scalar from the full (x + o); both inputs are stable
    since the previous barrier, so no barrier is needed here."""
    noff = tl.arange(0, NORMK)
    nmask = noff < H
    xall = tl.load(x_ptr + noff, mask=nmask, other=0.0).to(tl.float32) + tl.load(
        o_ptr + noff, mask=nmask, other=0.0
    ).to(tl.float32)
    ss = tl.sum(xall * xall, axis=0)
    rms = 1.0 / tl.sqrt(ss / H + eps)
    npid = tl.num_programs(0)
    mo = tl.arange(0, BLOCK_M)
    ko = tl.max_contiguous(tl.multiple_of(tl.arange(0, BLOCK_K), BLOCK_K), BLOCK_K)
    n_tiles = (M + BLOCK_M - 1) // BLOCK_M
    NK: tl.constexpr = (H + BLOCK_K - 1) // BLOCK_K
    tile = pid
    while tile < n_tiles:
        rows = tile * BLOCK_M + mo
        rmask = rows < M
        acc = tl.zeros((BLOCK_M, BLOCK_K), dtype=tl.float32)
        for ki in tl.range(0, NK, num_stages=NSTAGES):
            kk = ki * BLOCK_K + ko
            kmask = kk < H
            # recompute xnew = x + o inline (avoids depending on other programs'
            # persisted x_ptr writes -- no barrier needed before this read)
            xk = tl.load(x_ptr + kk, mask=kmask, other=0.0).to(tl.float32) + tl.load(
                o_ptr + kk, mask=kmask, other=0.0
            ).to(tl.float32)
            gk = tl.load(g_ptr + kk, mask=kmask, other=0.0).to(tl.float32)
            nk = xk * rms * gk
            w = tl.load(w_base + rows[:, None] * H + kk[None, :], mask=rmask[:, None] & kmask[None, :], other=0.0).to(
                tl.float32
            )
            acc += w * nk[None, :]
        s = tl.sum(acc, axis=1)
        if has_bias:
            s += tl.load(b_base + rows, mask=rmask, other=0.0).to(tl.float32)
        tl.store(y_ptr + rows, s, mask=rmask)
        tile += npid


@triton.jit
def _gemv_bf16_rmsnorm(
    w_base,
    x_ptr,
    g_ptr,
    y_ptr,
    has_bias,
    b_base,
    M,
    H: tl.constexpr,
    pid,
    eps,
    BLOCK_M: tl.constexpr,
    BLOCK_K: tl.constexpr,
    NORMK: tl.constexpr,
    NSTAGES: tl.constexpr = 3,
):
    """Fused RMSNorm + GEMV: y[r] = sum_k (rmsnorm(x)*g)[k] * w[r,k] (+b[r]).

    Each program computes the RMSNorm scale from x_ptr once (redundant across WGs,
    but cheap and barrier-free: x is stable since the previous barrier), then runs
    the tiled GEMV applying norm*gamma inline. Removes the separate pid0 RMSNorm
    phase AND its grid barrier. NORMK = round_up_pow2(H)."""
    noff = tl.arange(0, NORMK)
    nmask = noff < H
    xall = tl.load(x_ptr + noff, mask=nmask, other=0.0).to(tl.float32)
    ss = tl.sum(xall * xall, axis=0)
    rms = 1.0 / tl.sqrt(ss / H + eps)
    npid = tl.num_programs(0)
    mo = tl.arange(0, BLOCK_M)
    ko = tl.max_contiguous(tl.multiple_of(tl.arange(0, BLOCK_K), BLOCK_K), BLOCK_K)
    n_tiles = (M + BLOCK_M - 1) // BLOCK_M
    NK: tl.constexpr = (H + BLOCK_K - 1) // BLOCK_K
    tile = pid
    while tile < n_tiles:
        rows = tile * BLOCK_M + mo
        rmask = rows < M
        acc = tl.zeros((BLOCK_M, BLOCK_K), dtype=tl.float32)
        for ki in tl.range(0, NK, num_stages=NSTAGES):
            kk = ki * BLOCK_K + ko
            kmask = kk < H
            xk = tl.load(x_ptr + kk, mask=kmask, other=0.0).to(tl.float32)
            gk = tl.load(g_ptr + kk, mask=kmask, other=0.0).to(tl.float32)
            nk = xk * rms * gk  # normed activation chunk
            w = tl.load(w_base + rows[:, None] * H + kk[None, :], mask=rmask[:, None] & kmask[None, :], other=0.0).to(
                tl.float32
            )
            acc += w * nk[None, :]
        s = tl.sum(acc, axis=1)
        if has_bias:
            s += tl.load(b_base + rows, mask=rmask, other=0.0).to(tl.float32)
        tl.store(y_ptr + rows, s, mask=rmask)
        tile += npid


@triton.jit
def _gemv_bf16(w_base, x_ptr, y_ptr, has_bias, b_base, M, K: tl.constexpr, pid, BLOCK_K: tl.constexpr):
    """y[r] = sum_k w[r,k]*x[k] (+b[r]); rows strided across programs by pid. Legacy
    scalar path kept for reference/fallback; _gemv_bf16_tiled is the fast one."""
    koff = tl.arange(0, BLOCK_K)
    r = pid
    while r < M:
        acc = tl.zeros((BLOCK_K,), dtype=tl.float32)
        k0 = 0
        while k0 < K:
            kk = k0 + koff
            kmask = kk < K
            w = tl.load(w_base + r * K + kk, mask=kmask, other=0.0).to(tl.float32)
            x = tl.load(x_ptr + kk, mask=kmask, other=0.0).to(tl.float32)
            acc += w * x
            k0 += BLOCK_K
        s = tl.sum(acc, axis=0)
        if has_bias:
            s += tl.load(b_base + r).to(tl.float32)
        tl.store(y_ptr + r, s)
        r += _NWG


@triton.jit
def _gemv_fp8_rmsnorm(
    w_base,
    ws_base,
    x_ptr,
    g_ptr,
    y_ptr,
    has_bias,
    b_base,
    M,
    H: tl.constexpr,
    pid,
    eps,
    BLOCK_M: tl.constexpr,
    BLOCK_K: tl.constexpr,
    NORMK: tl.constexpr,
    NSTAGES: tl.constexpr = 3,
    SCALE_BLK: tl.constexpr = 1 << 30,
):
    """FP8-weight fused RMSNorm + GEMV: y[r] = sum_k ws[r,blk(k)]*W8[r,k]*(rmsnorm(x)*g)[k].

    W is FP8-e4m3 [M,H]; ws is a weight scale of granularity SCALE_BLK along K (one
    scale per [row, k-block of SCALE_BLK]). SCALE_BLK>=H is per-row (ws shape [M,1]);
    SCALE_BLK=32 is MXFP8-style 1x32 blocks ([M, H/32]). Activation stays fp32; each
    program recomputes the norm scale from the stable x_ptr (no barrier). BLOCK_K must
    be a multiple of SCALE_BLK (or >=H for per-row)."""
    noff = tl.arange(0, NORMK)
    nmask = noff < H
    xall = tl.load(x_ptr + noff, mask=nmask, other=0.0).to(tl.float32)
    ss = tl.sum(xall * xall, axis=0)
    rms = 1.0 / tl.sqrt(ss / H + eps)
    npid = tl.num_programs(0)
    ko = tl.max_contiguous(tl.multiple_of(tl.arange(0, BLOCK_K), BLOCK_K), BLOCK_K)
    n_tiles = (M + BLOCK_M - 1) // BLOCK_M
    NK: tl.constexpr = (H + BLOCK_K - 1) // BLOCK_K
    PERROW: tl.constexpr = SCALE_BLK >= H
    NSB: tl.constexpr = (H + SCALE_BLK - 1) // SCALE_BLK  # scale blocks along K (per matrix row)
    SBT: tl.constexpr = BLOCK_K // SCALE_BLK if not PERROW else 1  # scale blocks per K-tile
    tile = pid
    while tile < n_tiles:
        rows = tile * BLOCK_M + tl.arange(0, BLOCK_M)
        rmask = rows < M
        acc = tl.zeros((BLOCK_M, BLOCK_K), dtype=tl.float32)  # per-row path (no inner scale)
        bacc = tl.zeros((BLOCK_M,), dtype=tl.float32)  # per-block path (scale folded in K-loop)
        for ki in tl.range(0, NK, num_stages=NSTAGES):
            kk = ki * BLOCK_K + ko
            kmask = kk < H
            xk = tl.load(x_ptr + kk, mask=kmask, other=0.0).to(tl.float32)
            gk = tl.load(g_ptr + kk, mask=kmask, other=0.0).to(tl.float32)
            nk = xk * rms * gk
            w = tl.load(
                w_base + rows[:, None] * H + kk[None, :], mask=rmask[:, None] & kmask[None, :], other=0.0
            ).to(tl.float32)
            if PERROW:
                acc += w * nk[None, :]
            else:
                # per-block dequant: psum[m,blk] = sum_{k in blk} W8*nk; then *= scale[m,blk]
                part = (w * nk[None, :]).reshape(BLOCK_M, SBT, SCALE_BLK)
                psum = tl.sum(part, axis=2)  # [BLOCK_M, SBT]
                sbk = ki * SBT + tl.arange(0, SBT)
                sc = tl.load(
                    ws_base + rows[:, None] * NSB + sbk[None, :],
                    mask=rmask[:, None] & (sbk[None, :] < NSB),
                    other=0.0,
                ).to(tl.float32)
                bacc += tl.sum(psum * sc, axis=1)  # [BLOCK_M]
        if PERROW:
            s = tl.sum(acc, axis=1)
            s = s * tl.load(ws_base + rows, mask=rmask, other=0.0).to(tl.float32)
        else:
            s = bacc
        if has_bias:
            s += tl.load(b_base + rows, mask=rmask, other=0.0).to(tl.float32)
        tl.store(y_ptr + rows, s, mask=rmask)
        tile += npid


@triton.jit
def _gemv_fp8_tiled(
    w_base,
    ws_base,
    x_ptr,
    y_ptr,
    has_bias,
    b_base,
    M,
    K: tl.constexpr,
    pid,
    BLOCK_M: tl.constexpr,
    BLOCK_K: tl.constexpr,
    NSTAGES: tl.constexpr = 3,
    SCALE_BLK: tl.constexpr = 1 << 30,
):
    """FP8-weight GEMV (no norm): y[r] = sum_k ws[r,blk(k)]*W8[r,k]*x[k] (+b[r]).
    Activation stays fp32. SCALE_BLK>=K is per-row; SCALE_BLK=32 is MXFP8 1x32 blocks."""
    npid = tl.num_programs(0)
    ko = tl.max_contiguous(tl.multiple_of(tl.arange(0, BLOCK_K), BLOCK_K), BLOCK_K)
    n_tiles = (M + BLOCK_M - 1) // BLOCK_M
    NK: tl.constexpr = (K + BLOCK_K - 1) // BLOCK_K
    PERROW: tl.constexpr = SCALE_BLK >= K
    NSB: tl.constexpr = (K + SCALE_BLK - 1) // SCALE_BLK
    SBT: tl.constexpr = BLOCK_K // SCALE_BLK if not PERROW else 1
    tile = pid
    while tile < n_tiles:
        rows = tile * BLOCK_M + tl.arange(0, BLOCK_M)
        rmask = rows < M
        acc = tl.zeros((BLOCK_M, BLOCK_K), dtype=tl.float32)
        bacc = tl.zeros((BLOCK_M,), dtype=tl.float32)
        for ki in tl.range(0, NK, num_stages=NSTAGES):
            kk = ki * BLOCK_K + ko
            kmask = kk < K
            x = tl.load(x_ptr + kk, mask=kmask, other=0.0).to(tl.float32)
            w = tl.load(
                w_base + rows[:, None] * K + kk[None, :], mask=rmask[:, None] & kmask[None, :], other=0.0
            ).to(tl.float32)
            if PERROW:
                acc += w * x[None, :]
            else:
                part = (w * x[None, :]).reshape(BLOCK_M, SBT, SCALE_BLK)
                psum = tl.sum(part, axis=2)
                sbk = ki * SBT + tl.arange(0, SBT)
                sc = tl.load(
                    ws_base + rows[:, None] * NSB + sbk[None, :],
                    mask=rmask[:, None] & (sbk[None, :] < NSB),
                    other=0.0,
                ).to(tl.float32)
                bacc += tl.sum(psum * sc, axis=1)
        if PERROW:
            s = tl.sum(acc, axis=1)
            s = s * tl.load(ws_base + rows, mask=rmask, other=0.0).to(tl.float32)
        else:
            s = bacc
        if has_bias:
            s += tl.load(b_base + rows, mask=rmask, other=0.0).to(tl.float32)
        tl.store(y_ptr + rows, s, mask=rmask)
        tile += npid


@triton.jit
def _gemv_fp8_resid_rmsnorm(
    w_base,
    ws_base,
    x_ptr,
    o_ptr,
    g_ptr,
    y_ptr,
    has_bias,
    b_base,
    M,
    H: tl.constexpr,
    pid,
    eps,
    BLOCK_M: tl.constexpr,
    BLOCK_K: tl.constexpr,
    NORMK: tl.constexpr,
    NSTAGES: tl.constexpr = 3,
    SCALE_BLK: tl.constexpr = 1 << 30,
):
    """FP8-weight fused residual + RMSNorm + GEMV: y = GEMV8(rmsnorm(x + o) * g) with
    weight scale of granularity SCALE_BLK along K. SCALE_BLK>=H is per-row; 32 is
    MXFP8. Activation stays fp32."""
    noff = tl.arange(0, NORMK)
    nmask = noff < H
    xall = tl.load(x_ptr + noff, mask=nmask, other=0.0).to(tl.float32) + tl.load(
        o_ptr + noff, mask=nmask, other=0.0
    ).to(tl.float32)
    ss = tl.sum(xall * xall, axis=0)
    rms = 1.0 / tl.sqrt(ss / H + eps)
    npid = tl.num_programs(0)
    ko = tl.max_contiguous(tl.multiple_of(tl.arange(0, BLOCK_K), BLOCK_K), BLOCK_K)
    n_tiles = (M + BLOCK_M - 1) // BLOCK_M
    NK: tl.constexpr = (H + BLOCK_K - 1) // BLOCK_K
    PERROW: tl.constexpr = SCALE_BLK >= H
    NSB: tl.constexpr = (H + SCALE_BLK - 1) // SCALE_BLK
    SBT: tl.constexpr = BLOCK_K // SCALE_BLK if not PERROW else 1
    tile = pid
    while tile < n_tiles:
        rows = tile * BLOCK_M + tl.arange(0, BLOCK_M)
        rmask = rows < M
        acc = tl.zeros((BLOCK_M, BLOCK_K), dtype=tl.float32)
        bacc = tl.zeros((BLOCK_M,), dtype=tl.float32)
        for ki in tl.range(0, NK, num_stages=NSTAGES):
            kk = ki * BLOCK_K + ko
            kmask = kk < H
            xk = tl.load(x_ptr + kk, mask=kmask, other=0.0).to(tl.float32) + tl.load(
                o_ptr + kk, mask=kmask, other=0.0
            ).to(tl.float32)
            gk = tl.load(g_ptr + kk, mask=kmask, other=0.0).to(tl.float32)
            nk = xk * rms * gk
            w = tl.load(
                w_base + rows[:, None] * H + kk[None, :], mask=rmask[:, None] & kmask[None, :], other=0.0
            ).to(tl.float32)
            if PERROW:
                acc += w * nk[None, :]
            else:
                part = (w * nk[None, :]).reshape(BLOCK_M, SBT, SCALE_BLK)
                psum = tl.sum(part, axis=2)
                sbk = ki * SBT + tl.arange(0, SBT)
                sc = tl.load(
                    ws_base + rows[:, None] * NSB + sbk[None, :],
                    mask=rmask[:, None] & (sbk[None, :] < NSB),
                    other=0.0,
                ).to(tl.float32)
                bacc += tl.sum(psum * sc, axis=1)
        if PERROW:
            s = tl.sum(acc, axis=1)
            s = s * tl.load(ws_base + rows, mask=rmask, other=0.0).to(tl.float32)
        else:
            s = bacc
        if has_bias:
            s += tl.load(b_base + rows, mask=rmask, other=0.0).to(tl.float32)
        tl.store(y_ptr + rows, s, mask=rmask)
        tile += npid


@triton.jit
def _gemv_fp4(
    blk_base,
    scl_base,
    x_ptr,
    y_ptr,
    b_base,
    has_bias,
    M,
    K: tl.constexpr,
    NB: tl.constexpr,
    pid,
    gate_w,
    ACCUM: tl.constexpr,
):
    """FP4 weight GEMV. weight row r: blk_base + r*(K//2) bytes (K/2), scales
    scl_base + r*NB. Rows strided by pid -> each row owned by one program, so
    plain store / load-add-store (no atomics). ACCUM: y[r]+=gate_w*(s+b) else
    y[r]=gate_w*(s+b). Processes one 32-wide scale block at a time."""
    pos32 = tl.arange(0, 32)
    byte_idx = pos32 // 2
    hi = (pos32 % 2) == 1
    r = pid
    half = K // 2
    while r < M:
        acc = tl.zeros((32,), dtype=tl.float32)
        kb = 0
        while kb < NB:
            raw = tl.load(blk_base + r * half + kb * 16 + byte_idx).to(tl.int32)
            nib = tl.where(hi, (raw >> 4) & 0xF, raw & 0xF)
            sign = (nib & 0x8) != 0
            mag = _fp4_lut(nib & 0x7)
            val = tl.where(sign, -mag, mag)
            se = tl.load(scl_base + r * NB + kb).to(tl.int32)
            sc = tl.where(se > 0, tl.exp2((se - 127).to(tl.float32)), 0.0)
            xk = tl.load(x_ptr + kb * 32 + pos32).to(tl.float32)
            acc += val * sc * xk
            kb += 1
        s = tl.sum(acc, axis=0)
        if has_bias:
            s += tl.load(b_base + r).to(tl.float32)
        s = gate_w * s
        if ACCUM:
            s += tl.load(y_ptr + r).to(tl.float32)
        tl.store(y_ptr + r, s)
        r += _NWG


@triton.jit
def _quant_norm_fp8(
    x_ptr, o_ptr, g_ptr, fp8_ptr, scl_ptr, H: tl.constexpr, NB: tl.constexpr, pid, eps, NORMK: tl.constexpr
):
    """Fused residual + RMSNorm + FP8-E4M3 quantization, per-32-element E8M0.

    Reads the residual x and attention output o, applies RMSNorm(x + o) * g, and
    quantizes to FP8 in one pass, so the experts' shared input never has to be
    written to HBM as BF16 first. Each program computes the rms scalar from the full
    (x + o), then quantizes its 32-element blocks."""
    noff = tl.arange(0, NORMK)
    nmask = noff < H
    xall = tl.load(x_ptr + noff, mask=nmask, other=0.0).to(tl.float32) + tl.load(
        o_ptr + noff, mask=nmask, other=0.0
    ).to(tl.float32)
    ss = tl.sum(xall * xall, axis=0)
    rms = 1.0 / tl.sqrt(ss / H + eps)
    pos32 = tl.arange(0, 32)
    b = pid
    while b < NB:
        off = b * 32 + pos32
        xv = tl.load(x_ptr + off).to(tl.float32) + tl.load(o_ptr + off).to(tl.float32)
        gv = tl.load(g_ptr + off).to(tl.float32)
        x = xv * rms * gv
        amax = tl.max(tl.abs(x), axis=0)
        target = amax / 448.0
        u = target.to(tl.int32, bitcast=True)
        raw = (u >> 23) & 0xFF
        raw = raw + tl.where((u & 0x7FFFFF) != 0, 1, 0)
        raw = tl.where(amax > 0.0, tl.minimum(tl.maximum(raw, 0), 255), 0)
        sc = tl.where(raw > 0, tl.exp2((raw - 127).to(tl.float32)), 1.0)
        tl.store(fp8_ptr + off, (x / sc).to(tl.float8e4nv))
        tl.store(scl_ptr + b, raw.to(tl.uint8))
        b += _NWG


@triton.jit
def _gemv_fp4_scaled(
    blk_base,
    scl_base,
    afp8_ptr,
    ascl_ptr,
    y_ptr,
    b_base,
    has_bias,
    N,
    K: tl.constexpr,
    NB: tl.constexpr,
    pid,
    gate_w,
    ACCUM: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
    MTILE: tl.constexpr,
    NSTAGES: tl.constexpr = 3,
    FINALIZE: tl.constexpr = False,
    x_ptr=None,
    o_ptr=None,
):
    """y[n] = gate_w*(sum_k W[n,k]*a[k] + b[n]) via native FP4xFP8 scaled MFMA
    (tl.dot_scaled -> v_mfma_scale_f32_16x16x128_f8f6f4 on gfx950).
    W FP4 e2m1 packed [N, K//2] (low nibble = even k), weight scales e8m0 [N, NB].
    a FP8 e4m3 [K], act scales e8m0 [NB]. Output rows tiled BLOCK_N across programs.

    The weight is the dot_scaled lhs ([BLOCK_N, K]) so its contiguous-K bytes
    coalesce into wide loads; the single-token activation is the broadcast rhs
    ([K, MTILE], only column 0 is real)."""
    SB: tl.constexpr = BLOCK_K // 32
    NK: tl.constexpr = (K + BLOCK_K - 1) // BLOCK_K
    colsN = tl.arange(0, MTILE)
    tile = pid
    half = K // 2
    n_tiles = (N + BLOCK_N - 1) // BLOCK_N
    while tile < n_tiles:
        n = tile * BLOCK_N + tl.arange(0, BLOCK_N)
        nmask = n < N
        acc = tl.zeros((BLOCK_N, MTILE), dtype=tl.float32)
        # pipelined K-loop: tl.range(num_stages) overlaps the next tile's loads with
        # the current dot, which the plain while-loop did not do
        for ki in tl.range(0, NK, num_stages=NSTAGES):
            k0 = ki * BLOCK_K
            kk = k0 + tl.arange(0, BLOCK_K)
            kp = (k0 // 2) + tl.arange(0, BLOCK_K // 2)
            kpmask = kp < half
            sb = (k0 // 32) + tl.arange(0, SB)
            sbmask = sb < NB
            # lhs = weight [BLOCK_N, BLOCK_K] e2m1 (packed K, contiguous -> coalesced)
            w = tl.load(blk_base + n[:, None] * half + kp[None, :], mask=nmask[:, None] & kpmask[None, :], other=0).to(
                tl.uint8
            )
            wscl = tl.load(scl_base + n[:, None] * NB + sb[None, :], mask=nmask[:, None] & sbmask[None, :], other=0)
            # rhs = activation [BLOCK_K, MTILE] e4m3, only column 0 carries the token
            a = tl.load(afp8_ptr + kk[:, None], mask=(colsN[None, :] == 0) & (kk[:, None] < K), other=0.0)
            ascl = tl.load(ascl_ptr + sb[None, :], mask=sbmask[None, :], other=0)
            ascl = tl.broadcast_to(ascl, (MTILE, SB))
            acc = tl.dot_scaled(w, wscl, "e2m1", a, ascl, "e4m3", acc=acc, out_dtype=tl.float32)
        y = tl.sum(tl.where(colsN[None, :] == 0, acc, 0.0), axis=1)  # [BLOCK_N]
        if has_bias:
            y += tl.load(b_base + n, mask=nmask, other=0.0).to(tl.float32)
        y = gate_w * y
        if ACCUM:
            y += tl.load(y_ptr + n, mask=nmask, other=0.0).to(tl.float32)
        tl.store(y_ptr + n, y, mask=nmask)
        if FINALIZE:
            # this program owns rows n of the MoE output; finalize the residual
            # x[n] += o[n] + moe[n] here (same program wrote moe[n]) to fold the
            # final residual into the down phase and drop its barrier
            xv = tl.load(x_ptr + n, mask=nmask, other=0.0).to(tl.float32)
            ov = tl.load(o_ptr + n, mask=nmask, other=0.0).to(tl.float32)
            tl.store(x_ptr + n, xv + ov + y, mask=nmask)
        tile += _NWG


# ───────────────────────── the megakernel ─────────────────────────
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
):
    pid = tl.program_id(0)
    HALF: tl.constexpr = DH // 2
    GROUP: tl.constexpr = NH // NKV
    bars_per_layer = 11  # number of _barrier calls per layer
    base = 0

    for layer in range(L):
        lb = base
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

        # ---- P0+P1 FUSED: RMSNorm(attn) folded into QKV GEMV (no separate norm
        # phase / barrier). Each WG recomputes the norm scale from x_p (stable since
        # the prev-layer barrier) and applies it inline. ----
        if FP8_QKV:
            _gemv_fp8_rmsnorm(wq, wq_s, x_p, na, q_p, True, bq, q_dim, H, pid, eps, BLOCK_M, BLOCK_K, NORMK, NSTAGES, MXFP8_BLK)
            _gemv_fp8_rmsnorm(wk, wk_s, x_p, na, k_p, True, bk, kv_dim, H, pid, eps, BLOCK_M, BLOCK_K, NORMK, NSTAGES, MXFP8_BLK)
            _gemv_fp8_rmsnorm(wv, wv_s, x_p, na, v_p, True, bv, kv_dim, H, pid, eps, BLOCK_M, BLOCK_K, NORMK, NSTAGES, MXFP8_BLK)
        else:
            _gemv_bf16_rmsnorm(wq, x_p, na, q_p, True, bq, q_dim, H, pid, eps, BLOCK_M, BLOCK_K, NORMK)
            _gemv_bf16_rmsnorm(wk, x_p, na, k_p, True, bk, kv_dim, H, pid, eps, BLOCK_M, BLOCK_K, NORMK)
            _gemv_bf16_rmsnorm(wv, x_p, na, v_p, True, bv, kv_dim, H, pid, eps, BLOCK_M, BLOCK_K, NORMK)
        _barrier(bar_p, (lb + 1) * _NWG)

        # ---- P2+P3 FUSED: RoPE, KV-cache append, and attention in one phase (no
        # separate KV-append barrier). The current position's key/value are RoPE'd and
        # used in-register for its own attention term, and also written to the cache
        # for future tokens; the history [lo, pos-1] is read from the cache. Since no
        # attention head reads the just-written cache[pos], the append and the
        # attention can share a phase. ----
        d = tl.arange(0, DH)
        # the KV-head owners append the current k,v (RoPE'd) to the cache for next time
        if pid < NKV:
            h = tl.arange(0, HALF)
            cosv = tl.load(cos_p + h).to(tl.float32)
            sinv = tl.load(sin_p + h).to(tl.float32)
            bidx = pid * DH
            k1 = tl.load(k_p + bidx + h).to(tl.float32)
            k2 = tl.load(k_p + bidx + HALF + h).to(tl.float32)
            tl.store(kcache + pos * kv_dim + bidx + h, k1 * cosv - k2 * sinv)
            tl.store(kcache + pos * kv_dim + bidx + HALF + h, k2 * cosv + k1 * sinv)
            tl.store(vcache + pos * kv_dim + bidx + d, tl.load(v_p + bidx + d).to(tl.float32))
        if pid < NH:
            hh = pid
            kvh = hh // GROUP
            # RoPE this head's q in-register (NeoX half-split), full-width form:
            #   d <  HALF: qr[d] = q[d]*cos[d]      - q[d+HALF]*sin[d]
            #   d >= HALF: qr[d] = q[d]*cos[d-HALF] + q[d-HALF]*sin[d-HALF]
            dlo = d % HALF  # angle index for both halves
            cosd = tl.load(cos_p + dlo).to(tl.float32)
            sind = tl.load(sin_p + dlo).to(tl.float32)
            q_self = tl.load(q_p + hh * DH + d).to(tl.float32)
            partner = tl.where(d < HALF, d + HALF, d - HALF)
            q_part = tl.load(q_p + hh * DH + partner).to(tl.float32)
            qv = tl.where(d < HALF, q_self * cosd - q_part * sind, q_self * cosd + q_part * sind)
            lo = 0
            if SLIDING > 0:
                lo = tl.maximum(0, pos - SLIDING + 1)
            # Flash-decode over the KV history [lo, pos-1] from the cache, online softmax.
            m_i = -1e30
            l_i = 0.0
            acc = tl.zeros((DH,), dtype=tl.float32)
            toff = tl.arange(0, BLOCK_T)
            t0 = lo
            while t0 < pos:
                tt = t0 + toff
                tmask = tt < pos
                kblk = tl.load(
                    kcache + tt[:, None] * kv_dim + kvh * DH + d[None, :], mask=tmask[:, None], other=0.0
                ).to(tl.float32)
                sc = tl.sum(kblk * qv[None, :], axis=1) * scale  # [BLOCK_T]
                sc = tl.where(tmask, sc, -1e30)
                blk_max = tl.max(sc, axis=0)
                mn = tl.maximum(m_i, blk_max)
                al = tl.exp(m_i - mn)
                p = tl.exp(sc - mn)  # [BLOCK_T]
                l_i = l_i * al + tl.sum(p, axis=0)
                vblk = tl.load(
                    vcache + tt[:, None] * kv_dim + kvh * DH + d[None, :], mask=tmask[:, None], other=0.0
                ).to(tl.float32)
                acc = acc * al + tl.sum(p[:, None] * vblk, axis=0)
                m_i = mn
                t0 += BLOCK_T
            # current position pos: RoPE this kv-head's k in-register from k_p, which
            # avoids reading the just-written cache entry and the append barrier.
            kself = tl.load(k_p + kvh * DH + d).to(tl.float32)
            kpart = tl.load(k_p + kvh * DH + partner).to(tl.float32)
            k_pos = tl.where(d < HALF, kself * cosd - kpart * sind, kself * cosd + kpart * sind)
            sc_pos = tl.sum(qv * k_pos, axis=0) * scale
            mn = tl.maximum(m_i, sc_pos)
            al = tl.exp(m_i - mn)
            p_pos = tl.exp(sc_pos - mn)
            l_i = l_i * al + p_pos
            v_pos = tl.load(v_p + kvh * DH + d).to(tl.float32)
            acc = acc * al + p_pos * v_pos
            m_i = mn
            sink = tl.load(sinks + hh).to(tl.float32)
            mn = tl.maximum(m_i, sink)
            al = tl.exp(m_i - mn)
            l_i = l_i * al + tl.exp(sink - mn)
            acc = acc * al / l_i
            tl.store(attn_p + hh * DH + d, acc.to(tl.bfloat16))
        bc = 1  # barriers used so far this layer: QKV(1); KV-append fused into attention
        bc += 1
        _barrier(bar_p, (lb + bc) * _NWG)

        # ---- P4: O-proj -> o_p (the post-attention output). The residual add is
        # deferred to the layer's final accumulation, so there is no separate
        # residual phase or barrier. ----
        if FP8_O:
            _gemv_fp8_tiled(wo, wo_s, attn_p, o_p, True, bo, H, q_dim, pid, BLOCK_M, BLOCK_K, NSTAGES, MXFP8_BLK)
        else:
            _gemv_bf16_tiled(wo, attn_p, o_p, True, bo, H, q_dim, pid, BLOCK_M, BLOCK_K)
        rstep = _NWG * BLOCK_M
        bc += 1
        _barrier(bar_p, (lb + bc) * _NWG)

        # ---- P5: router GEMV, expert-input quant, and accumulator reset all run in
        # one phase -- they only depend on x + o, available since the O-proj barrier.
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
        bc += 1
        _barrier(bar_p, (lb + bc) * _NWG)

        # Every program does the top-k + softmax over the E router logits redundantly
        # in registers (E is tiny) and writes ids_p/gw_p. The writes are identical
        # across programs and each program reads back only what it wrote, so the
        # experts proceed without a separate top-k barrier.
        eoff = tl.arange(0, E)
        work = tl.load(logits_p + eoff).to(tl.float32)
        for kk in range(0, TOPK):
            mval = tl.max(work, axis=0)
            ismax = work == mval
            idx = tl.min(tl.where(ismax, eoff, E), axis=0)
            tl.store(ids_p + kk, idx)
            tl.store(gw_p + kk, mval)
            work = tl.where(eoff == idx, -1e30, work)
        tv = tl.load(gw_p + tl.arange(0, TOPK)).to(tl.float32)
        tmax = tl.max(tv, axis=0)
        ex = tl.exp(tv - tmax)
        sm = tl.sum(ex, axis=0)
        tl.store(gw_p + tl.arange(0, TOPK), ex / sm)

        # ---- P6: experts. The top-k experts are independent until the final
        # accumulation, so each expert phase (gate-up, SwiGLU, down) runs over all
        # experts before the next barrier -- 3 barriers per layer instead of 3 per
        # expert. QUANT uses the native FP4xFP8 scaled multiply; otherwise the FP4
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
        bc += 1
        _barrier(bar_p, (lb + bc) * _NWG)
        # --- phase B: SwiGLU for every expert. QUANT quantizes each 32-element block
        # in place (producer == consumer for that block, no barrier needed). ---
        for slot in range(0, TOPK):
            gu_out = gu_p + slot * (2 * I)
            if QUANT:
                afp8_out = afp8_p + slot * I
                afp8_scl_out = afp8_scl_p + slot * DN_NB
                pos32 = tl.arange(0, 32)
                blk = pid
                while blk < DN_NB:
                    base_i = blk * 32 + pos32
                    gate = tl.load(gu_out + 2 * base_i).to(tl.float32)
                    up = tl.load(gu_out + 2 * base_i + 1).to(tl.float32)
                    gate = tl.minimum(gate, limit)
                    up = tl.maximum(tl.minimum(up, limit), -limit)
                    glu = gate * (1.0 / (1.0 + tl.exp(-alpha * gate)))
                    act = (up + 1.0) * glu  # [32]
                    amax = tl.max(tl.abs(act), axis=0)
                    target = amax / 448.0
                    u = target.to(tl.int32, bitcast=True)
                    raw = (u >> 23) & 0xFF
                    raw = raw + tl.where((u & 0x7FFFFF) != 0, 1, 0)
                    raw = tl.where(amax > 0.0, tl.minimum(tl.maximum(raw, 0), 255), 0)
                    sc = tl.where(raw > 0, tl.exp2((raw - 127).to(tl.float32)), 1.0)
                    tl.store(afp8_out + base_i, (act / sc).to(tl.float8e4nv))
                    tl.store(afp8_scl_out + blk, raw.to(tl.uint8))
                    blk += _NWG
            else:
                act_out = act_p + slot * I
                ii = pid
                while ii < I:
                    gate = tl.load(gu_out + 2 * ii).to(tl.float32)
                    up = tl.load(gu_out + 2 * ii + 1).to(tl.float32)
                    gate = tl.minimum(gate, limit)
                    up = tl.maximum(tl.minimum(up, limit), -limit)
                    glu = gate * (1.0 / (1.0 + tl.exp(-alpha * gate)))
                    tl.store(act_out + ii, ((up + 1.0) * glu).to(tl.bfloat16))
                    ii += _NWG
        bc += 1
        _barrier(bar_p, (lb + bc) * _NWG)
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
            # phase and dropping its barrier.
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
        bc += 1
        _barrier(bar_p, (lb + bc) * _NWG)

        # ---- P7 (BF16 path only): final residual x = x + o + moe, striped. The
        # QUANT path folds this into the last down GEMV above. ----
        if not QUANT:
            base_w = pid * BLOCK_M
            while base_w < H:
                woff = base_w + tl.arange(0, BLOCK_M)
                wm = woff < H
                xv = tl.load(x_p + woff, mask=wm, other=0.0).to(tl.float32)
                ov = tl.load(o_p + woff, mask=wm, other=0.0).to(tl.float32)
                mv = tl.load(moe_p + woff, mask=wm, other=0.0).to(tl.float32)
                tl.store(x_p + woff, xv + ov + mv, mask=wm)
                base_w += rstep
            bc += 1
            _barrier(bar_p, (lb + bc) * _NWG)
        base = lb + bc

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
    base = base + 1
    _barrier(bar_p, base * _NWG)
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


# ───────────────────────── host wrapper ─────────────────────────
class MegaModel:
    def __init__(
        self,
        cfg: GptOssConfig,
        num_layers: int,
        dev="cuda",
        snapshot=None,
        _skip_load=False,
        quant=False,
        fp8_attn=False,
        fp8_components=None,
        fp8_scale_blk=0,
    ):
        self.cfg = cfg
        self.L = num_layers
        self.dev = dev
        self.quant = quant
        # fp8_components: subset of {"qkv","o","router"} to store in FP8. fp8_attn=True
        # is shorthand for all three; an explicit set overrides it (e.g. {"qkv","o"}
        # keeps the router in BF16, the DeepSeek recipe).
        if fp8_components is None:
            fp8_components = {"qkv", "o", "router"} if fp8_attn else set()
        self.fp8_components = set(fp8_components)
        self.fp8_attn = bool(self.fp8_components)
        # FP8 weight-scale granularity along K: 0 = per-row (one scale per output row);
        # 32 = MXFP8-style 1x32 blocks. Block scaling tracks per-block dynamic range.
        self.fp8_scale_blk = fp8_scale_blk if fp8_scale_blk else (1 << 30)
        if not _skip_load:
            w = load_hf_weights(
                GptOssConfig(), snapshot=snapshot, num_layers=num_layers, device="cpu", dtype=torch.bfloat16
            )
            self._pack(w)
            self.cos, self.sin = build_yarn_rope(GptOssConfig(), device=dev)
            self._alloc_buffers()

    @classmethod
    def from_iris(
        cls,
        iris_path: str,
        cfg: GptOssConfig,
        num_layers: int,
        dev="cuda",
        quant=False,
        fp8_attn=False,
        fp8_components=None,
        fp8_scale_blk=0,
    ):
        """Build directly from a converted .iris weight file (mmap -> device)."""
        from convert_to_iris import read_iris_header, load_iris_tensor

        self = cls(
            cfg, num_layers, dev=dev, _skip_load=True, quant=quant, fp8_attn=fp8_attn,
            fp8_components=fp8_components, fp8_scale_blk=fp8_scale_blk,
        )
        _, ents = read_iris_header(iris_path)
        g = lambda nm: load_iris_tensor(iris_path, ents[nm], device=dev)
        L = num_layers
        st = lambda key: torch.stack([g(f"L{l}.{key}") for l in range(L)]).contiguous()
        self.norm_attn = st("norm_attn")
        self.norm_moe = st("norm_moe")
        self.wq = st("w_q")
        self.bq = st("b_q")
        self.wk = st("w_k")
        self.bk = st("b_k")
        self.wv = st("w_v")
        self.bv = st("b_v")
        self.wo = st("w_o")
        self.bo = st("b_o")
        self.sinks = st("sinks")
        self.router_w = st("router_w")
        self.router_b = st("router_b")
        self.gu_blk = st("gate_up_blocks")
        self.gu_scl = st("gate_up_scales")
        self.gu_b = st("gate_up_b")
        self.dn_blk = st("down_blocks")
        self.dn_scl = st("down_scales")
        self.dn_b = st("down_b")
        self.gu_nb = self.gu_blk.shape[3]
        self.dn_nb = self.dn_blk.shape[3]
        self.embed = g("embed")
        self.final_norm = g("final_norm")
        self.lm_head = g("lm_head")
        self.cos, self.sin = build_yarn_rope(GptOssConfig(), device=dev)
        self._alloc_buffers()
        return self

    def _pack(self, w):
        cfg, L, dev = self.cfg, self.L, self.dev
        H, qd, kvd, NH, E, I = (
            cfg.hidden_dim,
            cfg.q_dim,
            cfg.kv_dim,
            cfg.num_heads,
            cfg.num_experts,
            cfg.intermediate_dim,
        )
        st = lambda key, dt: torch.stack([w.layers[l][key] for l in range(L)]).to(dev).to(dt).contiguous()
        self.norm_attn = st("norm_attn", torch.float32)
        self.norm_moe = st("norm_moe", torch.float32)
        self.wq = st("w_q", torch.bfloat16)
        self.bq = st("b_q", torch.bfloat16)
        self.wk = st("w_k", torch.bfloat16)
        self.bk = st("b_k", torch.bfloat16)
        self.wv = st("w_v", torch.bfloat16)
        self.bv = st("b_v", torch.bfloat16)
        self.wo = st("w_o", torch.bfloat16)
        self.bo = st("b_o", torch.bfloat16)
        self.sinks = st("sinks", torch.float32)
        self.router_w = st("router_w", torch.bfloat16)
        self.router_b = st("router_b", torch.bfloat16)
        # experts FP4 stay uint8; reshape to [L,E,...] flattened block dim
        self.gu_blk = torch.stack([w.layers[l]["gate_up_blocks"] for l in range(L)]).to(dev).contiguous()
        self.gu_scl = torch.stack([w.layers[l]["gate_up_scales"] for l in range(L)]).to(dev).contiguous()
        self.gu_b = torch.stack([w.layers[l]["gate_up_b"] for l in range(L)]).to(dev).to(torch.bfloat16).contiguous()
        self.dn_blk = torch.stack([w.layers[l]["down_blocks"] for l in range(L)]).to(dev).contiguous()
        self.dn_scl = torch.stack([w.layers[l]["down_scales"] for l in range(L)]).to(dev).contiguous()
        self.dn_b = torch.stack([w.layers[l]["down_b"] for l in range(L)]).to(dev).to(torch.bfloat16).contiguous()
        self.gu_nb = w.layers[0]["gate_up_blocks"].shape[2]  # n_blocks for K=H
        self.dn_nb = w.layers[0]["down_blocks"].shape[2]  # n_blocks for K=I
        self.embed = w.embed.to(dev)
        self.final_norm = w.final_norm.to(dev)
        self.lm_head = w.lm_head.to(dev)

    def _quant_attn_fp8(self):
        """Convert the BF16 attention/router weights (wq, wk, wv, wo, router_w) to
        FP8-e4m3. Weight scale granularity along K is set by fp8_scale_blk: per-row
        (default, scale shape [L,N]) or 1xB blocks (B=32 -> MXFP8, scale [L,N,K/B]).
        Always builds scale tensors so the kernel signature is uniform; only replaces
        the weights with FP8 for the selected components."""
        dev = self.dev
        B = self.fp8_scale_blk

        def q(W):  # W [L, N, K] bf16 -> (fp8 [L,N,K], scale)
            Wf = W.float()
            Lc, N, K = Wf.shape
            if B >= K:  # per-row: one scale per output row
                s = Wf.abs().amax(dim=2, keepdim=True) / 448.0
                s = torch.where(s > 0, s, torch.ones_like(s))
                return (Wf / s).to(torch.float8_e4m3fn).contiguous(), s.squeeze(2).contiguous()
            # 1xB blocks along K (K assumed divisible by B; attn dims are 2880/4096)
            nsb = K // B
            Wb = Wf.reshape(Lc, N, nsb, B)
            s = Wb.abs().amax(dim=3, keepdim=True) / 448.0
            s = torch.where(s > 0, s, torch.ones_like(s))
            Wq = (Wb / s).to(torch.float8_e4m3fn).reshape(Lc, N, K)
            return Wq.contiguous(), s.squeeze(3).contiguous()  # scale [L, N, nsb]

        # placeholder scales (used only for components left in BF16, so the kernel
        # signature stays uniform)
        z1 = lambda n: torch.ones(self.L, n, dtype=torch.float32, device=dev)
        self.wq_s = z1(self.cfg.q_dim)
        self.wk_s = z1(self.cfg.kv_dim)
        self.wv_s = z1(self.cfg.kv_dim)
        self.wo_s = z1(self.cfg.hidden_dim)
        self.router_w_s = z1(self.cfg.num_experts)
        if "qkv" in self.fp8_components:
            self.wq, self.wq_s = q(self.wq)
            self.wk, self.wk_s = q(self.wk)
            self.wv, self.wv_s = q(self.wv)
        if "o" in self.fp8_components:
            self.wo, self.wo_s = q(self.wo)
        if "router" in self.fp8_components:
            self.router_w, self.router_w_s = q(self.router_w)

    def _alloc_buffers(self):
        cfg, dev = self.cfg, self.dev
        self._quant_attn_fp8()
        H, qd, kvd, E, I, V = (
            cfg.hidden_dim,
            cfg.q_dim,
            cfg.kv_dim,
            cfg.num_experts,
            cfg.intermediate_dim,
            cfg.vocab_size,
        )
        z = lambda n, dt=torch.float32: torch.zeros(n, dtype=dt, device=dev)
        self.x = z(H)
        self.normed = z(H, torch.bfloat16)
        self.q = z(qd)
        self.k = z(kvd)
        self.v = z(kvd)
        self.kcache = torch.zeros(self.L, cfg.max_seq_len, kvd, device=dev)
        self.vcache = torch.zeros(self.L, cfg.max_seq_len, kvd, device=dev)
        self.attn = z(qd, torch.bfloat16)
        self.o = z(H)  # post-attention output; residual deferred to the layer end
        self.logits = z(E)
        self.ids = z(cfg.top_k, torch.int32)
        self.gw = z(cfg.top_k)
        K = cfg.top_k
        # Per-expert intermediates are kept for all top-k experts at once so the
        # expert phases (gate-up, SwiGLU, down) each run across every expert before
        # the next grid barrier, rather than one expert at a time.
        self.gu = z(K * 2 * I)
        self.act = z(K * I, torch.bfloat16)
        self.moe = z(H)
        # FP8 activation-quant scratch (quantized path). nfp8 = MoE-normed activation
        # (K=H), afp8 = SwiGLU output (K=I) per expert; scales are per-32 e8m0 bytes.
        self.nfp8 = z(H, torch.float8_e4m3fn)
        self.nfp8_scl = z(H // 32, torch.uint8)
        self.afp8 = z(K * I, torch.float8_e4m3fn)
        self.afp8_scl = z(K * (I // 32), torch.uint8)
        self.vlogits = z(V)
        self.amax_v = z(NUM_WG)
        self.amax_i = z(NUM_WG, torch.int32)
        self.next_tok = z(1, torch.int32)
        self.bar = z(1, torch.int32)

    @torch.no_grad()
    def step(self, token_id: int, pos: int, dump_logits: bool = False) -> int:
        cfg = self.cfg
        self.x.copy_(self.embed[token_id].float())
        self.bar.zero_()
        gpt_oss_megakernel[(NUM_WG,)](
            self.norm_attn,
            self.norm_moe,
            self.wq,
            self.bq,
            self.wk,
            self.bk,
            self.wv,
            self.bv,
            self.wo,
            self.bo,
            self.sinks,
            self.router_w,
            self.router_b,
            self.wq_s,
            self.wk_s,
            self.wv_s,
            self.wo_s,
            self.router_w_s,
            self.gu_blk,
            self.gu_scl,
            self.gu_b,
            self.dn_blk,
            self.dn_scl,
            self.dn_b,
            self.final_norm,
            self.lm_head,
            self.x,
            self.normed,
            self.q,
            self.k,
            self.v,
            self.kcache,
            self.vcache,
            self.attn,
            self.o,
            self.logits,
            self.ids,
            self.gw,
            self.gu,
            self.act,
            self.moe,
            self.nfp8,
            self.nfp8_scl,
            self.afp8,
            self.afp8_scl,
            self.vlogits,
            self.amax_v,
            self.amax_i,
            self.next_tok,
            self.cos[pos],
            self.sin[pos],
            self.bar,
            pos,
            1.0 / (cfg.head_dim**0.5),
            cfg.rms_eps,
            cfg.swiglu_alpha,
            cfg.swiglu_limit,
            L=self.L,
            H=cfg.hidden_dim,
            q_dim=cfg.q_dim,
            kv_dim=cfg.kv_dim,
            NH=cfg.num_heads,
            NKV=cfg.num_kv_heads,
            DH=cfg.head_dim,
            E=cfg.num_experts,
            TOPK=cfg.top_k,
            I=cfg.intermediate_dim,
            V=cfg.vocab_size,
            SLIDING=cfg.sliding_window,
            GU_NB=self.gu_nb,
            DN_NB=self.dn_nb,
            max_seq=cfg.max_seq_len,
            BLOCK_K=1024,
            BLOCK_KI=256,
            BLOCK_M=8,
            BLOCK_M_LM=16,
            NORMK=triton.next_power_of_2(cfg.hidden_dim),
            QUANT=self.quant,
            BLOCK_NQ=32,
            BLOCK_ND=16,
            BLOCK_KQ=1024,
            MTILE=16,
            BLOCK_T=64,
            NSTAGES=3,
            FP8_QKV=("qkv" in self.fp8_components),
            FP8_O=("o" in self.fp8_components),
            FP8_ROUTER=("router" in self.fp8_components),
            MXFP8_BLK=self.fp8_scale_blk,
            DUMP_LOGITS=dump_logits,
            num_warps=4,
        )
        torch.cuda.synchronize()
        if dump_logits:
            return int(self.next_tok.item()), self.vlogits.clone()
        return int(self.next_tok.item())


def main():
    import argparse

    ap = argparse.ArgumentParser()
    ap.add_argument("--prompt", default="The capital of France is")
    ap.add_argument("--max-new", type=int, default=5)
    ap.add_argument("--layers", type=int, default=0)
    ap.add_argument("--snapshot", default=None)
    ap.add_argument("--model", default=None, help="path to a converted .iris weight file")
    ap.add_argument("--quant", action="store_true", help="use native FP4xFP8 scaled-MFMA experts")
    ap.add_argument(
        "--fp8-attn",
        action="store_true",
        help="store attention/router weights in FP8 (weight-only; activations stay fp32). "
        "BF16 is the default and is more accurate.",
    )
    args = ap.parse_args()

    cfg = GptOssConfig()
    L = args.layers if args.layers > 0 else cfg.num_layers
    tok = load_tokenizer(args.snapshot)
    ids = tok.encode(args.prompt)
    print(f"prompt={args.prompt!r} ids={ids} quant={args.quant}")

    import time

    t0 = time.time()
    if args.model:
        model = MegaModel.from_iris(args.model, cfg, L, quant=args.quant, fp8_attn=args.fp8_attn)
        print(f"loaded {L} layers from {args.model} in {time.time()-t0:.1f}s")
    else:
        model = MegaModel(cfg, L, snapshot=args.snapshot, quant=args.quant, fp8_attn=args.fp8_attn)
        print(f"loaded {L} layers (HF) in {time.time()-t0:.1f}s")

    pos = 0
    nxt = None
    for tid in ids:
        nxt = model.step(tid, pos)
        pos += 1
    out = [nxt]
    for _ in range(args.max_new - 1):
        nxt = model.step(nxt, pos)
        pos += 1
        out.append(nxt)
    print("generated ids:", out)
    print("generated text:", repr(tok.decode(out)))


if __name__ == "__main__":
    main()
