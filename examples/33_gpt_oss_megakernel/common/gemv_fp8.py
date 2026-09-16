# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
"""FP8-weight batch-1 GEMV device helpers (weight-only FP8, fp32 activations).

Weights are FP8-e4m3 with a scale of granularity SCALE_BLK along K: SCALE_BLK>=K
is per-row (one scale per output row), SCALE_BLK=32 is MXFP8-style 1x32 blocks.
The activation stays fp32 so the win is purely halving the weight bytes; the dot
is a scalar-FMA over a [BLOCK_M, BLOCK_K] tile (an MFMA broadcast wastes the matrix
engine on a batch-1 GEMV). Output row tiles strided by tl.num_programs(0)."""

import triton
import triton.language as tl


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
