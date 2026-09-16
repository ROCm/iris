# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
"""BF16-weight batch-1 GEMV device helpers (plain, tiled, and RMSNorm-fused).

All variants keep weights and activations in BF16/fp32 and are tiled so the
contiguous-K loads vectorize to dwordx4. Output row tiles are strided across the
grid by tl.num_programs(0), so the helpers are grid-size agnostic and reusable
across the single-GPU megakernel and the per-GPU multi-GPU kernels."""

import triton
import triton.language as tl


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
    npid = tl.num_programs(0)
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
        r += npid
