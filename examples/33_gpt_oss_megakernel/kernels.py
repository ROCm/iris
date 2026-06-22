# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
"""
Triton phase kernels for GPT-OSS-120B batch-1 decode. These are validated one by
one against reference.py, then fused into the persistent megakernel. BF16-first:
non-expert weights are BF16; expert weights are MXFP4 dequantized in-kernel
(dequant_fp4 below) — same data the MXFP4-MFMA follow-on will consume directly.

All kernels are batch-1 (single token) GEMV / attention. Conventions:
  - residual stream x is fp32 [H]
  - weights W are bf16 [M, K] row-major; GEMV computes y[m]=sum_k W[m,k]*x[k]+b[m]
  - fp32 accumulation everywhere
"""

from __future__ import annotations

import torch
import triton
import triton.language as tl


# ───────────────────────── RMSNorm ─────────────────────────
@triton.jit
def _rmsnorm_kernel(x_ptr, g_ptr, out_ptr, H: tl.constexpr, eps, BLOCK: tl.constexpr):
    offs = tl.arange(0, BLOCK)
    mask = offs < H
    x = tl.load(x_ptr + offs, mask=mask, other=0.0).to(tl.float32)
    ss = tl.sum(x * x, axis=0)
    rms = 1.0 / tl.sqrt(ss / H + eps)
    g = tl.load(g_ptr + offs, mask=mask, other=0.0).to(tl.float32)
    tl.store(out_ptr + offs, x * rms * g, mask=mask)


def rmsnorm(x: torch.Tensor, gamma: torch.Tensor, eps: float) -> torch.Tensor:
    H = x.numel()
    out = torch.empty_like(x, dtype=torch.float32)
    BLOCK = triton.next_power_of_2(H)
    _rmsnorm_kernel[(1,)](x, gamma, out, H, eps, BLOCK=BLOCK, num_warps=8)
    return out


# ───────────────────────── GEMV (y = W@x + b) ─────────────────────────
@triton.jit
def _gemv_kernel(
    w_ptr, x_ptr, b_ptr, y_ptr, M, K: tl.constexpr, HAS_BIAS: tl.constexpr, BLOCK_M: tl.constexpr, BLOCK_K: tl.constexpr
):
    pid = tl.program_id(0)
    rows = pid * BLOCK_M + tl.arange(0, BLOCK_M)
    rmask = rows < M
    acc = tl.zeros((BLOCK_M,), dtype=tl.float32)
    for k0 in range(0, K, BLOCK_K):
        koff = k0 + tl.arange(0, BLOCK_K)
        kmask = koff < K
        xk = tl.load(x_ptr + koff, mask=kmask, other=0.0).to(tl.float32)  # [BK]
        w = tl.load(w_ptr + rows[:, None] * K + koff[None, :], mask=rmask[:, None] & kmask[None, :], other=0.0).to(
            tl.float32
        )
        acc += tl.sum(w * xk[None, :], axis=1)
    if HAS_BIAS:
        acc += tl.load(b_ptr + rows, mask=rmask, other=0.0).to(tl.float32)
    tl.store(y_ptr + rows, acc, mask=rmask)


def gemv(
    W: torch.Tensor, x: torch.Tensor, b: torch.Tensor | None = None, BLOCK_M: int = 64, BLOCK_K: int = 256
) -> torch.Tensor:
    M, K = W.shape
    y = torch.empty(M, dtype=torch.float32, device=x.device)
    grid = (triton.cdiv(M, BLOCK_M),)
    _gemv_kernel[grid](
        W, x, b if b is not None else x, y, M, K, HAS_BIAS=b is not None, BLOCK_M=BLOCK_M, BLOCK_K=BLOCK_K, num_warps=4
    )
    return y


# ───────────────────────── RoPE (NeoX) + KV append ─────────────────────────
@triton.jit
def _rope_qk_kernel(
    q_ptr,
    k_ptr,
    cos_ptr,
    sin_ptr,
    kcache_ptr,
    vcache_ptr,
    v_ptr,
    pos,
    NH: tl.constexpr,
    NKV: tl.constexpr,
    DH: tl.constexpr,
    kv_dim: tl.constexpr,
    HALF: tl.constexpr,
):
    # one program; rotate all heads. DH<=128 so vectorize over HALF.
    h = tl.arange(0, HALF)
    cos = tl.load(cos_ptr + h).to(tl.float32)
    sin = tl.load(sin_ptr + h).to(tl.float32)
    # q heads
    for hd in range(0, NH):
        base = hd * DH
        x1 = tl.load(q_ptr + base + h).to(tl.float32)
        x2 = tl.load(q_ptr + base + HALF + h).to(tl.float32)
        tl.store(q_ptr + base + h, x1 * cos - x2 * sin)
        tl.store(q_ptr + base + HALF + h, x2 * cos + x1 * sin)
    # k heads -> write into cache at pos
    for hd in range(0, NKV):
        base = hd * DH
        x1 = tl.load(k_ptr + base + h).to(tl.float32)
        x2 = tl.load(k_ptr + base + HALF + h).to(tl.float32)
        r1 = x1 * cos - x2 * sin
        r2 = x2 * cos + x1 * sin
        tl.store(kcache_ptr + pos * kv_dim + base + h, r1)
        tl.store(kcache_ptr + pos * kv_dim + base + HALF + h, r2)
    # v -> cache (no rope)
    vo = tl.arange(0, kv_dim)
    vv = tl.load(v_ptr + vo).to(tl.float32)
    tl.store(vcache_ptr + pos * kv_dim + vo, vv)


def rope_and_cache(q, k, v, cos_row, sin_row, kcache, vcache, pos, NH, NKV, DH):
    kv_dim = NKV * DH
    HALF = DH // 2
    _rope_qk_kernel[(1,)](q, k, cos_row, sin_row, kcache, vcache, v, pos, NH, NKV, DH, kv_dim, HALF, num_warps=4)


# ───────────────────────── Attention decode (GQA + sink + window) ─────────────────────────
@triton.jit
def _attn_kernel(
    q_ptr,
    kcache_ptr,
    vcache_ptr,
    sink_ptr,
    out_ptr,
    pos,
    lo,
    scale,
    NKV: tl.constexpr,
    DH: tl.constexpr,
    kv_dim: tl.constexpr,
    GROUP: tl.constexpr,
):
    h = tl.program_id(0)  # query head
    kvh = h // GROUP
    d = tl.arange(0, DH)
    q = tl.load(q_ptr + h * DH + d).to(tl.float32)
    m = -1e30
    l = 0.0
    acc = tl.zeros((DH,), dtype=tl.float32)
    for t in range(lo, pos + 1):
        kt = tl.load(kcache_ptr + t * kv_dim + kvh * DH + d).to(tl.float32)
        score = tl.sum(q * kt, axis=0) * scale
        m_new = tl.maximum(m, score)
        alpha = tl.exp(m - m_new)
        p = tl.exp(score - m_new)
        l = l * alpha + p
        vt = tl.load(vcache_ptr + t * kv_dim + kvh * DH + d).to(tl.float32)
        acc = acc * alpha + p * vt
        m = m_new
    # sink term: extra logit in denominator only
    sink = tl.load(sink_ptr + h).to(tl.float32)
    m_new = tl.maximum(m, sink)
    alpha = tl.exp(m - m_new)
    l = l * alpha + tl.exp(sink - m_new)
    acc = acc * alpha
    out = acc / l
    tl.store(out_ptr + h * DH + d, out)


def attention(q, kcache, vcache, sinks, pos, window, scale, NH, NKV, DH):
    kv_dim = NKV * DH
    GROUP = NH // NKV
    lo = max(0, pos - window + 1) if window > 0 else 0
    out = torch.empty(NH * DH, dtype=torch.float32, device=q.device)
    _attn_kernel[(NH,)](q, kcache, vcache, sinks, out, pos, lo, scale, NKV, DH, kv_dim, GROUP, num_warps=2)
    return out


# ───────────────────────── Router top-4 softmax ─────────────────────────
@triton.jit
def _topk4_kernel(logits_ptr, ids_ptr, w_ptr, E: tl.constexpr, K: tl.constexpr, BLOCK_E: tl.constexpr):
    offs = tl.arange(0, BLOCK_E)
    mask = offs < E
    lg = tl.load(logits_ptr + offs, mask=mask, other=-1e30).to(tl.float32)
    # iteratively extract top-K
    topv = tl.zeros((K,), dtype=tl.float32)  # not used as array store; do scalar loop
    # We can't easily index dynamic; emulate with K passes mutating lg
    # Pass k: find argmax, record, set to -inf
    work = lg
    for kk in range(0, K):
        mval = tl.max(work, axis=0)
        # index of first occurrence
        is_max = work == mval
        idx = tl.min(tl.where(is_max, offs, E), axis=0)
        tl.store(ids_ptr + kk, idx)
        tl.store(w_ptr + kk, mval)  # temporarily store logit; softmax below on host or 2nd pass
        work = tl.where(offs == idx, -1e30, work)


def router_topk(logits: torch.Tensor, E: int, K: int = 4):
    ids = torch.empty(K, dtype=torch.int32, device=logits.device)
    vals = torch.empty(K, dtype=torch.float32, device=logits.device)
    BLOCK_E = triton.next_power_of_2(E)
    _topk4_kernel[(1,)](logits, ids, vals, E, K, BLOCK_E=BLOCK_E, num_warps=4)
    # softmax over the K logits (tiny; do on device with torch)
    w = torch.softmax(vals, dim=0)
    return ids, w


# ───────────────────────── FP4 dequant ─────────────────────────
# E2M1 LUT as a small constant tensor; dequant a [R, n_blk, 16] uint8 + [R,n_blk] e8m0.
@triton.jit
def _dequant_fp4_kernel(blk_ptr, scale_ptr, out_ptr, R, C: tl.constexpr, NBLK: tl.constexpr, BLOCK_C: tl.constexpr):
    r = tl.program_id(0)
    # process row r, all C cols
    c = tl.arange(0, BLOCK_C)
    cmask = c < C
    byte_idx = c // 2  # which packed byte
    nib_hi = (c % 2) == 1
    raw = tl.load(blk_ptr + r * (C // 2) + byte_idx, mask=cmask, other=0).to(tl.int32)
    nib = tl.where(nib_hi, (raw >> 4) & 0xF, raw & 0xF)
    sign = (nib & 0x8) != 0
    mag_idx = nib & 0x7
    # LUT {0,.5,1,1.5,2,3,4,6}
    mag = tl.where(
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
    val = tl.where(sign, -mag, mag)
    blk = c // 32
    se = tl.load(scale_ptr + r * NBLK + blk, mask=cmask, other=0).to(tl.int32)
    sc = tl.where(se > 0, tl.exp2((se - 127).to(tl.float32)), 0.0)
    tl.store(out_ptr + r * C + c, val * sc, mask=cmask)


def dequant_fp4(blocks: torch.Tensor, scales: torch.Tensor) -> torch.Tensor:
    # blocks [R, nblk, 16] uint8 -> [R, C]; scales [R, nblk]
    R = blocks.shape[0]
    nblk = blocks.shape[1]
    C = nblk * 32
    blk2 = blocks.reshape(R, nblk * 16).contiguous()
    out = torch.empty(R, C, dtype=torch.float32, device=blocks.device)
    BLOCK_C = triton.next_power_of_2(C)
    _dequant_fp4_kernel[(R,)](blk2, scales.contiguous(), out, R, C, nblk, BLOCK_C=BLOCK_C, num_warps=8)
    return out


# ───────────────────────── SwiGLU-OAI ─────────────────────────
@triton.jit
def _swiglu_kernel(gu_ptr, out_ptr, I: tl.constexpr, alpha, limit, BLOCK: tl.constexpr):
    # gu layout: interleaved gate,up : gate=gu[2i], up=gu[2i+1]
    i = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    mask = i < I
    gate = tl.load(gu_ptr + 2 * i, mask=mask, other=0.0).to(tl.float32)
    up = tl.load(gu_ptr + 2 * i + 1, mask=mask, other=0.0).to(tl.float32)
    gate = tl.minimum(gate, limit)
    up = tl.maximum(tl.minimum(up, limit), -limit)
    glu = gate * (1.0 / (1.0 + tl.exp(-alpha * gate)))
    tl.store(out_ptr + i, (up + 1.0) * glu, mask=mask)


def swiglu(gu: torch.Tensor, I: int, alpha: float, limit: float) -> torch.Tensor:
    out = torch.empty(I, dtype=torch.float32, device=gu.device)
    BLOCK = 256
    _swiglu_kernel[(triton.cdiv(I, BLOCK),)](gu, out, I, alpha, limit, BLOCK=BLOCK)
    return out


# ───────────────────────── argmax ─────────────────────────
@triton.jit
def _argmax_kernel(x_ptr, out_ptr, N, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offs < N
    x = tl.load(x_ptr + offs, mask=mask, other=-1e30).to(tl.float32)
    mval = tl.max(x, axis=0)
    is_max = x == mval
    idx = tl.min(tl.where(is_max, offs, N), axis=0)
    # write partial (value,index) for this block
    tl.store(out_ptr + 2 * pid, mval)
    tl.store(out_ptr + 2 * pid + 1, idx.to(tl.float32))


def argmax(x: torch.Tensor) -> int:
    N = x.numel()
    BLOCK = 4096
    nb = triton.cdiv(N, BLOCK)
    partial = torch.empty(2 * nb, dtype=torch.float32, device=x.device)
    _argmax_kernel[(nb,)](x, partial, N, BLOCK=BLOCK, num_warps=8)
    pv = partial[0::2]
    pi = partial[1::2]
    best = int(pi[int(torch.argmax(pv))].item())
    return best
