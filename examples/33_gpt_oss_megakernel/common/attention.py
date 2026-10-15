# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
"""Decode-time attention device helpers: RoPE + KV-cache append and the per-head
YaRN-RoPE flash-decode with attention sinks.

These are split out so the future attention-GPU kernel can call the same code the
single-GPU megakernel uses. Both are called once per program under a pid guard
(pid < NKV for the KV-cache append, pid < NH for the per-head decode)."""

import triton
import triton.language as tl


@triton.jit
def _rope_kv_append(
    k_p,
    v_p,
    cos_p,
    sin_p,
    kcache,
    vcache,
    pos,
    pid,
    kv_dim: tl.constexpr,
    DH: tl.constexpr,
    HALF: tl.constexpr,
):
    """KV-head owner pid appends the current token's RoPE'd key and its value to the
    per-layer KV cache at position `pos` (NeoX half-split RoPE). Call under
    `if pid < NKV`."""
    d = tl.arange(0, DH)
    h = tl.arange(0, HALF)
    cosv = tl.load(cos_p + h).to(tl.float32)
    sinv = tl.load(sin_p + h).to(tl.float32)
    bidx = pid * DH
    k1 = tl.load(k_p + bidx + h).to(tl.float32)
    k2 = tl.load(k_p + bidx + HALF + h).to(tl.float32)
    tl.store(kcache + pos * kv_dim + bidx + h, k1 * cosv - k2 * sinv)
    tl.store(kcache + pos * kv_dim + bidx + HALF + h, k2 * cosv + k1 * sinv)
    tl.store(vcache + pos * kv_dim + bidx + d, tl.load(v_p + bidx + d).to(tl.float32))


@triton.jit
def _flash_decode_head(
    q_p,
    k_p,
    v_p,
    cos_p,
    sin_p,
    kcache,
    vcache,
    sinks,
    attn_p,
    pos,
    scale,
    pid,
    kv_dim: tl.constexpr,
    DH: tl.constexpr,
    HALF: tl.constexpr,
    GROUP: tl.constexpr,
    SLIDING: tl.constexpr,
    BLOCK_T: tl.constexpr,
):
    """Per-query-head flash decode: RoPE this head's q in-register, run an online
    softmax over the KV history [lo, pos-1] from the cache, fold in the current
    position (RoPE'd from k_p/v_p, avoiding a read of the just-written cache entry)
    and the per-head attention sink, and write the head output to attn_p. Call under
    `if pid < NH`; the query head is hh = pid, its KV head kvh = hh // GROUP."""
    d = tl.arange(0, DH)
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
        kblk = tl.load(kcache + tt[:, None] * kv_dim + kvh * DH + d[None, :], mask=tmask[:, None], other=0.0).to(
            tl.float32
        )
        sc = tl.sum(kblk * qv[None, :], axis=1) * scale  # [BLOCK_T]
        sc = tl.where(tmask, sc, -1e30)
        blk_max = tl.max(sc, axis=0)
        mn = tl.maximum(m_i, blk_max)
        al = tl.exp(m_i - mn)
        p = tl.exp(sc - mn)  # [BLOCK_T]
        l_i = l_i * al + tl.sum(p, axis=0)
        vblk = tl.load(vcache + tt[:, None] * kv_dim + kvh * DH + d[None, :], mask=tmask[:, None], other=0.0).to(
            tl.float32
        )
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
