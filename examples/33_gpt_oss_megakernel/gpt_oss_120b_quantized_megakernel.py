# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
"""
GPT-OSS 120B quantized megakernel — a SINGLE persistent Triton kernel that runs
BOTH attention AND mixture-of-experts for ALL 36 layers of GPT-OSS-120B on ONE
GPU, for the batch-1 (decode) GEMV case.

This collapses ROCm/cosmic's multi-GPU assembly design (1 GPU attention + 4 GPUs
MoE) into one Triton kernel. The kernel is launched once per token; internally it
loops over all layers, synchronizing its NUM_WG persistent programs at each phase
with a grid-wide monotonic-counter barrier. Attention and MoE are different phases
of the same resident kernel — never separate launches, never separate GPUs.

Precision: non-expert weights (attn/router/embed/lm_head) are BF16; the 128
experts are MXFP4 (E2M1 + per-32 E8M0 scales). Two expert-GEMV compute paths,
selected by QUANT:
  - QUANT=False (default): FP4 weights dequantized to BF16 in the GEMV inner loop,
    BF16 activations (W4A16). Bit-faithful to the BF16 reference.
  - QUANT=True (--quant): activations dynamically quantized to FP8-E4M3 (per-32
    E8M0) and multiplied with FP4 weights via tl.dot_scaled, which lowers to the
    native gfx950 v_mfma_scale_f32_16x16x128_f8f6f4 tensor-core op (W4A8). ~2.8x
    faster; not bit-identical to BF16 but the standard production regime.

Validated greedy output: "The capital of France is" -> token 12650 " Paris" in
both paths (matches the PyTorch reference reference.py).

Architecture (per layer, all inside the persistent kernel):
  RMSNorm -> QKV+bias -> NeoX YaRN RoPE -> KV append -> GQA flash-decode with
  per-head attention SINK and alternating sliding(128)/full window -> O-proj+bias
  + residual -> RMSNorm -> router top-4 (softmax-after-topk) -> 4x SwiGLU-OAI
  experts (FP4 weights) -> gate-weighted sum + residual.
Then: final RMSNorm -> lm_head -> argmax.

Performance notes:
  - GEMVs use block-of-rows tiling (2D [BLOCK_M, BLOCK_K] weight loads) with
    max_contiguous/multiple_of hints so the compiler emits wide (dwordx4) loads.
  - The attention RMSNorm is fused into the QKV GEMV and the MoE RMSNorm into the
    router GEMV (each program recomputes the tiny rms scalar from x, barrier-free).
    RoPE is folded into attention. These remove the serial pid0 phases + their grid
    barriers. Residual/zeroing are striped across all programs, not pid0-serial.
"""

from __future__ import annotations

import torch
import triton
import triton.language as tl

from reference import GptOssConfig, build_yarn_rope
from load_hf import load_hf_weights
from tokenizer_util import load_tokenizer

NUM_WG = 256  # one persistent program per CU on MI355X
_NWG = tl.constexpr(NUM_WG)


# ───────────────────────── device helpers ─────────────────────────
@triton.jit
def _barrier(bar_ptr, target):
    tl.debug_barrier()
    tl.atomic_add(bar_ptr, 1, sem="release")
    done = 0
    while done == 0:
        cur = tl.atomic_add(bar_ptr, 0, sem="acquire")
        if cur >= target:
            done = 1


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
    w_base, x_ptr, y_ptr, has_bias, b_base, M, K: tl.constexpr, pid, BLOCK_M: tl.constexpr, BLOCK_K: tl.constexpr
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
    tile = pid
    while tile < n_tiles:
        rows = tile * BLOCK_M + mo
        rmask = rows < M
        acc = tl.zeros((BLOCK_M, BLOCK_K), dtype=tl.float32)
        k0 = 0
        while k0 < K:
            kk = k0 + ko
            kmask = kk < K
            w = tl.load(w_base + rows[:, None] * K + kk[None, :], mask=rmask[:, None] & kmask[None, :], other=0.0).to(
                tl.float32
            )
            x = tl.load(x_ptr + kk, mask=kmask, other=0.0).to(tl.float32)
            acc += w * x[None, :]
            k0 += BLOCK_K
        s = tl.sum(acc, axis=1)  # [BLOCK_M]
        if has_bias:
            s += tl.load(b_base + rows, mask=rmask, other=0.0).to(tl.float32)
        tl.store(y_ptr + rows, s, mask=rmask)
        tile += npid


@triton.jit
def _store_rmsnorm(x_ptr, g_ptr, out_ptr, H: tl.constexpr, pid, eps, BLOCK_M: tl.constexpr, NORMK: tl.constexpr):
    """Materialize normed = rmsnorm(x)*g into out_ptr (bf16), striped across WGs.
    Each program computes the rms scalar from the full x (cheap), then writes its
    row slices. Replaces the pid0-serial RMSNorm; needs a barrier after (callers add)."""
    noff = tl.arange(0, NORMK)
    nmask = noff < H
    xall = tl.load(x_ptr + noff, mask=nmask, other=0.0).to(tl.float32)
    ss = tl.sum(xall * xall, axis=0)
    rms = 1.0 / tl.sqrt(ss / H + eps)
    base = pid * BLOCK_M
    step = tl.num_programs(0) * BLOCK_M
    while base < H:
        off = base + tl.arange(0, BLOCK_M)
        m = off < H
        xv = tl.load(x_ptr + off, mask=m, other=0.0).to(tl.float32)
        g = tl.load(g_ptr + off, mask=m, other=0.0).to(tl.float32)
        tl.store(out_ptr + off, (xv * rms * g).to(tl.bfloat16), mask=m)
        base += step


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
    tile = pid
    while tile < n_tiles:
        rows = tile * BLOCK_M + mo
        rmask = rows < M
        acc = tl.zeros((BLOCK_M, BLOCK_K), dtype=tl.float32)
        k0 = 0
        while k0 < H:
            kk = k0 + ko
            kmask = kk < H
            xk = tl.load(x_ptr + kk, mask=kmask, other=0.0).to(tl.float32)
            gk = tl.load(g_ptr + kk, mask=kmask, other=0.0).to(tl.float32)
            nk = xk * rms * gk  # normed activation chunk
            w = tl.load(w_base + rows[:, None] * H + kk[None, :], mask=rmask[:, None] & kmask[None, :], other=0.0).to(
                tl.float32
            )
            acc += w * nk[None, :]
            k0 += BLOCK_K
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
def _quant_act_fp8(x_ptr, fp8_ptr, scl_ptr, K, NB: tl.constexpr, pid):
    """Dynamic FP8-E4M3 activation quant, per-32 E8M0 (amax/448), matching cosmic.
    x_ptr fp32-ish [K] -> fp8_ptr (float8e4nv) [K], scl_ptr (uint8 e8m0) [NB].
    Each program handles a strided set of 32-element blocks."""
    pos32 = tl.arange(0, 32)
    b = pid
    while b < NB:
        x = tl.load(x_ptr + b * 32 + pos32).to(tl.float32)
        amax = tl.max(tl.abs(x), axis=0)
        target = amax / 448.0
        u = target.to(tl.int32, bitcast=True)  # raw IEEE-754 bits, not numeric cast
        raw = (u >> 23) & 0xFF
        raw = raw + tl.where((u & 0x7FFFFF) != 0, 1, 0)
        raw = tl.where(amax > 0.0, tl.minimum(tl.maximum(raw, 0), 255), 0)
        sc = tl.where(raw > 0, tl.exp2((raw - 127).to(tl.float32)), 1.0)
        q = (x / sc).to(tl.float8e4nv)
        tl.store(fp8_ptr + b * 32 + pos32, q)
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
):
    """y[n] = gate_w*(sum_k W[n,k]*a[k] + b[n]) via native FP4xFP8 scaled MFMA
    (tl.dot_scaled -> v_mfma_scale_f32_16x16x128_f8f6f4 on gfx950).
    W FP4 e2m1 packed [N, K//2] (low nibble = even k), weight scales e8m0 [N, NB].
    a FP8 e4m3 [K], act scales e8m0 [NB]. Output rows tiled BLOCK_N across programs."""
    SB: tl.constexpr = BLOCK_K // 32
    rowsM = tl.arange(0, MTILE)
    tile = pid
    half = K // 2
    n_tiles = (N + BLOCK_N - 1) // BLOCK_N
    while tile < n_tiles:
        n = tile * BLOCK_N + tl.arange(0, BLOCK_N)
        nmask = n < N
        acc = tl.zeros((MTILE, BLOCK_N), dtype=tl.float32)
        k0 = 0
        while k0 < K:
            kk = k0 + tl.arange(0, BLOCK_K)
            kmask = kk < K
            kp = (k0 // 2) + tl.arange(0, BLOCK_K // 2)
            kpmask = kp < half
            sb = (k0 // 32) + tl.arange(0, SB)
            sbmask = sb < NB
            a = tl.load(afp8_ptr + kk[None, :], mask=(rowsM[:, None] == 0) & kmask[None, :], other=0.0)
            ascl = tl.load(ascl_ptr + sb[None, :], mask=sbmask[None, :], other=0)
            ascl = tl.broadcast_to(ascl, (MTILE, SB))
            w = tl.load(blk_base + n[None, :] * half + kp[:, None], mask=nmask[None, :] & kpmask[:, None], other=0).to(
                tl.uint8
            )
            wscl = tl.load(scl_base + n[:, None] * NB + sb[None, :], mask=nmask[:, None] & sbmask[None, :], other=0)
            acc = tl.dot_scaled(a, ascl, "e4m3", w, wscl, "e2m1", acc=acc, out_dtype=tl.float32)
            k0 += BLOCK_K
        y = tl.sum(tl.where(rowsM[:, None] == 0, acc, 0.0), axis=0)  # [BLOCK_N]
        if has_bias:
            y += tl.load(b_base + n, mask=nmask, other=0.0).to(tl.float32)
        y = gate_w * y
        if ACCUM:
            y += tl.load(y_ptr + n, mask=nmask, other=0.0).to(tl.float32)
        tl.store(y_ptr + n, y, mask=nmask)
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
    NORMK: tl.constexpr,
    QUANT: tl.constexpr,
    BLOCK_NQ: tl.constexpr,
    BLOCK_KQ: tl.constexpr,
    MTILE: tl.constexpr,
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
        kcache = kcache_p + layer * max_seq * kv_dim
        vcache = vcache_p + layer * max_seq * kv_dim

        # ---- P0+P1 FUSED: RMSNorm(attn) folded into QKV GEMV (no separate norm
        # phase / barrier). Each WG recomputes the norm scale from x_p (stable since
        # the prev-layer barrier) and applies it inline. ----
        _gemv_bf16_rmsnorm(wq, x_p, na, q_p, True, bq, q_dim, H, pid, eps, BLOCK_M, BLOCK_K, NORMK)
        _gemv_bf16_rmsnorm(wk, x_p, na, k_p, True, bk, kv_dim, H, pid, eps, BLOCK_M, BLOCK_K, NORMK)
        _gemv_bf16_rmsnorm(wv, x_p, na, v_p, True, bv, kv_dim, H, pid, eps, BLOCK_M, BLOCK_K, NORMK)
        _barrier(bar_p, (lb + 1) * _NWG)

        # ---- P2+P3 FUSED: RoPE folded into attention. KV-cache append done by the
        # KV-head owners (pid < NKV); each attention head RoPEs its own q in-register.
        # No separate RoPE phase / barrier. ----
        if pid < NKV:
            # this program owns kv-head `pid`: RoPE k + append k,v to cache at pos
            h = tl.arange(0, HALF)
            cosv = tl.load(cos_p + h).to(tl.float32)
            sinv = tl.load(sin_p + h).to(tl.float32)
            bidx = pid * DH
            k1 = tl.load(k_p + bidx + h).to(tl.float32)
            k2 = tl.load(k_p + bidx + HALF + h).to(tl.float32)
            tl.store(kcache + pos * kv_dim + bidx + h, k1 * cosv - k2 * sinv)
            tl.store(kcache + pos * kv_dim + bidx + HALF + h, k2 * cosv + k1 * sinv)
            vd = tl.arange(0, DH)
            tl.store(vcache + pos * kv_dim + bidx + vd, tl.load(v_p + bidx + vd).to(tl.float32))
        _barrier(bar_p, (lb + 2) * _NWG)

        # ---- P3: attention (one head per program, heads 0..NH-1) ----
        if pid < NH:
            hh = pid
            kvh = hh // GROUP
            d = tl.arange(0, DH)
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
            m_i = -1e30
            l_i = 0.0
            acc = tl.zeros((DH,), dtype=tl.float32)
            t = lo
            while t <= pos:
                kt = tl.load(kcache + t * kv_dim + kvh * DH + d).to(tl.float32)
                sc = tl.sum(qv * kt, axis=0) * scale
                mn = tl.maximum(m_i, sc)
                al = tl.exp(m_i - mn)
                p = tl.exp(sc - mn)
                l_i = l_i * al + p
                vt = tl.load(vcache + t * kv_dim + kvh * DH + d).to(tl.float32)
                acc = acc * al + p * vt
                m_i = mn
                t += 1
            sink = tl.load(sinks + hh).to(tl.float32)
            mn = tl.maximum(m_i, sink)
            al = tl.exp(m_i - mn)
            l_i = l_i * al + tl.exp(sink - mn)
            acc = acc * al / l_i
            tl.store(attn_p + hh * DH + d, acc.to(tl.bfloat16))
        bc = 2  # barriers used so far this layer: QKV(1), KV-append(2)
        bc += 1
        _barrier(bar_p, (lb + bc) * _NWG)

        # ---- P4: O-proj (striped) -> moe_p, then striped residual into x_p ----
        _gemv_bf16_tiled(wo, attn_p, moe_p, True, bo, H, q_dim, pid, BLOCK_M, BLOCK_K)
        bc += 1
        _barrier(bar_p, (lb + bc) * _NWG)
        # residual x += o, striped across all WGs (no pid0 serial bottleneck)
        rstep = _NWG * BLOCK_M
        base_r = pid * BLOCK_M
        while base_r < H:
            roff = base_r + tl.arange(0, BLOCK_M)
            rm = roff < H
            xo = tl.load(x_p + roff, mask=rm, other=0.0).to(tl.float32)
            oo = tl.load(moe_p + roff, mask=rm, other=0.0).to(tl.float32)
            tl.store(x_p + roff, xo + oo, mask=rm)
            base_r += rstep
        bc += 1
        _barrier(bar_p, (lb + bc) * _NWG)

        # ---- P5: router GEMV with FUSED MoE-RMSNorm (no separate norm phase) ----
        _gemv_bf16_rmsnorm(rw, x_p, nm, logits_p, True, rb, E, H, pid, eps, BLOCK_M, BLOCK_K, NORMK)
        bc += 1
        _barrier(bar_p, (lb + bc) * _NWG)
        if pid == 0:
            eoff = tl.arange(0, E)
            work = tl.load(logits_p + eoff).to(tl.float32)
            # top-K extraction
            for kk in range(0, TOPK):
                mval = tl.max(work, axis=0)
                ismax = work == mval
                idx = tl.min(tl.where(ismax, eoff, E), axis=0)
                tl.store(ids_p + kk, idx)
                tl.store(gw_p + kk, mval)
                work = tl.where(eoff == idx, -1e30, work)
            # softmax over the TOPK stored logits
            tv = tl.load(gw_p + tl.arange(0, TOPK)).to(tl.float32)
            tmax = tl.max(tv, axis=0)
            ex = tl.exp(tv - tmax)
            sm = tl.sum(ex, axis=0)
            tl.store(gw_p + tl.arange(0, TOPK), ex / sm)
        # zero moe accumulator, striped across all WGs
        base_z = pid * BLOCK_M
        while base_z < H:
            zoff = base_z + tl.arange(0, BLOCK_M)
            tl.store(moe_p + zoff, tl.zeros((BLOCK_M,), dtype=tl.float32), mask=zoff < H)
            base_z += rstep
        # MoE-normed activation needed by ALL experts: materialize striped into
        # normed_p (each WG computes the rms scalar from x_p, stores its row slice).
        _store_rmsnorm(x_p, nm, normed_p, H, pid, eps, BLOCK_M, NORMK)
        bc += 1
        _barrier(bar_p, (lb + bc) * _NWG)

        # ---- P6: experts (loop TOPK). QUANT: native FP4xFP8 scaled MFMA
        # (dot_scaled); else: in-kernel FP4->BF16 dequant GEMV. ----
        # QUANT: pre-quantize the shared MoE-normed activation to FP8 once per layer.
        if QUANT:
            _quant_act_fp8(normed_p, nfp8_p, nfp8_scl_p, H, GU_NB, pid)
        for slot in range(0, TOPK):
            e_id = tl.load(ids_p + slot)
            gwv = tl.load(gw_p + slot).to(tl.float32)
            eidx = (layer * E + e_id).to(tl.int64)  # int64: expert blobs overflow int32
            gu_blk = gu_blk_p + eidx * (2 * I) * (H // 2)
            gu_scl = gu_scl_p + eidx * (2 * I) * GU_NB
            gu_b = gu_b_p + eidx * (2 * I)
            dn_blk = dn_blk_p + eidx * H * (I // 2)
            dn_scl = dn_scl_p + eidx * H * DN_NB
            dn_b = dn_b_p + eidx * H
            # --- sub-phase A: gate_up (2I rows, K=H) -> gu_p ---
            if QUANT:
                _gemv_fp4_scaled(
                    gu_blk,
                    gu_scl,
                    nfp8_p,
                    nfp8_scl_p,
                    gu_p,
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
                _gemv_fp4(gu_blk, gu_scl, normed_p, gu_p, gu_b, True, 2 * I, H, GU_NB, pid, 1.0, ACCUM=False)
            bc += 1
            _barrier(bar_p, (lb + bc) * _NWG)
            # --- sub-phase B: swiglu -> act_p. In QUANT mode, each program owns whole
            # 32-elem blocks and FP8-quantizes the block it just produced (no extra
            # barrier: producer == consumer for that block). ---
            if QUANT:
                pos32 = tl.arange(0, 32)
                blk = pid
                while blk < DN_NB:
                    base_i = blk * 32 + pos32
                    gate = tl.load(gu_p + 2 * base_i).to(tl.float32)
                    up = tl.load(gu_p + 2 * base_i + 1).to(tl.float32)
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
                    tl.store(afp8_p + base_i, (act / sc).to(tl.float8e4nv))
                    tl.store(afp8_scl_p + blk, raw.to(tl.uint8))
                    blk += _NWG
            else:
                ii = pid
                while ii < I:
                    gate = tl.load(gu_p + 2 * ii).to(tl.float32)
                    up = tl.load(gu_p + 2 * ii + 1).to(tl.float32)
                    gate = tl.minimum(gate, limit)
                    up = tl.maximum(tl.minimum(up, limit), -limit)
                    glu = gate * (1.0 / (1.0 + tl.exp(-alpha * gate)))
                    tl.store(act_p + ii, ((up + 1.0) * glu).to(tl.bfloat16))
                    ii += _NWG
            bc += 1
            _barrier(bar_p, (lb + bc) * _NWG)
            # --- sub-phase C: down (H rows, K=I) -> accumulate gw*ev into moe_p ---
            if QUANT:
                _gemv_fp4_scaled(
                    dn_blk,
                    dn_scl,
                    afp8_p,
                    afp8_scl_p,
                    moe_p,
                    dn_b,
                    True,
                    H,
                    I,
                    DN_NB,
                    pid,
                    gwv,
                    (slot > 0),
                    BLOCK_NQ,
                    BLOCK_KQ,
                    MTILE,
                )
            else:
                _gemv_fp4(dn_blk, dn_scl, act_p, moe_p, dn_b, True, H, I, DN_NB, pid, gwv, ACCUM=(slot > 0))
            bc += 1
            _barrier(bar_p, (lb + bc) * _NWG)

        # ---- P7: residual add moe -> x (program 0) ----
        if pid == 0:
            off = tl.arange(0, 4096)
            m = off < H
            xv = tl.load(x_p + off, mask=m, other=0.0).to(tl.float32)
            mv = tl.load(moe_p + off, mask=m, other=0.0).to(tl.float32)
            tl.store(x_p + off, xv + mv, mask=m)
        bc += 1
        base = lb + bc
        _barrier(bar_p, base * _NWG)

    # ===== final norm + lm_head + argmax =====
    if pid == 0:
        off = tl.arange(0, 4096)
        m = off < H
        xv = tl.load(x_p + off, mask=m, other=0.0).to(tl.float32)
        ss = tl.sum(xv * xv, axis=0)
        rms = 1.0 / tl.sqrt(ss / H + eps)
        g = tl.load(final_norm_p + off, mask=m, other=0.0).to(tl.float32)
        tl.store(normed_p + off, (xv * rms * g).to(tl.bfloat16), mask=m)
    base = base + 1
    _barrier(bar_p, base * _NWG)

    # lm_head GEMV: V rows -> vlogits ; then per-prog argmax slice
    _gemv_bf16_tiled(lm_head_p, normed_p, vlogits_p, False, lm_head_p, V, H, pid, BLOCK_M, BLOCK_K)
    base = base + 1
    _barrier(bar_p, base * _NWG)

    # argmax: each program reduces its strided slice, writes (val,idx); prog0 reduces
    best_v = -1e30
    best_i = 0
    r = pid
    while r < V:
        val = tl.load(vlogits_p + r).to(tl.float32)
        if val > best_v:
            best_v = val
            best_i = r
        r += _NWG
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
    def __init__(self, cfg: GptOssConfig, num_layers: int, dev="cuda", snapshot=None, _skip_load=False, quant=False):
        self.cfg = cfg
        self.L = num_layers
        self.dev = dev
        self.quant = quant
        if not _skip_load:
            w = load_hf_weights(
                GptOssConfig(), snapshot=snapshot, num_layers=num_layers, device="cpu", dtype=torch.bfloat16
            )
            self._pack(w)
            self.cos, self.sin = build_yarn_rope(GptOssConfig(), device=dev)
            self._alloc_buffers()

    @classmethod
    def from_iris(cls, iris_path: str, cfg: GptOssConfig, num_layers: int, dev="cuda", quant=False):
        """Build directly from a converted .iris weight file (mmap -> device)."""
        from convert_to_iris import read_iris_header, load_iris_tensor

        self = cls(cfg, num_layers, dev=dev, _skip_load=True, quant=quant)
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

    def _alloc_buffers(self):
        cfg, dev = self.cfg, self.dev
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
        self.logits = z(E)
        self.ids = z(cfg.top_k, torch.int32)
        self.gw = z(cfg.top_k)
        self.gu = z(2 * I)
        self.act = z(I, torch.bfloat16)
        self.moe = z(H)
        # FP8 activation-quant scratch (quantized path). nfp8 = MoE-normed activation
        # (K=H), afp8 = SwiGLU output (K=I); scales are per-32 e8m0 bytes.
        self.nfp8 = z(H, torch.float8_e4m3fn)
        self.nfp8_scl = z(H // 32, torch.uint8)
        self.afp8 = z(I, torch.float8_e4m3fn)
        self.afp8_scl = z(I // 32, torch.uint8)
        self.vlogits = z(V)
        self.amax_v = z(NUM_WG)
        self.amax_i = z(NUM_WG, torch.int32)
        self.next_tok = z(1, torch.int32)
        self.bar = z(1, torch.int32)

    @torch.no_grad()
    def step(self, token_id: int, pos: int) -> int:
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
            BLOCK_K=512,
            BLOCK_KI=256,
            BLOCK_M=8,
            NORMK=triton.next_power_of_2(cfg.hidden_dim),
            QUANT=self.quant,
            BLOCK_NQ=64,
            BLOCK_KQ=128,
            MTILE=16,
            num_warps=4,
        )
        torch.cuda.synchronize()
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
    args = ap.parse_args()

    cfg = GptOssConfig()
    L = args.layers if args.layers > 0 else cfg.num_layers
    tok = load_tokenizer(args.snapshot)
    ids = tok.encode(args.prompt)
    print(f"prompt={args.prompt!r} ids={ids} quant={args.quant}")

    import time

    t0 = time.time()
    if args.model:
        model = MegaModel.from_iris(args.model, cfg, L, quant=args.quant)
        print(f"loaded {L} layers from {args.model} in {time.time()-t0:.1f}s")
    else:
        model = MegaModel(cfg, L, snapshot=args.snapshot, quant=args.quant)
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
