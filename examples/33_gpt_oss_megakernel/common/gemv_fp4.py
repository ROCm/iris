# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
"""MXFP4-weight expert GEMV device helpers.

_gemv_fp4 dequantizes the FP4 (E2M1 + per-32 E8M0) weights to BF16 and multiplies
in fp32. _gemv_fp4_scaled uses the native FP4xFP8 scaled MFMA
(tl.dot_scaled -> v_mfma_scale_f32_16x16x128_f8f6f4 on gfx950) with an FP8 (E4M3)
activation. Both stride output rows/tiles by tl.num_programs(0)."""

import triton
import triton.language as tl

from common.fp4 import _fp4_lut


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
    npid = tl.num_programs(0)
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
        r += npid


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
    npid = tl.num_programs(0)
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
        tile += npid
