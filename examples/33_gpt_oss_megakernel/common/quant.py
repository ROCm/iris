# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
"""FP8-E4M3 activation quantization device helper (per-32-element E8M0 block scale)."""

import triton
import triton.language as tl


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
    npid = tl.num_programs(0)
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
        b += npid
