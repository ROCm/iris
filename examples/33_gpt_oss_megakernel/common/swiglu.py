# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
"""SwiGLU-OAI activation device helpers for the MoE expert path.

The gate-up GEMV interleaves (gate, up) pairs in gu_out: gate = gu_out[2*i],
up = gu_out[2*i + 1]. SwiGLU-OAI is act = (up + 1) * gate * sigmoid(alpha * gate),
with gate clamped to <= limit and up clamped to [-limit, limit].

_swiglu_quant_fp8 fuses the per-32-element FP8-E4M3 quantization of the activation
(producer == consumer for that block, so no barrier between this and the down GEMV).
_swiglu_bf16 writes the activation as BF16 for the dequantizing FP4 path. Both stride
their work blocks by tl.num_programs(0)."""

import triton
import triton.language as tl


@triton.jit
def _swiglu_quant_fp8(gu_out, afp8_out, afp8_scl_out, DN_NB: tl.constexpr, pid, alpha, limit):
    """SwiGLU-OAI over the (gate, up) pairs in gu_out, quantized to FP8-E4M3 with a
    per-32-element E8M0 block scale written to afp8_out / afp8_scl_out."""
    pos32 = tl.arange(0, 32)
    npid = tl.num_programs(0)
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
        blk += npid


@triton.jit
def _swiglu_bf16(gu_out, act_out, I: tl.constexpr, pid, alpha, limit):
    """SwiGLU-OAI over the (gate, up) pairs in gu_out, written as BF16 to act_out
    (for the dequantizing FP4 expert path)."""
    npid = tl.num_programs(0)
    ii = pid
    while ii < I:
        gate = tl.load(gu_out + 2 * ii).to(tl.float32)
        up = tl.load(gu_out + 2 * ii + 1).to(tl.float32)
        gate = tl.minimum(gate, limit)
        up = tl.maximum(tl.minimum(up, limit), -limit)
        glu = gate * (1.0 / (1.0 + tl.exp(-alpha * gate)))
        tl.store(act_out + ii, ((up + 1.0) * glu).to(tl.bfloat16))
        ii += npid
