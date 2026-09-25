# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
"""RMSNorm device helpers (the fused RMSNorm+GEMV variants live in the gemv_* files)."""

import triton
import triton.language as tl


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
