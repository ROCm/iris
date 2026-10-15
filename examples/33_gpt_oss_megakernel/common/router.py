# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
"""MoE router device helper: top-k selection + softmax over the selected experts."""

import triton
import triton.language as tl


@triton.jit
def _topk_softmax(logits_p, ids_p, gw_p, E: tl.constexpr, TOPK: tl.constexpr):
    """Top-k + softmax over the E router logits, done redundantly in registers by
    every program (E is tiny). Writes the selected expert ids to ids_p and their
    softmax weights to gw_p. The writes are identical across programs and each
    program reads back only what it wrote, so the experts proceed without a separate
    top-k barrier."""
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
