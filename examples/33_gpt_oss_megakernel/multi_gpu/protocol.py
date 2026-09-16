# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
"""Shared layout/protocol for the 1-attention + 4-MoE multi-GPU GPT-OSS decode.

Ranks
-----
  rank 0          : attention + tail. Embedding, QKV, RoPE/flash-decode, O-proj,
                    router + top-k, FP8 quant of the expert input, the gated sum +
                    residual, the final RMSNorm + LM head + argmax.
  ranks 1..TOPK   : one MoE expert each. rank r computes the (r-1)-th selected
                    expert's gate-up -> SwiGLU -> down for the current token.

Per-layer exchange (token decode, batch 1)
------------------------------------------
  1. rank 0 computes through the router and writes, into a symmetric-heap inbox on
     each MoE rank: the FP8 expert input (H e4m3 + H/32 e8m0 scales), the selected
     expert id, and the softmax gate weight. It then raises that rank's `in_flag`.
  2. MoE rank r spins on its `in_flag`, runs gate-up/SwiGLU/down for its expert,
     and writes the gate-weighted result vector (H fp32) back into rank 0's
     per-slot result inbox, then raises rank 0's `out_flag[r-1]`.
  3. rank 0 spins on all TOPK `out_flag`s, sums the result vectors into the
     residual, and proceeds to the next layer.

All inbox/flag buffers live on the iris symmetric heap so the producer can reach
the consumer's copy with iris.store(..., from_rank, to_rank, heap_bases).
"""

ATTN_RANK = 0


def moe_rank(slot: int) -> int:
    """The rank that owns expert slot `slot` (0..TOPK-1)."""
    return 1 + slot
