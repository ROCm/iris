#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.

"""
Benchmark iris collectives at exact shapes observed in vLLM inference.

Shapes extracted from vLLM profiling traces (MI300X, rocm/vllm:latest) across:
  - gpt-oss-20b (20B dense, mxfp4 quant, hidden=2880/3072)
  - gpt-oss-120b (120B dense, mxfp4 quant, hidden=2880/3072)
  - Mixtral-8x7B-v0.1 (MoE, hidden=4096)

Key findings from profiling:
  - All-reduce is the dominant collective (>90% of comm time)
  - vLLM uses custom P2P all-reduce, not RCCL — iris targets replacing this
  - All-gather is RCCL-only, used for embedding lookup (vocab_size/TP)
  - All shapes are bf16
  - Decode tokens: 8 (batch), Prefill tokens: 6-64 (variable)
"""

import torch
import iris.bench as bench
from iris.ccl import Config


# ─── All-Reduce shapes from vLLM traces ─────────────────────────────────────
#
# These are the exact (M, N) shapes passed to all-reduce in vLLM inference.
# M = number of tokens, N = hidden dimension (constant across TP for all-reduce).
#
# Decode (dominant — thousands of calls per inference):
#   gpt-oss-20b/120b:  [8, 2880] (attn out), [8, 3072] (MoE/FFN out)
#   Mixtral-8x7B:      [8, 4096] (attn + MoE out)
#
# Prefill (few calls, larger M):
#   gpt-oss: [6, 2880], [7, 2880], [14, 2880], [46, 2880], [53, 2880]
#   Mixtral: [7, 4096], [64, 4096]

ALLREDUCE_SHAPES = [
    # (M, N) — decode shapes (hot path)
    (8, 2880),  # gpt-oss attn output
    (8, 3072),  # gpt-oss MoE/FFN output
    (8, 4096),  # Mixtral attn + MoE output
    # prefill shapes (variable token counts)
    (6, 2880),
    (7, 2880),
    (14, 2880),
    (46, 2880),
    (53, 2880),
    (6, 3072),
    (7, 3072),
    (14, 3072),
    (46, 3072),
    (53, 3072),
    (7, 4096),
    (64, 4096),
]


@bench.register
@bench.axis("num_ranks", [2, 4, 8])
@bench.axis("shape", ALLREDUCE_SHAPES)
@bench.axis("variant", ["two_shot", "ll"])
def vllm_all_reduce(state, ctx):
    """All-reduce at exact vLLM inference shapes (bf16)."""
    M, N = state["shape"]
    variant = state["variant"]
    dtype = torch.bfloat16
    world_size = ctx.get_num_ranks()

    inp = ctx.zeros((M, N), dtype=dtype)
    out = ctx.zeros((M, N), dtype=dtype)
    inp.fill_(float(ctx.get_rank() + 1))

    state.set_bytes(int(M * N * inp.element_size() * 2 * (world_size - 1) / world_size))
    state.add_counter("tokens", M)
    state.add_counter("hidden", N)
    state.add_counter("msg_KB", M * N * inp.element_size() / 1024)

    config = Config(all_reduce_variant=variant)
    workspace = ctx.ccl.all_reduce_preamble(out, inp, config=config)

    state.exec(
        lambda: ctx.ccl.all_reduce(out, inp, config=config, workspace=workspace),
        preamble_fn=lambda: out.zero_(),
    )


# ─── All-Gather shapes from vLLM traces ─────────────────────────────────────
#
# All-gather is used for embedding gather (vocab parallel).
# Per-rank input: [tokens, vocab_size / TP]
# Output: [tokens * TP, vocab_size / TP] (gathered across ranks)
#
# gpt-oss vocab=201088:
#   TP=2: [8, 100544], TP=4: [8, 50272], TP=8: [8, 25136]
# Mixtral vocab=32000:
#   TP=4: [8, 8000], TP=8: [8, 4000]

ALLGATHER_SHAPES_BY_RANKS = {
    # num_ranks -> list of (M, N) per-rank input shapes
    2: [
        (8, 100544),  # gpt-oss vocab/2
    ],
    4: [
        (8, 50272),  # gpt-oss vocab/4
        (8, 8000),  # Mixtral vocab/4
    ],
    8: [
        (8, 25136),  # gpt-oss vocab/8
        (8, 4000),  # Mixtral vocab/8
    ],
}

# Flatten for axis — encode as (num_ranks, M, N) tuples
ALLGATHER_CONFIGS = []
for nr, shapes in ALLGATHER_SHAPES_BY_RANKS.items():
    for m, n in shapes:
        ALLGATHER_CONFIGS.append((nr, m, n))


@bench.register
@bench.axis("num_ranks", [2, 4, 8])
@bench.axis("config", ALLGATHER_CONFIGS)
def vllm_all_gather(state, ctx):
    """All-gather at exact vLLM embedding shapes (bf16)."""
    target_ranks, M, N = state["config"]
    dtype = torch.bfloat16
    world_size = ctx.get_num_ranks()

    # Skip if this config doesn't match the current rank count
    if world_size != target_ranks:
        state.skip(f"config targets {target_ranks} ranks, running {world_size}")
        return

    inp = ctx.zeros((M, N), dtype=dtype)
    out = ctx.zeros((world_size * M, N), dtype=dtype)
    inp.fill_(float(ctx.get_rank() + 1))

    state.set_bytes((world_size - 1) * M * N * inp.element_size())
    state.add_counter("tokens", M)
    state.add_counter("vocab_shard", N)
    state.add_counter("msg_KB", M * N * inp.element_size() / 1024)

    config = Config()
    state.exec(
        lambda: ctx.ccl.all_gather(out, inp, config=config),
        preamble_fn=lambda: out.zero_(),
    )


if __name__ == "__main__":
    bench.main()
