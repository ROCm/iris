#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.

"""
Tuning sweep for iris all-reduce at vLLM OSS shapes.

Sweeps variants, comm_sms, block sizes, and distribution to find the
best config for each shape. Includes RCCL baseline for comparison.

Usage:
    python benchmark/ccl/bench_oss_tune.py
    python benchmark/ccl/bench_oss_tune.py --axis_num_ranks=8
"""

import torch
import torch.distributed as dist
import iris.bench as bench
from iris.ccl import Config

# Focus on the hot-path decode shapes + the best prefill case
SHAPES = [
    (8, 2880),  # gpt-oss decode, attn proj (45 KB)
    (8, 3072),  # gpt-oss decode, MoE/FFN proj (48 KB)
    (8, 4096),  # Mixtral decode (64 KB)
    (64, 4096),  # Mixtral prefill (512 KB) — largest message
]


# ─── RCCL baseline ──────────────────────────────────────────────────────


@bench.register
@bench.axis("num_ranks", [8])
@bench.axis("shape", SHAPES)
def rccl(state, ctx):
    M, N = state["shape"]
    dtype = torch.bfloat16
    world_size = ctx.get_num_ranks()
    rank = ctx.get_rank()
    tensor = torch.full((M, N), float(rank + 1), dtype=dtype, device=f"cuda:{rank}")
    state.set_bytes(int(M * N * tensor.element_size() * 2 * (world_size - 1) / world_size))
    state.add_counter("msg_KB", M * N * tensor.element_size() / 1024)
    state.exec(
        lambda: dist.all_reduce(tensor, op=dist.ReduceOp.SUM),
        preamble_fn=lambda: tensor.fill_(float(rank + 1)),
    )


# ─── Variant sweep ──────────────────────────────────────────────────────


@bench.register
@bench.axis("num_ranks", [8])
@bench.axis("shape", SHAPES)
@bench.axis("variant", ["two_shot", "one_shot", "atomic", "ring", "spinlock"])
def iris_variant(state, ctx):
    M, N = state["shape"]
    variant = state["variant"]
    dtype = torch.bfloat16
    world_size = ctx.get_num_ranks()
    inp = ctx.zeros((M, N), dtype=dtype)
    out = ctx.zeros((M, N), dtype=dtype)
    inp.fill_(float(ctx.get_rank() + 1))
    state.set_bytes(int(M * N * inp.element_size() * 2 * (world_size - 1) / world_size))
    state.add_counter("msg_KB", M * N * inp.element_size() / 1024)
    config = Config(all_reduce_variant=variant)
    workspace = ctx.ccl.all_reduce_preamble(out, inp, config=config)
    state.exec(
        lambda: ctx.ccl.all_reduce(out, inp, config=config, workspace=workspace),
        preamble_fn=lambda: out.zero_(),
    )


# ─── comm_sms sweep (best variant from above will be clear) ─────────


@bench.register
@bench.axis("num_ranks", [8])
@bench.axis("shape", SHAPES)
@bench.axis("comm_sms", [8, 16, 32, 64, 128, 256])
def iris_comm_sms(state, ctx):
    M, N = state["shape"]
    sms = state["comm_sms"]
    dtype = torch.bfloat16
    world_size = ctx.get_num_ranks()
    inp = ctx.zeros((M, N), dtype=dtype)
    out = ctx.zeros((M, N), dtype=dtype)
    inp.fill_(float(ctx.get_rank() + 1))
    state.set_bytes(int(M * N * inp.element_size() * 2 * (world_size - 1) / world_size))
    state.add_counter("msg_KB", M * N * inp.element_size() / 1024)
    config = Config(all_reduce_variant="two_shot", comm_sms=sms)
    workspace = ctx.ccl.all_reduce_preamble(out, inp, config=config)
    state.exec(
        lambda: ctx.ccl.all_reduce(out, inp, config=config, workspace=workspace),
        preamble_fn=lambda: out.zero_(),
    )


# ─── block_size sweep ───────────────────────────────────────────────


@bench.register
@bench.axis("num_ranks", [8])
@bench.axis("shape", SHAPES)
@bench.axis("block_m_n", [(4, 32), (4, 64), (8, 32), (8, 64), (16, 32), (16, 64), (32, 32), (32, 64), (64, 64)])
def iris_block_size(state, ctx):
    M, N = state["shape"]
    bm, bn = state["block_m_n"]
    dtype = torch.bfloat16
    world_size = ctx.get_num_ranks()
    inp = ctx.zeros((M, N), dtype=dtype)
    out = ctx.zeros((M, N), dtype=dtype)
    inp.fill_(float(ctx.get_rank() + 1))
    state.set_bytes(int(M * N * inp.element_size() * 2 * (world_size - 1) / world_size))
    state.add_counter("msg_KB", M * N * inp.element_size() / 1024)
    config = Config(all_reduce_variant="two_shot", block_size_m=bm, block_size_n=bn)
    workspace = ctx.ccl.all_reduce_preamble(out, inp, config=config)
    state.exec(
        lambda: ctx.ccl.all_reduce(out, inp, config=config, workspace=workspace),
        preamble_fn=lambda: out.zero_(),
    )


# ─── distribution sweep ─────────────────────────────────────────────


@bench.register
@bench.axis("num_ranks", [8])
@bench.axis("shape", SHAPES)
@bench.axis("distribution", [0, 1])
def iris_distribution(state, ctx):
    M, N = state["shape"]
    dist_mode = state["distribution"]
    dtype = torch.bfloat16
    world_size = ctx.get_num_ranks()
    inp = ctx.zeros((M, N), dtype=dtype)
    out = ctx.zeros((M, N), dtype=dtype)
    inp.fill_(float(ctx.get_rank() + 1))
    state.set_bytes(int(M * N * inp.element_size() * 2 * (world_size - 1) / world_size))
    state.add_counter("msg_KB", M * N * inp.element_size() / 1024)
    config = Config(all_reduce_variant="two_shot", all_reduce_distribution=dist_mode)
    workspace = ctx.ccl.all_reduce_preamble(out, inp, config=config)
    state.exec(
        lambda: ctx.ccl.all_reduce(out, inp, config=config, workspace=workspace),
        preamble_fn=lambda: out.zero_(),
    )


# ─── one_shot tuning (likely best for small msgs) ───────────────────


@bench.register
@bench.axis("num_ranks", [8])
@bench.axis("shape", SHAPES)
@bench.axis("comm_sms", [8, 16, 32, 64])
@bench.axis("block_m_n", [(4, 32), (4, 64), (8, 32), (8, 64), (16, 64)])
def iris_one_shot_tune(state, ctx):
    M, N = state["shape"]
    sms = state["comm_sms"]
    bm, bn = state["block_m_n"]
    dtype = torch.bfloat16
    world_size = ctx.get_num_ranks()
    inp = ctx.zeros((M, N), dtype=dtype)
    out = ctx.zeros((M, N), dtype=dtype)
    inp.fill_(float(ctx.get_rank() + 1))
    state.set_bytes(int(M * N * inp.element_size() * 2 * (world_size - 1) / world_size))
    state.add_counter("msg_KB", M * N * inp.element_size() / 1024)
    config = Config(all_reduce_variant="one_shot", comm_sms=sms, block_size_m=bm, block_size_n=bn)
    workspace = ctx.ccl.all_reduce_preamble(out, inp, config=config)
    state.exec(
        lambda: ctx.ccl.all_reduce(out, inp, config=config, workspace=workspace),
        preamble_fn=lambda: out.zero_(),
    )


if __name__ == "__main__":
    bench.main()
