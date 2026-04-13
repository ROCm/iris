#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.

"""
All-reduce benchmark at exact shapes from vLLM OSS model inference.

Compares RCCL (torch.distributed) vs iris.ccl (two_shot) at the tensor
shapes actually observed in production vLLM serving of:
  - gpt-oss-20b  (hidden=2880, MoE_hidden=3072)
  - gpt-oss-120b (hidden=2880, MoE_hidden=3072)
  - Mixtral-8x7B (hidden=4096)

Shapes extracted from MI300X profiling traces (bf16, TP=2/4/8).

Usage:
    python benchmark/ccl/bench_oss_allreduce.py
    python benchmark/ccl/bench_oss_allreduce.py --axis_num_ranks=8
    python benchmark/ccl/bench_oss_allreduce.py --benchmark_format=json --benchmark_out=results.json
"""

import torch
import torch.distributed as dist
import iris.bench as bench
from iris.ccl import Config

# Exact (M, N) shapes from vLLM traces.
# M = tokens, N = hidden dim. All bf16.
# Shapes are constant across TP — only rank count changes.
SHAPES = [
    (8, 2880),  # gpt-oss decode, attn proj
    (8, 3072),  # gpt-oss decode, MoE/FFN proj
    (8, 4096),  # Mixtral decode
    (7, 2880),  # prefill
    (14, 2880),  # prefill
    (46, 2880),  # prefill
    (53, 2880),  # prefill
    (64, 4096),  # Mixtral prefill
]


@bench.register
@bench.axis("num_ranks", [2, 4, 8])
@bench.axis("shape", SHAPES)
def rccl_all_reduce(state, ctx):
    """RCCL baseline: torch.distributed.all_reduce."""
    M, N = state["shape"]
    dtype = torch.bfloat16
    world_size = ctx.get_num_ranks()
    rank = ctx.get_rank()

    tensor = torch.full((M, N), float(rank + 1), dtype=dtype, device=f"cuda:{rank}")

    state.set_bytes(int(M * N * tensor.element_size() * 2 * (world_size - 1) / world_size))
    state.add_counter("tokens", M)
    state.add_counter("hidden", N)
    state.add_counter("msg_KB", M * N * tensor.element_size() / 1024)

    def reset():
        tensor.fill_(float(rank + 1))

    state.exec(
        lambda: dist.all_reduce(tensor, op=dist.ReduceOp.SUM),
        preamble_fn=reset,
    )


@bench.register
@bench.axis("num_ranks", [2, 4, 8])
@bench.axis("shape", SHAPES)
@bench.axis("variant", ["two_shot"])
def iris_all_reduce(state, ctx):
    """iris.ccl all-reduce."""
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


if __name__ == "__main__":
    bench.main()
