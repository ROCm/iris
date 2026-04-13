#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.

"""
Benchmark comparing RCCL (torch.distributed) vs iris all-reduce at vLLM shapes.

Focuses on the decode-dominant shapes where latency matters most.
All shapes are bf16, matching vLLM inference.
"""

import torch
import torch.distributed as dist
import iris.bench as bench
from iris.ccl import Config


SHAPES = [
    (8, 2880),  # gpt-oss attn output (45 KB)
    (8, 3072),  # gpt-oss MoE/FFN output (48 KB)
    (8, 4096),  # Mixtral (64 KB)
    (64, 4096),  # Mixtral prefill (512 KB)
]


@bench.register
@bench.axis("num_ranks", [2, 4, 8])
@bench.axis("shape", SHAPES)
def rccl_all_reduce(state, ctx):
    """RCCL baseline via torch.distributed.all_reduce."""
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
@bench.axis("variant", ["two_shot", "ll"])
def iris_all_reduce(state, ctx):
    """iris.ccl all-reduce at vLLM shapes."""
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
