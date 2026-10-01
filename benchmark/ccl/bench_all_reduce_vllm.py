#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.

"""
All-reduce benchmark with vLLM/GPT-OSS shapes.

N=2880 (GPT-OSS hidden dimension). M values cover decode-like (1-512 tokens)
and prefill-like (2048-8192 tokens). Two sections: RCCL baseline and iris
variants, so each has its own parameter space.
"""

import torch
import torch.distributed as dist
import iris.bench as bench
from iris.ccl import Config

DECODE_MS = [1, 32, 64, 128, 512]
PREFILL_MS = [2048, 4096, 8192]
ALL_MS = DECODE_MS + PREFILL_MS


# ── RCCL baseline ────────────────────────────────────────────────────────


@bench.register
@bench.axis("num_ranks", [8])
@bench.axis("M", ALL_MS)
@bench.axis("N", [2880])
@bench.axis("dtype", [torch.bfloat16])
def rccl_all_reduce(state, ctx):
    M, N, dtype = state["M"], state["N"], state["dtype"]
    world_size = ctx.get_num_ranks()
    rank = ctx.get_rank()

    element_size = torch.tensor([], dtype=dtype).element_size()
    state.set_bytes(int(M * N * element_size * 2 * (world_size - 1) / world_size))

    tensor = torch.full((M, N), float(rank + 1), dtype=dtype, device=torch.device("cuda"))

    def preamble():
        tensor.fill_(float(rank + 1))

    state.exec(
        lambda: dist.all_reduce(tensor, op=dist.ReduceOp.SUM),
        preamble_fn=preamble,
    )


# ── Iris variants ────────────────────────────────────────────────────────


@bench.register
@bench.axis("num_ranks", [8])
@bench.axis("M", ALL_MS)
@bench.axis("N", [2880])
@bench.axis("dtype", [torch.bfloat16])
@bench.axis("variant", ["two_shot", "ring", "one_shot"])
def iris_all_reduce(state, ctx):
    M, N, dtype = state["M"], state["N"], state["dtype"]
    variant = state["variant"]
    world_size = ctx.get_num_ranks()

    element_size = torch.tensor([], dtype=dtype).element_size()
    state.set_bytes(int(M * N * element_size * 2 * (world_size - 1) / world_size))

    inp = ctx.zeros((M, N), dtype=dtype)
    out = ctx.zeros((M, N), dtype=dtype)
    inp.fill_(float(ctx.get_rank() + 1))

    config = Config(all_reduce_variant=variant)
    workspace = ctx.ccl.all_reduce_preamble(out, inp, config=config)

    def preamble():
        out.zero_()
        ctx.ccl.all_reduce_preamble(out, inp, config=config, workspace=workspace)

    state.exec(
        lambda: ctx.ccl.all_reduce(out, inp, config=config, workspace=workspace),
        preamble_fn=preamble,
    )


if __name__ == "__main__":
    bench.main()
