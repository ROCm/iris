#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.

"""Benchmark for iris-ccl all-reduce collective."""

import torch
import iris.bench as bench
from iris.ccl import Config


@bench.register
@bench.axis("num_ranks", [2, 4, 8])
@bench.axis("M", bench.power_of_two(10, 14))
@bench.axis("N", bench.power_of_two(10, 14))
@bench.axis("dtype", [torch.float16, torch.bfloat16])
@bench.axis("variant", ["two_shot"])
def all_reduce(state, ctx):
    M, N, dtype = state["M"], state["N"], state["dtype"]
    variant = state["variant"]
    world_size = ctx.get_num_ranks()

    inp = ctx.zeros((M, N), dtype=dtype)
    out = ctx.zeros((M, N), dtype=dtype)
    inp.fill_(float(ctx.get_rank() + 1))

    # All-reduce bus bandwidth: 2 * (W-1)/W * data_size
    state.set_bytes(int(M * N * inp.element_size() * 2 * (world_size - 1) / world_size))

    config = Config(all_reduce_variant=variant)
    workspace = ctx.ccl.all_reduce_preamble(out, inp, config=config)

    state.exec(
        lambda: ctx.ccl.all_reduce(out, inp, config=config, workspace=workspace),
        preamble_fn=lambda: out.zero_(),
    )


if __name__ == "__main__":
    bench.main()
