#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.

"""
Autotune iris all-gather collective operation.

Sweeps block_size_m, block_size_n, comm_sms, num_warps, and swizzle_size
to find the fastest Config for a given (M, N, dtype, num_ranks) combination.

Usage:
    python tune_all_gather.py
    python tune_all_gather.py --num_ranks=4 --top_k=5
    python tune_all_gather.py --benchmark_format=json --benchmark_out=results.json
"""

import torch
import iris.tune as tune
from iris.ccl import Config


@tune.register
@tune.search_space("block_size_m", [16, 32, 64, 128])
@tune.search_space("block_size_n", [32, 64, 128, 256])
@tune.search_space("comm_sms", [32, 64, 108])
@tune.search_space("num_warps", [2, 4, 8])
@tune.search_space("swizzle_size", [2, 4, 8])
@tune.param("M", 4096)
@tune.param("N", 4096)
@tune.param("dtype", torch.float16)
def all_gather(state, ctx):
    M = state["M"]
    N = state["N"]
    dtype = state["dtype"]
    world_size = ctx.get_num_ranks()

    inp = ctx.zeros((M, N), dtype=dtype)
    out = ctx.zeros((world_size * M, N), dtype=dtype)
    inp.fill_(float(ctx.get_rank() + 1))

    config = Config(**state.config_kwargs())
    state.set_bytes((world_size - 1) * M * N * inp.element_size())
    state.exec(
        lambda: ctx.ccl.all_gather(out, inp, config=config),
        preamble_fn=lambda: out.zero_(),
    )


if __name__ == "__main__":
    tune.main()
