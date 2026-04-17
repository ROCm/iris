#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.

"""Benchmark for iris-ccl all-to-all collective."""

import torch
import iris.bench as bench
from iris.ccl import Config


@bench.register
@bench.axis("num_ranks", [2, 4, 8])
@bench.axis("M", bench.power_of_two(10, 14))
@bench.axis("N", bench.power_of_two(10, 14))
@bench.axis("dtype", [torch.float16, torch.bfloat16])
def all_to_all(state, ctx):
    M, N, dtype = state["M"], state["N"], state["dtype"]
    world_size = ctx.get_num_ranks()

    # All-to-all: input/output are (M, N * world_size) concatenated
    inp = ctx.zeros((M, N * world_size), dtype=dtype)
    out = ctx.zeros((M, N * world_size), dtype=dtype)

    rank = ctx.get_rank()
    for target in range(world_size):
        inp[:, target * N : (target + 1) * N] = float(rank * 1000 + target)

    state.set_bytes((world_size - 1) * M * N * inp.element_size())

    config = Config()
    state.exec(
        lambda: ctx.ccl.all_to_all(out, inp, config=config),
        preamble_fn=lambda: out.zero_(),
    )


if __name__ == "__main__":
    bench.main()
