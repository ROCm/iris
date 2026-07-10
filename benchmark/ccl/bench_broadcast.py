#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
# Copyright (c) 2025 Advanced Micro Devices, Inc. All rights reserved.

"""Benchmark for iris-ccl broadcast collective."""

import torch
import iris.bench as bench
from iris.ccl import Config


@bench.register
@bench.axis("num_ranks", [2, 4, 8])
@bench.axis("M", bench.power_of_two(10, 14))
@bench.axis("N", bench.power_of_two(10, 14))
@bench.axis("dtype", [torch.float16, torch.bfloat16])
def broadcast(state, ctx):
    M, N, dtype = state["M"], state["N"], state["dtype"]
    world_size = ctx.get_num_ranks()

    inp = ctx.zeros((M, N), dtype=dtype)
    out = ctx.zeros((M, N), dtype=dtype)
    inp.fill_(float(ctx.get_rank() + 1))

    # Broadcast transfers M*N elements from src to (world_size - 1) ranks
    state.set_bytes((world_size - 1) * M * N * inp.element_size())

    config = Config()
    state.exec(
        lambda: ctx.ccl.broadcast(out, inp, src=0, config=config),
        preamble_fn=lambda: out.zero_(),
    )


if __name__ == "__main__":
    bench.main()
