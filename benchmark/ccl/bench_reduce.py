#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
# Copyright (c) 2025 Advanced Micro Devices, Inc. All rights reserved.

"""Benchmark for iris-ccl reduce collective."""

import torch
import iris.bench as bench
from iris.ccl import Config


@bench.register
@bench.axis("num_ranks", [2, 4, 8])
@bench.axis("M", bench.power_of_two(10, 14))
@bench.axis("N", bench.power_of_two(10, 14))
@bench.axis("dtype", [torch.float16, torch.bfloat16])
def reduce(state, ctx):
    M, N, dtype = state["M"], state["N"], state["dtype"]
    world_size = ctx.get_num_ranks()
    root = 0

    inp = ctx.zeros((M, N), dtype=dtype)
    out = ctx.zeros((M, N), dtype=dtype)
    inp.fill_(float(ctx.get_rank() + 1) * 0.1)

    # Reduce bus bandwidth: (W-1)/W * data_size
    state.set_bytes(int(M * N * inp.element_size() * (world_size - 1) / world_size))

    config = Config(reduce_variant="two_shot")
    workspace = ctx.ccl.reduce_preamble(out, inp, root=root, config=config)

    def preamble():
        out.zero_()
        ctx.ccl.reduce_preamble(out, inp, root=root, config=config, workspace=workspace)

    state.exec(
        lambda: ctx.ccl.reduce(out, inp, root=root, config=config, workspace=workspace),
        preamble_fn=preamble,
    )


if __name__ == "__main__":
    bench.main()
