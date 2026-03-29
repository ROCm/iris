#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.

"""Benchmark for iris-ccl reduce collective."""

import torch
import iris.bench as bench
from iris.ccl import Config


@bench.register
@bench.axis("num_ranks", [2, 4, 8])
@bench.axis("numel", bench.power_of_two(10, 26))
@bench.axis("dtype", [torch.bfloat16])
def reduce(state, ctx):
    numel, dtype = state["numel"], state["dtype"]
    world_size = ctx.get_num_ranks()

    tensor = ctx.zeros((numel,), dtype=dtype)
    tensor.fill_(float(ctx.get_rank() + 1))

    # Reduce data moved: root reads (W-1) chunks of numel * element_size
    state.set_bytes(numel * tensor.element_size() * (world_size - 1))

    config = Config()
    state.exec(
        lambda: ctx.ccl.reduce(tensor, dst=0, config=config),
    )


if __name__ == "__main__":
    bench.main()
