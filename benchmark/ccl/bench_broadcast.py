#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.

"""Benchmark for iris-ccl broadcast collective."""

import torch
import iris.bench as bench
from iris.ccl import Config


@bench.register
@bench.axis("num_ranks", [2, 4, 8])
@bench.axis("numel", bench.power_of_two(10, 26))
@bench.axis("dtype", [torch.bfloat16])
def broadcast(state, ctx):
    numel, dtype = state["numel"], state["dtype"]

    tensor = ctx.zeros((numel,), dtype=dtype)
    rank = ctx.get_rank()
    if rank == 0:
        tensor.fill_(1.0)

    # Broadcast data moved: numel * element_size (one copy from root)
    state.set_bytes(numel * tensor.element_size())

    config = Config()
    state.exec(
        lambda: ctx.ccl.broadcast(tensor, src=0, config=config),
    )


if __name__ == "__main__":
    bench.main()
