#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.

"""Sample benchmark using iris.bench — all-gather collective."""

import torch
import iris.bench as bench
from iris.ccl import Config


@bench.register
@bench.axis("num_ranks", [2, 4, 8])
@bench.axis("M", [1024, 4096, 16384])
@bench.axis("N", [1024, 4096])
@bench.axis("dtype", [torch.float16, torch.bfloat16])
def all_gather(state, ctx):
    M, N, dtype = state["M"], state["N"], state["dtype"]
    world_size = ctx.get_num_ranks()

    inp = ctx.zeros((M, N), dtype=dtype)
    out = ctx.zeros((world_size * M, N), dtype=dtype)
    inp.fill_(float(ctx.get_rank() + 1))

    total_bytes = (world_size - 1) * M * N * inp.element_size()
    state.set_bytes(total_bytes)

    config = Config(use_gluon=False)
    state.exec(lambda: ctx.ccl.all_gather(out, inp, config=config))


if __name__ == "__main__":
    bench.main()
