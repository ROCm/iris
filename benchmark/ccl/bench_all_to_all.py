#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.

"""Benchmark for iris-ccl all-to-all collective."""

import torch.distributed as dist
import iris.bench as bench
from iris.ccl import Config
from benchmark.ccl._common import DTYPES, M_VALUES, N_VALUES, NUM_RANKS, torch_tensor


@bench.register
@bench.axis("num_ranks", NUM_RANKS)
@bench.axis("M", M_VALUES)
@bench.axis("N", N_VALUES)
@bench.axis("dtype", DTYPES)
@bench.axis("backend", ["iris", "rccl"])
def all_to_all(state, ctx):
    M, N, dtype = state["M"], state["N"], state["dtype"]
    world_size = ctx.get_num_ranks()

    state.set_bytes((world_size - 1) * M * N * dtype.itemsize)
    rank = ctx.get_rank()

    if state["backend"] == "rccl":
        t_in = torch_tensor(ctx, (M * world_size, N), dtype)
        t_out = torch_tensor(ctx, (M * world_size, N), dtype)
        for target in range(world_size):
            t_in[target * M : (target + 1) * M, :] = float(rank * 1000 + target)
        state.exec(
            lambda: dist.all_to_all_single(t_out, t_in),
            preamble_fn=lambda: t_out.zero_(),
        )
        return

    # All-to-all: input/output are (M, N * world_size) concatenated
    inp = ctx.zeros((M, N * world_size), dtype=dtype)
    out = ctx.zeros((M, N * world_size), dtype=dtype)

    for target in range(world_size):
        inp[:, target * N : (target + 1) * N] = float(rank * 1000 + target)

    config = Config()
    state.exec(
        lambda: ctx.ccl.all_to_all(out, inp, config=config),
        preamble_fn=lambda: out.zero_(),
    )


if __name__ == "__main__":
    bench.main()
