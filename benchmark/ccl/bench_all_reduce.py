#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.

"""Benchmark for iris-ccl all-reduce collective."""

import torch.distributed as dist
import iris.bench as bench
from iris.ccl import Config
from benchmark.ccl._common import DTYPES, M_VALUES, N_VALUES, NUM_RANKS, torch_tensor


@bench.register
@bench.axis("num_ranks", NUM_RANKS)
@bench.axis("M", M_VALUES)
@bench.axis("N", N_VALUES)
@bench.axis("dtype", DTYPES)
@bench.axis("variant", ["two_shot"])
@bench.axis("backend", ["iris", "rccl"])
def all_reduce(state, ctx):
    M, N, dtype = state["M"], state["N"], state["dtype"]
    variant = state["variant"]
    world_size = ctx.get_num_ranks()

    # All-reduce bus bandwidth: 2 * (W-1)/W * data_size
    state.set_bytes(int(M * N * dtype.itemsize * 2 * (world_size - 1) / world_size))

    if state["backend"] == "rccl":
        # dist.all_reduce is in-place; copy in the preamble so the timed region
        # holds only the collective, matching the Iris path.
        t_in = torch_tensor(ctx, (M, N), dtype)
        t_out = torch_tensor(ctx, (M, N), dtype)
        t_in.fill_(float(ctx.get_rank() + 1))
        state.exec(
            lambda: dist.all_reduce(t_out, op=dist.ReduceOp.SUM),
            preamble_fn=lambda: t_out.copy_(t_in),
        )
        return

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
