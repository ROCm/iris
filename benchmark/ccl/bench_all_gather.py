#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.

"""Benchmark for iris-ccl all-gather collective."""

import torch
import torch.distributed as dist
import iris.bench as bench
from iris.ccl import Config
from benchmark.ccl._common import DTYPES, M_VALUES, N_VALUES, NUM_RANKS, is_fp8, torch_tensor


@bench.register
@bench.axis("num_ranks", NUM_RANKS)
@bench.axis("M", M_VALUES)
@bench.axis("N", N_VALUES)
@bench.axis("dtype", DTYPES)
@bench.axis("backend", ["iris", "iris_tdm", "rccl"])
def all_gather(state, ctx):
    M, N, dtype = state["M"], state["N"], state["dtype"]
    world_size = ctx.get_num_ranks()

    state.set_bytes((world_size - 1) * M * N * dtype.itemsize)

    if state["backend"] == "rccl":
        if is_fp8(dtype):
            # fp8 is an Iris-only series: there is no RCCL baseline to
            # compare against, so record the gap rather than inventing one.
            state.skip("RCCL has no fp8 support; fp8 is measured for Iris only")
        t_in = torch_tensor(ctx, (M, N), dtype)
        t_out = torch_tensor(ctx, (world_size * M, N), dtype)
        t_in.fill_(float(ctx.get_rank() + 1))
        state.exec(
            lambda: dist.all_gather_into_tensor(t_out, t_in),
            preamble_fn=lambda: t_out.zero_(),
        )
        return

    if state["backend"] == "iris_tdm":
        if not torch.cuda.get_device_properties(torch.cuda.current_device()).gcnArchName.startswith("gfx1250"):
            state.skip("the TDM all-gather needs gfx1250")
        config = Config(use_gluon=True, all_gather_variant="tdm", block_size_m=128, block_size_n=256, num_warps=16)
    else:
        config = Config()

    inp = ctx.zeros((M, N), dtype=dtype)
    out = ctx.zeros((world_size * M, N), dtype=dtype)
    inp.fill_(float(ctx.get_rank() + 1))

    state.exec(
        lambda: ctx.ccl.all_gather(out, inp, config=config),
        preamble_fn=lambda: out.zero_(),
    )


if __name__ == "__main__":
    bench.main()
