#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.

"""Benchmark for iris-ccl all-to-all collective."""

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
@bench.axis("variant", ["persistent", "tdm"])
# Only the TDM kernel reads these two. Sweep them with e.g. --axis_comm_sms=32,64,128,256.
@bench.axis("comm_sms", [64])
@bench.axis("num_stages", [6])
@bench.axis("backend", ["iris", "rccl"])
def all_to_all(state, ctx):
    M, N, dtype = state["M"], state["N"], state["dtype"]
    variant = state["variant"]
    world_size = ctx.get_num_ranks()

    state.set_bytes((world_size - 1) * M * N * dtype.itemsize)
    rank = ctx.get_rank()

    if variant != "tdm" and (state["comm_sms"] != 64 or state["num_stages"] != 6):
        # The persistent kernel and RCCL ignore these axes; measure them once.
        state.skip("comm_sms and num_stages configure the TDM kernel; measured once otherwise")

    if state["backend"] == "rccl":
        if variant != "persistent":
            state.skip("variant configures Iris; RCCL is measured once, at variant=persistent")
        if is_fp8(dtype):
            # fp8 is an Iris-only series: there is no RCCL baseline to
            # compare against, so record the gap rather than inventing one.
            state.skip("RCCL has no fp8 support; fp8 is measured for Iris only")
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

    if variant == "tdm":
        if not torch.cuda.get_device_properties(torch.cuda.current_device()).gcnArchName.startswith("gfx1250"):
            state.skip("the TDM all-to-all needs gfx1250")
        # Unlike all-gather, each loaded tile is stored once, so the bytes in flight are
        # bounded by the LDS ring itself: about num_stages - 2 tiles. A deep ring of
        # 32 KiB tiles uses the same 192 KiB of LDS as the TDM all-gather's config.
        config = Config(
            use_gluon=True,
            all_to_all_variant="tdm",
            block_size_m=128,
            block_size_n=128,
            num_warps=16,
            comm_sms=state["comm_sms"],
            num_stages=state["num_stages"],
        )
    else:
        config = Config()
    state.exec(
        lambda: ctx.ccl.all_to_all(out, inp, config=config),
        preamble_fn=lambda: out.zero_(),
    )


if __name__ == "__main__":
    bench.main()
