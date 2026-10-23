#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.

"""Benchmark for iris-ccl reduce-scatter collective."""

import torch
import iris.bench as bench
from iris.ccl import Config


@bench.register
@bench.axis("num_ranks", [2, 4, 8])
@bench.axis("M", bench.power_of_two(10, 14))
@bench.axis("N", bench.power_of_two(10, 14))
@bench.axis("dtype", [torch.float16, torch.bfloat16])
@bench.axis("variant", ["two_shot", "tdm"])
# Only the TDM kernel reads these two. Sweep them with e.g. --axis_comm_sms=32,64,128,256.
# num_stages=0 picks the deepest ring that fits in 192 KiB of LDS.
@bench.axis("comm_sms", [64])
@bench.axis("num_stages", [0])
def reduce_scatter(state, ctx):
    M, N, dtype = state["M"], state["N"], state["dtype"]
    variant = state["variant"]
    world_size = ctx.get_num_ranks()

    if variant != "tdm" and (state["comm_sms"] != 64 or state["num_stages"] != 0):
        # The two-shot kernel ignores these axes; measure it once.
        state.skip("comm_sms and num_stages configure the TDM kernel; measured once otherwise")

    inp = ctx.zeros((M, N), dtype=dtype)
    out = ctx.zeros((M, N), dtype=dtype)
    inp.fill_(float(ctx.get_rank() + 1) * 0.1)

    # Reduce-scatter bus bandwidth: (W-1)/W * data_size
    state.set_bytes(int(M * N * inp.element_size() * (world_size - 1) / world_size))

    if variant == "tdm":
        if not torch.cuda.get_device_properties(torch.cuda.current_device()).gcnArchName.startswith("gfx1250"):
            state.skip("the TDM reduce-scatter needs gfx1250")
        # A ring slot holds world_size input tiles and one output tile. 192 KiB is the
        # LDS the TDM all-gather's config uses: 4 stages at 2 ranks, 2 at 4, 1 at 8.
        block_size_m, block_size_n = 64, 128
        slot_bytes = (world_size + 1) * block_size_m * block_size_n * dtype.itemsize
        num_stages = state["num_stages"] or max(1, (192 << 10) // slot_bytes)
        config = Config(
            use_gluon=True,
            reduce_scatter_variant="tdm",
            block_size_m=block_size_m,
            block_size_n=block_size_n,
            num_warps=16,
            comm_sms=state["comm_sms"],
            num_stages=num_stages,
        )
    else:
        config = Config()
    state.exec(
        lambda: ctx.ccl.reduce_scatter(out, inp, config=config),
        preamble_fn=lambda: out.zero_(),
    )


if __name__ == "__main__":
    bench.main()
