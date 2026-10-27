#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.

"""Benchmark for iris-ccl reduce-scatter collective."""

import torch
import iris.bench as bench
from iris.ccl import Config

# The symmetric heap never frees, and every variant, comm_sms and num_stages point of a
# shape would otherwise allocate its own pair: a comm_sms sweep multiplies heap use by its
# length. Points of one shape run back to back, so they share the last pair allocated.
_iris_buffers = {}

# Default TDM ring depth per world size for 64 x 128 tiles, measured on gfx1250.
_TDM_NUM_STAGES = {2: 4, 4: 3}


def _iris_buffers_for(ctx, M, N, dtype):
    key = (M, N, dtype)
    if key not in _iris_buffers:
        _iris_buffers.clear()
        _iris_buffers[key] = (ctx.zeros((M, N), dtype=dtype), ctx.zeros((M, N), dtype=dtype))
    return _iris_buffers[key]


@bench.register
@bench.axis("num_ranks", [2, 4, 8])
@bench.axis("M", bench.power_of_two(10, 14))
@bench.axis("N", bench.power_of_two(10, 14))
@bench.axis("dtype", [torch.float16, torch.bfloat16])
@bench.axis("variant", ["two_shot", "tdm"])
# Only the TDM kernel reads these two. Sweep them with e.g. --axis_comm_sms=32,64,128,256.
# num_stages=0 picks the default ring depth below.
@bench.axis("comm_sms", [64])
@bench.axis("num_stages", [0])
def reduce_scatter(state, ctx):
    M, N, dtype = state["M"], state["N"], state["dtype"]
    variant = state["variant"]
    world_size = ctx.get_num_ranks()

    if variant != "tdm" and (state["comm_sms"] != 64 or state["num_stages"] != 0):
        # The two-shot kernel ignores these axes; measure it once.
        state.skip("comm_sms and num_stages configure the TDM kernel; measured once otherwise")

    inp, out = _iris_buffers_for(ctx, M, N, dtype)
    inp.fill_(float(ctx.get_rank() + 1) * 0.1)

    # Reduce-scatter bus bandwidth: (W-1)/W * data_size
    state.set_bytes(int(M * N * inp.element_size() * (world_size - 1) / world_size))

    if variant == "tdm":
        if not torch.cuda.get_device_properties(torch.cuda.current_device()).gcnArchName.startswith("gfx1250"):
            state.skip("the TDM reduce-scatter needs gfx1250")
        # A ring slot holds world_size input tiles and one output tile. Deeper is not
        # always better: on gfx1250 throughput peaks at 4 stages for 2 ranks and 3 for
        # 4. The deeper rings that are slower are also the ones where the longer peel
        # spills SGPRs. Other rank counts are unmeasured and take the deepest ring that
        # fits in 192 KiB, the LDS the TDM all-gather's config uses.
        block_size_m, block_size_n = 64, 128
        slot_bytes = (world_size + 1) * block_size_m * block_size_n * dtype.itemsize
        num_stages = state["num_stages"] or _TDM_NUM_STAGES.get(world_size, max(1, (192 << 10) // slot_bytes))
        max_lds = torch.cuda.get_device_properties(torch.cuda.current_device()).shared_memory_per_block
        if num_stages * slot_bytes > max_lds:
            # The launcher would raise, ending the whole sweep.
            state.skip(f"a {num_stages * slot_bytes}-byte LDS ring does not fit in {max_lds} bytes")
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
