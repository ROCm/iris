#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.

"""Benchmark for iris-ccl all-to-all collective."""

import torch
import torch.distributed as dist
import iris.bench as bench
from iris.ccl import Config
from benchmark.ccl._common import DTYPES, M_VALUES, N_VALUES, NUM_RANKS, is_fp8, torch_tensor

# The symmetric heap never frees, and every variant, comm_sms and num_stages point of a
# shape would otherwise allocate its own pair: a comm_sms sweep multiplies heap use by its
# length. Points of one shape run back to back, so they share the last pair allocated.
_iris_buffers = {}

# The TDM axes' defaults; the other backends are measured only at these.
_TDM_DEFAULTS = {"comm_sms": 64, "num_stages": 6, "block_size_m": 128}
_TDM_BLOCK_SIZE_N = 128


def _iris_buffers_for(ctx, M, N, dtype):
    key = (M, N, dtype)
    if key not in _iris_buffers:
        _iris_buffers.clear()
        world_size = ctx.get_num_ranks()
        _iris_buffers[key] = (
            ctx.zeros((M, N * world_size), dtype=dtype),
            ctx.zeros((M, N * world_size), dtype=dtype),
        )
    return _iris_buffers[key]


@bench.register
@bench.axis("num_ranks", NUM_RANKS)
@bench.axis("M", M_VALUES)
@bench.axis("N", N_VALUES)
@bench.axis("dtype", DTYPES)
@bench.axis("variant", ["persistent", "tdm"])
# Only the TDM kernel reads these. Sweep them with e.g. --axis_comm_sms=32,64,128,256.
@bench.axis("comm_sms", [_TDM_DEFAULTS["comm_sms"]])
@bench.axis("num_stages", [_TDM_DEFAULTS["num_stages"]])
@bench.axis("block_size_m", [_TDM_DEFAULTS["block_size_m"]])
@bench.axis("backend", ["iris", "rccl"])
def all_to_all(state, ctx):
    M, N, dtype = state["M"], state["N"], state["dtype"]
    variant = state["variant"]
    world_size = ctx.get_num_ranks()

    state.set_bytes((world_size - 1) * M * N * dtype.itemsize)
    rank = ctx.get_rank()

    if variant != "tdm" and any(state[name] != value for name, value in _TDM_DEFAULTS.items()):
        # The persistent kernel and RCCL ignore these axes; measure them once.
        state.skip("comm_sms, num_stages and block_size_m configure the TDM kernel; measured once otherwise")

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

    arch = torch.cuda.get_device_properties(torch.cuda.current_device()).gcnArchName
    if variant == "tdm":
        if not arch.startswith("gfx1250"):
            state.skip("the TDM all-to-all needs gfx1250")
        ring_bytes = state["num_stages"] * state["block_size_m"] * _TDM_BLOCK_SIZE_N * dtype.itemsize
        max_lds = torch.cuda.get_device_properties(torch.cuda.current_device()).shared_memory_per_block
        if ring_bytes > max_lds:
            # The launcher would raise, ending the whole sweep.
            state.skip(f"a {ring_bytes}-byte LDS ring does not fit in {max_lds} bytes")

    # All-to-all: input/output are (M, N * world_size) concatenated
    inp, out = _iris_buffers_for(ctx, M, N, dtype)

    for target in range(world_size):
        inp[:, target * N : (target + 1) * N] = float(rank * 1000 + target)

    if variant == "tdm":
        # Unlike all-gather, each loaded tile is stored once, so the stores in flight are
        # bounded by the LDS ring itself: num_stages // 2 tiles. A deep ring of 32 KiB
        # tiles uses the same 192 KiB of LDS as the TDM all-gather's config.
        config = Config(
            use_gluon=True,
            all_to_all_variant="tdm",
            block_size_m=state["block_size_m"],
            block_size_n=_TDM_BLOCK_SIZE_N,
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
