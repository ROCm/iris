#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.

"""Benchmark for iris-ccl all-reduce collective."""

import torch
import torch.distributed as dist
import iris.bench as bench
from iris.ccl import Config


# The RCCL baseline deliberately uses ordinary torch tensors rather than Iris
# heap memory. Iris allocates its symmetric heap with flags RMA requires (the
# banner reports the allocator), and timing RCCL against that would measure
# Iris's allocation choice rather than RCCL. A user weighing "Iris or RCCL?"
# would call RCCL on normal tensors, so that is what is measured.
def _torch_tensor(ctx, shape, dtype):
    return torch.zeros(shape, dtype=dtype, device=f"cuda:{ctx.get_rank()}")


@bench.register
@bench.axis("num_ranks", [2, 4, 8])
@bench.axis("M", bench.power_of_two(10, 14))
@bench.axis("N", bench.power_of_two(10, 14))
@bench.axis("dtype", [torch.float16, torch.bfloat16])
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
        t_in = _torch_tensor(ctx, (M, N), dtype)
        t_out = _torch_tensor(ctx, (M, N), dtype)
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
