#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.

"""Benchmark for iris-ccl all-gather collective."""

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
@bench.axis("backend", ["iris", "rccl"])
def all_gather(state, ctx):
    M, N, dtype = state["M"], state["N"], state["dtype"]
    world_size = ctx.get_num_ranks()

    state.set_bytes((world_size - 1) * M * N * dtype.itemsize)

    if state["backend"] == "rccl":
        t_in = _torch_tensor(ctx, (M, N), dtype)
        t_out = _torch_tensor(ctx, (world_size * M, N), dtype)
        t_in.fill_(float(ctx.get_rank() + 1))
        state.exec(
            lambda: dist.all_gather_into_tensor(t_out, t_in),
            preamble_fn=lambda: t_out.zero_(),
        )
        return

    inp = ctx.zeros((M, N), dtype=dtype)
    out = ctx.zeros((world_size * M, N), dtype=dtype)
    inp.fill_(float(ctx.get_rank() + 1))

    config = Config()
    state.exec(
        lambda: ctx.ccl.all_gather(out, inp, config=config),
        preamble_fn=lambda: out.zero_(),
    )


if __name__ == "__main__":
    bench.main()
