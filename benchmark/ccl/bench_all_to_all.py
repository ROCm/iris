#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.

"""Benchmark for iris-ccl all-to-all collective."""

import torch
import torch.distributed as dist
import iris.bench as bench
from iris.ccl import Config


# The RCCL baseline deliberately uses ordinary torch tensors rather than Iris
# heap memory. Iris allocates its symmetric heap with flags RMA requires, and
# timing RCCL against that would measure Iris's allocation choice rather than
# RCCL. A user weighing "Iris or RCCL?" would call RCCL on normal tensors.
#
# Layout note: Iris all_to_all uses (M, N*W) split along dim 1, while
# dist.all_to_all_single splits along dim 0. The RCCL path therefore uses
# (M*W, N). The communication pattern and total bytes are identical -- W-1
# peer messages of M*N elements each -- so the bandwidth comparison holds; only
# the in-memory layout differs.
def _torch_tensor(ctx, shape, dtype):
    return torch.zeros(shape, dtype=dtype, device=f"cuda:{ctx.get_rank()}")


@bench.register
@bench.axis("num_ranks", [2, 4, 8])
@bench.axis("M", bench.power_of_two(10, 14))
@bench.axis("N", bench.power_of_two(10, 14))
@bench.axis("dtype", [torch.float16, torch.bfloat16])
@bench.axis("backend", ["iris", "rccl"])
def all_to_all(state, ctx):
    M, N, dtype = state["M"], state["N"], state["dtype"]
    world_size = ctx.get_num_ranks()

    state.set_bytes((world_size - 1) * M * N * dtype.itemsize)
    rank = ctx.get_rank()

    if state["backend"] == "rccl":
        t_in = _torch_tensor(ctx, (M * world_size, N), dtype)
        t_out = _torch_tensor(ctx, (M * world_size, N), dtype)
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
