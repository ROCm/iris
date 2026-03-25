# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.

"""
iris.bench — GPU Benchmarking Framework

A declarative benchmarking framework for iris that eliminates boilerplate.
Write ~25 lines instead of ~350 to benchmark a GPU kernel.

Example::

    import torch
    import iris.bench as bench
    from iris.ccl import Config

    @bench.register
    @bench.axis("num_ranks", [2, 4, 8])
    @bench.axis("M", bench.power_of_two(8, 13))
    @bench.axis("N", [256, 512, 1024])
    @bench.axis("dtype", [torch.float16, torch.float32])
    def all_gather(state, ctx):
        M, N, dtype = state["M"], state["N"], state["dtype"]
        world_size = ctx.get_num_ranks()

        inp = ctx.zeros((M, N), dtype=dtype)
        out = ctx.zeros((world_size * M, N), dtype=dtype)
        inp.fill_(float(ctx.get_rank() + 1))

        state.set_bytes((world_size - 1) * M * N * inp.element_size())

        config = Config(use_gluon=False)
        state.exec(lambda: ctx.ccl.all_gather(out, inp, config=config))

    if __name__ == "__main__":
        bench.main()

Run::

    python bench_all_gather.py
    python bench_all_gather.py --skip_num_ranks=2
"""

from ._core import (
    AxisDef,
    BenchmarkDef,
    Result,
    State,
    axis,
    linear_range,
    power_of_two,
    register,
)
from ._runner import main

__all__ = [
    "AxisDef",
    "BenchmarkDef",
    "Result",
    "State",
    "axis",
    "linear_range",
    "main",
    "power_of_two",
    "register",
]
