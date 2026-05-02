#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.

"""Benchmark RCCL all-reduce with environment variable sweeps.

Demonstrates ``@bench.env()`` — each env var combination triggers a
separate process spawn because RCCL reads these at init time.

Usage::

    python bench_rccl_env_sweep.py
    python bench_rccl_env_sweep.py --axis_M=1024,4096
"""

import torch
import torch.distributed as dist
import iris.bench as bench


@bench.register
@bench.env("NCCL_MIN_CTAS", [2, 4, 8])
@bench.axis("M", [1024, 4096])
@bench.axis("N", [2880])
@bench.axis("dtype", [torch.bfloat16])
def rccl_cta_sweep(state, ctx):
    M, N, dtype = state["M"], state["N"], state["dtype"]
    world_size = ctx.get_num_ranks()

    inp = torch.randn(M, N, dtype=dtype, device=f"cuda:{ctx.get_rank()}")
    out = inp.clone()

    state.set_bytes(int(M * N * inp.element_size() * 2 * (world_size - 1) / world_size))

    def run():
        out.copy_(inp)
        dist.all_reduce(out)

    state.exec(run)


if __name__ == "__main__":
    bench.main()
