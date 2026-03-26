#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.

"""Benchmark for fused GEMM + reduce-scatter (iris.ops)."""

import torch
import iris.bench as bench
from iris.ops import FusedConfig


@bench.register
@bench.axis("num_ranks", [2, 4, 8])
@bench.axis("M", [1024, 4096, 16384])
@bench.axis("N", [3584])
@bench.axis("K", [8192])
@bench.axis("dtype", [torch.float16])
def matmul_reduce_scatter(state, ctx):
    M, N, K = state["M"], state["N"], state["K"]
    dtype = state["dtype"]

    A = ctx.zeros((M, K), dtype=dtype)
    A.fill_(float(ctx.get_rank() + 1) * 0.01)
    B = torch.randn((K, N), device="cuda", dtype=dtype)
    C = ctx.zeros((M, N), dtype=dtype)

    config = FusedConfig()

    state.set_flops(2 * M * N * K)

    state.exec(
        lambda: ctx.ops.matmul_reduce_scatter(C, A, B, config=config),
    )


if __name__ == "__main__":
    bench.main()
