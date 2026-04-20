#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.

"""Benchmark for fused GEMM + all-gather (iris.ops)."""

import torch
import iris.bench as bench
from iris.ops import FusedConfig


@bench.register
@bench.axis("num_ranks", [2, 4, 8])
@bench.axis("M_local", [1024, 4096, 16384])
@bench.axis("N", [3584])
@bench.axis("K", [8192])
@bench.axis("dtype", [torch.float16])
def matmul_all_gather(state, ctx):
    M_local, N, K = state["M_local"], state["N"], state["K"]
    dtype = state["dtype"]
    world_size = ctx.get_num_ranks()
    M = M_local * world_size

    A = ctx.zeros((M_local, K), dtype=dtype)
    A.fill_(1.0)
    B = torch.randn((K, N), device="cuda", dtype=dtype)
    C = ctx.zeros((M, N), dtype=dtype)

    config = FusedConfig()

    state.set_flops(2 * M_local * N * K)
    state.set_bytes((world_size - 1) * M_local * N * A.element_size())

    state.exec(
        lambda: ctx.ops.matmul_all_gather(C, A, B, config=config),
        preamble_fn=lambda: C.zero_(),
    )


if __name__ == "__main__":
    bench.main()
