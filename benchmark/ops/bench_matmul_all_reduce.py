#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.

"""Benchmark for fused GEMM + all-reduce (iris.ops)."""

import torch
import iris.bench as bench
from iris.ops import FusedConfig, matmul_all_reduce_preamble


@bench.register
@bench.axis("num_ranks", [2, 4, 8])
@bench.axis("M", [1024, 4096, 16384])
@bench.axis("N", [3584])
@bench.axis("K", [8192])
@bench.axis("dtype", [torch.float16])
@bench.axis("variant", ["atomic", "two_shot"])
def matmul_all_reduce(state, ctx):
    M, N, K = state["M"], state["N"], state["K"]
    dtype = state["dtype"]
    variant = state["variant"]
    world_size = ctx.get_num_ranks()

    A = ctx.zeros((M, K), dtype=dtype)
    A.fill_(float(ctx.get_rank() + 1) * 0.01)
    B = torch.randn((K, N), device="cuda", dtype=dtype)
    C = ctx.zeros((M, N), dtype=dtype)

    config = FusedConfig(all_reduce_variant=variant)
    workspace = matmul_all_reduce_preamble(ctx, C, A, B, config=config)

    state.set_flops(2 * M * N * K)

    def _run():
        workspace.prepared = False
        ctx.ops.matmul_all_reduce(C, A, B, config=config, workspace=workspace)

    def _preamble():
        C.zero_()
        if workspace.locks is not None:
            workspace.locks.zero_()
        if workspace.aux_buffer is not None:
            workspace.aux_buffer.zero_()
        workspace.prepared = True

    state.exec(_run, preamble_fn=_preamble)


if __name__ == "__main__":
    bench.main()
