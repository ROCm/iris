#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.

"""
Stage profiler for fused GEMM + All-Reduce.

Sweeps all-reduce variants (two_shot → one_shot) across
vLLM-shaped workloads (decode / hybrid / prefill) and compares
against the unfused baseline (torch.mm + dist.all_reduce).

"""

import torch
import torch.distributed as dist
import iris.bench as bench
from iris.ops import FusedConfig, matmul_all_reduce_preamble


# --- Unfused baseline: torch.mm + RCCL all-reduce ---


@bench.register
@bench.axis("num_ranks", [2, 4, 8])
@bench.axis("M", [32, 896, 2048])
@bench.axis("N", [2880])
@bench.axis("K", [4096])
@bench.axis("dtype", [torch.float16])
def unfused_mm_allreduce(state, ctx):
    M, N, K_global = state["M"], state["N"], state["K"]
    dtype = state["dtype"]
    K_local = K_global // ctx.get_num_ranks()

    A = torch.randn((M, K_local), device="cuda", dtype=dtype)
    B = torch.randn((K_local, N), device="cuda", dtype=dtype)
    C = torch.empty((M, N), device="cuda", dtype=dtype)

    state.set_flops(2 * M * N * K_local)

    def _run():
        torch.mm(A, B, out=C)
        dist.all_reduce(C, op=dist.ReduceOp.SUM)

    state.exec(_run)


# --- Fused GEMM+AR: sweep variant (tile config is chosen by the tritonBLAS selector) ---


@bench.register
@bench.axis("num_ranks", [2, 4, 8])
@bench.axis("M", [32, 896, 2048])
@bench.axis("N", [2880])
@bench.axis("K", [4096])
@bench.axis("dtype", [torch.float16])
@bench.axis("variant", ["two_shot", "one_shot"])
def fused_gemm_allreduce(state, ctx):
    M, N, K_global = state["M"], state["N"], state["K"]
    dtype = state["dtype"]
    variant = state["variant"]
    K_local = K_global // ctx.get_num_ranks()

    A = ctx.zeros((M, K_local), dtype=dtype)
    A.fill_(float(ctx.get_rank() + 1) * 0.01)
    B = torch.randn((K_local, N), device="cuda", dtype=dtype)
    C = ctx.zeros((M, N), dtype=dtype)

    config = FusedConfig(all_reduce_variant=variant)
    workspace = matmul_all_reduce_preamble(ctx, C, A, B, config=config)

    state.set_flops(2 * M * N * K_local)

    def _run():
        ctx.ops.matmul_all_reduce(C, A, B, config=config, workspace=workspace)

    def _preamble():
        C.zero_()

    state.exec(_run, preamble_fn=_preamble)


if __name__ == "__main__":
    bench.main()
