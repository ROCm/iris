#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.

"""
Stage profiler for fused GEMM + All-Reduce.

Sweeps optimization stages (atomic → two_shot → one_shot) across
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


# --- Fused GEMM+AR: sweep variant × tile config ---


@bench.register
@bench.axis("num_ranks", [2, 4, 8])
@bench.axis("M", [32, 896, 2048])
@bench.axis("N", [2880])
@bench.axis("K", [4096])
@bench.axis("dtype", [torch.float16])
@bench.axis("variant", ["atomic", "two_shot", "one_shot"])
@bench.axis("bm", [32, 64, 128])
@bench.axis("bn", [64, 128])
def fused_gemm_allreduce(state, ctx):
    M, N, K_global = state["M"], state["N"], state["K"]
    dtype = state["dtype"]
    variant = state["variant"]
    bm, bn = state["bm"], state["bn"]
    K_local = K_global // ctx.get_num_ranks()

    # Skip configs where block > problem size
    if bm > M:
        state.skip(f"bm={bm} > M={M}")
        return
    if bn > N:
        state.skip(f"bn={bn} > N={N}")
        return

    A = ctx.zeros((M, K_local), dtype=dtype)
    A.fill_(float(ctx.get_rank() + 1) * 0.01)
    B = torch.randn((K_local, N), device="cuda", dtype=dtype)
    C = ctx.zeros((M, N), dtype=dtype)

    config = FusedConfig(
        block_size_m=bm,
        block_size_n=bn,
        block_size_k=64,
        group_size_m=4,
        num_xcds=8,
        all_reduce_variant=variant,
    )
    workspace = matmul_all_reduce_preamble(ctx, C, A, B, config=config)

    state.set_flops(2 * M * N * K_local)

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
