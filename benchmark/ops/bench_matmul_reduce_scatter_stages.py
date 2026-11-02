#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.

"""
Stage profiler for fused GEMM + Reduce-Scatter.

Models the O_proj (row-parallel GEMM) + Reduce-Scatter for GPT-OSS-120B in
Sequence Parallelism mode.  GPT-OSS-120B is an MoE model (128 experts, top-4)
whose MoE layer handles its own sequence parallelism internally, so there is
no All-Gather after RMSNorm and no column-parallel FFN GEMM.

SP dataflow (GPT-OSS-120B):
  O_proj: [M, K_local] x [K_local, N] → partial [M, N]
    → Reduce-Scatter → [M/tp, N]
    → RMSNorm on [M/tp, N]
    → (MoE receives [M/tp, N] directly — handles its own AG at exit)

  K = num_heads * head_dim = 64 * 64 = 4096
  N = hidden_size = 2880

Sweeps optimization stages (atomic → two_shot → one_shot) across
vLLM-shaped workloads (decode / hybrid / prefill) and compares
against the unfused baseline (torch.mm + dist.reduce_scatter_tensor).
"""

import torch
import torch.distributed as dist
import iris.bench as bench
from iris.ops import FusedConfig, matmul_reduce_scatter, matmul_reduce_scatter_preamble


# --- Unfused baseline: torch.mm + RCCL reduce_scatter_tensor ---


@bench.register
@bench.axis("num_ranks", [2, 4, 8])
@bench.axis("M", [32, 896, 2048])
@bench.axis("N", [2880])
@bench.axis("K", [4096])
@bench.axis("dtype", [torch.float16])
def unfused_mm_reduce_scatter(state, ctx):
    M, N, K_global = state["M"], state["N"], state["K"]
    dtype = state["dtype"]
    num_ranks = ctx.get_num_ranks()
    K_local = K_global // num_ranks
    M_local = M // num_ranks  # sequence shard each rank owns after RS

    A = torch.randn((M, K_local), device="cuda", dtype=dtype)
    B = torch.randn((K_local, N), device="cuda", dtype=dtype)
    C_full = torch.empty((M, N), device="cuda", dtype=dtype)
    C_local = torch.empty((M_local, N), device="cuda", dtype=dtype)

    state.set_flops(2 * M * N * K_local)
    state.set_bytes((num_ranks - 1) * M_local * N * A.element_size())

    def _run():
        torch.mm(A, B, out=C_full)
        dist.reduce_scatter_tensor(C_local, C_full, op=dist.ReduceOp.SUM)

    state.exec(_run)


# --- Fused GEMM+RS: sweep variant × tile config ---


@bench.register
@bench.axis("num_ranks", [2, 4, 8])
@bench.axis("M", [32, 896, 2048])
@bench.axis("N", [2880])
@bench.axis("K", [4096])
@bench.axis("dtype", [torch.float16])
@bench.axis("bm", [32, 64, 128])
@bench.axis("bn", [64, 128])
def fused_gemm_reduce_scatter(state, ctx):
    M, N, K_global = state["M"], state["N"], state["K"]
    dtype = state["dtype"]
    bm, bn = state["bm"], state["bn"]
    num_ranks = ctx.get_num_ranks()
    K_local = K_global // num_ranks
    M_local = M // num_ranks

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
    C = ctx.zeros((M_local, N), dtype=dtype)  # RS output: each rank owns M_local rows

    config = FusedConfig(
        block_size_m=bm,
        block_size_n=bn,
        block_size_k=64,
        group_size_m=4,
        num_xcds=8,
        all_reduce_variant="two_shot",  # RS kernel only implements two_shot
    )
    workspace = matmul_reduce_scatter_preamble(ctx, C, A, B, config=config)

    state.set_flops(2 * M * N * K_local)
    state.set_bytes((num_ranks - 1) * M_local * N * A.element_size())

    def _run():
        workspace.prepared = False
        matmul_reduce_scatter(ctx, C, A, B, config=config, workspace=workspace)

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
