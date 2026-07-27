#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.

"""
Benchmark: fused GEMM+ReduceScatter with HBM buffer staging.

Compares three variants for the GPT-OSS-120B MoE O_proj → ReduceScatter segment:
  1. Unfused: torch.mm + dist.reduce_scatter_tensor (RCCL baseline)
  2. Fused (existing): iris matmul_reduce_scatter
  3. Fused HBM buffer: iris matmul_reduce_scatter_hbm_buffer (this PR)

Shapes from aporva's PR #513:
  K = 4096, N = 2880
  M = 32 (decode), 896 (hybrid), 2048 (prefill)
"""

import torch
import torch.distributed as dist
import iris.bench as bench
from iris.ops import FusedConfig
from iris.ops.matmul_reduce_scatter_hbm_buffer import (
    matmul_reduce_scatter_hbm_buffer,
    matmul_reduce_scatter_hbm_buffer_preamble,
)


# --- Unfused baseline: torch.mm + RCCL reduce_scatter_tensor ---

@bench.register
@bench.axis("num_ranks", [2, 4, 8])
@bench.axis("M", [2048])
@bench.axis("N", [2880])
@bench.axis("K", [4096])
@bench.axis("dtype", [torch.float16])
def unfused_mm_reduce_scatter(state, ctx):
    M, N, K_global = state["M"], state["N"], state["K"]
    dtype = state["dtype"]
    num_ranks = ctx.get_num_ranks()
    K_local = K_global // num_ranks
    M_local = M // num_ranks

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


# --- Fused HBM buffer GEMM+RS ---

@bench.register
@bench.axis("num_ranks", [2, 4, 8])
@bench.axis("M", [2048])
@bench.axis("N", [2880])
@bench.axis("K", [4096])
@bench.axis("dtype", [torch.float16])
@bench.axis("bm", [128])
@bench.axis("bn", [64])
@bench.axis("scatter_sms", [16, 32, 64])
def fused_hbm_buffer_gemm_rs(state, ctx):
    M, N, K_global = state["M"], state["N"], state["K"]
    dtype = state["dtype"]
    bm, bn = state["bm"], state["bn"]
    scatter_sms = state["scatter_sms"]
    num_ranks = ctx.get_num_ranks()
    K_local = K_global // num_ranks
    M_local = M // num_ranks

    if M % (num_ranks * bm) != 0:
        state.skip(f"M={M} not divisible by num_ranks*bm={num_ranks * bm}")
        return

    A = ctx.zeros((M, K_local), dtype=dtype)
    A.fill_(float(ctx.get_rank() + 1) * 0.01)
    B = torch.randn((K_local, N), device="cuda", dtype=dtype)
    C = torch.zeros((M_local, N), device="cuda", dtype=dtype)

    config = FusedConfig(
        block_size_m=bm,
        block_size_n=bn,
        block_size_k=64,
        group_size_m=4,
    )
    workspace = matmul_reduce_scatter_hbm_buffer_preamble(ctx, A, B, config)

    state.set_flops(2 * M * N * K_local)
    state.set_bytes((num_ranks - 1) * M_local * N * A.element_size())

    def _run():
        matmul_reduce_scatter_hbm_buffer(
            ctx, C, A, B, config=config, workspace=workspace,
            num_scatter_sms=scatter_sms,
        )

    def _preamble():
        C.zero_()
        workspace.locks.zero_()

    state.exec(_run, preamble_fn=_preamble)


if __name__ == "__main__":
    bench.main()
