#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.

"""
End-to-end SP segment benchmark for GPT-OSS-120B.

Models the SP segment between attention output and MoE input for GPT-OSS-120B:

  O_proj (row-parallel): [M, K_local] x [K_local, N] → partial [M, N]
    → Reduce-Scatter: [M/tp, N]
    → RMSNorm: [M/tp, N]
    → (MoE handles its own SP from here)

  K = num_heads * head_dim = 64 * 64 = 4096
  N = hidden_size = 2880

GPT-OSS-120B is an MoE model (128 experts, top-4) whose MoE layer handles its
own sequence parallelism internally (sequence_parallel_chunk at entry +
tensor_model_parallel_all_gather at exit).  There is no All-Gather after
RMSNorm and no column-parallel FFN GEMM — the MoE receives [M/tp, N] directly.

Two benchmarks:

  unfused_gemm_rs_rmsnorm
      torch.mm + dist.reduce_scatter_tensor + aiter RMSNorm
      Covers all M (decode / hybrid / prefill).

  fused_gemm_rs_rmsnorm_iris
      iris matmul_reduce_scatter (GEMM+RS fused, tile-sweep) + aiter RMSNorm.
      Covers all M sizes; sweeps bm/bn tile configs.

Shapes:
  M:       sequence length  (32=decode, 896=hybrid, 2048=prefill)
  N=2880:  hidden dimension
  K=4096:  global K split across ranks (K_local = K // num_ranks)
  M_local = M // num_ranks

FLOPs = 2 * M * N * K_local  (single O_proj GEMM)
Bytes = (num_ranks - 1) * M_local * N * element_size  (RS only)
"""

import torch
import torch.distributed as dist
import torch.nn.functional as F
import iris.bench as bench
from iris.ops import (
    FusedConfig,
    matmul_reduce_scatter, matmul_reduce_scatter_preamble,
)

# Use aiter's Triton RMSNorm where available, fall back to torch
try:
    from aiter.ops.triton.rmsnorm import rms_norm as _aiter_rms_norm
    def _rmsnorm(x, weight, eps=1e-6):
        return _aiter_rms_norm(x, weight, eps)
except Exception:
    def _rmsnorm(x, weight, eps=1e-6):
        return F.rms_norm(x, [x.shape[-1]], weight=weight, eps=eps)


# ---------------------------------------------------------------------------
# Unfused baseline: torch.mm + NCCL RS + RMSNorm
# ---------------------------------------------------------------------------


@bench.register
@bench.axis("num_ranks", [2, 4, 8])
@bench.axis("M", [32, 896, 2048])
@bench.axis("N", [2880])
@bench.axis("K", [4096])
@bench.axis("dtype", [torch.float16])
def unfused_gemm_rs_rmsnorm(state, ctx):
    M, N, K_global = state["M"], state["N"], state["K"]
    dtype = state["dtype"]
    num_ranks = ctx.get_num_ranks()

    if M % num_ranks != 0:
        state.skip(f"M={M} not divisible by num_ranks={num_ranks}")
        return

    K_local = K_global // num_ranks
    M_local = M // num_ranks

    # --- Allocate buffers ---
    # O_proj: each rank holds K_local columns of the weight
    A = torch.randn((M, K_local), device="cuda", dtype=dtype)
    B_out = torch.randn((K_local, N), device="cuda", dtype=dtype)
    C_full = torch.empty((M, N), device="cuda", dtype=dtype)    # intermediate full GEMM output
    C_local = torch.empty((M_local, N), device="cuda", dtype=dtype)  # RS output shard

    # RMSNorm weight (applied to the M_local shard in the SP region)
    gamma = torch.ones(N, device="cuda", dtype=dtype)

    # Single O_proj GEMM only (no FFN — MoE handles that)
    state.set_flops(2 * M * N * K_local)
    state.set_bytes((num_ranks - 1) * M_local * N * A.element_size())

    def _run():
        # 1. O_proj GEMM  (row-parallel: partial sum per rank)
        torch.mm(A, B_out, out=C_full)
        # 2. Reduce-Scatter  (sum partials; each rank receives M_local rows)
        dist.reduce_scatter_tensor(C_local, C_full, op=dist.ReduceOp.SUM)
        # 3. RMSNorm on local shard  (SP region: M_local rows per rank)
        _rmsnorm(C_local, gamma)

    state.exec(_run)


# ---------------------------------------------------------------------------
# iris GEMM+RS  |  aiter RMSNorm
# ---------------------------------------------------------------------------


@bench.register
@bench.axis("num_ranks", [2, 4, 8])
@bench.axis("M", [32, 896, 2048])
@bench.axis("N", [2880])
@bench.axis("K", [4096])
@bench.axis("dtype", [torch.float16])
@bench.axis("bm", [32, 64, 128])
@bench.axis("bn", [64, 128])
def fused_gemm_rs_rmsnorm_iris(state, ctx):
    M, N, K_global = state["M"], state["N"], state["K"]
    dtype = state["dtype"]
    bm, bn = state["bm"], state["bn"]
    num_ranks = ctx.get_num_ranks()

    if M % num_ranks != 0:
        state.skip(f"M={M} not divisible by num_ranks={num_ranks}")
        return

    if bm > M:
        state.skip(f"bm={bm} > M={M}")
        return
    if bn > N:
        state.skip(f"bn={bn} > N={N}")
        return

    K_local = K_global // num_ranks
    M_local = M // num_ranks

    # --- Allocate buffers in iris shared memory where required ---
    A = ctx.zeros((M, K_local), dtype=dtype)
    A.fill_(float(ctx.get_rank() + 1) * 0.01)
    B_out = torch.randn((K_local, N), device="cuda", dtype=dtype)
    # iris matmul_reduce_scatter writes the RS shard directly to C_local
    C_local = ctx.zeros((M_local, N), dtype=dtype)

    # RMSNorm weight
    gamma = torch.ones(N, device="cuda", dtype=dtype)

    # Tile config for the fused GEMM+RS kernel
    rs_config = FusedConfig(
        block_size_m=bm,
        block_size_n=bn,
        block_size_k=64,
        group_size_m=4,
        num_xcds=8,
        all_reduce_variant="two_shot",  # RS kernel only implements two_shot
    )
    workspace = matmul_reduce_scatter_preamble(ctx, C_local, A, B_out, config=rs_config)

    # Single O_proj GEMM only (no FFN — MoE handles that)
    state.set_flops(2 * M * N * K_local)
    state.set_bytes((num_ranks - 1) * M_local * N * A.element_size())

    def _run():
        # 1+2. Fused O_proj GEMM + Reduce-Scatter (single iris kernel)
        workspace.prepared = False
        matmul_reduce_scatter(ctx, C_local, A, B_out, config=rs_config, workspace=workspace)

        # 3. RMSNorm on the RS output shard (M_local, N)
        _rmsnorm(C_local, gamma)

    def _preamble():
        C_local.zero_()
        if workspace.locks is not None:
            workspace.locks.zero_()
        if workspace.aux_buffer is not None:
            workspace.aux_buffer.zero_()
        workspace.prepared = True

    state.exec(_run, preamble_fn=_preamble)


if __name__ == "__main__":
    bench.main()
