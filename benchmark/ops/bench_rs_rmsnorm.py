#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.

"""
Stage profiler for the SP middle section: Reduce-Scatter + RMSNorm.

Models the section between O_proj output and MoE input for GPT-OSS-120B in
Sequence Parallelism mode.  GPT-OSS-120B is an MoE model (128 experts, top-4)
whose MoE layer handles its own sequence parallelism internally
(sequence_parallel_chunk at entry + tensor_model_parallel_all_gather at exit),
so there is no All-Gather after RMSNorm.

    O_proj output [M, N]  (partial sums, one per rank)
      -> Reduce-Scatter       each rank receives M_local = M/tp contiguous rows
      -> RMSNorm              normalize the M_local-row shard
      -> (MoE receives [M_local, N] directly — handles its own AG at exit)

  K = num_heads * head_dim = 64 * 64 = 4096
  N = hidden_size = 2880

Two benchmarks:

  unfused_rs_rmsnorm
      dist.reduce_scatter_tensor  (NCCL)
      + aiter triton rms_norm

  fused_ready_rs_rmsnorm
      Same ops but allocated with iris shared memory so the buffer is ready
      for the eventual aiter fused kernel swap-in.

Shapes match the O_proj benchmarks (N=2880 hidden dim, M=32/896/2048).
"""

import torch
import torch.distributed as dist
import torch.nn.functional as F
import iris.bench as bench

# Use aiter's Triton RMSNorm if available; fall back to torch
try:
    from aiter.ops.triton.rmsnorm import rms_norm as _aiter_rms_norm

    def rmsnorm(x, weight, eps=1e-6):
        return _aiter_rms_norm(x, weight, eps)

    _RMSNORM_BACKEND = "aiter-triton"
except Exception:

    def rmsnorm(x, weight, eps=1e-6):
        return F.rms_norm(x, [x.shape[-1]], weight=weight, eps=eps)

    _RMSNORM_BACKEND = "torch"


# ---------------------------------------------------------------------------
# Unfused: NCCL RS  +  RMSNorm
# Covers all M values (32 / 896 / 2048) and all TP degrees.
# ---------------------------------------------------------------------------


@bench.register
@bench.axis("num_ranks", [2, 4, 8])
@bench.axis("M", [32, 896, 2048])
@bench.axis("N", [2880])
@bench.axis("dtype", [torch.float16])
def unfused_rs_rmsnorm(state, ctx):
    M, N = state["M"], state["N"]
    dtype = state["dtype"]
    num_ranks = ctx.get_num_ranks()

    if M % num_ranks != 0:
        state.skip(f"M={M} not divisible by num_ranks={num_ranks}")
        return

    M_local = M // num_ranks

    # Input: full [M, N] partial-sum tensor (output of O_proj GEMM on each rank)
    inp = torch.randn((M, N), device="cuda", dtype=dtype)
    # RS output: contiguous [M_local, N] shard
    rs_out = torch.empty((M_local, N), device="cuda", dtype=dtype)
    # RMSNorm weight
    gamma = torch.ones(N, device="cuda", dtype=dtype)

    # RS comm volume: each rank sends (num_ranks-1) shards of size M_local * N
    state.set_bytes((num_ranks - 1) * M_local * N * inp.element_size())

    def _run():
        # 1. Reduce-Scatter: sum partial outputs, each rank gets M_local rows
        dist.reduce_scatter_tensor(rs_out, inp, op=dist.ReduceOp.SUM)
        # 2. RMSNorm on the M_local-row shard
        rmsnorm(rs_out, gamma)

    state.exec(_run)


# ---------------------------------------------------------------------------
# "Fused-ready" variant: same ops but input lives in iris shared memory.
# This buffer layout is compatible with the aiter fused kernel calling convention.
# Once the aiter comms module is released, replace the two-op _run body with
# a single kernel call.
# ---------------------------------------------------------------------------


@bench.register
@bench.axis("num_ranks", [2, 4, 8])
@bench.axis("M", [32, 896, 2048])
@bench.axis("N", [2880])
@bench.axis("dtype", [torch.float16])
def fused_ready_rs_rmsnorm(state, ctx):
    """
    Like unfused_rs_rmsnorm but all tensors allocated in iris shared memory,
    matching the calling convention of aiter's fused kernel.

    Swapping in the fused kernel is a one-line change once aiter.comms is
    available in the installed package.
    """
    M, N = state["M"], state["N"]
    dtype = state["dtype"]
    num_ranks = ctx.get_num_ranks()

    if M % num_ranks != 0:
        state.skip(f"M={M} not divisible by num_ranks={num_ranks}")
        return

    M_local = M // num_ranks

    # Allocate in iris shmem (required by the fused aiter kernel)
    inp = ctx.zeros((M, N), dtype=dtype)
    inp.fill_(float(ctx.get_rank() + 1) * 0.01)
    rs_out = ctx.zeros((M_local, N), dtype=dtype)
    gamma = torch.ones(N, device="cuda", dtype=dtype)

    # RS comm volume only (no AG)
    state.set_bytes((num_ranks - 1) * M_local * N * inp.element_size())

    def _run():
        # --- replace these two lines with the fused kernel call ---
        dist.reduce_scatter_tensor(rs_out, inp, op=dist.ReduceOp.SUM)
        rmsnorm(rs_out, gamma)
        # --- end replacement site ---

    def _preamble():
        rs_out.zero_()

    state.exec(_run, preamble_fn=_preamble)


if __name__ == "__main__":
    import iris.bench as _bench

    print(f"[info] RMSNorm backend: {_RMSNORM_BACKEND}")
    _bench.main()
