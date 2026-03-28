#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.

"""Benchmark for iris-ccl fused AllReduce + RMSNorm vs separate RCCL AR + torch RMSNorm."""

import torch
import torch.distributed as dist
import iris.bench as bench
from iris.ccl import Config


def _rmsnorm_torch(x: torch.Tensor, weight: torch.Tensor, eps: float) -> torch.Tensor:
    """Reference RMSNorm in float32."""
    x_fp32 = x.to(torch.float32)
    variance = x_fp32.pow(2).mean(dim=-1, keepdim=True)
    x_normed = x_fp32 * torch.rsqrt(variance + eps)
    return (x_normed * weight.to(torch.float32)).to(x.dtype)


@bench.register
@bench.axis("num_ranks", [8])
@bench.axis("tokens", [1, 4, 32, 128, 256, 512])
@bench.axis("hidden", [1024, 2048, 4096, 5120])
@bench.axis("dtype", [torch.bfloat16])
def fused_ar_rmsnorm(state, ctx):
    """Iris fused AllReduce + Residual Add + RMSNorm."""
    tokens, hidden, dtype = state["tokens"], state["hidden"], state["dtype"]
    rank = ctx.get_rank()
    world_size = ctx.get_num_ranks()
    eps = 1e-6

    partial = ctx.zeros((tokens, hidden), dtype=dtype)
    partial.fill_(float(rank + 1))
    residual = ctx.zeros((tokens, hidden), dtype=dtype)
    residual.fill_(1.0)
    weight = torch.ones(hidden, dtype=dtype, device=f"cuda:{rank}")

    # Bus bandwidth: allreduce moves 2*(W-1)/W * data, plus residual read/write + norm output
    data_bytes = tokens * hidden * partial.element_size()
    ar_bytes = int(data_bytes * 2 * (world_size - 1) / world_size)
    state.set_bytes(ar_bytes + 3 * data_bytes)  # AR + res_read + res_write + norm_out

    config = Config(all_reduce_variant="two_shot", all_reduce_distribution=1)

    def run():
        ctx.ccl.all_reduce_rmsnorm(partial, residual, weight, eps=eps, config=config)

    def preamble():
        residual.fill_(1.0)

    state.exec(run, preamble_fn=preamble)


@bench.register
@bench.axis("num_ranks", [8])
@bench.axis("tokens", [1, 4, 32, 128, 256, 512])
@bench.axis("hidden", [1024, 2048, 4096, 5120])
@bench.axis("dtype", [torch.bfloat16])
def separate_ar_rmsnorm(state, ctx):
    """Separate RCCL AllReduce + torch RMSNorm (baseline)."""
    tokens, hidden, dtype = state["tokens"], state["hidden"], state["dtype"]
    rank = ctx.get_rank()
    world_size = ctx.get_num_ranks()
    eps = 1e-6

    partial = torch.randn(tokens, hidden, dtype=dtype, device=f"cuda:{rank}")
    residual = torch.randn(tokens, hidden, dtype=dtype, device=f"cuda:{rank}")
    weight = torch.ones(hidden, dtype=dtype, device=f"cuda:{rank}")

    data_bytes = tokens * hidden * partial.element_size()
    ar_bytes = int(data_bytes * 2 * (world_size - 1) / world_size)
    state.set_bytes(ar_bytes + 3 * data_bytes)

    def run():
        dist.all_reduce(partial, op=dist.ReduceOp.SUM)
        residual.add_(partial)
        _rmsnorm_torch(residual, weight, eps)

    def preamble():
        partial.fill_(float(rank + 1))
        residual.fill_(1.0)

    state.exec(run, preamble_fn=preamble)


if __name__ == "__main__":
    bench.main()
