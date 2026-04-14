#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.

"""
Benchmark for K-split fused GEMM + reduce-scatter (SpatialK approach).

Compares:
  1. ksplit_fused_reduce_scatter: K-split GEMM with atomic_add RS fusion
  2. unfused_mm_reduce_scatter: torch.mm + torch.distributed.reduce_scatter_tensor
  3. fused_all_gather_matmul: existing iris all_gather+GEMM (pull pattern)
"""

import torch
import torch.distributed as dist
import iris.bench as bench
from iris.ops import FusedConfig


@bench.register
@bench.axis("num_ranks", [4])
@bench.axis("M", [1024, 2048, 4096, 8192])
@bench.axis("N", [2880, 7168])
@bench.axis("K", [8192, 14336])
@bench.axis("dtype", [torch.float16])
def ksplit_fused_reduce_scatter(state, ctx):
    """K-split GEMM with fused reduce-scatter via atomic_add."""
    M, N, K = state["M"], state["N"], state["K"]
    dtype = state["dtype"]
    world_size = ctx.get_num_ranks()
    K_local = K // world_size
    N_local = N // world_size

    A_shard = torch.randn((M, K_local), device="cuda", dtype=dtype)
    B = torch.randn((K_local, N), device="cuda", dtype=dtype)
    C = ctx.zeros((M, N_local), dtype=dtype)

    config = FusedConfig(ksplit=True)
    state.set_flops(2 * M * N * K_local)

    def _run():
        ctx.ops.matmul_reduce_scatter(C, A_shard, B, config=config)

    def _preamble():
        C.zero_()
        ctx.barrier()

    state.exec(_run, preamble_fn=_preamble)


@bench.register
@bench.axis("num_ranks", [4])
@bench.axis("M", [1024, 2048, 4096, 8192])
@bench.axis("N", [2880, 7168])
@bench.axis("K", [8192, 14336])
@bench.axis("dtype", [torch.float16])
def unfused_mm_reduce_scatter(state, ctx):
    """Baseline: torch.mm + separate RCCL reduce-scatter."""
    M, N, K = state["M"], state["N"], state["K"]
    dtype = state["dtype"]
    world_size = ctx.get_num_ranks()
    K_local = K // world_size
    N_local = N // world_size

    A_shard = torch.randn((M, K_local), device="cuda", dtype=dtype)
    B = torch.randn((K_local, N), device="cuda", dtype=dtype)
    C_full = torch.empty((M, N), device="cuda", dtype=dtype)
    C_local = torch.empty((M, N_local), device="cuda", dtype=dtype)

    state.set_flops(2 * M * N * K_local)

    def _run():
        torch.mm(A_shard, B, out=C_full)
        dist.reduce_scatter_tensor(C_local, C_full, op=dist.ReduceOp.SUM)

    state.exec(_run)


@bench.register
@bench.axis("num_ranks", [4])
@bench.axis("M", [1024, 2048, 4096, 8192])
@bench.axis("N", [2880, 7168])
@bench.axis("K", [8192, 14336])
@bench.axis("dtype", [torch.float16])
def fused_all_gather_matmul(state, ctx):
    """Existing iris fused all-gather + GEMM (K-sharded pull pattern)."""
    M, N, K = state["M"], state["N"], state["K"]
    dtype = state["dtype"]
    world_size = ctx.get_num_ranks()
    K_local = K // world_size

    A_shard = ctx.zeros((M, K_local), dtype=dtype)
    A_shard.copy_(torch.randn_like(A_shard))
    B = torch.randn((K, N), device="cuda", dtype=dtype)
    C = torch.empty((M, N), device="cuda", dtype=dtype)

    config = FusedConfig()
    state.set_flops(2 * M * N * K)

    state.exec(
        lambda: ctx.ops.all_gather_matmul(C, A_shard, B, config=config),
    )


if __name__ == "__main__":
    bench.main()
