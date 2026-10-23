#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.

"""Benchmarks for fused GEMM + all-gather (iris.ops) and its RCCL baselines."""

import torch
import torch.distributed as dist
import tritonblas
import iris.bench as bench
from iris.ops import FusedConfig
from tritonblas.matmul import persistent_matmul_lt


def _make_inputs(ctx, M_local, N, K, dtype):
    # Per-rank seed for A (each rank holds a different shard); shared seed for B.
    # Same seeding as the all-gather + matmul benchmarks.
    rank = ctx.get_rank()
    A = ctx.randn((M_local, K), dtype=dtype, generator=torch.Generator("cuda").manual_seed(42 + rank))
    B = ctx.randn((K, N), dtype=dtype, generator=torch.Generator("cuda").manual_seed(123))
    return A, B


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

    A, B = _make_inputs(ctx, M_local, N, K, dtype)
    C = ctx.zeros((M, N), dtype=dtype)

    config = FusedConfig()

    state.set_flops(2 * M_local * N * K)
    state.set_bytes((world_size - 1) * M_local * N * A.element_size())

    state.exec(
        lambda: ctx.ops.matmul_all_gather(C, A, B, config=config),
        preamble_fn=lambda: C.zero_(),
    )


@bench.register
@bench.axis("num_ranks", [2, 4, 8])
@bench.axis("M_local", [1024, 4096, 16384])
@bench.axis("N", [3584])
@bench.axis("K", [8192])
@bench.axis("dtype", [torch.float16])
def pytorch_matmul_all_gather(state, ctx):
    """PyTorch/RCCL baseline: torch.mm + all_gather_into_tensor."""
    M_local, N, K = state["M_local"], state["N"], state["K"]
    dtype = state["dtype"]
    world_size = ctx.get_num_ranks()
    M = M_local * world_size

    A, B = _make_inputs(ctx, M_local, N, K, dtype)
    C_local = ctx.zeros((M_local, N), dtype=dtype)
    C = ctx.zeros((M, N), dtype=dtype)

    state.set_flops(2 * M_local * N * K)
    state.set_bytes((world_size - 1) * M_local * N * A.element_size())

    state.exec(
        lambda: (
            torch.mm(A, B, out=C_local),
            dist.all_gather_into_tensor(C, C_local),
        ),
    )


@bench.register
@bench.axis("num_ranks", [2, 4, 8])
@bench.axis("M_local", [1024, 4096, 16384])
@bench.axis("N", [3584])
@bench.axis("K", [8192])
@bench.axis("dtype", [torch.float16])
def tritonblas_matmul_all_gather(state, ctx):
    """tritonBLAS GEMM + RCCL all_gather_into_tensor baseline (same GEMM as the iris ops)."""
    M_local, N, K = state["M_local"], state["N"], state["K"]
    dtype = state["dtype"]
    world_size = ctx.get_num_ranks()
    M = M_local * world_size

    A, B = _make_inputs(ctx, M_local, N, K, dtype)
    C_local = ctx.zeros((M_local, N), dtype=dtype)
    C = ctx.zeros((M, N), dtype=dtype)
    selector = tritonblas.OrigamiMatmulSelector(M_local, N, K, A.dtype, B.dtype, C_local.dtype, A.device)

    state.set_flops(2 * M_local * N * K)
    state.set_bytes((world_size - 1) * M_local * N * A.element_size())

    state.exec(
        lambda: (
            persistent_matmul_lt(A, B, C_local, selector, config=None, work_stealing=False),
            dist.all_gather_into_tensor(C, C_local),
        ),
    )


if __name__ == "__main__":
    bench.main()
