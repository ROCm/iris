#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.

"""Benchmarks for GEMM + all-gather and related baselines."""

import torch
import torch.distributed as dist
import tritonblas
import iris.bench as bench

from iris.ops.matmul_all_gather import matmul_all_gather_preamble
from tritonblas.matmul import persistent_matmul_lt


def _register_fused_matmul_all_gather(state, ctx) -> None:
    M_local, N, K = state["M_local"], state["N"], state["K"]
    dtype = state["dtype"]
    world_size = ctx.get_num_ranks()
    rank = ctx.get_rank()
    M = M_local * world_size

    torch.manual_seed(123 + rank)
    A = ctx.randn((M_local, K), dtype=dtype)
    torch.manual_seed(456)
    B = ctx.randn((K, N), dtype=dtype)
    C = ctx.zeros((M, N), dtype=dtype)
    workspace = matmul_all_gather_preamble(ctx, A, B, out_dtype=C.dtype)
    launch = workspace.launch_params

    state.set_flops(2 * M_local * N * K)
    state.set_bytes((world_size - 1) * M_local * N * A.element_size())
    state.add_counter("block_m", launch["block_size_m"])
    state.add_counter("block_n", launch["block_size_n"])
    state.add_counter("block_k", launch["block_size_k"])
    state.add_counter("group_size_m", launch["group_size_m"])
    state.add_counter("num_xcds", launch["num_xcds"])
    state.add_counter("chunk_size", launch["chunk_size"])
    state.add_counter("grid_size", launch["num_sms"])
    state.add_counter("total_tiles", launch["total_tiles"])

    state.exec(
        lambda: ctx.ops.matmul_all_gather(C, A, B, workspace=workspace),
    )


def _register_pytorch_matmul_all_gather(state, ctx) -> None:
    M_local, N, K = state["M_local"], state["N"], state["K"]
    dtype = state["dtype"]
    world_size = ctx.get_num_ranks()
    rank = ctx.get_rank()
    M = M_local * world_size

    torch.manual_seed(123 + rank)
    A = ctx.randn((M_local, K), dtype=dtype)
    torch.manual_seed(456)
    B = ctx.randn((K, N), dtype=dtype)
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


def _register_tritonblas_matmul_all_gather(state, ctx) -> None:
    M_local, N, K = state["M_local"], state["N"], state["K"]
    dtype = state["dtype"]
    world_size = ctx.get_num_ranks()
    rank = ctx.get_rank()
    M = M_local * world_size

    torch.manual_seed(123 + rank)
    A = ctx.randn((M_local, K), dtype=dtype)

    torch.manual_seed(456)
    B = ctx.randn((K, N), dtype=dtype)

    C_local = ctx.zeros((M_local, N), dtype=dtype)
    C = ctx.zeros((M, N), dtype=dtype)
    selector = tritonblas.OrigamiMatmulSelector(
        M_local,
        N,
        K,
        A.dtype,
        B.dtype,
        C_local.dtype,
        A.device,
    )

    state.set_flops(2 * M_local * N * K)
    state.set_bytes((world_size - 1) * M_local * N * A.element_size())

    state.exec(
        lambda: (
            persistent_matmul_lt(A, B, C_local, selector, config=None, work_stealing=False),
            dist.all_gather_into_tensor(C, C_local),
        ),
    )


@bench.register
@bench.axis("num_ranks", [2, 4, 8])
@bench.axis("M_local", [1024, 4096, 16384])
@bench.axis("N", [3584])
@bench.axis("K", [8192])
@bench.axis("dtype", [torch.float16])
def pytorch_matmul_all_gather(state, ctx):
    _register_pytorch_matmul_all_gather(state, ctx)


@bench.register
@bench.axis("num_ranks", [2, 4, 8])
@bench.axis("M_local", [1024, 4096, 16384])
@bench.axis("N", [3584])
@bench.axis("K", [8192])
@bench.axis("dtype", [torch.float16])
def tritonblas_matmul_all_gather(state, ctx):
    _register_tritonblas_matmul_all_gather(state, ctx)


@bench.register
@bench.axis("num_ranks", [2, 4, 8])
@bench.axis("M_local", [1024, 4096, 16384])
@bench.axis("N", [3584])
@bench.axis("K", [8192])
@bench.axis("dtype", [torch.float16])
def matmul_all_gather(state, ctx):
    _register_fused_matmul_all_gather(state, ctx)


if __name__ == "__main__":
    bench.main()
