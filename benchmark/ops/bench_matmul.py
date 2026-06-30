#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.

"""Benchmarks for standalone GEMM."""

import torch
import iris.bench as bench

from iris.ops.matmul import matmul as _matmul
from iris.ops.matmul import matmul_preamble as _matmul_preamble


def _register_local_matmul(state, ctx, *, m_key: str, pytorch: bool, work_stealing: bool = False, enable_streamk: bool = False) -> None:
    M, N, K = state[m_key], state["N"], state["K"]
    dtype = state["dtype"]
    rank = ctx.get_rank()

    state.set_flops(2 * M * N * K)
    state.set_bytes(((M * K) + (K * N) + (M * N)) * torch.tensor([], dtype=dtype).element_size())

    torch.manual_seed(123 + rank)
    A_data = torch.randn((M, K), device="cuda", dtype=dtype)
    torch.manual_seed(456)
    B_data = torch.randn((K, N), device="cuda", dtype=dtype)

    if pytorch:
        C_torch = torch.empty((M, N), device="cuda", dtype=dtype)
        state.exec(lambda: torch.mm(A_data, B_data, out=C_torch))
    else:
        # Use iris heap allocation (same as tritonblas_rcclbaseline)
        torch.manual_seed(123 + rank)
        A = ctx.randn((M, K), dtype=dtype)
        torch.manual_seed(456)
        B = ctx.randn((K, N), dtype=dtype)
        C = ctx.zeros((M, N), dtype=dtype)

        workspace = _matmul_preamble(ctx, A, B)
        # Using async_op=True to match torch
        state.exec(
            lambda: _matmul(ctx, C, A, B, workspace=workspace, async_op=True, work_stealing=work_stealing, enable_streamk=enable_streamk),
        )


@bench.register
@bench.axis("num_ranks", [8])
@bench.axis("M_local", [1024, 4096, 16384])
@bench.axis("N", [3584])
@bench.axis("K", [8192])
@bench.axis("dtype", [torch.float16])
def matmul_only_local(state, ctx):
    _register_local_matmul(state, ctx, m_key="M_local", pytorch=False)


@bench.register
@bench.axis("num_ranks", [8])
@bench.axis("M_local", [1024, 4096, 16384])
@bench.axis("N", [3584])
@bench.axis("K", [8192])
@bench.axis("dtype", [torch.float16])
def pytorch_matmul_only_local(state, ctx):
    _register_local_matmul(state, ctx, m_key="M_local", pytorch=True)


@bench.register
@bench.axis("num_ranks", [8])
@bench.axis("M", [1024, 4096, 16384])
@bench.axis("N", [3584])
@bench.axis("K", [8192])
@bench.axis("dtype", [torch.float16])
def matmul_only(state, ctx):
    _register_local_matmul(state, ctx, m_key="M", pytorch=False, work_stealing=False)


@bench.register
@bench.axis("num_ranks", [8])
@bench.axis("M", [1024, 4096, 16384])
@bench.axis("N", [3584])
@bench.axis("K", [8192])
@bench.axis("dtype", [torch.float16])
def matmul_work_stealing(state, ctx):
    _register_local_matmul(state, ctx, m_key="M", pytorch=False, work_stealing=True)


@bench.register
@bench.axis("num_ranks", [8])
@bench.axis("M", [1024, 4096, 16384])
@bench.axis("N", [3584])
@bench.axis("K", [8192])
@bench.axis("dtype", [torch.float16])
def matmul_streamk(state, ctx):
    _register_local_matmul(state, ctx, m_key="M", pytorch=False, enable_streamk=True)


@bench.register
@bench.axis("num_ranks", [8])
@bench.axis("M", [1024, 4096, 16384])
@bench.axis("N", [3584])
@bench.axis("K", [8192])
@bench.axis("dtype", [torch.float16])
def pytorch_matmul_only(state, ctx):
    _register_local_matmul(state, ctx, m_key="M", pytorch=True)


if __name__ == "__main__":
    bench.main()
