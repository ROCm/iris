#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.

"""Benchmarks for standalone GEMM."""

import torch
import iris.bench as bench

from iris.ops import FusedConfig
from iris.ops.matmul import matmul as _matmul
from iris.ops.matmul import matmul_preamble as _matmul_preamble


def _register_local_matmul(state, ctx, *, m_key: str, pytorch: bool) -> None:
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
        A = ctx.zeros((M, K), dtype=dtype)
        A.copy_(A_data)
        C = ctx.zeros((M, N), dtype=dtype)

        workspace = _matmul_preamble(ctx, A, B_data, FusedConfig())
        state.exec(
            lambda: _matmul(ctx, C, A, B_data, workspace=workspace),
            preamble_fn=lambda: C.zero_(),
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
    _register_local_matmul(state, ctx, m_key="M", pytorch=False)


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
