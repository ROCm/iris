#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.

"""Benchmarks for fused GEMM + all-gather copy-engine variants."""

import torch
import iris.bench as bench

from iris.ops.matmul_all_gather_copy_engine import (
    matmul_all_gather_copy_engine as _device_copy_engine,
    matmul_all_gather_copy_engine_preamble as _device_preamble,
)
from iris.ops.matmul_all_gather_host_copy_engine import (
    matmul_all_gather_host_copy_engine as _host_copy_engine,
    matmul_all_gather_host_copy_engine_preamble as _host_preamble,
)
from tritonblas.matmul import _make_matmul_selector


def _make_selector(M_local: int, N: int, K: int, dtype: torch.dtype, device: torch.device):
    return _make_matmul_selector(
        M_local,
        N,
        K,
        dtype,
        dtype,
        dtype,
        device,
        streamk=False,
    )


def _register_copy_engine(state, ctx, *, device_initiated: bool) -> None:
    M_local, N, K = state["M_local"], state["N"], state["K"]
    dtype = state["dtype"]
    rank = ctx.get_rank()
    world_size = ctx.get_num_ranks()
    M = M_local * world_size

    device = torch.device(f"cuda:{torch.cuda.current_device()}")
    selector = _make_selector(M_local, N, K, dtype, device)

    if M_local % selector.block_m != 0:
        state.skip(f"M_local={M_local} must be divisible by block_size_m={selector.block_m}")
    if K % selector.block_k != 0:
        state.skip(f"K={K} must be divisible by block_size_k={selector.block_k}")

    torch.manual_seed(123 + rank)
    A = ctx.randn((M_local, K), dtype=dtype)
    torch.manual_seed(456)
    B = ctx.randn((K, N), dtype=dtype)
    C = ctx.zeros((M, N), dtype=dtype)

    flag_iteration = [0]

    if device_initiated:
        workspace = _device_preamble(
            ctx,
            A,
            B,
            selector=selector,
        )

        def _run():
            _device_copy_engine(
                ctx,
                C,
                A,
                B,
                async_op=False,
                workspace=workspace,
                flag_iteration=flag_iteration[0],
            )
            flag_iteration[0] += 1

    else:
        workspace = _host_preamble(
            ctx,
            A,
            B,
            trace=False,
            selector=selector,
        )

        def _run():
            _host_copy_engine(
                ctx,
                C,
                A,
                B,
                async_op=False,
                workspace=workspace,
                flag_iteration=flag_iteration[0],
                trace=False,
            )
            flag_iteration[0] += 1

    state.set_flops(2 * M_local * N * K)
    state.set_bytes((world_size - 1) * M_local * N * A.element_size())
    state.add_counter("group_size_m", float(selector.group_m))
    state.add_counter("m_tiles_per_batch", float(workspace.m_tiles_per_batch))
    state.add_counter("device_initiated", 1.0 if device_initiated else 0.0)

    state.exec(_run)


@bench.register
@bench.axis("num_ranks", [8])
@bench.axis("M_local", [1024, 4096, 16384])
@bench.axis("N", [3584])
@bench.axis("K", [8192])
@bench.axis("dtype", [torch.float16])
def matmul_all_gather_copy_engine_host(state, ctx):
    _register_copy_engine(state, ctx, device_initiated=False)


@bench.register
@bench.axis("num_ranks", [8])
@bench.axis("M_local", [1024, 4096, 16384])
@bench.axis("N", [3584])
@bench.axis("K", [8192])
@bench.axis("dtype", [torch.float16])
def matmul_all_gather_copy_engine_device(state, ctx):
    _register_copy_engine(state, ctx, device_initiated=True)


if __name__ == "__main__":
    bench.main()
