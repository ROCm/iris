#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.

"""Benchmarks for fused GEMM + all-gather copy-engine variants."""

import torch
import iris.bench as bench

from iris.ops import FusedConfig
from iris.ops.matmul_all_gather_copy_engine import (
    matmul_all_gather_copy_engine as _device_copy_engine,
    matmul_all_gather_copy_engine_preamble as _device_preamble,
)
from iris.ops.matmul_all_gather_host_copy_engine import (
    matmul_all_gather_host_copy_engine as _host_copy_engine,
    matmul_all_gather_host_copy_engine_preamble as _host_preamble,
)
from tritonblas.matmul import _make_matmul_selector


def _selector_and_config(M_local: int, N: int, K: int, dtype: torch.dtype, device: torch.device) -> tuple:
    selector = _make_matmul_selector(
        M_local,
        N,
        K,
        dtype,
        dtype,
        dtype,
        device,
        streamk=False,
    )
    config = FusedConfig(
        block_size_m=selector.block_m,
        block_size_n=selector.block_n,
        block_size_k=selector.block_k,
        group_size_m=selector.group_m,
        num_xcds=max(1, int(getattr(selector, "num_sms", 1))),
    )
    return selector, config


def _m_tiles_per_batch(selector, M_local: int, N: int) -> int:
    num_tiles_m = (M_local + selector.block_m - 1) // selector.block_m
    num_tiles_n = (N + selector.block_n - 1) // selector.block_n
    active_cus = getattr(selector, "_ACTIVE_CU", None)
    if active_cus is None or active_cus <= 0:
        active_cus = torch.cuda.get_device_properties(torch.cuda.current_device()).multi_processor_count
    tiles_per_group = max(1, selector.group_m * num_tiles_n)
    groups_per_wave = max(1, int(active_cus) // tiles_per_group)
    return max(1, min(num_tiles_m, groups_per_wave * selector.group_m))


def _register_copy_engine(state, ctx, *, device_initiated: bool) -> None:
    M_local, N, K = state["M_local"], state["N"], state["K"]
    dtype = state["dtype"]
    rank = ctx.get_rank()
    world_size = ctx.get_num_ranks()
    M = M_local * world_size

    device = torch.device(f"cuda:{torch.cuda.current_device()}")
    selector, config = _selector_and_config(M_local, N, K, dtype, device)
    m_tiles_per_batch = _m_tiles_per_batch(selector, M_local, N)

    if M_local % config.block_size_m != 0:
        state.skip(f"M_local={M_local} must be divisible by block_size_m={config.block_size_m}")
    if K % config.block_size_k != 0:
        state.skip(f"K={K} must be divisible by block_size_k={config.block_size_k}")

    A = ctx.zeros((M_local, K), dtype=dtype)
    torch.manual_seed(123 + rank)
    A_data = torch.randn((M_local, K), device="cuda", dtype=dtype)
    A.copy_(A_data)
    torch.manual_seed(456)
    B = torch.randn((K, N), device="cuda", dtype=dtype)
    C = ctx.zeros((M, N), dtype=dtype)

    flag_iteration = [0]

    if device_initiated:
        workspace = _device_preamble(
            ctx,
            A,
            B,
            config,
            m_tiles_per_batch=m_tiles_per_batch,
        )
        workspace.selector = selector

        def _run():
            _device_copy_engine(
                ctx,
                C,
                A,
                B,
                config=config,
                async_op=False,
                workspace=workspace,
                use_copy_engine=True,
                flag_iteration=flag_iteration[0],
                m_tiles_per_batch=m_tiles_per_batch,
            )
            flag_iteration[0] += 1

    else:
        workspace = _host_preamble(
            ctx,
            A,
            B,
            config,
            m_tiles_per_batch=m_tiles_per_batch,
            trace=False,
            use_tritonblas=True,
        )

        def _run():
            _host_copy_engine(
                ctx,
                C,
                A,
                B,
                config=config,
                async_op=False,
                workspace=workspace,
                flag_iteration=flag_iteration[0],
                m_tiles_per_batch=m_tiles_per_batch,
                trace=False,
                use_tritonblas=True,
            )
            flag_iteration[0] += 1

    state.set_flops(2 * M_local * N * K)
    state.set_bytes((world_size - 1) * M_local * N * A.element_size())
    state.add_counter("group_size_m", float(config.group_size_m))
    state.add_counter("m_tiles_per_batch", float(m_tiles_per_batch))
    state.add_counter("device_initiated", 1.0 if device_initiated else 0.0)

    state.exec(_run, preamble_fn=lambda: C.zero_())


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
