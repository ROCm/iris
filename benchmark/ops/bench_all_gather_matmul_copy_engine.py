#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.

"""Benchmarks for all-gather + GEMM copy-engine variants."""

import torch
import iris.bench as bench

from iris.ops import FusedConfig
from iris.ops.all_gather_matmul_copy_engine import (
    all_gather_matmul_copy_engine as _copy_engine,
    all_gather_matmul_copy_engine_preamble,
)
from tritonblas.matmul import _make_matmul_selector


def _selector_and_config(M: int, N: int, K: int, dtype: torch.dtype, device: torch.device) -> tuple:
    selector = _make_matmul_selector(
        M,
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


def _register_copy_engine(state, ctx, *, device_initiated: bool) -> None:
    M, N, K = state["M"], state["N"], state["K"]
    dtype = state["dtype"]
    rank = ctx.get_rank()
    world_size = ctx.get_num_ranks()
    if K % world_size != 0:
        state.skip(f"K={K} must be divisible by world_size={world_size}")

    K_local = K // world_size
    device = torch.device(f"cuda:{torch.cuda.current_device()}")
    selector, config = _selector_and_config(M, N, K, dtype, device)

    if M % config.block_size_m != 0:
        state.skip(f"M={M} must be divisible by block_size_m={config.block_size_m}")
    if K % config.block_size_k != 0:
        state.skip(f"K={K} must be divisible by block_size_k={config.block_size_k}")
    if K_local % config.block_size_k != 0:
        state.skip(f"K_local={K_local} must be divisible by block_size_k={config.block_size_k}")

    m_tiles_per_batch = config.group_size_m
    k_per_flag = 4

    A_sharded = ctx.zeros((M, K_local), dtype=dtype)
    torch.manual_seed(123 + rank)
    A_sharded_data = torch.randn((M, K_local), device="cuda", dtype=dtype)
    A_sharded.copy_(A_sharded_data)
    torch.manual_seed(456)
    B = torch.randn((K, N), device="cuda", dtype=dtype)
    C = ctx.zeros((M, N), dtype=dtype)

    workspace = all_gather_matmul_copy_engine_preamble(
        ctx,
        A_sharded,
        B,
        config,
        k_per_flag=k_per_flag,
        m_tiles_per_batch=m_tiles_per_batch,
    )
    workspace.selector = selector

    flag_iteration = [0]

    def _run():
        _copy_engine(
            ctx,
            C,
            A_sharded,
            B,
            config=config,
            async_op=False,
            workspace=workspace,
            flag_iteration=flag_iteration[0],
            k_per_flag=k_per_flag,
            m_tiles_per_batch=m_tiles_per_batch,
            device_initiated=device_initiated,
        )
        flag_iteration[0] += 1

    state.set_flops(2 * M * N * K)
    state.set_bytes((world_size - 1) * M * K_local * A_sharded.element_size())
    state.add_counter("group_size_m", float(config.group_size_m))
    state.add_counter("m_tiles_per_batch", float(m_tiles_per_batch))
    state.add_counter("device_initiated", 1.0 if device_initiated else 0.0)

    state.exec(_run, preamble_fn=lambda: C.zero_())


@bench.register
@bench.axis("num_ranks", [8])
@bench.axis("M", [1024, 4096, 16384])
@bench.axis("N", [3584])
@bench.axis("K", [8192])
@bench.axis("dtype", [torch.float16])
def all_gather_matmul_copy_engine_host(state, ctx):
    _register_copy_engine(state, ctx, device_initiated=False)


@bench.register
@bench.axis("num_ranks", [8])
@bench.axis("M", [1024, 4096, 16384])
@bench.axis("N", [3584])
@bench.axis("K", [8192])
@bench.axis("dtype", [torch.float16])
def all_gather_matmul_copy_engine_device(state, ctx):
    _register_copy_engine(state, ctx, device_initiated=True)


if __name__ == "__main__":
    bench.main()
