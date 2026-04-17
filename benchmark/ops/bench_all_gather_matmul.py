#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.

"""Benchmark for all-gather + GEMM: RCCL baseline vs iris HBM-buffer prefetch.

The HBM-buffer benchmark automatically loads tuned kernel parameters from
configs/{arch}/{transpose}/ws{N}.json when available. Run with --list-configs
to see which shapes have tuned configs for the current GPU.
"""

import sys
import os

import torch
import torch.distributed as dist
import iris.bench as bench
from iris.ops.all_gather_matmul_hbm_buffer import (
    all_gather_matmul_hbm_buffer as _hbm_buffer,
    all_gather_matmul_hbm_buffer_preamble,
)

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "all_gather_matmul"))
from auto_config import select_ag_mm_config


@bench.register
@bench.axis("num_ranks", [2, 4, 8])
@bench.axis("M", [1024, 4096, 16384])
@bench.axis("N", [3584])
@bench.axis("K", [8192])
@bench.axis("dtype", [torch.float16])
def rccl_all_gather_matmul(state, ctx):
    """PyTorch/RCCL baseline: all_gather_into_tensor + torch.mm"""
    M, N, K = state["M"], state["N"], state["K"]
    dtype = state["dtype"]
    world_size = dist.get_world_size()
    K_local = K // world_size

    A_sharded = torch.ones((M, K_local), device="cuda", dtype=dtype)
    B = torch.randn((K, N), device="cuda", dtype=dtype)
    A_gathered = torch.empty((M, K), device="cuda", dtype=dtype)
    C = torch.empty((M, N), device="cuda", dtype=dtype)

    state.set_flops(2 * M * N * K)
    state.set_bytes((world_size - 1) * M * K_local * A_sharded.element_size())

    state.exec(
        lambda: (
            dist.all_gather_into_tensor(A_gathered, A_sharded),
            torch.mm(A_gathered, B, out=C),
        ),
    )


@bench.register
@bench.axis("num_ranks", [2, 4, 8])
@bench.axis("M", [1024, 4096, 16384])
@bench.axis("N", [3584])
@bench.axis("K", [8192])
@bench.axis("dtype", [torch.float16])
def all_gather_matmul_hbm_buffer(state, ctx):
    """Iris HBM-buffer AG+MM with auto-tuned config from configs/ JSON files."""
    M, N, K = state["M"], state["N"], state["K"]
    dtype = state["dtype"]
    world_size = ctx.get_num_ranks()
    K_local = K // world_size

    result = select_ag_mm_config(M, N, K, world_size=world_size)
    config = result.to_fused_config()
    hbm = result.hbm_buffer_params

    A_sharded = ctx.zeros((M, K_local), dtype=dtype)
    A_sharded.fill_(1.0)
    B = torch.randn((K, N), device="cuda", dtype=dtype)
    C = ctx.zeros((M, N), dtype=dtype)

    workspace = all_gather_matmul_hbm_buffer_preamble(
        ctx,
        A_sharded,
        B,
        config,
        k_per_flag=hbm.get("k_per_flag", 8),
    )

    state.set_flops(2 * M * N * K)
    state.set_bytes((world_size - 1) * M * K_local * A_sharded.element_size())

    state.exec(
        lambda: _hbm_buffer(
            ctx,
            C,
            A_sharded,
            B,
            config=config,
            workspace=workspace,
            num_fetch_sms=hbm.get("num_fetch_sms", 16),
            k_per_flag=hbm.get("k_per_flag", 8),
            fetch_block_m=hbm.get("fetch_block_m"),
            fetch_block_k=hbm.get("fetch_block_k"),
            num_warps=hbm.get("num_warps", 8),
            num_stages=hbm.get("num_stages", 2),
            num_fetch_stages=hbm.get("num_fetch_stages"),
            first_stage_fetch_sms=hbm.get("first_stage_fetch_sms"),
        ),
        preamble_fn=lambda: (C.zero_(), workspace.locks.zero_()),
    )


if __name__ == "__main__":
    bench.main()
