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
import tritonblas
import iris.bench as bench
from iris.ops.all_gather_matmul_hbm_buffer import (
    all_gather_matmul_hbm_buffer as _hbm_buffer,
    all_gather_matmul_hbm_buffer_preamble,
)
from iris.ops.all_gather_matmul import all_gather_matmul_preamble

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "all_gather_matmul"))
from auto_config import select_ag_mm_config


@bench.register
@bench.axis("num_ranks", [2, 4, 8])
@bench.axis("M", [1024, 4096, 16384])
@bench.axis("N", [3584])
@bench.axis("K", [8192])
@bench.axis("dtype", [torch.float16])
def all_gather_matmul(state, ctx):
    """Iris fused all-gather + GEMM baseline.

    Tensor Parallelism pattern (Megatron-LM row-parallel): shard K (reduction dimension).
    Used for: MLP down-projection, attention output projection.
    """
    M, N, K = state["M"], state["N"], state["K"]
    dtype = state["dtype"]
    world_size = ctx.get_num_ranks()
    rank = ctx.get_rank()
    K_local = K // world_size

    torch.manual_seed(123 + rank)
    A_sharded = ctx.randn((M, K_local), dtype=dtype)  # Shard along K
    torch.manual_seed(456)
    B = ctx.randn((K, N), dtype=dtype)
    C = ctx.zeros((M, N), dtype=dtype)
    workspace = all_gather_matmul_preamble(ctx, A_sharded, B, out_dtype=C.dtype)
    launch = workspace.launch_params

    state.set_flops(2 * M * N * K)
    state.set_bytes((world_size - 1) * M * K_local * A_sharded.element_size())
    state.add_counter("block_m", launch["block_size_m"])
    state.add_counter("block_n", launch["block_size_n"])
    state.add_counter("block_k", launch["block_size_k"])
    state.add_counter("group_size_m", launch["group_size_m"])
    state.add_counter("num_xcds", launch["num_xcds"])
    state.add_counter("chunk_size", launch["chunk_size"])
    state.add_counter("grid_size", launch["num_sms"])
    state.add_counter("total_tiles", launch["total_tiles"])

    state.exec(
        lambda: ctx.ops.all_gather_matmul(C, A_sharded, B, workspace=workspace),
    )


@bench.register
@bench.axis("num_ranks", [2, 4, 8])
@bench.axis("M", [1024, 4096, 16384])
@bench.axis("N", [3584])
@bench.axis("K", [8192])
@bench.axis("dtype", [torch.float16])
def rccl_all_gather_matmul(state, ctx):
    """PyTorch/RCCL baseline: all_gather + torch.cat + torch.mm.

    Tensor Parallelism pattern (Megatron-LM row-parallel): shard K (reduction dimension).
    """
    M, N, K = state["M"], state["N"], state["K"]
    dtype = state["dtype"]
    world_size = dist.get_world_size()
    rank = ctx.get_rank()
    K_local = K // world_size

    torch.manual_seed(123 + rank)
    A_sharded = ctx.randn((M, K_local), dtype=dtype)  # Shard along K
    torch.manual_seed(456)
    B = ctx.randn((K, N), dtype=dtype)
    A_gathered_parts = [ctx.zeros((M, K_local), dtype=dtype) for _ in range(world_size)]
    A_gathered = ctx.zeros((M, K), dtype=dtype)
    C = ctx.zeros((M, N), dtype=dtype)

    state.set_flops(2 * M * N * K)
    state.set_bytes((world_size - 1) * M * K_local * A_sharded.element_size())

    state.exec(
        lambda: (
            dist.all_gather(A_gathered_parts, A_sharded),
            A_gathered.copy_(torch.cat(A_gathered_parts, dim=1)),  # Concat along K (dim=1)
            torch.mm(A_gathered, B, out=C),
        ),
    )


@bench.register
@bench.axis("num_ranks", [2, 4, 8])
@bench.axis("M", [1024, 4096, 16384])
@bench.axis("N", [3584])
@bench.axis("K", [8192])
@bench.axis("dtype", [torch.float16])
def tritonblas_rccl_all_gather_matmul(state, ctx):
    """RCCL all_gather + tritonBLAS matmul baseline.

    Tensor Parallelism pattern (Megatron-LM row-parallel): shard K (reduction dimension).
    """
    M, N, K = state["M"], state["N"], state["K"]
    dtype = state["dtype"]
    world_size = dist.get_world_size()
    rank = ctx.get_rank()
    K_local = K // world_size

    torch.manual_seed(123 + rank)
    A_sharded = ctx.randn((M, K_local), dtype=dtype)  # Shard along K
    torch.manual_seed(456)
    B = ctx.randn((K, N), dtype=dtype)
    A_gathered_parts = [ctx.zeros((M, K_local), dtype=dtype) for _ in range(world_size)]
    A_gathered = ctx.zeros((M, K), dtype=dtype)
    C = ctx.zeros((M, N), dtype=dtype)
    selector = tritonblas.OrigamiMatmulSelector(
        M,
        N,
        K,
        A_gathered.dtype,
        B.dtype,
        C.dtype,
        A_gathered.device,
    )
    config = tritonblas.matmul_preamble(selector)

    state.set_flops(2 * M * N * K)
    state.set_bytes((world_size - 1) * M * K_local * A_sharded.element_size())

    state.exec(
        lambda: (
            dist.all_gather(A_gathered_parts, A_sharded),
            A_gathered.copy_(torch.cat(A_gathered_parts, dim=1)),  # Concat along K (dim=1)
            tritonblas.matmul_lt(A_gathered, B, C, selector, config),
        ),
    )


@bench.register
@bench.axis("num_ranks", [2, 4, 8])
@bench.axis("M", [1024, 4096, 16384])
@bench.axis("N", [3584])
@bench.axis("K", [8192])
@bench.axis("dtype", [torch.float16])
def all_gather_matmul_hbm_buffer(state, ctx):
    """Iris HBM-buffer AG+MM with auto-tuned config from configs/ JSON files.

    Tensor Parallelism pattern (Megatron-LM row-parallel): shard K (reduction dimension).
    """
    M, N, K = state["M"], state["N"], state["K"]
    dtype = state["dtype"]
    world_size = ctx.get_num_ranks()
    rank = ctx.get_rank()
    K_local = K // world_size

    result = select_ag_mm_config(M, N, K, world_size=world_size)
    config = result.to_fused_config()
    hbm = result.hbm_buffer_params

    torch.manual_seed(123 + rank)
    A_sharded = ctx.randn((M, K_local), dtype=dtype)  # Shard along K
    torch.manual_seed(456)
    B = ctx.randn((K, N), dtype=dtype)
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

    flag_iter = [0]  # Mutable counter for flag_iteration

    def _run():
        _hbm_buffer(
            ctx,
            C,
            A_sharded,
            B,
            config=config,
            workspace=workspace,
            flag_iteration=flag_iter[0],
            num_fetch_sms=hbm.get("num_fetch_sms", 16),
            k_per_flag=hbm.get("k_per_flag", 8),
            fetch_block_m=hbm.get("fetch_block_m"),
            fetch_block_k=hbm.get("fetch_block_k"),
            num_warps=hbm.get("num_warps", 8),
            num_stages=hbm.get("num_stages", 2),
            num_fetch_stages=hbm.get("num_fetch_stages"),
            first_stage_fetch_sms=hbm.get("first_stage_fetch_sms"),
        )
        flag_iter[0] += 1

    state.exec(_run)


if __name__ == "__main__":
    bench.main()
