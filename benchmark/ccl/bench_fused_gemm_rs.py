#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
# Copyright (c) 2025 Advanced Micro Devices, Inc. All rights reserved.

"""Benchmark for fused GEMM + reduce-scatter: iris vs unfused torch.matmul + dist.reduce_scatter_tensor."""

import torch
import iris.bench as bench


@bench.register
@bench.axis("num_ranks", [2, 4, 8])
@bench.axis("tokens", [1, 32, 128, 512, 2048])
@bench.axis("H", [4096, 8192])
@bench.axis("K", [4096, 8192])
@bench.axis("dtype", [torch.bfloat16])
def fused_gemm_rs(state, ctx):
    tokens = state["tokens"]
    H, K, dtype = state["H"], state["K"], state["dtype"]
    world_size = ctx.get_num_ranks()
    rank = ctx.get_rank()

    H_shard = H // world_size
    shard_size = K // world_size

    inp = ctx.zeros((tokens, H_shard), dtype=dtype)
    inp.fill_(float(rank + 1) * 0.01)
    weight = torch.randn(H_shard, K, dtype=dtype, device=f"cuda:{rank}")

    # Bytes: input read + weight read + output write (per-rank)
    # GEMM: tokens * H_shard * K * 2 FLOPs
    # Comm: tokens * shard_size * element_size * (W-1)/W bytes moved
    state.set_bytes(int(tokens * shard_size * inp.element_size()))

    workspace = ctx.ccl.gemm_reduce_scatter_preamble(inp, weight)
    ctx.barrier()

    state.exec(
        lambda: ctx.ccl.gemm_reduce_scatter(inp, weight, workspace=workspace),
    )


@bench.register
@bench.axis("num_ranks", [2, 4, 8])
@bench.axis("tokens", [1, 32, 128, 512, 2048])
@bench.axis("H", [4096, 8192])
@bench.axis("K", [4096, 8192])
@bench.axis("dtype", [torch.bfloat16])
def unfused_gemm_rs_rccl(state, ctx):
    """Baseline: torch.matmul + RCCL all_reduce + column slice."""
    import torch.distributed as dist

    tokens = state["tokens"]
    H, K, dtype = state["H"], state["K"], state["dtype"]
    world_size = ctx.get_num_ranks()
    rank = ctx.get_rank()

    H_shard = H // world_size
    shard_size = K // world_size

    inp = torch.randn(tokens, H_shard, dtype=dtype, device=f"cuda:{rank}")
    weight = torch.randn(H_shard, K, dtype=dtype, device=f"cuda:{rank}")
    partial = torch.empty(tokens, K, dtype=dtype, device=f"cuda:{rank}")

    state.set_bytes(int(tokens * shard_size * inp.element_size()))

    def run():
        torch.matmul(inp, weight, out=partial)
        dist.all_reduce(partial, op=dist.ReduceOp.SUM)
        # Column slice: rank gets columns rank*shard_size : (rank+1)*shard_size
        _ = partial[:, rank * shard_size : (rank + 1) * shard_size].contiguous()

    state.exec(run)


if __name__ == "__main__":
    bench.main()
