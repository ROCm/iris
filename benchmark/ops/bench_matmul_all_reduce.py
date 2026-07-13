#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.

"""Benchmark for fused GEMM + all-reduce (iris.ops)."""

import torch
import torch.distributed as dist
import tritonblas
import iris.bench as bench
from iris.ops import FusedConfig, matmul_all_reduce_preamble
from tritonblas.matmul import persistent_matmul_lt


_MAX_ONE_SHOT_INBOX_ELEMENTS = 2**32


@bench.register
@bench.axis("num_ranks", [2, 4, 8])
@bench.axis("M", [1024, 4096, 16384])
@bench.axis("N", [3584])
@bench.axis("K", [8192])
@bench.axis("dtype", [torch.float16])
def pytorch_matmul_all_reduce(state, ctx):
    """PyTorch/RCCL baseline: torch.mm + dist.all_reduce.

    Standard distributed training pattern: local matmul followed by all-reduce.
    """
    M, N, K = state["M"], state["N"], state["K"]
    dtype = state["dtype"]
    rank = ctx.get_rank()
    world_size = ctx.get_num_ranks()

    torch.manual_seed(123 + rank)
    A = ctx.randn((M, K), dtype=dtype)
    torch.manual_seed(456)
    B = ctx.randn((K, N), dtype=dtype)
    C = ctx.zeros((M, N), dtype=dtype)

    state.set_flops(2 * M * N * K)
    state.set_bytes((world_size - 1) * M * N * C.element_size())

    state.exec(
        lambda: (
            torch.mm(A, B, out=C),
            dist.all_reduce(C, op=dist.ReduceOp.SUM),
        ),
    )


@bench.register
@bench.axis("num_ranks", [2, 4, 8])
@bench.axis("M", [1024, 4096, 16384])
@bench.axis("N", [3584])
@bench.axis("K", [8192])
@bench.axis("dtype", [torch.float16])
def tritonblas_rccl_matmul_all_reduce(state, ctx):
    """TritonBLAS + RCCL baseline: tritonblas.matmul + dist.all_reduce.

    Optimized matmul with standard RCCL all-reduce.
    """
    M, N, K = state["M"], state["N"], state["K"]
    dtype = state["dtype"]
    rank = ctx.get_rank()
    world_size = ctx.get_num_ranks()

    torch.manual_seed(123 + rank)
    A = ctx.randn((M, K), dtype=dtype)
    torch.manual_seed(456)
    B = ctx.randn((K, N), dtype=dtype)
    C = ctx.zeros((M, N), dtype=dtype)

    selector = tritonblas.OrigamiMatmulSelector(
        M,
        N,
        K,
        A.dtype,
        B.dtype,
        C.dtype,
        A.device,
    )

    state.set_flops(2 * M * N * K)
    state.set_bytes((world_size - 1) * M * N * C.element_size())

    state.exec(
        lambda: (
            persistent_matmul_lt(A, B, C, selector, config=None, work_stealing=False),
            dist.all_reduce(C, op=dist.ReduceOp.SUM),
        ),
    )


@bench.register
@bench.axis("num_ranks", [2, 4, 8])
@bench.axis("M", [1024, 4096, 16384])
@bench.axis("N", [3584])
@bench.axis("K", [8192])
@bench.axis("dtype", [torch.float16])
@bench.axis("variant", ["one_shot", "two_shot"])
def matmul_all_reduce(state, ctx):
    """Fused GEMM + all-reduce with configurable variant."""
    M, N, K = state["M"], state["N"], state["K"]
    dtype = state["dtype"]
    variant = state["variant"]
    world_size = ctx.get_num_ranks()
    rank = ctx.get_rank()

    if variant == "one_shot" and world_size * M * N >= _MAX_ONE_SHOT_INBOX_ELEMENTS:
        state.skip(
            "one_shot requires world_size * M * N < 2**32 elements "
            f"(got {world_size * M * N})"
        )

    torch.manual_seed(123 + rank)
    A = ctx.randn((M, K), dtype=dtype)
    torch.manual_seed(456)
    B = ctx.randn((K, N), dtype=dtype)
    C = ctx.zeros((M, N), dtype=dtype)

    config = FusedConfig(all_reduce_variant=variant)
    workspace = matmul_all_reduce_preamble(ctx, C, A, B, config=config)
    launch = workspace.launch_params

    state.set_flops(2 * M * N * K)
    state.set_bytes((world_size - 1) * M * N * C.element_size())
    state.add_counter("uses_selector", int(workspace.selector is not None))
    state.add_counter("block_m", launch["block_size_m"])
    state.add_counter("block_n", launch["block_size_n"])
    state.add_counter("block_k", launch["block_size_k"])
    state.add_counter("group_size_m", launch["group_size_m"])
    state.add_counter("num_xcds", launch["num_xcds"])
    state.add_counter("chunk_size", launch["chunk_size"])
    state.add_counter("grid_size", launch["num_sms"])
    state.add_counter("total_tiles", launch["total_tiles"])
    state.add_counter("num_stages", launch["num_stages"] or 0)
    state.add_counter("selector_fallback", int(launch.get("selector_fallback", False)))
    state.add_counter("publish_tiles", launch.get("publish_tiles", launch["total_tiles"]))
    state.add_counter("publish_programs", launch.get("publish_programs", launch.get("publish_tiles", 0)))

    def _run():
        ctx.ops.matmul_all_reduce(C, A, B, config=config, workspace=workspace)

    state.exec(_run)


if __name__ == "__main__":
    bench.main()
