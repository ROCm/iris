# SPDX-License-Identifier: MIT
# Copyright (c) 2025-2026 Advanced Micro Devices, Inc. All rights reserved.

"""
Test suite for matmul_reduce_scatter: rank r gets rows [r * M / W, (r + 1) * M / W) of the sum of
A @ B over the ranks, with the partials moved by compute units or by the SDMA copy engines.
"""

import gc
import importlib.util

import pytest
import torch
import torch.distributed as dist

import iris
import iris.ops as ops

HAS_SDMA = importlib.util.find_spec("xio") is not None
ENGINES = ["cu", pytest.param("sdma", marks=pytest.mark.skipif(not HAS_SDMA, reason="rocm-xio not installed"))]


@pytest.fixture(scope="module")
def ctx():
    if not dist.is_initialized():
        pytest.skip("torch.distributed not initialized")
    ctx = iris.iris(1 << 30, engines=("sdma",) if HAS_SDMA else ())
    yield ctx
    ctx.barrier()
    del ctx
    gc.collect()


def _run_and_check(ctx, M, N, K, dtype, config, calls=3, namespace=False, async_op=False):
    """Run ``calls`` times on one workspace with fresh inputs, against reduce_scatter(A @ B) in fp32."""
    rank, world_size = ctx.get_rank(), ctx.get_num_ranks()
    rows = slice(rank * M // world_size, (rank + 1) * M // world_size)
    C = torch.empty(M // world_size, N, dtype=dtype, device=ctx.get_device())
    workspace = None
    for call in range(calls):
        A = (torch.randn(M, K, device=C.device) / 4).to(dtype)
        B = (torch.randn(K, N, device=C.device) / 4).to(dtype)
        ref = A.float() @ B.float()
        dist.all_reduce(ref)
        C.fill_(float("nan"))
        if namespace:
            workspace = ctx.ops.matmul_reduce_scatter(C, A, B, async_op=async_op, config=config, workspace=workspace)
        else:
            workspace = ops.matmul_reduce_scatter(ctx, C, A, B, async_op=async_op, config=config, workspace=workspace)
        torch.cuda.synchronize()
        if async_op:
            ctx.barrier()
        torch.testing.assert_close(
            C.float(), ref[rows], atol=2e-2 * float(ref.abs().max()), rtol=0, msg=lambda m: f"call {call}: {m}"
        )
    ctx.barrier()


@pytest.mark.parametrize("engine", ENGINES)
@pytest.mark.parametrize(
    "dtype, block",
    [
        (torch.float16, (64, 64, 32)),
        (torch.bfloat16, (256, 256, 64)),
        (torch.float32, (128, 128, 32)),
    ],
)
@pytest.mark.parametrize(
    "rows_per_rank, N, K",
    [
        ("2 tiles", 512, 256),  # Each rank's rows are whole tile rows
        ("tile + 37", 200, 100),  # Rows straddle tile rows; N and K not multiples of the blocks
    ],
)
def test_matmul_reduce_scatter(ctx, engine, dtype, block, rows_per_rank, N, K):
    block_m, block_n, block_k = block
    rows = 2 * block_m if rows_per_rank == "2 tiles" else block_m + 37
    config = ops.FusedConfig(engine=engine, block_size_m=block_m, block_size_n=block_n, block_size_k=block_k)
    _run_and_check(ctx, ctx.get_num_ranks() * rows, N, K, dtype, config)


@pytest.mark.parametrize("engine", ENGINES)
def test_matmul_reduce_scatter_namespace_async(ctx, engine):
    """shmem.ops forwards to iris.ops.matmul_reduce_scatter; async_op leaves synchronization to the caller."""
    config = ops.FusedConfig(engine=engine, block_size_m=128, block_size_n=128, block_size_k=64)
    _run_and_check(ctx, ctx.get_num_ranks() * 256, 384, 128, torch.bfloat16, config, namespace=True, async_op=True)


def test_matmul_reduce_scatter_validation(ctx):
    world_size = ctx.get_num_ranks()
    with pytest.raises(ValueError, match="engine"):
        ops.FusedConfig(engine="dma").validate()

    A = torch.zeros(64 * world_size, 32, dtype=torch.float16, device=ctx.get_device())
    B = torch.zeros(32, 64, dtype=torch.float16, device=ctx.get_device())
    with pytest.raises(ValueError, match="expected"):
        ops.matmul_reduce_scatter(ctx, torch.zeros(65, 64, dtype=torch.float16, device=A.device), A, B)
    if world_size > 1:
        with pytest.raises(ValueError, match="multiple of the number of ranks"):
            ops.matmul_reduce_scatter(ctx, torch.zeros(1, 64, dtype=torch.float16, device=A.device), A[:-1], B)

    config = ops.FusedConfig(engine="sdma")
    C = torch.zeros(64, 64, dtype=torch.float16, device=A.device)
    for op in [ops.matmul_all_reduce, ops.matmul_all_gather]:
        with pytest.raises(ValueError, match="only matmul_reduce_scatter"):
            op(ctx, C, A[:64], B, config=config)
    ctx.barrier()
