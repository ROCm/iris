# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.

import gc
import importlib.util

import pytest
import torch
import torch.distributed as dist
import triton
import triton.language as tl

import iris
from iris.experimental.plan import TriggeredView, tile_chunks, whole

pytestmark = pytest.mark.skipif(
    importlib.util.find_spec("xio") is None,
    reason="rocm-xio not installed (pip install iris[sdma])",
)

EMPTY = -1


@pytest.fixture(scope="module")
def ctx():
    if not dist.is_initialized():
        pytest.skip("torch.distributed not initialized")
    ctx = iris.iris(1 << 28, engines=("sdma",))
    if ctx.get_num_ranks() < 2:
        pytest.skip("needs at least two ranks")
    yield ctx
    ctx.barrier()
    del ctx
    gc.collect()


@triton.jit
def _drive(handle):
    view = TriggeredView.initialize(handle)
    c = tl.program_id(0)
    if view.owns(c):
        view.mark_ready(c)
    view.wait(c)


def _payload(round, part, rows, cols, dtype):
    idx = torch.arange(rows * cols, dtype=torch.int32, device="cuda") % 100
    return ((round * 16 + part + 1) * 100 + idx).to(dtype).view(rows, cols)


def _run(ctx, plan, tensor, parts, holds, lands, rounds=2):
    n = len(parts)
    rows, cols, dtype = parts[0].shape[0], tensor.shape[1], tensor.dtype
    tensor.fill_(EMPTY)
    with plan:
        for round in range(1, rounds + 1):
            for p in holds:
                parts[p].copy_(_payload(round, p, rows, cols, dtype))
            handle = plan.start()
            _drive[(plan.num_chunks,)](handle)
            plan.wait()
            for p in range(n):
                if p in lands or p in holds:
                    assert torch.equal(parts[p], _payload(round, p, rows, cols, dtype)), f"round {round}: part {p}"
                else:
                    assert (parts[p] == EMPTY).all(), f"round {round}: part {p} should be untouched"
            tensor.fill_(EMPTY)
            plan.next_round()


def _split(tensor, n):
    rows = tensor.shape[0] // n
    return [tensor[i * rows : (i + 1) * rows] for i in range(n)]


@pytest.mark.parametrize("root", [0, 1])
def test_prepare_scatter(ctx, root):
    rank, world = ctx.get_rank(), ctx.get_num_ranks()
    tensor = ctx.zeros((world * 15, 70), dtype=torch.int32)
    parts = _split(tensor, world)
    holds = range(world) if rank == root else []
    _run(ctx, ctx.ccl.prepare_scatter(tensor, root), tensor, parts, holds, [rank])


@pytest.mark.parametrize("root", [0, 1])
def test_prepare_gather(ctx, root):
    rank, world = ctx.get_rank(), ctx.get_num_ranks()
    tensor = ctx.zeros((world * 16, 64), dtype=torch.int32)
    parts = _split(tensor, world)
    lands = range(world) if rank == root else []
    _run(ctx, ctx.ccl.prepare_gather(tensor, root), tensor, parts, [rank], lands)


@pytest.mark.parametrize("root", [0, 1])
def test_prepare_broadcast(ctx, root):
    rank = ctx.get_rank()
    tensor = ctx.zeros((10, 40), dtype=torch.int32)
    _run(ctx, ctx.ccl.prepare_broadcast(tensor, root), tensor, [tensor], [0] if rank == root else [], [0])


def test_prepare_all_to_all(ctx):
    rank, world = ctx.get_rank(), ctx.get_num_ranks()
    tensor = ctx.zeros((world * world * 4, 24), dtype=torch.int32)
    blocks = _split(tensor, world * world)
    holds = [rank * world + j for j in range(world)]
    lands = [r * world + rank for r in range(world)]
    _run(ctx, ctx.ccl.prepare_all_to_all(tensor), tensor, blocks, holds, lands)


def test_prepare_scatter_tiles(ctx):
    rank, world = ctx.get_rank(), ctx.get_num_ranks()
    tensor = ctx.zeros((world * 15, 70), dtype=torch.int32)
    parts = _split(tensor, world)
    schedule = None
    if rank == 0:
        schedule = [c for r in range(1, world) for c in tile_chunks(parts[r], (4, 32), 1, base=tensor, peers=(r,))]
    holds = range(world) if rank == 0 else []
    _run(ctx, ctx.ccl.prepare_scatter(tensor, 0, schedule), tensor, parts, holds, [rank])


def test_rejects_bad_input(ctx):
    rank, world = ctx.get_rank(), ctx.get_num_ranks()
    tensor = ctx.zeros((world * 8, 16), dtype=torch.int32)
    with pytest.raises(ValueError, match="root must be a rank"):
        ctx.ccl.prepare_gather(tensor, world)
    partial = whole(_split(tensor, world)[1][:-1], base=tensor, peers=(0,)) if rank == 1 else None
    with pytest.raises(ValueError if rank == 1 else RuntimeError, match="leaves part" if rank == 1 else "could not"):
        ctx.ccl.prepare_gather(tensor, 0, partial)
    ctx.barrier()
