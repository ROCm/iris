# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.

"""
All-gather on the SDMA copy engines: ``all_gather(config=Config(engine="sdma"))``.
"""

import gc
import importlib.util

import pytest
import torch
import torch.distributed as dist

import iris
import iris.ccl.sdma.all_gather as sdma_all_gather
from iris.ccl import Config

pytestmark = pytest.mark.skipif(
    importlib.util.find_spec("xio") is None,
    reason="rocm-xio not installed (pip install iris[sdma])",
)

EMPTY = -1


@pytest.fixture(scope="module")
def ctx():
    if not dist.is_initialized():
        pytest.skip("torch.distributed not initialized")
    ctx = iris.iris(1 << 30, engines=("sdma",))
    yield ctx
    ctx.barrier()
    del ctx
    gc.collect()


def _gather_and_check(ctx, M, N, dtype, calls=3, async_op=False, config=None):
    """Run ``calls`` all-gathers with fresh input and a wiped output each time, against RCCL."""
    rank = ctx.get_rank()
    world_size = ctx.get_num_ranks()
    inp = ctx.zeros((M, N), dtype=dtype)
    out = ctx.zeros((world_size * M, N), dtype=dtype)
    ref = torch.empty_like(out)
    for call in range(calls):
        inp.copy_(torch.randn(M, N, device=inp.device).to(dtype) + rank * 100 + call)
        out.fill_(EMPTY)
        dist.all_gather_into_tensor(ref, inp)
        ctx.barrier()
        ctx.ccl.all_gather(out, inp, async_op=async_op, config=config or Config(engine="sdma"))
        torch.cuda.synchronize()
        assert torch.equal(out, ref), f"rank {rank}, call {call}: output differs from all_gather_into_tensor"
    ctx.barrier()


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16, torch.float32, torch.int8])
@pytest.mark.parametrize(
    "M, N",
    [
        (1, 1),
        (128, 64),
        (1000, 333),  # Odd byte counts
        (4096, 1024),
    ],
)
def test_all_gather_sdma(ctx, dtype, M, N):
    _gather_and_check(ctx, M, N, dtype)


def test_all_gather_sdma_async(ctx):
    # async_op skips the trailing barrier: completion is stream-ordered
    _gather_and_check(ctx, 256, 128, torch.float32, calls=5, async_op=True)


def test_all_gather_sdma_after_cu(ctx):
    # Interleaving with the compute-unit all-gather does not disturb the arrival counters
    for engine in ["cu", "sdma", "cu", "sdma"]:
        _gather_and_check(ctx, 64, 64, torch.float32, calls=1, config=Config(engine=engine))


def test_all_gather_sdma_in_place(ctx):
    rank = ctx.get_rank()
    world_size = ctx.get_num_ranks()
    M, N = 300, 70
    out = ctx.zeros((world_size * M, N), dtype=torch.float32)
    for call in range(3):
        out.fill_(EMPTY)
        mine = out[rank * M : (rank + 1) * M]
        mine.fill_(rank * 10 + call)
        ctx.ccl.all_gather(out, mine, config=Config(engine="sdma"))
        for r in range(world_size):
            assert (out[r * M : (r + 1) * M] == r * 10 + call).all(), f"rank {rank}: slice {r} wrong"
    ctx.barrier()


def test_all_gather_sdma_split_copies(ctx, monkeypatch):
    # Slices above the per-copy limit go as several copies: 4 KiB copies, one partial
    monkeypatch.setattr(sdma_all_gather, "MAX_COPY_BYTES", 4096)
    _gather_and_check(ctx, 33, 1000, torch.float32)


def test_engine_validation(ctx):
    with pytest.raises(ValueError, match="engine"):
        Config(engine="dma")
    world_size = ctx.get_num_ranks()
    t = ctx.zeros((64, 64 * world_size), dtype=torch.float32)
    out = ctx.zeros((64, 64 * world_size), dtype=torch.float32)
    config = Config(engine="sdma")
    for collective in [ctx.ccl.all_reduce, ctx.ccl.reduce_scatter, ctx.ccl.all_to_all]:
        with pytest.raises(ValueError, match="only all_gather"):
            collective(out, t, config=config)
    with pytest.raises(ValueError, match="contiguous"):
        ctx.ccl.all_gather(ctx.zeros((world_size * 64, 64), dtype=torch.float32), t[:, :64].t(), config=config)
    ctx.barrier()
