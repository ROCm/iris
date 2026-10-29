# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.

"""
All-gather on the SDMA copy engines: ``all_gather(config=Config(engine="sdma"))`` and the
triggered ``prepare_all_gather``.
"""

import gc
import importlib.util

import pytest
import torch
import torch.distributed as dist
import triton
import triton.language as tl

import iris
import iris.ccl.sdma.all_gather as sdma_all_gather
from iris.ccl import Config
from iris.experimental.plan import TriggeredView, chunks, tile_chunks, whole

pytestmark = pytest.mark.skipif(
    importlib.util.find_spec("xio") is None,
    reason="rocm-xio not installed (pip install iris[sdma])",
)

EMPTY = -1
# test_all_gather_sdma_large_slice: 1536 rows of 1 MiB per rank, at 2 ranks
LARGE_ROWS, LARGE_ROW_BYTES = 1536, 1 << 20


@pytest.fixture(scope="module")
def ctx():
    if not dist.is_initialized():
        pytest.skip("torch.distributed not initialized")
    large = 2 * LARGE_ROWS * LARGE_ROW_BYTES if dist.get_world_size() == 2 else 0
    ctx = iris.iris((1 << 30) + large, engines=("sdma",))
    yield ctx
    ctx.barrier()
    del ctx
    gc.collect()


def _gather_and_check(ctx, M, N, dtype, calls=3, async_op=False, config=None, input_on_heap=True):
    """Run ``calls`` all-gathers with fresh input and a wiped output each time, against RCCL."""
    rank = ctx.get_rank()
    world_size = ctx.get_num_ranks()
    if input_on_heap:
        inp = ctx.zeros((M, N), dtype=dtype)
    else:
        inp = torch.zeros(M, N, dtype=dtype, device=ctx.get_device())
        assert not ctx.heap.is_symmetric(inp)
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


def test_all_gather_sdma_input_off_heap(ctx):
    # The copy engines read the input at its local address, so it need not be on the symmetric heap
    _gather_and_check(ctx, 300, 70, torch.bfloat16, input_on_heap=False)


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


def test_all_gather_sdma_large_slice(ctx):
    # A 1.5 GiB slice goes as two copies per peer; the copy loop's byte offset must be 64-bit, since
    # an i32 offset wraps to a negative value past 2 GiB
    world_size = ctx.get_num_ranks()
    if world_size != 2:
        pytest.skip("3 GiB of output per rank: run at 2 ranks only")
    rank = ctx.get_rank()

    def pattern(r):
        return ((torch.arange(LARGE_ROWS, device=ctx.get_device()) * 7 + r) % 127).to(torch.int8)

    inp = pattern(rank)[:, None].expand(LARGE_ROWS, LARGE_ROW_BYTES).contiguous()
    out = ctx.zeros((world_size * LARGE_ROWS, LARGE_ROW_BYTES), dtype=torch.int8)
    ctx.ccl.all_gather(out, inp, config=Config(engine="sdma"))
    expected = torch.cat([pattern(r) for r in range(world_size)])
    assert torch.equal(out, expected[:, None].expand_as(out))
    ctx.barrier()


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
    gathered = ctx.zeros((world_size * 64, 64), dtype=torch.float32)
    for ignored in [
        dict(use_gluon=True),
        dict(all_gather_variant="partitioned"),
        dict(all_gather_variant="persistent"),
    ]:
        with pytest.raises(ValueError, match="does not take"):
            ctx.ccl.all_gather(gathered, t[:, :64].contiguous(), config=Config(engine="sdma", **ignored))
    ctx.barrier()


@triton.jit
def _fill_rows(out, start, n, round, handle, chunk, BLOCK: tl.constexpr):
    """One program writes elements [start, start + n) of ``out``, then marks ``chunk`` ready."""
    for off in range(0, n, BLOCK):
        idx = start + off + tl.arange(0, BLOCK)
        tl.store(out + idx, (round << 24) | idx, mask=off + tl.arange(0, BLOCK) < n)
    TriggeredView.initialize(handle).mark_ready(chunk)


@triton.jit
def _tile_producer(
    out, row0, rows, cols, round, handle, first, num_tiles, BM: tl.constexpr, BN: tl.constexpr, P: tl.constexpr
):
    """Persistent tiled producer: program p writes tiles p, p + P, ... and marks each one ready."""
    view = TriggeredView.initialize(handle)
    tiles_n = tl.cdiv(cols, BN)
    for step in range(tl.program_id(0), num_tiles, P):
        r = row0 + (step // tiles_n) * BM + tl.arange(0, BM)[:, None]
        c = (step % tiles_n) * BN + tl.arange(0, BN)[None, :]
        idx = r * cols + c
        tl.store(out + idx, (round << 24) | idx, mask=(r < row0 + rows) & (c < cols))
        view.mark_ready(first + step // P)


def _check_round(out, round):
    expected = (round << 24) | torch.arange(out.numel(), dtype=torch.int32, device=out.device)
    assert torch.equal(out.view(-1), expected), f"round {round}: {(out.view(-1) != expected).sum()} elements wrong"


def test_prepare_all_gather_whole(ctx):
    rank = ctx.get_rank()
    world_size = ctx.get_num_ranks()
    M, N = 100, 37
    out = ctx.zeros((world_size * M, N), dtype=torch.int32)
    out.fill_(EMPTY)
    with ctx.ccl.prepare_all_gather(out) as plan:
        for round in range(1, 4):
            handle = plan.start()
            _fill_rows[(1,)](out, rank * M * N, M * N, round, handle, plan.chunk_id(rank, 0), BLOCK=1024)
            plan.wait()
            _check_round(out, round)
            out.fill_(EMPTY)
            plan.next_round()


def test_prepare_all_gather_tiles(ctx):
    rank = ctx.get_rank()
    world_size = ctx.get_num_ranks()
    M, N, BM, BN, P = 96, 80, 32, 32, 4  # Edge tiles clipped; 9 tiles, so the last chunk has one
    out = ctx.zeros((world_size * M, N), dtype=torch.int32)
    out.fill_(EMPTY)
    num_tiles = triton.cdiv(M, BM) * triton.cdiv(N, BN)
    schedule = tile_chunks(out[rank * M : (rank + 1) * M], (BM, BN), tiles_per_chunk=P, base=out)
    with ctx.ccl.prepare_all_gather(out, schedule) as plan:
        for round in range(1, 4):
            handle = plan.start()
            _tile_producer[(P,)](
                out, rank * M, M, N, round, handle, plan.chunk_id(rank, 0), num_tiles, BM=BM, BN=BN, P=P
            )
            plan.wait()
            _check_round(out, round)
            out.fill_(EMPTY)
            plan.next_round()


def test_prepare_all_gather_rejects_bad_schedule(ctx):
    rank = ctx.get_rank()
    world_size = ctx.get_num_ranks()
    if world_size < 2:
        pytest.skip("needs another rank's rows")
    M, N = 16, 16
    out = ctx.zeros((world_size * M, N), dtype=torch.int32)
    mine = out[rank * M : (rank + 1) * M]
    theirs = (rank + 1) % world_size
    cases = [
        # (what is wrong, this rank's schedule, whether this rank's schedule is the bad one)
        ("whole buffer on rank 0", chunks(out, 2, base=out) if rank == 0 else whole(mine, base=out), rank == 0),
        # Offsets from mine itself: right on rank 0 only
        ("forgotten base", whole(mine), rank != 0),
        ("another rank's rows", whole(out[theirs * M : (theirs + 1) * M], base=out), True),
        ("partial", whole(mine[:-1], base=out), True),
        ("empty", [], True),
    ]
    for what, schedule, bad in cases:
        with pytest.raises(ValueError if bad else RuntimeError, match=None if bad else "could not prepare"):
            ctx.ccl.prepare_all_gather(out, schedule)
    with pytest.raises(ValueError, match="multiple"):
        ctx.ccl.prepare_all_gather(ctx.zeros((world_size * M + 1, N), dtype=torch.int32))
    ctx.barrier()
