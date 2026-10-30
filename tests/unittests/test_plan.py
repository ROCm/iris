# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.

# Triggered plans on the copy engines. Each rank owns a slice of a symmetric buffer and sends it to
# every peer; chunks are released by kernels with TriggeredView.mark_ready (Triton or Gluon).
#
# Every test uses its plans as context managers, so no chain is left parked on a queue.

import time
from types import SimpleNamespace

import pytest
import torch
import triton
import triton.language as tl
from triton.experimental import gluon
from triton.experimental.gluon import language as gl
from triton.language.extra.hip import utils as hip_utils

import iris
from iris.experimental.plan.gluon import TriggeredView as GluonView
from iris.host import queue_ledger
from iris.experimental.plan import triton as dv
from iris.mem.utils import read_realtime
from iris.experimental.plan import (
    Chunk,
    Region,
    State,
    TriggeredPlan,
    TriggeredView,
    armer,
    chunks,
    tile_chunks,
    whole,
)

EMPTY = -1


def _ctx(allocator_type="vmem_chunked"):
    ctx = iris.iris(1 << 26, allocator_type=allocator_type, engines=("sdma",))
    if ctx.get_num_ranks() < 2:
        pytest.skip("triggered plans need at least two ranks")
    return ctx


def _expected(round, n, start=0, device="cuda"):
    # Round in the top byte, element index below: stale or misplaced data never matches
    return (round << 24) | torch.arange(start, start + n, dtype=torch.int32, device=device)


def _slice(buffer, rank, slice_elems):
    return buffer.view(-1)[rank * slice_elems : (rank + 1) * slice_elems]


def _check_before_release(ctx, plan, buffer, slice_elems):
    """After start(), before any producer runs: nothing has arrived from any peer."""
    ctx.barrier()
    rank = ctx.get_rank()
    others = [p for p in range(ctx.get_num_ranks()) if p != rank]
    consume = [plan.chunk_id(p, i) for p in others for i in range(len(plan.schedules[p]))]
    assert (plan.gates[consume, dv.G_ARRIVAL.value] == plan.round - 1).all()
    for p in others:
        assert (_slice(buffer, p, slice_elems) == EMPTY).all()
    ctx.barrier()


def _check_and_wipe(ctx, buffer, round, slice_elems):
    assert torch.equal(buffer, _expected(round, buffer.numel()).view_as(buffer))
    mine = _slice(buffer, ctx.get_rank(), slice_elems).clone()
    buffer.fill_(EMPTY)
    _slice(buffer, ctx.get_rank(), slice_elems).copy_(mine)


@triton.jit
def _regions_producer(buf, regs, max_regs, handle, first, round, BLOCK: tl.constexpr):
    """Program c writes every region (element offset, count) of its chunk, then marks it ready."""
    c = tl.program_id(0)
    for r in range(max_regs):
        start = tl.load(regs + (c * max_regs + r) * 2)
        count = tl.load(regs + (c * max_regs + r) * 2 + 1)
        for off in range(0, count, BLOCK):
            idx = start + off + tl.arange(0, BLOCK)
            tl.store(buf + idx, (round << 24) | idx, mask=off + tl.arange(0, BLOCK) < count)
    TriggeredView.initialize(handle).mark_ready(first + c)


def _region_table(schedule, elem):
    max_regs = max(len(c.regions) for c in schedule)
    table = torch.zeros(len(schedule), max_regs, 2, dtype=torch.int64)
    for i, chunk in enumerate(schedule):
        for j, r in enumerate(chunk.regions):
            table[i, j] = torch.tensor([r.offset_bytes // elem, r.row_bytes // elem])
    return table.cuda(), max_regs


def _strided(slice_elems, rank, num_chunks, blocks_per_chunk):
    """Chunk j holds blocks j, j + num_chunks, ...: several separate regions per chunk."""
    block = slice_elems // (num_chunks * blocks_per_chunk) * 4
    base = rank * slice_elems * 4
    return [
        Chunk(tuple(Region(base + (j + k * num_chunks) * block, block) for k in range(blocks_per_chunk)))
        for j in range(num_chunks)
    ]


@pytest.mark.parametrize("kind", ["chunks", "strided"])
def test_linear_chunks_across_rounds(kind):
    ctx = _ctx()
    rank, world = ctx.get_rank(), ctx.get_num_ranks()
    slice_elems = 16 * 1024
    buffer = ctx.zeros(world * slice_elems, dtype=torch.int32)
    buffer.fill_(EMPTY)
    if kind == "chunks":
        schedule = chunks(_slice(buffer, rank, slice_elems), 4, base=buffer)
    else:
        schedule = _strided(slice_elems, rank, 4, 4)
    regs, max_regs = _region_table(schedule, 4)

    with TriggeredPlan(ctx, buffer, schedule, timeout=20) as plan:
        for round in range(1, 4):
            if round > 1:
                plan.next_round()
            handle = plan.start()
            _check_before_release(ctx, plan, buffer, slice_elems)
            _regions_producer[(len(schedule),)](
                buffer, regs, max_regs, handle, plan.chunk_id(rank, 0), round, BLOCK=1024
            )
            plan.wait()
            _check_and_wipe(ctx, buffer, round, slice_elems)
    ctx.barrier()
    del ctx


def test_heap_growth_while_prepared():
    """The chunked heap keeps peer addresses fixed as it grows, so prepared copies stay valid."""
    ctx = _ctx()
    rank, world = ctx.get_rank(), ctx.get_num_ranks()
    alloc = ctx.heap.allocator
    slice_elems = 4096
    buffer = ctx.zeros(world * slice_elems, dtype=torch.int32)
    buffer.fill_(EMPTY)
    schedule = chunks(_slice(buffer, rank, slice_elems), 2, base=buffer)
    regs, max_regs = _region_table(schedule, 4)

    with TriggeredPlan(ctx, buffer, schedule, timeout=20) as plan:
        num_chunks = alloc.get_num_chunks()
        ctx.zeros(alloc.chunk_size, dtype=torch.int8)  # mapped on every rank while copies are prepared
        assert alloc.get_num_chunks() > num_chunks
        handle = plan.start()
        _regions_producer[(2,)](buffer, regs, max_regs, handle, plan.chunk_id(rank, 0), 1, BLOCK=1024)
        plan.wait()
        _check_and_wipe(ctx, buffer, 1, slice_elems)

    # A buffer larger than a heap chunk spans a heap-chunk boundary on every rank
    slice_elems = -(-(alloc.chunk_size // 4 + 1024) // world)
    buffer = ctx.zeros(world * slice_elems, dtype=torch.int32)
    buffer.fill_(EMPTY)
    mine = _slice(buffer, rank, slice_elems)
    mine.copy_(torch.arange(rank * slice_elems, (rank + 1) * slice_elems, dtype=torch.int32, device=mine.device))
    with TriggeredPlan(ctx, buffer, whole(mine, base=buffer), timeout=20) as plan:
        plan.run()
        assert torch.equal(buffer, torch.arange(buffer.numel(), dtype=torch.int32, device=buffer.device))
    ctx.barrier()
    del ctx


@triton.jit
def _grouped(i, tiles_m, tiles_n, GROUP_M: tl.constexpr):
    """Step -> tile id with a GROUP_M-row swizzle; also called on the host for the schedule."""
    if GROUP_M == 0:
        return i
    group = i // (GROUP_M * tiles_n)
    size_m = min(tiles_m - group * GROUP_M, GROUP_M)
    within = i % (GROUP_M * tiles_n)
    return (group * GROUP_M + within % size_m) * tiles_n + within // size_m


@triton.jit
def _tiled_producer(
    buf,
    handle,
    first,
    round,
    row0,
    M,
    N,
    BM: tl.constexpr,
    BN: tl.constexpr,
    NUM_PROGRAMS: tl.constexpr,
    GROUP_M: tl.constexpr,
):
    """Persistent tiled producer: program p takes steps p, p + NUM_PROGRAMS, ... and marks each step's chunk ready."""
    cv = TriggeredView.initialize(handle)
    tiles_m, tiles_n = tl.cdiv(M, BM), tl.cdiv(N, BN)
    for i in range(tl.program_id(0), tiles_m * tiles_n, NUM_PROGRAMS):
        tile = _grouped(i, tiles_m, tiles_n, GROUP_M)
        rows = (tile // tiles_n) * BM + tl.arange(0, BM)[:, None]
        cols = (tile % tiles_n) * BN + tl.arange(0, BN)[None, :]
        idx = (row0 + rows) * N + cols
        tl.store(buf + idx, (round << 24) | idx, mask=(rows < M) & (cols < N))
        cv.mark_ready(first + i // NUM_PROGRAMS)


@pytest.mark.parametrize("group_m", [0, 2])
def test_tile_chunks_2d_unaligned(group_m):
    """Row-major and grouped tile orders; the schedule takes its order from the kernel's own mapping."""
    ctx = _ctx()
    rank, world = ctx.get_rank(), ctx.get_num_ranks()
    m, n, bm, bn, programs = 40, 72, 16, 32, 4  # 3 x 3 tiles, clipped at the edges; chunks of 4, 4, 1
    buffer = ctx.zeros(world * m, n, dtype=torch.int32)
    buffer.fill_(EMPTY)
    order = lambda i: _grouped.fn(i, 3, 3, group_m)  # noqa: E731
    schedule = tile_chunks(buffer[rank * m : (rank + 1) * m], (bm, bn), programs, order=order, base=buffer)

    with TriggeredPlan(ctx, buffer, schedule, timeout=20) as plan:
        for round in range(1, 4):
            if round > 1:
                plan.next_round()
            handle = plan.start()
            _check_before_release(ctx, plan, buffer, m * n)
            _tiled_producer[(programs,)](
                buffer,
                handle,
                plan.chunk_id(rank, 0),
                round,
                rank * m,
                m,
                n,
                BM=bm,
                BN=bn,
                NUM_PROGRAMS=programs,
                GROUP_M=group_m,
            )
            plan.wait()
            _check_and_wipe(ctx, buffer, round, m * n)
    ctx.barrier()
    del ctx


@triton.jit
def _staggered_producer(buf, handle, chunk, round, start, part, delay, BLOCK: tl.constexpr):
    """Program p waits p * delay ticks, writes its part of the chunk, then marks it ready."""
    pid = tl.program_id(0)
    t0 = read_realtime()
    while read_realtime() - t0 < pid * delay:
        pass
    for off in range(0, part, BLOCK):
        idx = start + pid * part + off + tl.arange(0, BLOCK)
        tl.store(buf + idx, (round << 24) | idx)
    TriggeredView.initialize(handle).mark_ready(chunk)


def test_many_writers_one_chunk():
    """The copy must wait for the last writer, and see every writer's data."""
    ctx = _ctx()
    rank, world = ctx.get_rank(), ctx.get_num_ranks()
    programs, slice_elems = 8, 1 << 20  # 4 MiB per rank, one chunk
    buffer = ctx.zeros(world * slice_elems, dtype=torch.int32)
    buffer.fill_(EMPTY)
    schedule = whole(_slice(buffer, rank, slice_elems), base=buffer, writers=programs)

    with TriggeredPlan(ctx, buffer, schedule, timeout=20) as plan:
        for round in range(1, 11):
            if round > 1:
                plan.next_round()
            handle = plan.start()
            _staggered_producer[(programs,)](
                buffer,
                handle,
                plan.chunk_id(rank, 0),
                round,
                rank * slice_elems,
                slice_elems // programs,
                20_000,  # 200 us between writers
                BLOCK=1024,
            )
            plan.wait()
            _check_and_wipe(ctx, buffer, round, slice_elems)
    ctx.barrier()
    del ctx


@gluon.jit
def _flat_layout(BLOCK: gl.constexpr):
    THREADS: gl.constexpr = hip_utils.num_threads()
    WARPS: gl.constexpr = hip_utils.num_warps()
    return gl.BlockedLayout([BLOCK // THREADS], [THREADS // WARPS], [WARPS], [0])


@gluon.jit
def _gluon_producer(buf, handle, first, round, start, part, WRITERS: gl.constexpr, BLOCK: gl.constexpr):
    """Program p writes part p of the slice and marks chunk p // WRITERS ready."""
    layout: gl.constexpr = _flat_layout(BLOCK)
    pid = gl.program_id(0)
    for off in range(0, part, BLOCK):
        idx = start + pid * part + off + gl.arange(0, BLOCK, layout=layout)
        gl.store(buf + idx, (round << 24) | idx)
    GluonView.initialize(handle).mark_ready(first + pid // WRITERS)


@gluon.jit
def _gluon_consumer(buf, out, status, handle, chunk_elems, budget, BLOCK: gl.constexpr):
    """Program q waits for chunk q (up to ``budget`` polls), then copies it from ``buf`` to ``out``."""
    layout: gl.constexpr = _flat_layout(BLOCK)
    cv = GluonView.initialize(handle)
    chunk = gl.program_id(0)
    # Every wave must poll the same number of times: arrived() is a CTA-wide operation
    i = 0
    ok = cv.arrived(chunk)
    while (ok == 0) & (i < budget):
        ok = cv.arrived(chunk)
        i += 1
    if ok:
        for off in range(0, chunk_elems, BLOCK):
            idx = chunk * chunk_elems + off + gl.arange(0, BLOCK, layout=layout)
            gl.store(out + idx, gl.load(buf + idx))
    gl.store(status + chunk, ok.to(gl.int32))


@pytest.mark.parametrize("allocator_type", ["torch", "vmem_chunked"])
def test_gluon_producer_and_consumer(allocator_type):
    """A Gluon kernel marks this rank's chunks ready; another waits for every chunk and reads it."""
    ctx = _ctx(allocator_type)
    rank, world = ctx.get_rank(), ctx.get_num_ranks()
    slice_elems, num_chunks, k, block = 16 * 1024, 4, 2, 1024
    buffer = ctx.zeros(world * slice_elems, dtype=torch.int32)
    buffer.fill_(EMPTY)
    schedule = chunks(_slice(buffer, rank, slice_elems), num_chunks, base=buffer, writers=k)
    out = torch.empty_like(buffer)
    status = torch.zeros(world * num_chunks, dtype=torch.int32, device=buffer.device)

    with TriggeredPlan(ctx, buffer, schedule, timeout=20) as plan:
        assert [plan.chunk_id(r, 0) for r in range(world)] == [r * num_chunks for r in range(world)]
        for round in range(1, 4):
            if round > 1:
                plan.next_round()
            out.fill_(EMPTY)
            status.zero_()
            handle = plan.start()
            _check_before_release(ctx, plan, buffer, slice_elems)
            _gluon_producer[(num_chunks * k,)](
                buffer,
                handle,
                plan.chunk_id(rank, 0),
                round,
                rank * slice_elems,
                slice_elems // (num_chunks * k),
                WRITERS=k,
                BLOCK=block,
                num_warps=4,
            )
            _gluon_consumer[(world * num_chunks,)](
                buffer, out, status, handle, slice_elems // num_chunks, 20_000_000, BLOCK=block, num_warps=4
            )
            plan.wait()
            assert status.tolist() == [1] * (world * num_chunks)
            assert torch.equal(out, _expected(round, out.numel()))
            _check_and_wipe(ctx, buffer, round, slice_elems)
    ctx.barrier()
    del ctx


@triton.jit
def _mark_ready_kernel(handle, chunk):
    TriggeredView.initialize(handle).mark_ready(chunk)


def test_head_of_line_order():
    """Copies on one queue run in schedule order: a chunk released early waits for the one before it."""
    ctx = _ctx()
    rank, world = ctx.get_rank(), ctx.get_num_ranks()
    slice_elems = 4096
    buffer = ctx.zeros(world * slice_elems, dtype=torch.int32)
    buffer.fill_(EMPTY)
    if rank == 0:
        buffer[:slice_elems] = _expected(1, slice_elems)
    schedule = chunks(buffer[:slice_elems], 2) if rank == 0 else []

    with TriggeredPlan(ctx, buffer, schedule, timeout=20) as plan:
        handle = plan.start()
        if rank == 0:
            _mark_ready_kernel[(1,)](handle, 1)
        ctx.barrier()
        if rank != 0:
            time.sleep(0.1)
            assert (plan.gates[:, dv.G_ARRIVAL.value] == 0).all()
            assert (buffer[:slice_elems] == EMPTY).all()
        ctx.barrier()
        if rank == 0:
            _mark_ready_kernel[(1,)](handle, 0)
        plan.wait()
        assert torch.equal(buffer[:slice_elems], _expected(1, slice_elems))
        if rank != 0:
            assert (plan.gates[:, dv.G_ARRIVAL.value] == 1).all()
    ctx.barrier()
    del ctx


@pytest.mark.parametrize("marks", [1, 3])
def test_wrong_mark_count_raises(marks):
    """A chunk marked ready too few or too many times fails wait() with the counts, instead of hanging or shipping early."""
    ctx = _ctx()
    rank, world = ctx.get_rank(), ctx.get_num_ranks()
    slice_elems = 1024
    buffer = ctx.zeros(world * slice_elems, dtype=torch.int32)
    schedule = whole(_slice(buffer, rank, slice_elems), base=buffer, writers=2)

    with TriggeredPlan(ctx, buffer, schedule, timeout=20) as plan:
        handle = plan.start()
        _mark_ready_kernel[(marks,)](handle, plan.chunk_id(rank, 0))
        if marks < 2:
            with pytest.raises(TimeoutError, match="1 of 2"):
                plan.wait(timeout=2)
        else:
            with pytest.raises(RuntimeError, match="marked ready 3 times, but has 2 writers"):
                plan.wait()
        assert plan.state is State.FAILED
    ctx.barrier()
    del ctx


def test_misuse_raises():
    ctx = _ctx()
    rank, world = ctx.get_rank(), ctx.get_num_ranks()
    slice_elems = 1024
    buffer = ctx.zeros(world * slice_elems, dtype=torch.int32)
    schedule = whole(_slice(buffer, rank, slice_elems), base=buffer)
    peer = (rank + 1) % world

    with TriggeredPlan(ctx, buffer, schedule, timeout=20) as a, TriggeredPlan(ctx, buffer, schedule, timeout=20) as b:
        with pytest.raises(RuntimeError, match="copies prepared"):
            ctx.put(buffer[:slice_elems], to_rank=peer)
        with pytest.raises(RuntimeError, match="needs"):
            a.wait()
        with pytest.raises(RuntimeError, match="prepared earlier"):
            b.start()
        # Used in the order they were prepared, two plans on the same queues are fine
        for round in range(1, 3):
            if round > 1:
                a.next_round()
                b.next_round()
            a.run()
            b.run()
            with pytest.raises(RuntimeError, match="needs"):
                a.start()
    ctx.barrier()
    del ctx


def test_nested_plans_close_in_any_order():
    """
    Nested with blocks close the inner plan first. Its copies queue behind the outer plan's waiting
    ones, so it stays CLOSING until the outer plan closes, which finishes both.
    """
    ctx = _ctx()
    rank, world = ctx.get_rank(), ctx.get_num_ranks()
    slice_elems = 4096
    buffers = [ctx.zeros(world * slice_elems, dtype=torch.int32) for _ in range(2)]
    ledger = queue_ledger.ledger()

    with TriggeredPlan(ctx, buffers[0], chunks(_slice(buffers[0], rank, slice_elems), 2, base=buffers[0])) as a:
        with TriggeredPlan(ctx, buffers[1], whole(_slice(buffers[1], rank, slice_elems), base=buffers[1])) as b:
            a.run()
            b.run()
            # Both prepared again, as at the end of a loop iteration
            a.next_round()
            b.next_round()
            for buffer in buffers:
                buffer.fill_(EMPTY)
                _slice(buffer, rank, slice_elems).copy_(_expected(2, slice_elems, start=rank * slice_elems))
            ctx.barrier()  # every rank's data is in place before any rank's copies are released
        assert b.state is State.CLOSING
        assert a.state is State.PREPARED
    assert a.state is State.CLOSED and b.state is State.CLOSED
    assert not any(ledger.entries.values())
    for buffer in buffers:
        assert torch.equal(buffer, _expected(2, buffer.numel()))
    ctx.barrier()
    del ctx


def test_invalid_schedule_fails_every_rank():
    """A schedule rejected on one rank raises on every rank, before anything is allocated or prepared."""
    ctx = _ctx()
    rank, world = ctx.get_rank(), ctx.get_num_ranks()
    slice_elems = 1024
    buffer = ctx.zeros(world * slice_elems, dtype=torch.int32)
    schedule = whole(_slice(buffer, rank, slice_elems), base=buffer)
    if rank == 0:
        schedule = [Chunk((Region(0, world * slice_elems * 4 + 4),))]  # runs past the buffer
    with pytest.raises(ValueError if rank == 0 else RuntimeError, match="past the buffer" if rank == 0 else "validate"):
        TriggeredPlan(ctx, buffer, schedule, timeout=20)
    ctx.barrier()
    del ctx


def test_overlapping_schedules_rejected():
    """Every rank describing the whole buffer, not its own slice, fails on every rank before anything is prepared."""
    ctx = _ctx()
    world = ctx.get_num_ranks()
    buffer = ctx.zeros(world * 1024, dtype=torch.int32)
    with pytest.raises(ValueError, match="ranks 0 and 1 both send"):
        TriggeredPlan(ctx, buffer, chunks(buffer, 2), timeout=20)
    assert not any(queue_ledger.ledger().entries.values())
    ctx.barrier()
    del ctx


def test_capacity_failure_fails_every_rank():
    """A rank that cannot reserve queue space fails all ranks before anything is prepared."""
    ctx = _ctx()
    rank, world = ctx.get_rank(), ctx.get_num_ranks()
    slice_elems = 1024
    buffer = ctx.zeros(world * slice_elems, dtype=torch.int32)
    schedule = whole(_slice(buffer, rank, slice_elems), base=buffer)
    ledger = queue_ledger.ledger()
    assert ledger is queue_ledger.ledger()

    hog = SimpleNamespace(state=State.PREPARED)
    if rank == 0:
        ledger.reserve(hog, 1, rank, {p: ledger.capacity for p in range(world) if p != rank})
    try:
        with pytest.raises(RuntimeError, match="would exceed" if rank == 0 else "could not reserve"):
            TriggeredPlan(ctx, buffer, schedule, timeout=20)
    finally:
        ledger.release(hog)
    assert not any(ledger.entries.values())

    with TriggeredPlan(ctx, buffer, schedule, timeout=20) as plan:
        plan.run()
    ctx.barrier()
    del ctx


def test_prepare_failure_flushes_and_fails_every_rank(monkeypatch):
    """A failure while preparing raises on every rank and flushes what was prepared, leaving the queues usable."""
    ctx = _ctx()
    rank, world = ctx.get_rank(), ctx.get_num_ranks()
    slice_elems = 1024
    buffer = ctx.zeros(world * slice_elems, dtype=torch.int32)
    schedule = chunks(_slice(buffer, rank, slice_elems), 4, base=buffer)

    arm_chain, armed = armer.arm_chain, []

    def flaky(*args, **kwargs):
        if rank == 0 and len(armed) == 2:
            raise RuntimeError("injected arm failure")
        armed.append(args)
        return arm_chain(*args, **kwargs)

    monkeypatch.setattr(armer, "arm_chain", flaky)
    with pytest.raises(RuntimeError, match="injected arm failure" if rank == 0 else "could not prepare"):
        TriggeredPlan(ctx, buffer, schedule, timeout=20)
    monkeypatch.undo()
    assert not any(queue_ledger.ledger().entries.values())

    with TriggeredPlan(ctx, buffer, schedule, timeout=20) as plan:
        plan.run()
    ctx.barrier()
    del ctx


def test_close_flushes_unfinished_round():
    """Closing a plan whose round never ran sends each rank's chunks as they stand, and frees the queues."""
    ctx = _ctx()
    rank, world = ctx.get_rank(), ctx.get_num_ranks()
    slice_elems = 4096
    buffer = ctx.zeros(world * slice_elems, dtype=torch.int32)
    buffer.fill_(EMPTY)
    _slice(buffer, rank, slice_elems).copy_(_expected(1, slice_elems, start=rank * slice_elems))
    schedule = chunks(_slice(buffer, rank, slice_elems), 2, base=buffer)

    with pytest.raises(KeyError):
        with TriggeredPlan(ctx, buffer, schedule, timeout=20) as plan:
            assert plan.state is State.PREPARED
            raise KeyError("leave the block without running the round")
    assert plan.state is State.CLOSED
    assert not any(queue_ledger.ledger().entries.values())
    assert torch.equal(buffer, _expected(1, buffer.numel()))
    ctx.barrier()
    del ctx


def test_close_waits_for_every_rank():
    """Peers may still touch a rank's buffer and gates, so close() returns only once all ranks call it."""
    ctx = _ctx()
    rank, world = ctx.get_rank(), ctx.get_num_ranks()
    slice_elems = 1024
    buffer = ctx.zeros(world * slice_elems, dtype=torch.int32)
    plan = TriggeredPlan(ctx, buffer, whole(_slice(buffer, rank, slice_elems), base=buffer), timeout=20)
    plan.run()
    if rank != 0:
        time.sleep(1.0)
    start = time.monotonic()
    plan.close()
    if rank == 0:
        assert time.monotonic() - start > 0.5
    assert plan.state is State.CLOSED
    plan.close()  # idempotent
    ctx.barrier()
    del ctx


@triton.jit
def _release_flag(flag, value):
    tl.atomic_xchg(flag, value, sem="release", scope="sys")


def test_host_put_wait_flag():
    """Host put(wait_flag=, signal_flag=): the copy waits for the flag (>=), never reset between rounds."""
    ctx = _ctx()
    rank = ctx.get_rank()
    n = 4096
    src = ctx.zeros(n, dtype=torch.int32)
    dst = ctx.zeros(n, dtype=torch.int32)
    flag = ctx.zeros(1, dtype=torch.int32)
    signal = ctx.zeros(1, dtype=torch.int32)
    round = 0
    try:
        for round in range(1, 6):
            if rank == 0:
                src.copy_(_expected(round, n))
                ctx.put(
                    src, to_rank=1, to_tensor=dst, wait_flag=flag, wait_value=round, signal_flag=signal, async_op=True
                )
            ctx.barrier()
            if rank == 1:
                time.sleep(0.05)
                assert signal.item() == round - 1
            ctx.barrier()
            if rank == 0:
                _release_flag[(1,)](flag, round)
                ctx.quiet(to_rank=1)
            ctx.barrier()
            if rank == 1:
                assert signal.item() == round
                assert torch.equal(dst, _expected(round, n))
    finally:
        if rank == 0:
            _release_flag[(1,)](flag, round)
            ctx.quiet(to_rank=1)
    ctx.barrier()
    del ctx


def test_requires_sdma_engine():
    ctx = iris.iris(1 << 20)
    buffer = ctx.zeros(1024, dtype=torch.int32)
    with pytest.raises(RuntimeError, match="SDMA is disabled"):
        TriggeredPlan(ctx, buffer, whole(buffer))
    ctx.barrier()
    del ctx


def test_torch_allocator():
    """The torch allocator imports peers once, so allocating while copies are prepared is safe."""
    ctx = iris.iris(1 << 26, engines=("sdma",))
    if ctx.get_num_ranks() < 2:
        pytest.skip("triggered plans need at least two ranks")
    rank, world = ctx.get_rank(), ctx.get_num_ranks()
    slice_elems = 4096
    buffer = ctx.zeros(world * slice_elems, dtype=torch.int32)
    buffer.fill_(EMPTY)
    schedule = chunks(_slice(buffer, rank, slice_elems), 2, base=buffer)
    regs, max_regs = _region_table(schedule, 4)

    with TriggeredPlan(ctx, buffer, schedule, timeout=20) as plan:
        for round in range(1, 3):
            if round > 1:
                plan.next_round()
            ctx.zeros(1 << 20, dtype=torch.int8)  # allocated on every rank while copies are prepared
            handle = plan.start()
            _regions_producer[(2,)](buffer, regs, max_regs, handle, plan.chunk_id(rank, 0), round, BLOCK=1024)
            plan.wait()
            _check_and_wipe(ctx, buffer, round, slice_elems)
    ctx.barrier()
    del ctx
