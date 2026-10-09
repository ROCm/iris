# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.

# Triggered plans on the copy engines. Each rank owns a slice of a symmetric buffer and sends it to
# every peer; batches are released by kernels with TriggeredView.publish.
#
# Every test that arms a plan aborts it on failure, so no chain is left parked on a queue.

import time
from contextlib import contextmanager

import pytest
import torch
import triton
import triton.language as tl

import iris
from iris.mem.triton import triggered as dv
from iris.mem.utils import read_realtime
from iris.triggered import Batch, Segment, State, TriggeredPlan, TriggeredView, chunks, waves, whole

EMPTY = -1


def _ctx():
    ctx = iris.iris(1 << 26, allocator_type="vmem_chunked")
    if ctx.get_num_ranks() < 2:
        pytest.skip("triggered plans need at least two ranks")
    return ctx


@contextmanager
def _armed(plan):
    try:
        yield plan
    finally:
        if plan.state in (State.ARMED, State.ACTIVE, State.FAILED):
            plan.abort(timeout=10)
        plan.destroy()


def _expected(epoch, n, start=0, device="cuda"):
    # Epoch in the top byte, element index below: stale or misplaced data never matches
    return (epoch << 24) | torch.arange(start, start + n, dtype=torch.int32, device=device)


def _check_before_release(ctx, plan, buffer, slice_elems):
    """After begin_device_epoch(), before any producer runs: nothing has arrived from any peer."""
    ctx.barrier()
    rank = ctx.get_rank()
    others = [p for p in range(ctx.get_num_ranks()) if p != rank]
    consume = [plan.batch_id(p, i) for p in others for i in range(len(plan.schedules[p]))]
    assert (plan.gates[consume, dv.G_ARRIVAL.value] == plan.epoch - 1).all()
    for p in others:
        assert (buffer[p * slice_elems : (p + 1) * slice_elems] == EMPTY).all()
    ctx.barrier()


def _check_and_wipe(ctx, buffer, epoch, slice_elems):
    assert torch.equal(buffer, _expected(epoch, buffer.numel()).view_as(buffer))
    rank = ctx.get_rank()
    mine = buffer.view(-1)[rank * slice_elems : (rank + 1) * slice_elems].clone()
    buffer.fill_(EMPTY)
    buffer.view(-1)[rank * slice_elems : (rank + 1) * slice_elems] = mine


@triton.jit
def _segments_producer(buf, segs, max_segs, view, first, epoch, BLOCK: tl.constexpr):
    """Program b writes every segment (element offset, count) of its batch, then publishes it."""
    b = tl.program_id(0)
    for s in range(max_segs):
        start = tl.load(segs + (b * max_segs + s) * 2)
        count = tl.load(segs + (b * max_segs + s) * 2 + 1)
        for off in range(0, count, BLOCK):
            idx = start + off + tl.arange(0, BLOCK)
            tl.store(buf + idx, (epoch << 24) | idx, mask=off + tl.arange(0, BLOCK) < count)
    TriggeredView.initialize(view).publish(first + b)


def _segment_table(schedule, elem):
    max_segs = max(len(b.segments) for b in schedule)
    table = torch.zeros(len(schedule), max_segs, 2, dtype=torch.int64)
    for i, batch in enumerate(schedule):
        for j, s in enumerate(batch.segments):
            table[i, j] = torch.tensor([s.offset // elem, s.width // elem])
    return table.cuda(), max_segs


def _strided(slice_elems, rank, num_batches, blocks_per_batch):
    """Batch j holds blocks j, j + num_batches, ...: several separate segments per batch."""
    block = slice_elems // (num_batches * blocks_per_batch) * 4
    base = rank * slice_elems * 4
    return [
        Batch(tuple(Segment(base + (j + k * num_batches) * block, block) for k in range(blocks_per_batch)))
        for j in range(num_batches)
    ]


@pytest.mark.parametrize("kind", ["chunks", "strided"])
def test_linear_batches_across_epochs(kind):
    ctx = _ctx()
    rank, world = ctx.get_rank(), ctx.get_num_ranks()
    slice_elems = 16 * 1024
    buffer = ctx.zeros(world * slice_elems, dtype=torch.int32)
    buffer.fill_(EMPTY)
    if kind == "chunks":
        schedules = [chunks(buffer[r * slice_elems : (r + 1) * slice_elems], 4, base=buffer) for r in range(world)]
    else:
        schedules = [_strided(slice_elems, r, 4, 4) for r in range(world)]
    segs, max_segs = _segment_table(schedules[rank], 4)

    with _armed(TriggeredPlan(ctx, buffer, schedules, timeout=20)) as plan:
        for epoch in range(1, 4):
            if epoch > 1:
                plan.next_epoch()
            view = plan.begin_device_epoch()
            _check_before_release(ctx, plan, buffer, slice_elems)
            _segments_producer[(len(schedules[rank]),)](
                buffer, segs, max_segs, view, plan.batch_id(rank, 0), epoch, BLOCK=1024
            )
            plan.end_device_epoch()
            _check_and_wipe(ctx, buffer, epoch, slice_elems)
    ctx.barrier()
    del ctx


def test_heap_growth_while_armed():
    """The chunked heap keeps peer addresses fixed as it grows, so armed chains stay valid."""
    ctx = _ctx()
    rank, world = ctx.get_rank(), ctx.get_num_ranks()
    alloc = ctx.heap.allocator
    slice_elems = 4096
    buffer = ctx.zeros(world * slice_elems, dtype=torch.int32)
    buffer.fill_(EMPTY)
    schedules = [chunks(buffer[r * slice_elems : (r + 1) * slice_elems], 2, base=buffer) for r in range(world)]
    segs, max_segs = _segment_table(schedules[rank], 4)

    with _armed(TriggeredPlan(ctx, buffer, schedules, timeout=20)) as plan:
        num_chunks = alloc.get_num_chunks()
        ctx.zeros(alloc.chunk_size, dtype=torch.int8)  # mapped on every rank while chains are armed
        assert alloc.get_num_chunks() > num_chunks
        view = plan.begin_device_epoch()
        _segments_producer[(2,)](buffer, segs, max_segs, view, plan.batch_id(rank, 0), 1, BLOCK=1024)
        plan.end_device_epoch()
        _check_and_wipe(ctx, buffer, 1, slice_elems)

    # A buffer larger than a chunk spans a chunk boundary on every rank
    slice_elems = -(-(alloc.chunk_size // 4 + 1024) // world)
    buffer = ctx.zeros(world * slice_elems, dtype=torch.int32)
    buffer.fill_(EMPTY)
    mine = buffer[rank * slice_elems : (rank + 1) * slice_elems]
    mine.copy_(torch.arange(rank * slice_elems, (rank + 1) * slice_elems, dtype=torch.int32, device=mine.device))
    schedules = [whole(buffer[r * slice_elems : (r + 1) * slice_elems], base=buffer) for r in range(world)]
    with _armed(TriggeredPlan(ctx, buffer, schedules, timeout=20)) as plan:
        plan.run()
        assert torch.equal(buffer, torch.arange(buffer.numel(), dtype=torch.int32, device=buffer.device))
    ctx.barrier()
    del ctx


@triton.jit
def _waves_producer(
    buf, view, first, epoch, row0, M, N, BM: tl.constexpr, BN: tl.constexpr, NUM_PROGRAMS: tl.constexpr
):
    """Persistent tiled producer: program p makes tiles p, p + NUM_PROGRAMS, ... and publishes each tile's wave."""
    tv = TriggeredView.initialize(view)
    tiles_n = tl.cdiv(N, BN)
    for i in range(tl.program_id(0), tl.cdiv(M, BM) * tiles_n, NUM_PROGRAMS):
        rows = (i // tiles_n) * BM + tl.arange(0, BM)[:, None]
        cols = (i % tiles_n) * BN + tl.arange(0, BN)[None, :]
        idx = (row0 + rows) * N + cols
        tl.store(buf + idx, (epoch << 24) | idx, mask=(rows < M) & (cols < N))
        tv.publish(first + i // NUM_PROGRAMS)


def test_waves_2d_unaligned():
    ctx = _ctx()
    rank, world = ctx.get_rank(), ctx.get_num_ranks()
    m, n, bm, bn, programs = 40, 72, 16, 32, 4  # 3 x 3 tiles, clipped at the edges; waves of 4, 4, 1
    buffer = ctx.zeros(world * m, n, dtype=torch.int32)
    buffer.fill_(EMPTY)
    schedules = [waves(buffer[r * m : (r + 1) * m], (bm, bn), programs, base=buffer) for r in range(world)]

    with _armed(TriggeredPlan(ctx, buffer, schedules, timeout=20)) as plan:
        for epoch in range(1, 4):
            if epoch > 1:
                plan.next_epoch()
            view = plan.begin_device_epoch()
            _check_before_release(ctx, plan, buffer.view(-1), m * n)
            _waves_producer[(programs,)](
                buffer, view, plan.batch_id(rank, 0), epoch, rank * m, m, n, BM=bm, BN=bn, NUM_PROGRAMS=programs
            )
            plan.end_device_epoch()
            _check_and_wipe(ctx, buffer, epoch, m * n)
    ctx.barrier()
    del ctx


@triton.jit
def _staggered_producer(buf, view, batch, epoch, start, part, delay, BLOCK: tl.constexpr):
    """Program p waits p * delay ticks, writes its part of the batch, then publishes it."""
    pid = tl.program_id(0)
    t0 = read_realtime()
    while read_realtime() - t0 < pid * delay:
        pass
    for off in range(0, part, BLOCK):
        idx = start + pid * part + off + tl.arange(0, BLOCK)
        tl.store(buf + idx, (epoch << 24) | idx)
    TriggeredView.initialize(view).publish(batch)


def test_many_contributors_one_gate():
    """The chain must wait for the last contributor, and see every contributor's data."""
    ctx = _ctx()
    rank, world = ctx.get_rank(), ctx.get_num_ranks()
    programs, slice_elems = 8, 1 << 20  # 4 MiB per rank, one batch
    buffer = ctx.zeros(world * slice_elems, dtype=torch.int32)
    buffer.fill_(EMPTY)
    schedules = [
        whole(buffer[r * slice_elems : (r + 1) * slice_elems], base=buffer, contributors=programs) for r in range(world)
    ]

    with _armed(TriggeredPlan(ctx, buffer, schedules, timeout=20)) as plan:
        for epoch in range(1, 11):
            if epoch > 1:
                plan.next_epoch()
            view = plan.begin_device_epoch()
            _staggered_producer[(programs,)](
                buffer,
                view,
                plan.batch_id(rank, 0),
                epoch,
                rank * slice_elems,
                slice_elems // programs,
                20_000,  # 200 us between contributors
                BLOCK=1024,
            )
            plan.end_device_epoch()
            _check_and_wipe(ctx, buffer, epoch, slice_elems)
    ctx.barrier()
    del ctx


@triton.jit
def _publish_kernel(view, batch):
    TriggeredView.initialize(view).publish(batch)


def test_head_of_line_order():
    """Chains on one queue run in arming order: a batch released early waits for the one armed before it."""
    ctx = _ctx()
    rank, world = ctx.get_rank(), ctx.get_num_ranks()
    slice_elems = 4096
    buffer = ctx.zeros(world * slice_elems, dtype=torch.int32)
    buffer.fill_(EMPTY)
    if rank == 0:
        buffer[:slice_elems] = _expected(1, slice_elems)
    schedules = [chunks(buffer[:slice_elems], 2)] + [[] for _ in range(world - 1)]

    with _armed(TriggeredPlan(ctx, buffer, schedules, timeout=20)) as plan:
        view = plan.begin_device_epoch()
        if rank == 0:
            _publish_kernel[(1,)](view, 1)
        ctx.barrier()
        if rank != 0:
            time.sleep(0.1)
            assert (plan.gates[:, dv.G_ARRIVAL.value] == 0).all()
            assert (buffer[:slice_elems] == EMPTY).all()
        ctx.barrier()
        if rank == 0:
            _publish_kernel[(1,)](view, 0)
        plan.end_device_epoch()
        assert torch.equal(buffer[:slice_elems], _expected(1, slice_elems))
        if rank != 0:
            assert (plan.gates[:, dv.G_ARRIVAL.value] == 1).all()
    ctx.barrier()
    del ctx


def test_misuse_raises():
    ctx = _ctx()
    rank, world = ctx.get_rank(), ctx.get_num_ranks()
    slice_elems = 1024
    buffer = ctx.zeros(world * slice_elems, dtype=torch.int32)
    schedules = [whole(buffer[r * slice_elems : (r + 1) * slice_elems], base=buffer) for r in range(world)]
    peer = (rank + 1) % world

    with _armed(TriggeredPlan(ctx, buffer, schedules, timeout=20)) as a, _armed(
        TriggeredPlan(ctx, buffer, schedules, timeout=20)
    ) as b:
        with pytest.raises(RuntimeError, match="armed"):
            ctx.put(buffer[:slice_elems], to_rank=peer)
        with pytest.raises(RuntimeError, match="needs"):
            a.end_device_epoch()
        with pytest.raises(RuntimeError, match="armed earlier"):
            b.begin_device_epoch()
        with pytest.raises(RuntimeError, match="abort in the order"):
            b.abort()
        # Used in arming order, two plans on the same queues are fine
        for epoch in range(1, 3):
            if epoch > 1:
                a.next_epoch()
                b.next_epoch()
            a.run()
            b.run()
            with pytest.raises(RuntimeError, match="needs"):
                a.begin_device_epoch()
    ctx.barrier()
    del ctx


def test_asymmetric_buffer_raises():
    """A buffer at different heap offsets on different ranks would send every copy to the wrong place."""
    ctx = _ctx()
    rank, world = ctx.get_rank(), ctx.get_num_ranks()
    slice_elems = 1024
    if rank == 0:
        pad = ctx.zeros(slice_elems, dtype=torch.int32)  # noqa: F841  shifts rank 0's later allocations
    buffer = ctx.zeros(world * slice_elems, dtype=torch.int32)
    schedules = [whole(buffer[r * slice_elems : (r + 1) * slice_elems], base=buffer) for r in range(world)]
    with pytest.raises(RuntimeError, match="different heap offsets"):
        TriggeredPlan(ctx, buffer, schedules, timeout=20)
    ctx.barrier()
    del ctx


@triton.jit
def _release_flag(flag, value):
    tl.atomic_xchg(flag, value, sem="release", scope="sys")


def test_host_put_wait_flag():
    """Host put(wait_flag=, signal_flag=): the copy waits for the flag (>=), never reset between epochs."""
    ctx = _ctx()
    rank = ctx.get_rank()
    n = 4096
    src = ctx.zeros(n, dtype=torch.int32)
    dst = ctx.zeros(n, dtype=torch.int32)
    flag = ctx.zeros(1, dtype=torch.int32)
    signal = ctx.zeros(1, dtype=torch.int32)
    epoch = 0
    try:
        for epoch in range(1, 6):
            if rank == 0:
                src.copy_(_expected(epoch, n))
                ctx.put(
                    src, to_rank=1, to_tensor=dst, wait_flag=flag, wait_value=epoch, signal_flag=signal, async_op=True
                )
            ctx.barrier()
            if rank == 1:
                time.sleep(0.05)
                assert signal.item() == epoch - 1
            ctx.barrier()
            if rank == 0:
                _release_flag[(1,)](flag, epoch)
                ctx.quiet(to_rank=1)
            ctx.barrier()
            if rank == 1:
                assert signal.item() == epoch
                assert torch.equal(dst, _expected(epoch, n))
    finally:
        if rank == 0:
            _release_flag[(1,)](flag, epoch)
            ctx.quiet(to_rank=1)
    ctx.barrier()
    del ctx


def test_torch_allocator_rejected():
    ctx = iris.iris(1 << 20)
    buffer = ctx.zeros(1024, dtype=torch.int32)
    with pytest.raises(ValueError, match="vmem"):
        TriggeredPlan(ctx, buffer, [whole(buffer)] * ctx.get_num_ranks())
    ctx.barrier()
    del ctx
