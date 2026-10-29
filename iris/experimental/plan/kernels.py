# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.

"""Kernels a triggered plan launches itself: sending its own chunks and checking completion."""

import triton
import triton.language as tl

from iris.experimental.plan.triton import HEADER_WORDS, R_ARRIVAL, R_EXPECTED, R_WRITERS, RECORD_WORDS, TriggeredView
from iris.mem.utils import read_realtime

# Error word codes (see wait_kernel)
TIMED_OUT = tl.constexpr(1)
WRONG_MARKS = tl.constexpr(2)


@triton.jit
def release_kernel(view, chunks):
    """Open the gate of chunk ``chunks[pid]`` this round, however many of its writers have marked it."""
    cv = TriggeredView.initialize(view)
    chunk = tl.load(chunks + tl.program_id(0))
    writers = tl.load(view + HEADER_WORDS + chunk * RECORD_WORDS + R_WRITERS)
    tl.atomic_max(cv.gates + chunk, (writers * cv.round).to(tl.int32), sem="release", scope="sys")


@triton.jit
def _reached(ptr, target, t0, ticks):
    while (tl.atomic_add(ptr, 0, sem="acquire", scope="sys") < target) & (read_realtime() - t0 < ticks):
        pass
    return tl.atomic_add(ptr, 0, sem="acquire", scope="sys") >= target


@triton.jit
def _record_error(error, code, round, index, value):
    """Keep the first error as ``[code, round, index, value]``."""
    zero = tl.zeros((), tl.int64)
    if tl.atomic_cas(error, zero, zero + code) == 0:
        tl.store(error + 1, round)
        tl.store(error + 2, index)
        tl.store(error + 3, value)


@triton.jit
def wait_kernel(view, sent_to, status, error, ticks, OWN_ONLY: tl.constexpr, STREAM: tl.constexpr):
    """
    Program ``pid < num_chunks``: chunk ``pid`` has arrived here this round (always true of this
    rank's own chunks, of chunks not sent here, and of every chunk if ``OWN_ONLY``). Program ``num_chunks + p``: the queue to
    peer ``p`` has finished this round (always true if bit ``p`` of ``sent_to`` is clear). Reads only
    this rank's gates, for up to ``ticks`` of the 100 MHz clock. Launch with one warp, since the
    loop exits on a clock read.

    Writes 1 to ``status[pid]`` when done, 0 if the time ran out. With ``STREAM`` (queued behind the
    producer, nobody waiting on the host), records a timeout or a wrong mark count in ``error``.
    """
    pid = tl.program_id(0)
    cv = TriggeredView.initialize(view)
    t0 = read_realtime()
    done = tl.full((), True, tl.int1)
    if pid < cv.num_chunks:
        if cv.owns(pid):
            if STREAM:
                # The producer ran before this kernel on the stream, so all of its marks are in
                writers = tl.load(view + HEADER_WORDS + pid * RECORD_WORDS + R_WRITERS)
                marks = tl.atomic_add(cv.gates + pid, 0, sem="acquire", scope="sys") - writers * (cv.round - 1)
                if marks != writers:
                    _record_error(error, WRONG_MARKS, cv.round, pid, marks)
        elif not OWN_ONLY:
            record = view + HEADER_WORDS + pid * RECORD_WORDS
            target = tl.load(record + R_EXPECTED) * cv.round
            done = _reached(cv.gates + tl.load(record + R_ARRIVAL), target, t0, ticks)
    else:
        peer = pid - cv.num_chunks
        if ((sent_to >> peer) & 1) != 0:
            done = _reached(cv.gates + 2 * cv.num_chunks + peer, cv.round, t0, ticks)
    tl.store(status + pid, done.to(tl.int32))
    if STREAM:
        if not done:
            _record_error(error, TIMED_OUT, cv.round, pid, 0)
