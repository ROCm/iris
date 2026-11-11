# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.

"""Kernels a triggered plan launches itself: sending its own chunks, checking completion, and fencing rounds."""

import triton
import triton.language as tl

from iris.experimental.plan.triton import (
    H_WORLD_SIZE,
    HEADER_WORDS,
    R_ARRIVAL,
    R_EXPECTED,
    R_WRITERS,
    RECORD_WORDS,
    TriggeredView,
)
from iris.mem.triton.context import __translate
from iris.mem.utils import read_realtime

# Error word codes: the word is [code, round, index, value]
TIMED_OUT = tl.constexpr(1)  # index: a chunk, or num_chunks + peer for a queue; value: an own chunk's marks
WRONG_MARKS = tl.constexpr(2)  # index: a chunk marked more than writers times; value: its marks
FENCE_TIMED_OUT = tl.constexpr(3)  # index: a peer that did not reach next_round()
PEER_FAILED = tl.constexpr(4)  # index: a peer that failed this round or an earlier one


@triton.jit
def _fence_words(cv):
    """This rank's fence counters (one per peer: next_round() calls it has seen), then the abort pair."""
    return cv.gates + 2 * cv.num_chunks + tl.load(cv.view + H_WORLD_SIZE)


@triton.jit
def _load(ptr):
    return tl.atomic_add(ptr, 0, sem="acquire", scope="sys")


@triton.jit
def _failed(abort):
    """The abort pair in one load: the round a peer marked failed (0 if none), and that peer."""
    pair = _load(abort.to(tl.pointer_type(tl.int64)))
    return pair.to(tl.int32), (pair >> 32).to(tl.int32) - 1


@triton.jit
def _aborted(abort, round):
    """True once a peer has said it failed ``round`` (or an earlier one)."""
    failed, _ = _failed(abort)
    return (failed != 0) & (failed <= round)


@triton.jit
def _mark_failed(abort, round, rank):
    """
    Record in a peer's abort pair that ``rank`` failed ``round``, unless an earlier round is
    recorded there already: a peer still waiting on that round must keep seeing it. The pair is
    swapped as one 64-bit word (round in the low half, rank + 1 in the high half), so the rank always
    matches the round. The plan's gates are heap-aligned and the pair starts at an even word.
    """
    word = abort.to(tl.pointer_type(tl.int64))
    new = ((rank + 1).to(tl.int64) << 32) | round.to(tl.int64)
    cur = tl.atomic_add(word, 0, sem="relaxed", scope="sys")
    while (cur == 0) | (cur.to(tl.int32) > round):
        seen = tl.atomic_cas(word, cur, new, sem="relaxed", scope="sys")
        cur = tl.where(seen == cur, new, seen)


@triton.jit
def release_kernel(view, chunks):
    """Open the gate of chunk ``chunks[pid]`` this round, however many of its writers have marked it."""
    cv = TriggeredView.initialize(view)
    chunk = tl.load(chunks + tl.program_id(0))
    writers = tl.load(view + HEADER_WORDS + chunk * RECORD_WORDS + R_WRITERS)
    tl.atomic_max(cv.gates + chunk, (writers * cv.round).to(tl.int32), sem="release", scope="sys")


@triton.jit
def _reached(ptr, target, t0, ticks, abort, round):
    while (_load(ptr) < target) & (read_realtime() - t0 < ticks) & (_aborted(abort, round) == 0):
        pass
    return _load(ptr) >= target


@triton.jit
def _record_error(error, code, round, index, value):
    """Keep the first error as ``[code, round, index, value]``."""
    zero = tl.zeros((), tl.int64)
    if tl.atomic_cas(error, zero, zero + code) == 0:
        tl.store(error + 1, round)
        tl.store(error + 2, index)
        tl.store(error + 3, value)


@triton.jit
def wait_kernel(view, status, error, ticks, OWN_ONLY: tl.constexpr, STREAM: tl.constexpr):
    """
    Program ``pid < num_chunks``: chunk ``pid`` has arrived here this round (always true of chunks
    not sent here), or for this rank's own chunks, been marked ready by all of its writers (other
    ranks' chunks are skipped if ``OWN_ONLY``). Program ``num_chunks + p``: the queue to peer ``p``
    has finished this round (always true if this rank's chains do not go to ``p``). Reads only this
    rank's gates, for up to ``ticks`` of the read_realtime() clock. Launch with one warp, since the
    loop exits on a clock read.

    Writes 1 to ``status[pid]`` when done, 0 if not. Unless ``OWN_ONLY`` (a flush, which must wait
    for this rank's queues whatever its peers did), stops at and records in ``error`` a peer that
    failed the round (see fence_kernel). With ``STREAM`` (queued behind the producer, nobody waiting on the
    host), also records a timeout or extra marks, and returns at once if ``error`` already holds an
    error: it is reported anyway, and a loop that never synchronizes would otherwise spin a full
    timeout every round.
    """
    pid = tl.program_id(0)
    if STREAM:
        if tl.load(error, volatile=True) != 0:
            tl.store(status + pid, 0)
            return
    cv = TriggeredView.initialize(view)
    abort = _fence_words(cv) + tl.load(view + H_WORLD_SIZE)
    # Round 0 never counts as aborted
    abort_round = 0 if OWN_ONLY else cv.round
    t0 = read_realtime()
    done = tl.full((), True, tl.int1)
    marks = tl.zeros((), tl.int64)
    if pid < cv.num_chunks:
        if cv.owns(pid):
            # So wait() checks the mark count after the producer, even with no peers or on another stream
            writers = tl.load(view + HEADER_WORDS + pid * RECORD_WORDS + R_WRITERS)
            done = _reached(cv.gates + pid, writers * cv.round, t0, ticks, abort, abort_round)
            marks = _load(cv.gates + pid) - writers * (cv.round - 1)
            # Too few marks is a timeout, as in the blocking wait(); too many open later rounds early
            if STREAM:
                if done & (marks != writers):
                    _record_error(error, WRONG_MARKS, cv.round, pid, marks)
        elif not OWN_ONLY:
            record = view + HEADER_WORDS + pid * RECORD_WORDS
            target = tl.load(record + R_EXPECTED) * cv.round
            done = _reached(cv.gates + tl.load(record + R_ARRIVAL), target, t0, ticks, abort, abort_round)
    else:
        peer = pid - cv.num_chunks
        queues = cv._queues_to(peer)
        if queues != 0:
            done = _reached(cv.gates + 2 * cv.num_chunks + peer, queues * cv.round, t0, ticks, abort, abort_round)
    # Even if done: a peer that failed the round may still send it, such as its old data from a flush
    failed, failed_rank = _failed(abort)
    aborted = (failed != 0) & (failed <= abort_round)
    done = done & (aborted == 0)
    tl.store(status + pid, done.to(tl.int32))
    if aborted:
        _record_error(error, PEER_FAILED, cv.round, failed_rank, 0)
    elif STREAM and not done:
        _record_error(error, TIMED_OUT, cv.round, pid, marks)


@triton.jit
def fence_kernel(view, heap_bases, error, ticks, ABORT: tl.constexpr):
    """
    The round fence of next_round(): tell every peer this rank is done with the round, then wait up
    to ``ticks`` until every peer has said the same, recording a peer that has not. One program, one
    warp.

    With ``ABORT`` (this rank could not prepare the next round), or once this rank's plan holds an
    error, it also marks the next round failed on every peer, so their wait() for it fails at once
    instead of timing out on this rank. Such a rank still waits for its peers, so its next producer
    cannot overwrite a peer that is still reading. A rank that learned of a peer's failure
    (``PEER_FAILED``) waits without marking, so that peers keep naming the rank that failed.

    The counters belong to the plan, so a fence that times out leaves no other barrier out of step.
    A peer it times out on is marked failed for the next round, since this rank's next copies may
    land while that peer still reads; nor is that peer waited for again, so a rank that has gone
    away costs one timeout, not one per round.
    """
    cv = TriggeredView.initialize(view)
    held = tl.load(error, volatile=True)
    mark = ((held != 0) & (held != PEER_FAILED)) | ABORT
    world_size = tl.load(view + H_WORLD_SIZE)
    fence = _fence_words(cv)
    abort = fence + world_size
    for peer in range(world_size):
        if peer != cv.rank:
            if mark:
                _mark_failed(__translate(abort, cv.rank, peer, heap_bases), cv.round + 1, cv.rank)
            # Release orders the abort above first; earlier work on this stream finished before the kernel began
            tl.atomic_add(__translate(fence + cv.rank, cv.rank, peer, heap_bases), 1, sem="release", scope="sys")
    t0 = read_realtime()
    for peer in range(world_size):
        # Behind by a round: this rank timed out on it at an earlier fence
        if (peer != cv.rank) & (_load(fence + peer) >= cv.round - 1):
            while (_load(fence + peer) < cv.round) & (read_realtime() - t0 < ticks):
                pass
            if _load(fence + peer) < cv.round:
                _record_error(error, FENCE_TIMED_OUT, cv.round, peer, 0)
                # This rank's next copies may land while that peer still reads: fail its next round
                _mark_failed(__translate(abort, cv.rank, peer, heap_bases), cv.round + 1, cv.rank)
