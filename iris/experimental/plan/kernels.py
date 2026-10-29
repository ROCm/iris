# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.

"""Kernels a triggered plan launches itself: sending its own chunks and checking completion."""

import triton
import triton.language as tl

from iris.experimental.plan.triton import HEADER_WORDS, R_WRITERS, RECORD_WORDS, TriggeredView


@triton.jit
def release_kernel(view, chunks):
    """Open the gate of chunk ``chunks[pid]`` this round, however many of its writers have marked it."""
    cv = TriggeredView.initialize(view)
    chunk = tl.load(chunks + tl.program_id(0))
    writers = tl.load(view + HEADER_WORDS + chunk * RECORD_WORDS + R_WRITERS)
    tl.atomic_max(cv.gates + chunk, (writers * cv.round).to(tl.int32), sem="release", scope="sys")


@triton.jit
def _reached(ptr, round, budget: tl.constexpr):
    i = 0
    while (tl.atomic_add(ptr, 0, sem="acquire", scope="sys") < round) & (i < budget):
        i += 1
    return tl.atomic_add(ptr, 0, sem="acquire", scope="sys") >= round


@triton.jit
def wait_kernel(view, sent_to, status, BUDGET: tl.constexpr, OWN_ONLY: tl.constexpr):
    """
    Program ``pid < num_chunks``: chunk ``pid`` has arrived here this round (always true of this
    rank's own chunks, and of every chunk if ``OWN_ONLY``). Program ``num_chunks + p``: the queue to
    peer ``p`` has finished this round (always true if bit ``p`` of ``sent_to`` is clear). Writes 1 to
    ``status[pid]`` when done, 0 if ``BUDGET`` ran out. Reads only this rank's gates.
    """
    pid = tl.program_id(0)
    cv = TriggeredView.initialize(view)
    done = tl.full((), True, tl.int1)
    if pid < cv.num_chunks:
        if not OWN_ONLY:
            if cv.owns(pid) == 0:
                done = _reached(cv.gates + cv.num_chunks + pid, cv.round, BUDGET)
    else:
        peer = pid - cv.num_chunks
        if ((sent_to >> peer) & 1) != 0:
            done = _reached(cv.gates + 2 * cv.num_chunks + peer, cv.round, BUDGET)
    tl.store(status + pid, done.to(tl.int32))
