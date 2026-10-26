# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.

"""Kernels a triggered plan launches itself: sending its own chunks and checking completion."""

import triton
import triton.language as tl

from iris.mem.triton.context import __translate
from iris.experimental.plan.triton import (
    G_ARRIVAL,
    G_COUNTER,
    G_READY,
    GATE_WORDS,
    HEADER_WORDS,
    R_WRITERS,
    RECORD_WORDS,
    TriggeredView,
)


@triton.jit
def release_kernel(view, chunks):
    """Open the gate of chunk ``chunks[pid]``, counting a mark from each of its writers."""
    cv = TriggeredView.initialize(view)
    chunk = tl.load(chunks + tl.program_id(0))
    gate = cv.gates + chunk * GATE_WORDS
    writers = tl.load(view + HEADER_WORDS + chunk * RECORD_WORDS + R_WRITERS)
    tl.atomic_add(gate + G_COUNTER, writers, sem="relaxed", scope="sys")
    tl.atomic_xchg(gate + G_READY, cv.round, sem="release", scope="sys")


@triton.jit
def _reached(ptr, round, budget: tl.constexpr):
    i = 0
    while (tl.atomic_add(ptr, 0, sem="acquire", scope="sys") < round) & (i < budget):
        i += 1
    return tl.atomic_add(ptr, 0, sem="acquire", scope="sys") >= round


@triton.jit
def progress_kernel(
    view, heap_bases, armed, status, BUDGET: tl.constexpr, WORLD_SIZE: tl.constexpr, OWN_ONLY: tl.constexpr
):
    """
    Mark chunk ``pid`` complete once it is done this round: for this rank's own chunks, marked ready
    and arrived at every peer it was armed for (``armed[chunk * WORLD_SIZE + peer]``; each chain
    ends with that signal, so it has fully run); for the others, arrived here. Writes 1 to
    ``status[pid]`` when complete, 0 if ``BUDGET`` ran out.
    """
    chunk = tl.program_id(0)
    cv = TriggeredView.initialize(view)
    gate = cv.gates + chunk * GATE_WORDS
    done = tl.full((), True, tl.int1)
    if cv.owns(chunk):
        done = _reached(gate + G_READY, cv.round, BUDGET)
        for peer in tl.static_range(WORLD_SIZE):
            if done & (peer != cv.rank) & (tl.load(armed + chunk * WORLD_SIZE + peer) != 0):
                done = _reached(__translate(gate + G_ARRIVAL, cv.rank, peer, heap_bases), cv.round, BUDGET)
    elif not OWN_ONLY:
        done = _reached(gate + G_ARRIVAL, cv.round, BUDGET)
    tl.store(status + chunk, done.to(tl.int32))
