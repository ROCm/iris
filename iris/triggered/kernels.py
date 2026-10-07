# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.

"""Kernels a triggered plan launches itself: releasing its own batches and checking completion."""

import triton
import triton.language as tl

from iris.mem.triton.context import __translate
from iris.mem.triton.triggered import (
    G_ARRIVAL,
    G_COMPLETION,
    G_READY,
    GATE_WORDS,
    HEADER_WORDS,
    R_OWNER,
    RECORD_WORDS,
    TriggeredView,
)


@triton.jit
def release_kernel(view, batches):
    """Open the gate of batch ``batches[pid]``, as its last contributor would."""
    tv = TriggeredView.initialize(view)
    batch = tl.load(batches + tl.program_id(0))
    tl.atomic_xchg(tv.gates + batch * GATE_WORDS + G_READY, tv.epoch, sem="release", scope="sys")


@triton.jit
def _reached(ptr, epoch, budget: tl.constexpr):
    i = 0
    while (tl.atomic_add(ptr, 0, sem="acquire", scope="sys") < epoch) & (i < budget):
        i += 1
    return tl.atomic_add(ptr, 0, sem="acquire", scope="sys") >= epoch


@triton.jit
def progress_kernel(view, heap_bases, status, BUDGET: tl.constexpr, WORLD_SIZE: tl.constexpr, OWN_ONLY: tl.constexpr):
    """
    Mark batch ``pid`` complete once it is done this epoch: for this rank's own batches, published
    and arrived at every peer (each chain ends with that signal, so it has fully run); for the
    others, arrived here. Writes 1 to ``status[pid]`` when complete, 0 if ``BUDGET`` ran out.
    """
    batch = tl.program_id(0)
    tv = TriggeredView.initialize(view)
    gate = tv.gates + batch * GATE_WORDS
    own = tl.load(view + HEADER_WORDS + batch * RECORD_WORDS + R_OWNER) == tv.rank
    done = tl.full((), True, tl.int1)
    if own:
        done = _reached(gate + G_READY, tv.epoch, BUDGET)
        for peer in tl.static_range(WORLD_SIZE):
            if done & (peer != tv.rank):
                done = _reached(__translate(gate + G_ARRIVAL, tv.rank, peer, heap_bases), tv.epoch, BUDGET)
    elif not OWN_ONLY:
        done = _reached(gate + G_ARRIVAL, tv.epoch, BUDGET)
    if done:
        tl.atomic_xchg(gate + G_COMPLETION, tv.epoch, sem="release", scope="sys")
    tl.store(status + batch, done.to(tl.int32))
