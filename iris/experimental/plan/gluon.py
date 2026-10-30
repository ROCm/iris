# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.

"""
Gluon device side of triggered plans (see ``iris.experimental.plan``).

Same handle and gate layout as ``iris.experimental.plan.triton``; only the kernel language differs.
"""

from triton.experimental import gluon
from triton.experimental.gluon import language as gl
from triton.language.core import _aggregate as aggregate

from iris.experimental.plan.triton import (
    G_ARRIVAL,
    G_COUNTER,
    G_READY,
    GATE_WORDS,
    H_GATES,
    H_RANK,
    H_ROUND,
    HEADER_WORDS,
    R_OWNER,
    R_WRITERS,
    RECORD_WORDS,
)
from iris.mem.utils import wait_cnt


@aggregate
class TriggeredView:
    """
    Gluon device handle for one round of a triggered plan.

    Usage::

        @gluon.jit
        def producer(out, handle, ...):
            cv = TriggeredView.initialize(handle)
            ...                          # write this program's part of chunk c
            cv.mark_ready(c)             # whole CTA, after its writes

        @gluon.jit
        def consumer(out, handle, ...):
            cv = TriggeredView.initialize(handle)
            cv.wait(c)                   # chunk c's data is now visible
            x = gl.load(out + ...)
    """

    view: gl.tensor
    gates: gl.tensor
    round: gl.tensor
    rank: gl.tensor

    @gluon.constexpr_function
    def __init__(self, view, gates, round, rank):
        self.view = view
        self.gates = gates
        self.round = round
        self.rank = rank

    @staticmethod
    @gluon.jit
    def initialize(handle):
        """Read the plan's handle tensor (from ``TriggeredPlan.start()``)."""
        gates = gl.cast(gl.load(handle + H_GATES), gl.pointer_type(gl.int64))
        return TriggeredView(handle, gates, gl.load(handle + H_ROUND), gl.load(handle + H_RANK))

    @gluon.jit
    def _record(self, chunk, word):
        return gl.load(self.view + HEADER_WORDS + chunk * RECORD_WORDS + word)

    @gluon.jit
    def owns(self, chunk):
        """True if this rank produces chunk ``chunk``."""
        return self._record(chunk, R_OWNER) == self.rank

    @gluon.jit
    def mark_ready(self, chunk):
        """
        Declare this CTA's writes to chunk ``chunk`` done; the last of its writers sends it.

        Call from the whole CTA, after the CTA's writes to the chunk: outside divergent branches and
        outside ``gl.warp_specialize`` partitions.
        """
        # Copy engines read behind L2: every wave's stores must land before the release writes back
        wait_cnt()
        gl.barrier()
        gate = self.gates + chunk * GATE_WORDS
        writers = self._record(chunk, R_WRITERS)
        # Each writer's release writes back its own L2; the last writer's ticket follows all of them
        ticket = gl.atomic_add(gate + G_COUNTER, 1, sem="release", scope="sys")
        if ticket % writers == writers - 1:
            gl.atomic_xchg(gate + G_READY, self.round, sem="relaxed", scope="sys")

    @gluon.jit
    def arrived(self, chunk):
        """
        True once chunk ``chunk`` is visible here this round (marked ready, for this rank's own).

        Call uniformly across the CTA: the result reaches the other waves through a CTA barrier, so
        every wave must make the same number of calls (true of ``wait`` too).
        """
        gate = self.gates + chunk * GATE_WORDS
        word = gl.where(self.owns(chunk), G_READY, G_ARRIVAL)
        return gl.atomic_add(gate + word, 0, sem="acquire", scope="sys") >= self.round

    @gluon.jit
    def wait(self, chunk):
        """Spin until ``arrived(chunk)``."""
        while self.arrived(chunk) == 0:
            pass
