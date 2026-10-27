# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.

"""
Device side of triggered plans (see ``iris.experimental.plan``).

For each (chunk, peer), the host arms a chain on the copy engine: poll the chunk's ``ready`` gate,
copy the chunk's regions to the peer, then add 1 to the chunk's ``arrival`` gate on the peer.
A kernel opens the gate with ``mark_ready`` and observes arrivals with ``arrived`` / ``wait``.

View layout (int64 words): an 8-word header, then one 2-word record per chunk. Gate table: one
64-byte row per chunk.
"""

import triton
import triton.language as tl
from triton.language.core import _aggregate as aggregate

from iris.mem.utils import wait_cnt

ABI_VERSION = 1

# Header words
HEADER_WORDS = tl.constexpr(8)
H_VERSION = tl.constexpr(0)
H_NUM_CHUNKS = tl.constexpr(1)
H_RANK = tl.constexpr(2)
H_ROUND = tl.constexpr(3)
H_GATES = tl.constexpr(4)
H_WORLD_SIZE = tl.constexpr(5)

# Chunk record words
RECORD_WORDS = tl.constexpr(2)
R_WRITERS = tl.constexpr(0)
R_OWNER = tl.constexpr(1)

# Gate row words
GATE_WORDS = tl.constexpr(8)
G_READY = tl.constexpr(0)
G_ARRIVAL = tl.constexpr(1)
G_COUNTER = tl.constexpr(2)


@aggregate
class TriggeredView:
    """
    Device handle for one round of a triggered plan.

    Usage::

        @triton.jit
        def producer(out, handle, ...):
            cv = TriggeredView.initialize(handle)
            ...                          # write this program's part of chunk c
            cv.mark_ready(c)             # whole CTA, after its writes

        @triton.jit
        def consumer(out, handle, ...):
            cv = TriggeredView.initialize(handle)
            cv.wait(c)                   # chunk c's data is now visible
            x = tl.load(out + ...)
    """

    view: tl.tensor
    gates: tl.tensor
    round: tl.tensor
    rank: tl.tensor

    @triton.constexpr_function
    def __init__(self, view, gates, round, rank):
        self.view = view
        self.gates = gates
        self.round = round
        self.rank = rank

    @staticmethod
    @triton.jit
    def initialize(handle):
        """Read the plan's handle tensor (from ``TriggeredPlan.start()``)."""
        gates = tl.cast(tl.load(handle + H_GATES), tl.pointer_type(tl.int64))
        return TriggeredView(handle, gates, tl.load(handle + H_ROUND), tl.load(handle + H_RANK))

    @triton.jit
    def _record(self, chunk, word):
        return tl.load(self.view + HEADER_WORDS + chunk * RECORD_WORDS + word)

    @triton.jit
    def owns(self, chunk):
        """True if this rank produces chunk ``chunk``."""
        return self._record(chunk, R_OWNER) == self.rank

    @triton.jit
    def mark_ready(self, chunk):
        """
        Declare this CTA's writes to chunk ``chunk`` done; the last of its writers sends it.

        Call from the whole CTA, outside divergent branches, after the CTA's writes to the chunk.
        """
        # Copy engines read behind L2: every wave's stores must land before the release writes back
        wait_cnt()
        tl.debug_barrier()
        gate = self.gates + chunk * GATE_WORDS
        writers = self._record(chunk, R_WRITERS)
        # Each writer's release writes back its own L2; the last writer's ticket follows all of them
        ticket = tl.atomic_add(gate + G_COUNTER, 1, sem="release", scope="sys")
        if ticket % writers == writers - 1:
            tl.atomic_xchg(gate + G_READY, self.round, sem="relaxed", scope="sys")

    @triton.jit
    def arrived(self, chunk):
        """
        True once chunk ``chunk`` is visible here this round (marked ready, for this rank's own).

        Call uniformly across the CTA: the result reaches the other waves through a CTA barrier, so
        every wave must make the same number of calls (true of ``wait`` too).
        """
        gate = self.gates + chunk * GATE_WORDS
        word = tl.where(self.owns(chunk), G_READY, G_ARRIVAL)
        return tl.atomic_add(gate + word, 0, sem="acquire", scope="sys") >= self.round

    @triton.jit
    def wait(self, chunk):
        """Spin until ``arrived(chunk)``."""
        while self.arrived(chunk) == 0:
            pass
