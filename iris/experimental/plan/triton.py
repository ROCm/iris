# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.

"""
Device side of triggered plans (see ``iris.experimental.plan``).

For each (chunk, peer), the host arms a chain on the copy engine: wait until the chunk's counter
reaches ``writers * round``, copy the chunk's regions to the peer, then add 1 to the chunk's arrival
counter on the peer. After a round's last chain to a peer, the copy engine adds 1 to this rank's
completion counter for that peer. Kernels add to the counter with ``mark_ready``, observe arrivals
with ``arrived`` / ``wait`` and completion with ``sent`` / ``wait_sent``.

View layout (int64 words): an 8-word header, then one 2-word record per chunk. Gates (int32, one
symmetric tensor): ``num_chunks`` counters, then ``num_chunks`` arrivals, then one completion per rank.
"""

import triton
import triton.language as tl
from triton.language.core import _aggregate as aggregate

from iris.mem.utils import wait_cnt

ABI_VERSION = 2

# Header words
HEADER_WORDS = tl.constexpr(8)
H_VERSION = tl.constexpr(0)
H_NUM_CHUNKS = tl.constexpr(1)
H_RANK = tl.constexpr(2)
H_ROUND = tl.constexpr(3)
H_GATES = tl.constexpr(4)
H_WORLD_SIZE = tl.constexpr(5)
H_SENDS_TO = tl.constexpr(6)  # bitmask of the peers this rank's chains go to

# Chunk record words
RECORD_WORDS = tl.constexpr(2)
R_WRITERS = tl.constexpr(0)
R_OWNER = tl.constexpr(1)


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
    num_chunks: tl.tensor

    @triton.constexpr_function
    def __init__(self, view, gates, round, rank, num_chunks):
        self.view = view
        self.gates = gates
        self.round = round
        self.rank = rank
        self.num_chunks = num_chunks

    @staticmethod
    @triton.jit
    def initialize(handle):
        """Read the plan's handle tensor (from ``TriggeredPlan.start()``)."""
        gates = tl.cast(tl.load(handle + H_GATES), tl.pointer_type(tl.int32))
        return TriggeredView(
            handle, gates, tl.load(handle + H_ROUND), tl.load(handle + H_RANK), tl.load(handle + H_NUM_CHUNKS)
        )

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
        Declare this CTA's writes to chunk ``chunk`` done; the chunk is sent once all of its writers have.

        Call from the whole CTA, outside divergent branches, after the CTA's writes to the chunk.
        """
        # Copy engines read behind L2: every wave's stores must land before the release writes back
        wait_cnt()
        tl.debug_barrier()
        # Each writer's release writes back its own L2 before its increment, so the increment that
        # reaches writers * round follows every writer's data
        tl.atomic_add(self.gates + chunk, 1, sem="release", scope="sys")

    @triton.jit
    def arrived(self, chunk):
        """
        True once chunk ``chunk`` is visible here this round (marked ready, for this rank's own).

        Call uniformly across the CTA: the result reaches the other waves through a CTA barrier, so
        every wave must make the same number of calls (true of ``wait``, ``sent`` and ``wait_sent`` too).
        """
        own = self.owns(chunk)
        word = tl.where(own, chunk, self.num_chunks + chunk)
        target = tl.where(own, self._record(chunk, R_WRITERS) * self.round, self.round)
        return tl.atomic_add(self.gates + word, 0, sem="acquire", scope="sys") >= target

    @triton.jit
    def wait(self, chunk):
        """Spin until ``arrived(chunk)``."""
        while self.arrived(chunk) == 0:
            pass

    @triton.jit
    def sent(self):
        """True once every peer has received this rank's chunks this round, so they may be overwritten."""
        sends_to = tl.load(self.view + H_SENDS_TO)
        pending = 0
        for peer in range(tl.load(self.view + H_WORLD_SIZE)):
            if ((sends_to >> peer) & 1) != 0:
                completion = self.gates + 2 * self.num_chunks + peer
                pending += (tl.atomic_add(completion, 0, sem="acquire", scope="sys") < self.round).to(tl.int32)
        return pending == 0

    @triton.jit
    def wait_sent(self):
        """Spin until ``sent()``."""
        while self.sent() == 0:
            pass
