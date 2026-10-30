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
    H_GATES,
    H_NUM_CHUNKS,
    H_RANK,
    H_ROUND,
    H_SENDS_TO,
    H_WORLD_SIZE,
    HEADER_WORDS,
    R_ARRIVAL,
    R_EXPECTED,
    R_OWNER,
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
    num_chunks: gl.tensor

    @gluon.constexpr_function
    def __init__(self, view, gates, round, rank, num_chunks):
        self.view = view
        self.gates = gates
        self.round = round
        self.rank = rank
        self.num_chunks = num_chunks

    @staticmethod
    @gluon.jit
    def initialize(handle):
        """Read the plan's handle tensor (from ``TriggeredPlan.start()``)."""
        gates = gl.cast(gl.load(handle + H_GATES), gl.pointer_type(gl.int32))
        return TriggeredView(
            handle, gates, gl.load(handle + H_ROUND), gl.load(handle + H_RANK), gl.load(handle + H_NUM_CHUNKS)
        )

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
        Declare this CTA's writes to chunk ``chunk`` done; the chunk is sent once all of its writers have.

        Call from the whole CTA, after the CTA's writes to the chunk: outside divergent branches and
        outside ``gl.warp_specialize`` partitions.
        """
        # Copy engines read behind L2: every wave's stores must land before the release writes back
        wait_cnt()
        gl.barrier()
        # Each writer's release writes back its own L2 before its increment, so the increment that
        # reaches writers * round follows every writer's data
        gl.atomic_add(self.gates + chunk, 1, sem="release", scope="sys")

    @gluon.jit
    def arrived(self, chunk):
        """
        True once chunk ``chunk`` is visible here this round: for this rank's own chunks, once marked
        ready; for another rank's chunk in a group, once the whole group has arrived here. Always
        true of chunks not sent to this rank.

        Call uniformly across the CTA: the result reaches the other waves through a CTA barrier, so
        every wave must make the same number of calls (true of the other waits too).
        """
        word = self._record(chunk, R_ARRIVAL)
        target = self._record(chunk, R_EXPECTED) * self.round
        return gl.atomic_add(self.gates + word, 0, sem="acquire", scope="sys") >= target

    @gluon.jit
    def wait(self, chunk):
        """Spin until ``arrived(chunk)``."""
        while self.arrived(chunk) == 0:
            pass

    @gluon.jit
    def group_arrived(self, group):
        """True once every chunk of ``group`` sent to this rank has arrived this round."""
        expected = gl.load(self.view + HEADER_WORDS + self.num_chunks * RECORD_WORDS + group)
        word = 2 * self.num_chunks + gl.load(self.view + H_WORLD_SIZE) + group
        return gl.atomic_add(self.gates + word, 0, sem="acquire", scope="sys") >= expected * self.round

    @gluon.jit
    def wait_group(self, group):
        """Spin until ``group_arrived(group)``."""
        while self.group_arrived(group) == 0:
            pass

    @gluon.jit
    def sent(self):
        """True once every peer has received this rank's chunks this round, so they may be overwritten."""
        sends_to = gl.load(self.view + H_SENDS_TO)
        pending = 0
        for peer in range(gl.load(self.view + H_WORLD_SIZE)):
            if ((sends_to >> peer) & 1) != 0:
                completion = self.gates + 2 * self.num_chunks + peer
                pending += (gl.atomic_add(completion, 0, sem="acquire", scope="sys") < self.round).to(gl.int32)
        return pending == 0

    @gluon.jit
    def wait_sent(self):
        """Spin until ``sent()``."""
        while self.sent() == 0:
            pass
