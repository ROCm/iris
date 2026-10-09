# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.

"""
Device side of triggered SDMA plans (see ``iris.triggered``).

The host arms one chain per (batch, peer) on the copy engine: poll the batch's ``ready`` gate,
copy the batch's segments to the peer, then add 1 to the batch's ``arrival`` gate on the peer.
A kernel opens the gate with ``publish`` and observes arrivals with ``arrived`` / ``wait``.

View layout (int64 words): an 8-word header, then one 4-word record per batch. Gate table: one
64-byte row per batch.
"""

import triton
import triton.language as tl
from triton.language.core import _aggregate as aggregate

from iris.mem.utils import wait_cnt

ABI_VERSION = 1

# Header words
HEADER_WORDS = tl.constexpr(8)
H_VERSION = tl.constexpr(0)
H_NUM_BATCHES = tl.constexpr(1)
H_RANK = tl.constexpr(2)
H_EPOCH = tl.constexpr(3)
H_GATES = tl.constexpr(4)
H_WORLD_SIZE = tl.constexpr(5)

# Batch record words
RECORD_WORDS = tl.constexpr(4)
R_ROLE = tl.constexpr(0)
R_CONTRIBUTORS = tl.constexpr(1)
R_OWNER = tl.constexpr(2)

ROLE_PUBLISH = tl.constexpr(1)
ROLE_CONSUME = tl.constexpr(2)

# Gate row words
GATE_WORDS = tl.constexpr(8)
G_READY = tl.constexpr(0)
G_ARRIVAL = tl.constexpr(1)
G_COMPLETION = tl.constexpr(2)
G_ERROR = tl.constexpr(3)
G_COUNTER = tl.constexpr(4)


@aggregate
class TriggeredView:
    """
    Device handle for one epoch of a triggered plan.

    Usage::

        @triton.jit
        def producer(out, view, ...):
            tv = TriggeredView.initialize(view)
            ...                          # write this program's part of batch b
            tv.publish(b)                # whole CTA, after its writes

        @triton.jit
        def consumer(out, view, ...):
            tv = TriggeredView.initialize(view)
            tv.wait(b)                   # batch b's data is now visible
            x = tl.load(out + ...)
    """

    view: tl.tensor
    gates: tl.tensor
    epoch: tl.tensor
    rank: tl.tensor

    @triton.constexpr_function
    def __init__(self, view, gates, epoch, rank):
        self.view = view
        self.gates = gates
        self.epoch = epoch
        self.rank = rank

    @staticmethod
    @triton.jit
    def initialize(view):
        """Read the plan's view tensor (from ``TriggeredPlan.begin_device_epoch()``)."""
        gates = tl.cast(tl.load(view + H_GATES), tl.pointer_type(tl.int64))
        return TriggeredView(view, gates, tl.load(view + H_EPOCH), tl.load(view + H_RANK))

    @triton.jit
    def _record(self, batch, word):
        return tl.load(self.view + HEADER_WORDS + batch * RECORD_WORDS + word)

    @triton.jit
    def can_publish(self, batch):
        return (self._record(batch, R_ROLE) & ROLE_PUBLISH) != 0

    @triton.jit
    def can_consume(self, batch):
        return (self._record(batch, R_ROLE) & ROLE_CONSUME) != 0

    @triton.jit
    def publish(self, batch):
        """
        Contribute to batch ``batch``; the last of its contributors opens the gate.

        Call from the whole CTA, outside divergent branches, after the CTA's writes to the batch.
        """
        # Copy engines read behind L2: every wave's stores must land before the release writes back
        wait_cnt()
        tl.debug_barrier()
        gate = self.gates + batch * GATE_WORDS
        contributors = self._record(batch, R_CONTRIBUTORS)
        ticket = tl.atomic_add(gate + G_COUNTER, 1, sem="acq_rel", scope="sys")
        if ticket % contributors == contributors - 1:
            tl.atomic_xchg(gate + G_READY, self.epoch, sem="release", scope="sys")

    @triton.jit
    def arrived(self, batch):
        """
        True once batch ``batch`` is visible here this epoch (published, for this rank's own).

        Call uniformly across the CTA: the result reaches the other waves through a CTA barrier, so
        every wave must make the same number of calls (true of ``wait`` and ``reusable`` too).
        """
        gate = self.gates + batch * GATE_WORDS
        word = tl.where(self._record(batch, R_OWNER) == self.rank, G_READY, G_ARRIVAL)
        return tl.atomic_add(gate + word, 0, sem="acquire", scope="sys") >= self.epoch

    @triton.jit
    def wait(self, batch):
        """Spin until ``arrived(batch)``."""
        while self.arrived(batch) == 0:
            pass

    @triton.jit
    def reusable(self, batch):
        """True once batch ``batch`` is complete this epoch (set by ``end_device_epoch()``)."""
        gate = self.gates + batch * GATE_WORDS
        return tl.atomic_add(gate + G_COMPLETION, 0, sem="acquire", scope="sys") >= self.epoch
