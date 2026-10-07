# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.

"""
Triggered SDMA plans: the host arms copy-engine chains, a kernel releases them.

Lifecycle:

    plan = TriggeredPlan(ctx, buffer, schedules)    # collective: allocate gates, arm epoch 1
    view = plan.begin_device_epoch()
    producer[grid](..., view)                       # TriggeredView.publish(batch)
    plan.end_device_epoch()                         # every batch complete, or TimeoutError
    plan.next_epoch()                               # barrier -> re-arm -> barrier
    plan.destroy()

All of a rank's chains to one peer share that peer's host-initiated copy-engine queue, and run in
the order they were armed. Plans armed on the same queues must therefore be released in the order
they were armed, and nothing else may use those queues while chains are armed on them.
"""

import enum
import hashlib
import time
from collections import defaultdict

import numpy as np
import torch

from iris.host.distributed.helpers import distributed_allgather
from iris.host.memory.allocators.torch_allocator import TorchAllocator
from iris.mem.triton import triggered as dv
from iris.triggered import armer, kernels
from iris.triggered.schedule import validate

UINT32_MAX = (1 << 32) - 1
_PROGRESS_BUDGET = 100_000


class State(enum.Enum):
    ARMED = "armed"
    ACTIVE = "active"
    COMPLETE = "complete"
    FAILED = "failed"
    ABORTED = "aborted"
    DESTROYED = "destroyed"


class _QueueLedger:
    """What is armed on each of this process's host queues, in arming order."""

    def __init__(self, capacity):
        self.capacity = capacity
        self.entries = defaultdict(list)  # peer -> [(plan, epoch, nbytes)]

    def reserve(self, plan, epoch, nbytes):
        for peer, n in nbytes.items():
            used = sum(e[2] for e in self.entries[peer])
            if used + n > self.capacity:
                raise RuntimeError(
                    f"arming {n} bytes on the queue to rank {peer} would exceed {self.capacity} armed bytes "
                    f"({used} already armed); complete or destroy other plans first"
                )
        for peer, n in nbytes.items():
            self.entries[peer].append((plan, epoch, n))

    def release(self, plan):
        for peer in self.entries:
            self.entries[peer] = [e for e in self.entries[peer] if e[0] is not plan]

    def check_order(self, plan, allowed, action):
        for peer, entries in self.entries.items():
            for other, epoch, _ in entries:
                if other is plan:
                    break
                if other.state not in allowed:
                    raise RuntimeError(
                        f"plan armed earlier on the queue to rank {peer} (epoch {epoch}) is {other.state.value}; "
                        f"plans sharing queues must {action} in the order they were armed"
                    )

    def check_idle(self, peer):
        if self.entries.get(peer):
            raise RuntimeError(
                f"triggered chains are armed on the copy-engine queue to rank {peer}; "
                "complete, abort or destroy those plans before using it"
            )


def ledger(ctx):
    if getattr(ctx, "_triggered_ledger", None) is None:
        from xio import sdma_ep

        # Leave half the ring for wrap padding and work queued before the plan
        ctx._triggered_ledger = _QueueLedger(sdma_ep.SDMA_QUEUE_SIZE // 2)
    return ctx._triggered_ledger


def _digest(buffer_bytes, schedules):
    text = repr((buffer_bytes, schedules)).encode()
    return np.frombuffer(hashlib.sha1(text).digest()[:8], dtype=np.int64).copy()


def view_words(rank, world_size, gates_ptr, schedules):
    """The device view (see ``iris.mem.triton.triggered``) as a list of int64 words."""
    num_batches = sum(len(s) for s in schedules)
    words = [0] * (dv.HEADER_WORDS.value + num_batches * dv.RECORD_WORDS.value)
    words[dv.H_VERSION.value] = dv.ABI_VERSION
    words[dv.H_NUM_BATCHES.value] = num_batches
    words[dv.H_RANK.value] = rank
    words[dv.H_GATES.value] = gates_ptr
    words[dv.H_WORLD_SIZE.value] = world_size
    rec = dv.HEADER_WORDS.value
    for owner, schedule in enumerate(schedules):
        for batch in schedule:
            words[rec + dv.R_ROLE.value] = dv.ROLE_PUBLISH.value if owner == rank else dv.ROLE_CONSUME.value
            words[rec + dv.R_CONTRIBUTORS.value] = batch.contributors
            words[rec + dv.R_OWNER.value] = owner
            rec += dv.RECORD_WORDS.value
    return words


class TriggeredPlan:
    """
    Copies ``buffer`` regions between ranks with pre-armed copy-engine chains.

    Rank ``r`` owns the batches in ``schedules[r]`` and sends each of them to every other rank, at
    the same offsets in the peer's ``buffer``. Batch ids are global: rank ``r``'s batches come after
    those of ranks ``0..r-1`` (see ``batch_id``).

    Collective: every rank passes the same ``schedules`` for the same symmetric ``buffer``.

    Args:
        ctx: Iris instance.
        buffer: Contiguous tensor on the symmetric heap. The heap must keep peer mappings stable while
            chains are armed, which rules out the default torch allocator: use ``allocator_type="vmem_chunked"`` (or ``"vmem"``).
        schedules: One list of ``Batch`` per rank, each in the order that rank completes them.
        timeout: Seconds ``end_device_epoch()`` waits before raising.
    """

    def __init__(self, ctx, buffer, schedules, timeout=30.0):
        self.ctx = ctx
        self.rank = ctx.get_rank()
        self.world_size = ctx.get_num_ranks()
        self.timeout = timeout
        self.state = None

        # Armed chains hold peer addresses; the torch allocator re-maps peer heaps on every allocation
        if isinstance(ctx.heap.allocator, TorchAllocator):
            raise ValueError(
                "triggered plans need stable peer mappings: create the Iris context with allocator_type='vmem_chunked' (or 'vmem')"
            )
        if not ctx.heap.is_symmetric(buffer) or not buffer.is_contiguous():
            raise ValueError("buffer must be a contiguous tensor on the symmetric heap")
        buffer_bytes = buffer.numel() * buffer.element_size()
        if len(schedules) != self.world_size:
            raise ValueError(f"need one schedule per rank ({self.world_size}), got {len(schedules)}")
        self.schedules = [list(validate(s, buffer_bytes)) for s in schedules]
        digests = distributed_allgather(_digest(buffer_bytes, self.schedules))
        if not (digests == digests[0]).all():
            raise ValueError("ranks passed different buffers sizes or schedules")

        self._first = np.cumsum([0] + [len(s) for s in self.schedules]).tolist()
        self.num_batches = self._first[-1]
        if self.num_batches == 0:
            raise ValueError("schedules contain no batches")

        self.buffer = buffer
        self.gates = ctx.zeros(self.num_batches, dv.GATE_WORDS.value, dtype=torch.int64)
        words = view_words(self.rank, self.world_size, self.gates.data_ptr(), self.schedules)
        self.view = torch.tensor(words, dtype=torch.int64, device=buffer.device)
        self._own = torch.arange(
            self._first[self.rank], self._first[self.rank + 1], dtype=torch.int64, device=buffer.device
        )
        self._status = torch.zeros(self.num_batches, dtype=torch.int32, device=buffer.device)
        self._ledger = ledger(ctx)
        self.epoch = 0
        self._arm(1)
        ctx.barrier()

    def batch_id(self, rank, index):
        """Global id of batch ``index`` in ``schedules[rank]``."""
        return self._first[rank] + index

    def _arm(self, epoch):
        if epoch > UINT32_MAX:
            raise RuntimeError("epoch exceeds 32 bits; destroy the plan and create a new one")
        peers = [p for p in range(self.world_size) if p != self.rank]
        own = self.schedules[self.rank]
        nbytes = sum(armer.chain_bytes(b) for b in own)
        self._ledger.reserve(self, epoch, {p: nbytes for p in peers})
        self.epoch = epoch
        self.state = State.ARMED

        heap = self.ctx.heap
        self._heap_bases = heap.heap_bases_cpu.copy()
        base = self.buffer.data_ptr()
        gate_row = self.gates.element_size() * dv.GATE_WORDS.value
        try:
            for i, batch in enumerate(own):
                gate = self.gates.data_ptr() + self.batch_id(self.rank, i) * gate_row
                for peer in peers:
                    copies = [
                        armer.Copy(
                            base + s.offset,
                            heap.translate(base + s.offset, self.rank, peer),
                            s.width,
                            s.height,
                            s.row_pitch,
                        )
                        for s in batch.segments
                    ]
                    arrival = heap.translate(gate + 8 * dv.G_ARRIVAL.value, self.rank, peer)
                    armer.arm_chain(self.rank, peer, gate, self.epoch, copies, arrival, batch.is_2d)
        except Exception:
            # Chains armed so far stay parked: abort() releases them
            self.state = State.FAILED
            raise

    def _require(self, *states):
        if self.state not in states:
            names = " or ".join(s.value for s in states)
            raise RuntimeError(f"plan is {self.state.value if self.state else 'uninitialized'}, needs {names}")

    def begin_device_epoch(self):
        """Start the armed epoch and return the view tensor its kernels take."""
        self._require(State.ARMED)
        if not np.array_equal(self.ctx.heap.heap_bases_cpu, self._heap_bases):
            raise RuntimeError("symmetric heap was re-mapped since the plan was armed")
        self._ledger.check_order(self, (State.ACTIVE,), "begin")
        self.view[dv.H_EPOCH.value] = self.epoch
        self.state = State.ACTIVE
        return self.view

    def _wait(self, own_only, timeout):
        deadline = time.monotonic() + (self.timeout if timeout is None else timeout)
        while True:
            kernels.progress_kernel[(self.num_batches,)](
                self.view,
                self.ctx.get_heap_bases(),
                self._status,
                BUDGET=_PROGRESS_BUDGET,
                WORLD_SIZE=self.world_size,
                OWN_ONLY=own_only,
            )
            status = self._status.cpu()
            mine = slice(self._first[self.rank], self._first[self.rank + 1])
            pending = (status[mine] if own_only else status) == 0
            if not pending.any():
                return
            if time.monotonic() > deadline:
                missing = (pending.nonzero().flatten() + (mine.start if own_only else 0)).tolist()
                raise TimeoutError(f"epoch {self.epoch}: batches {missing[:16]} not complete")

    def end_device_epoch(self, timeout=None):
        """Wait until every batch is complete this epoch: own ones arrived at every peer, others here."""
        self._require(State.ACTIVE)
        try:
            self._wait(own_only=False, timeout=timeout)
        except TimeoutError:
            self.state = State.FAILED
            raise
        self._ledger.release(self)
        self.state = State.COMPLETE

    def next_epoch(self):
        """Collective: re-arm every chain for the next epoch."""
        self._require(State.COMPLETE)
        self.ctx.barrier()
        self._arm(self.epoch + 1)
        self.ctx.barrier()

    def run(self, timeout=None):
        """Run one epoch with no producer kernel: publish every own batch as it stands, then wait."""
        self.begin_device_epoch()
        if len(self._own):
            kernels.release_kernel[(len(self._own),)](self.view, self._own)
        self.end_device_epoch(timeout)

    def abort(self, timeout=None):
        """
        Release this rank's armed chains and wait for them to drain. The plan can only be destroyed
        afterwards. Peers' chains to this rank are theirs to abort. Plans sharing queues must be
        aborted in the order they were armed: later chains wait behind earlier ones.
        """
        if self.state not in (State.ARMED, State.ACTIVE, State.FAILED):
            return
        self._ledger.check_order(self, (State.ACTIVE, State.ABORTED), "abort")
        self.view[dv.H_EPOCH.value] = self.epoch
        if len(self._own):
            kernels.release_kernel[(len(self._own),)](self.view, self._own)
        self.state = State.ABORTED
        self._wait(own_only=True, timeout=timeout)
        self._ledger.release(self)

    def destroy(self):
        """Forget the plan. Its chains must have drained: abort() an armed or failed plan first."""
        if self.state in (State.ARMED, State.ACTIVE, State.FAILED):
            raise RuntimeError(f"plan is {self.state.value}; abort() it before destroying")
        self._ledger.release(self)
        self.state = State.DESTROYED
        self.gates = self.view = self.buffer = None
