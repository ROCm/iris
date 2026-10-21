# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.

"""
Collective plans: the host prepares copy-engine chains, a kernel releases them.

Lifecycle:

    with CollectivePlan(ctx, buffer, schedule) as plan:   # collective: allocate gates, prepare round 1
        for step in range(steps):
            handle = plan.start()
            producer[grid](..., handle)                   # CollectiveView.mark_ready(chunk)
            plan.wait()                                   # every chunk delivered, or TimeoutError
            ...                                           # read this round's data
            plan.next_round()                             # collective: barrier -> prepare -> barrier

All of a rank's chains to one peer share that peer's host-initiated copy-engine queue, and run in
the order they were prepared. Plans sharing queues must therefore start and close in the order they
were prepared, and nothing else may use those queues while copies are prepared on them.
"""

import enum
import time

import numpy as np
import torch
import torch.distributed as dist

from iris.host import queue_ledger
from iris.host.distributed.helpers import distributed_allgather
from iris.host.memory.allocators.torch_allocator import TorchAllocator
from iris.mem.triton import plan as dv
from iris.plan import armer, kernels
from iris.plan.schedule import validate

UINT32_MAX = (1 << 32) - 1
_PROGRESS_BUDGET = 100_000


class State(enum.Enum):
    PREPARED = "prepared"
    RUNNING = "running"
    DONE = "done"
    FAILED = "failed"
    CLOSING = "closing"
    CLOSED = "closed"


def view_words(rank, world_size, gates_ptr, schedules):
    """The device view (see ``iris.mem.triton.plan``) as a list of int64 words."""
    num_chunks = sum(len(s) for s in schedules)
    words = [0] * (dv.HEADER_WORDS.value + num_chunks * dv.RECORD_WORDS.value)
    words[dv.H_VERSION.value] = dv.ABI_VERSION
    words[dv.H_NUM_CHUNKS.value] = num_chunks
    words[dv.H_RANK.value] = rank
    words[dv.H_GATES.value] = gates_ptr
    words[dv.H_WORLD_SIZE.value] = world_size
    rec = dv.HEADER_WORDS.value
    for owner, schedule in enumerate(schedules):
        for chunk in schedule:
            words[rec + dv.R_WRITERS.value] = chunk.writers
            words[rec + dv.R_OWNER.value] = owner
            rec += dv.RECORD_WORDS.value
    return words


class CollectivePlan:
    """
    Sends regions of ``buffer`` from the rank that writes them to every other rank, using copy
    engines that a kernel releases chunk by chunk.

    Each rank passes its own ``schedule``: the chunks it writes, in the order its producer completes
    them. Every chunk is copied to the same offsets in each peer's ``buffer``. Chunk ids are global:
    rank ``r``'s chunks come after those of ranks ``0..r-1`` (see ``chunk_id``), and ``schedules[r]``
    holds rank ``r``'s schedule. A rank that writes nothing passes ``[]``.

    Collective: every rank creates the plan for the same symmetric ``buffer``. Use it as a context
    manager, or call ``close()`` on every rank when done.

    Args:
        ctx: Iris instance, created with ``engines=("sdma",)``.
        buffer: Contiguous tensor on the symmetric heap. The heap must keep peer mappings stable while
            copies are prepared, which rules out the default torch allocator: use ``allocator_type="vmem_chunked"`` (or ``"vmem"``).
        schedule: This rank's list of ``Chunk``, in the order it completes them.
        timeout: Seconds ``wait()`` and ``close()`` wait before raising.
    """

    def __init__(self, ctx, buffer, schedule, timeout=30.0):
        self.ctx = ctx
        self.rank = ctx.get_rank()
        self.world_size = ctx.get_num_ranks()
        self.timeout = timeout
        self.state = None
        ctx._require_copy_engine()

        # Armed chains hold peer addresses; the torch allocator re-maps peer heaps on every allocation
        if isinstance(ctx.heap.allocator, TorchAllocator):
            raise ValueError(
                "collective plans need stable peer mappings: create the Iris context with allocator_type='vmem_chunked' (or 'vmem')"
            )
        if not ctx.heap.is_symmetric(buffer) or not buffer.is_contiguous():
            raise ValueError("buffer must be a contiguous tensor on the symmetric heap")
        buffer_bytes = buffer.numel() * buffer.element_size()
        error = None
        try:
            schedule = list(validate(schedule, buffer_bytes))
        except Exception as e:
            error = e
        self._agree(error, "validate its schedule")
        gathered = [None] * self.world_size
        dist.all_gather_object(gathered, (buffer_bytes, schedule))
        if any(nbytes != buffer_bytes for nbytes, _ in gathered):
            raise ValueError("ranks passed buffers of different sizes")
        self.schedules = [s for _, s in gathered]

        self._first = np.cumsum([0] + [len(s) for s in self.schedules]).tolist()
        self.num_chunks = self._first[-1]
        if self.num_chunks == 0:
            raise ValueError("no rank's schedule has any chunks")

        self.buffer = buffer
        self.gates = ctx.zeros(self.num_chunks, dv.GATE_WORDS.value, dtype=torch.int64)
        words = view_words(self.rank, self.world_size, self.gates.data_ptr(), self.schedules)
        self.view = torch.tensor(words, dtype=torch.int64, device=buffer.device)
        self._own = torch.arange(
            self._first[self.rank], self._first[self.rank + 1], dtype=torch.int64, device=buffer.device
        )
        self._status = torch.zeros(self.num_chunks, dtype=torch.int32, device=buffer.device)
        self._armed = torch.zeros(self.num_chunks * self.world_size, dtype=torch.int32, device=buffer.device)
        self._ledger = queue_ledger.ledger()
        self.round = 0
        try:
            self._arm(1)
        except Exception as error:
            self._unwind_construction(error)
        ctx.barrier()

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()

    def chunk_id(self, rank, index):
        """Global id of chunk ``index`` in rank ``rank``'s schedule."""
        return self._first[rank] + index

    def _agree(self, error, what):
        """Collective: raise on every rank if any rank has an error, so none is left waiting for the others."""
        failed = distributed_allgather(np.array([error is not None], dtype=np.int64))[:, 0].nonzero()[0].tolist()
        if failed:
            if error is not None:
                raise error
            raise RuntimeError(f"rank(s) {failed} could not {what}")

    def _unwind_construction(self, error):
        """Raise ``error`` from the constructor, after flushing the chains this rank armed."""
        if self.state is not None:
            # No plan reaches the caller to close(), so flush here
            try:
                self._flush(None)
            except Exception as flush_error:
                raise RuntimeError(f"{error}; the copies it prepared could not be flushed: {flush_error}") from error
            finally:
                # Peers may still be flushing into this rank's buffer and gates
                self.ctx.barrier()
        raise error

    def _arm(self, round):
        """Collective: every rank arms ``round``, or every rank raises (``FAILED`` if chains are left armed)."""
        peers = [p for p in range(self.world_size) if p != self.rank]
        nbytes = sum(armer.chain_bytes(c) for c in self.schedules[self.rank])
        error, reserved = None, False
        try:
            if round > UINT32_MAX:
                raise RuntimeError("round exceeds 32 bits; close the plan and create a new one")
            self._ledger.reserve(self, round, self.rank, {p: nbytes for p in peers})
            reserved = True
        except Exception as e:
            error = e
        try:
            self._agree(error, "reserve queue space")
        except Exception:
            # Nothing is armed anywhere
            if reserved:
                self._ledger.release(self)
            raise
        self.round = round
        self.state = State.PREPARED

        try:
            self._submit(peers)
        except Exception as e:
            error = e
        try:
            self._agree(error, "prepare its copies")
        except Exception:
            # Chains armed so far stay parked: close() flushes them
            self.state = State.FAILED
            raise

    def _submit(self, peers):
        heap = self.ctx.heap
        self._heap_bases = heap.heap_bases_cpu.copy()
        base = self.buffer.data_ptr()
        gate_row = self.gates.element_size() * dv.GATE_WORDS.value
        # Which chains exist, so a failed arm flushes only those
        armed = np.zeros(self.num_chunks * self.world_size, dtype=np.int32)
        try:
            for i, chunk in enumerate(self.schedules[self.rank]):
                chunk_id = self.chunk_id(self.rank, i)
                gate = self.gates.data_ptr() + chunk_id * gate_row
                for peer in peers:
                    copies = [
                        armer.Copy(
                            base + r.offset_bytes,
                            heap.translate(base + r.offset_bytes, self.rank, peer),
                            r.row_bytes,
                            r.rows,
                            r.row_stride,
                        )
                        for r in chunk.regions
                    ]
                    arrival = heap.translate(gate + 8 * dv.G_ARRIVAL.value, self.rank, peer)
                    armer.arm_chain(self.rank, peer, gate, self.round, copies, arrival, chunk.is_2d)
                    armed[chunk_id * self.world_size + peer] = 1
        finally:
            self._armed.copy_(torch.from_numpy(armed))

    def _require(self, *states):
        if self.state not in states:
            names = " or ".join(s.value for s in states)
            raise RuntimeError(f"plan is {self.state.value if self.state else 'uninitialized'}, needs {names}")

    def start(self):
        """Start the prepared round and return the handle tensor its kernels take."""
        self._require(State.PREPARED)
        if not np.array_equal(self.ctx.heap.heap_bases_cpu, self._heap_bases):
            raise RuntimeError("symmetric heap was re-mapped since the plan was prepared")
        self._ledger.check_order(self, (State.RUNNING,), "start")
        self.view[dv.H_ROUND.value] = self.round
        self.state = State.RUNNING
        return self.view

    def _wait(self, own_only, timeout):
        deadline = time.monotonic() + (self.timeout if timeout is None else timeout)
        while True:
            kernels.progress_kernel[(self.num_chunks,)](
                self.view,
                self.ctx.get_heap_bases(),
                self._armed,
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
                raise TimeoutError(f"round {self.round}: chunks {missing[:16]} not complete")

    def wait(self, timeout=None):
        """Wait until every chunk is delivered this round: own ones at every peer, others here."""
        self._require(State.RUNNING)
        try:
            self._wait(own_only=False, timeout=timeout)
        except TimeoutError:
            self.state = State.FAILED
            raise
        self._ledger.release(self)
        self.state = State.DONE

    def next_round(self):
        """
        Collective: prepare every chain for the next round. If any rank cannot, every rank raises.

        Calling it declares that this rank is done reading the current round's data: once it
        returns, a peer may start the next round, and its copies can overwrite ``buffer`` at any
        time. Work queued before the call is covered (it synchronizes the device first); anything
        issued after it must not read ``buffer`` until the next ``wait()``.
        """
        self._require(State.DONE)
        self.ctx.barrier()
        self._arm(self.round + 1)
        self.ctx.barrier()

    def run(self, timeout=None):
        """Run one round with no producer kernel: send every own chunk as it stands, then wait."""
        self.start()
        if len(self._own):
            kernels.release_kernel[(len(self._own),)](self.view, self._own)
        self.wait(timeout)

    def _flush(self, timeout):
        """Release this rank's prepared chains and wait for them to drain."""
        if self.state in (State.PREPARED, State.RUNNING, State.FAILED):
            self._ledger.check_order(self, (State.RUNNING, State.CLOSING), "close")
            self.view[dv.H_ROUND.value] = self.round
            if len(self._own):
                kernels.release_kernel[(len(self._own),)](self.view, self._own)
            self.state = State.CLOSING
        if self.state is State.CLOSING:
            self._wait(own_only=True, timeout=timeout)
            self._ledger.release(self)

    def close(self, timeout=None):
        """
        Collective: wait until every rank is done with the plan, then free it.

        Prepared copies cannot be withdrawn from the queue, so closing a plan whose round has not
        finished sends this rank's chunks as they stand. Plans sharing queues must close in the
        order they were prepared. A peer's copies and polls may still touch this rank's buffer and
        gates, so every rank must call close() before any of them is freed.
        """
        if self.state is State.CLOSED:
            return
        self._flush(timeout)
        self._ledger.release(self)
        self.ctx.barrier()
        self.state = State.CLOSED
        self.gates = self.view = self.buffer = None
