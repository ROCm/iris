# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.

"""
Triggered plans: the host prepares copy-engine chains, a kernel releases them.

Lifecycle:

    with TriggeredPlan(ctx, buffer, schedule) as plan:   # collective: allocate gates, prepare round 1
        for step in range(steps):
            handle = plan.start()
            producer[grid](..., handle)                   # TriggeredView.mark_ready(chunk)
            plan.prepare_next()                           # optional: prepare round + 1 now
            plan.wait()                                   # every chunk delivered, or TimeoutError
            ...                                           # read this round's data
            plan.next_round()                             # collective, stream-ordered: device barrier

All of a rank's chains to one peer on one lane share that lane's host-initiated copy-engine queue,
and run in the order they were prepared. Plans sharing queues must therefore start and close in the
order they were prepared, and nothing else may use those queues while copies are prepared on them.
Chunks default to lane 0; ``Chunk.lane`` puts a chunk on another of the context's ``sdma_lanes``
lanes, each on its own host queue, and lanes do not wait for each other.
"""

import enum
import time
import warnings

import numpy as np
import torch
import torch.distributed as dist

from iris.host import queue_ledger
from iris.host.distributed.helpers import distributed_allgather
from iris.experimental.plan import triton as dv
from iris.experimental.plan import armer, kernels
from iris.experimental.plan.queues import queue_index
from iris.experimental.plan.schedule import check_disjoint, validate
from iris.mem.utils import REALTIME_HZ

# The copy engine compares the low 32 bits unsigned, but Triton's and torch's int32 compares and
# atomic_max are signed, so counters stay below 2^31
INT32_MAX = (1 << 31) - 1
_WAIT_SLICE = REALTIME_HZ // 100  # 10 ms per wait_kernel launch


class State(enum.Enum):
    PREPARED = "prepared"
    RUNNING = "running"
    DONE = "done"
    FAILED = "failed"
    CLOSING = "closing"
    CLOSED = "closed"


def group_gate(num_chunks, world_size, group):
    """
    Index in the gates of ``group``'s arrival: after the counters, arrivals, completion, fence counts
    and abort pair. The device views compute the same in ``_group_word``.
    """
    return 2 * num_chunks + 2 * world_size + 2 + group


def group_counts(schedules, world_size):
    """
    ``counts[rank][group]``: how many chunks of each group are sent to each rank, per round.

    Unlike a chunk's arrival counter, which one sender's queue adds to in order, a group's counter
    sums several senders, so reaching ``counts × round`` shows round ``round`` landed only while no
    sender's next-round copy can arrive first. The round fence ensures that: no rank marks round
    ``r + 1`` before every rank's ``wait()`` for round ``r`` has seen its copies land. Past a fence
    timeout that no longer holds, and the late rank's next ``wait()`` raises instead (see
    ``kernels.fence_kernel``). A change that loosens the fence or pipelines rounds must keep it for
    groups.
    """
    num_groups = 1 + max((c.group for s in schedules for c in s if c.group is not None), default=-1)
    counts = np.zeros((world_size, num_groups), dtype=np.int64)
    for sender, schedule in enumerate(schedules):
        for chunk in schedule:
            if chunk.group is not None:
                for peer in chunk.recipients(sender, world_size):
                    counts[peer, chunk.group] += 1
    return counts


def view_words(rank, world_size, gates_ptr, schedules):
    """The device view (see ``iris.experimental.plan.triton``) as a list of int64 words."""
    num_chunks = sum(len(s) for s in schedules)
    expected = group_counts(schedules, world_size)[rank].tolist()
    words = [0] * (dv.HEADER_WORDS.value + num_chunks * dv.RECORD_WORDS.value + world_size + len(expected))
    words[dv.H_VERSION.value] = dv.ABI_VERSION
    words[dv.H_NUM_CHUNKS.value] = num_chunks
    words[dv.H_RANK.value] = rank
    words[dv.H_GATES.value] = gates_ptr
    words[dv.H_WORLD_SIZE.value] = world_size
    words[dv.H_NUM_GROUPS.value] = len(expected)
    rec, chunk_id = dv.HEADER_WORDS.value, 0
    for owner, schedule in enumerate(schedules):
        for chunk in schedule:
            if owner == rank:
                arrival, per_round = chunk_id, chunk.writers  # its counter
            elif rank not in chunk.recipients(owner, world_size):
                arrival, per_round = 0, 0
            elif chunk.group is None:
                arrival, per_round = num_chunks + chunk_id, 1
            else:
                arrival, per_round = group_gate(num_chunks, world_size, chunk.group), expected[chunk.group]
            words[rec + dv.R_WRITERS.value] = chunk.writers
            words[rec + dv.R_OWNER.value] = owner
            words[rec + dv.R_ARRIVAL.value] = arrival
            words[rec + dv.R_EXPECTED.value] = per_round
            rec += dv.RECORD_WORDS.value
            chunk_id += 1
    # How many queues this rank's chains use to each peer (its completion grows that much per round),
    # then how much each group's arrival grows here per round
    queues = {(p, queue_index(chunk.lane)) for chunk in schedules[rank] for p in chunk.recipients(rank, world_size)}
    for peer, _ in queues:
        words[rec + peer] += 1
    words[rec + world_size :] = expected
    return words


# Plans whose copies may still run after close(), kept so their memory is never reused
_leaked = []


def _mark_closed(plans):
    for plan in plans:
        plan.state = State.CLOSED
        if plan._leaked:
            warnings.warn("a triggered plan could not fence every copy-engine queue; its memory is kept, not freed")
            _leaked.append((plan.gates, plan.view, plan.buffer, plan.destination))
        plan.gates = plan.counters = plan.arrivals = plan.completion = plan.group_arrivals = None
        plan.view = plan.buffer = plan.destination = None


class TriggeredPlan:
    """
    Sends regions of ``buffer`` from the rank that writes them to other ranks, using copy engines
    that a kernel releases chunk by chunk.

    Each rank passes its own ``schedule``: the chunks it writes, in the order its producer completes
    them. A chunk goes to its ``peers`` (by default every other rank) and lands at each region's
    destination offset in their ``destination`` (by default the same offset in their ``buffer``).
    Chunk ids are global: rank ``r``'s chunks come after those of ranks ``0..r-1`` (see
    ``chunk_id``), and ``schedules[r]`` holds rank ``r``'s schedule. A rank that writes nothing
    passes ``[]``. No two ranks may write the same bytes of a rank's destination.

    Collective: every rank creates the plan for the same symmetric ``buffer`` and ``destination``.
    Use it as a context manager, or call ``close()`` on every rank when done.

    Args:
        ctx: Iris instance, created with ``engines=("sdma",)``.
        buffer: Contiguous tensor on the symmetric heap.
        schedule: This rank's list of ``Chunk``, in the order it completes them.
        timeout: Seconds ``wait()``, ``close()`` and the round fence of ``next_round()`` wait before
            raising.
        destination: Contiguous tensor on the symmetric heap that peers' chunks land in; ``buffer``
            if ``None``. It must not overlap ``buffer`` unless it is ``buffer``.
        prepare_ahead: Allow ``prepare_next()``. The plan then holds queue space for two rounds
            instead of one, so one round may use half as much of each queue's ring.

    Attributes:
        counters: Contiguous int32 tensor, one counter per chunk id. Adding 1 to a chunk's counter is
            what ``mark_ready`` does, so another kernel (such as a GEMM epilogue that signals per
            tile) may increment this rank's counters directly, under the same rules.
    """

    def __init__(self, ctx, buffer, schedule, timeout=30.0, destination=None, prepare_ahead=False):
        self.ctx = ctx
        self.rank = ctx.get_rank()
        self.world_size = ctx.get_num_ranks()
        self.timeout = timeout
        self.state = None
        ctx._require_copy_engine()

        destination = buffer if destination is None else destination
        for name, t in (("buffer", buffer), ("destination", destination)):
            if not ctx.heap.is_symmetric(t) or not t.is_contiguous():
                raise ValueError(f"{name} must be a contiguous tensor on the symmetric heap")
        buffer_bytes = buffer.numel() * buffer.element_size()
        destination_bytes = destination.numel() * destination.element_size()
        same = destination.data_ptr() == buffer.data_ptr() and destination_bytes == buffer_bytes
        if not same and max(buffer.data_ptr(), destination.data_ptr()) < min(
            buffer.data_ptr() + buffer_bytes, destination.data_ptr() + destination_bytes
        ):
            raise ValueError("destination overlaps buffer; pass the same tensor, or memory apart from it")
        error = None
        try:
            schedule = list(
                validate(schedule, buffer_bytes, destination_bytes, self.rank, self.world_size, lanes=ctx.sdma_lanes)
            )
        except Exception as e:
            error = e
        self._agree(error, "validate its schedule")
        gathered = [None] * self.world_size
        dist.all_gather_object(gathered, (buffer_bytes, destination_bytes, schedule))
        if any(g[:2] != (buffer_bytes, destination_bytes) for g in gathered):
            raise ValueError("ranks passed buffers or destinations of different sizes")
        self.schedules = [s for *_, s in gathered]
        check_disjoint(self.schedules, same)
        groups = [c.group for s in self.schedules for c in s if c.group is not None]
        total = sum(len(s) for s in self.schedules)
        if groups and max(groups) >= total:
            # Every id up to the largest gets a gate word and a view word on each rank
            raise ValueError(f"group ids go up to {max(groups)}, but the plan has {total} chunks; number groups from 0")

        self._first = np.cumsum([0] + [len(s) for s in self.schedules]).tolist()
        self.num_chunks = self._first[-1]
        if self.num_chunks == 0:
            raise ValueError("no rank's schedule has any chunks")

        self.buffer, self.destination = buffer, destination
        counts = group_counts(self.schedules, self.world_size)
        # Each chunk's counter (marks) and arrival, each peer queue's completion, each peer's round
        # fence count, the earliest round a peer failed and which peer, then each group's arrival,
        # all int32
        n, w = self.num_chunks, self.world_size
        self.gates = ctx.zeros(group_gate(n, w, counts.shape[1]), dtype=torch.int32)
        self.counters = self.gates[:n]
        self.arrivals = self.gates[n : 2 * n]
        self.completion = self.gates[2 * n : 2 * n + w]
        self.group_arrivals = self.gates[group_gate(n, w, 0) :]
        words = view_words(self.rank, self.world_size, self.gates.data_ptr(), self.schedules)
        self.view = torch.tensor(words, dtype=torch.int64, device=buffer.device)
        self._own = torch.arange(
            self._first[self.rank], self._first[self.rank + 1], dtype=torch.int64, device=buffer.device
        )
        self._writers = [c.writers for c in self.schedules[self.rank]]
        # The most any gate grows per round, on any rank: a counter, a group's arrival, or a completion
        # (one per queue, so per lane, to the peer)
        lanes_to_a_peer = max(
            len({queue_index(c.lane) for c in s if p in c.recipients(r, self.world_size)})
            for r, s in enumerate(self.schedules)
            for p in range(self.world_size)
        )
        self._max_per_round = max(
            max(c.writers for s in self.schedules for c in s), int(counts.max(initial=0)), lanes_to_a_peer
        )
        self._status = torch.zeros(self.num_chunks + self.world_size, dtype=torch.int32, device=buffer.device)
        # Every round ends with a completion signal on each of these (peer, queue), even after a failed arm
        self._queues = sorted(
            {
                (p, queue_index(c.lane))
                for c in self.schedules[self.rank]
                for p in c.recipients(self.rank, self.world_size)
            }
        )
        self._leaked = False
        self._prepare_error = None
        self._chains = None
        # First error of a stream-ordered wait(): [code, round, index, value]
        self._error = torch.zeros(4, dtype=torch.int64, device=buffer.device)
        self._next_prepared = False
        self._prepare_ahead = prepare_ahead
        self._ledger = queue_ledger.ledger()
        self.round = 0
        try:
            self._arm_first()
        except Exception as error:
            self._unwind_construction(error)
        ctx.barrier()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, *exc):
        # Leaving on an exception: close, but let that exception through rather than a deferred error
        self.close(check=exc_type is None)

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
            # No plan reaches the caller to close(), so flush here; if queued behind an earlier plan's
            # waiting copies, the ledger keeps this plan until that plan's close() drains it
            drained = []
            try:
                self._release_own()
                if not self._blocked():
                    drained = self._drain_closing(None)
            except Exception as flush_error:
                raise RuntimeError(f"{error}; the copies it prepared could not be flushed: {flush_error}") from error
            finally:
                # Peers may still be flushing into this rank's buffer and gates
                self.ctx.barrier()
            _mark_closed(drained)
        raise error

    def _arm_first(self):
        """Collective: hold queue space (two rounds with prepare_ahead) and prepare round 1 on every rank, or raise on every rank."""
        self._round_bytes = nbytes = armer.round_bytes(self.schedules[self.rank], self.rank, self.world_size)
        error, held = None, False
        try:
            # With prepare_next(), a round may be prepared while the one before it still is
            rounds = 2 if self._prepare_ahead else 1
            self._ledger.hold(self, self.rank, {p: rounds * n for p, n in nbytes.items()})
            held = True
            self._ledger.reserve(self, 1, self.rank, nbytes)
        except Exception as e:
            error = e
        try:
            self._agree(error, "reserve queue space")
        except Exception:
            # Nothing is armed anywhere
            if held:
                self._ledger.release(self)
            raise
        self.round = 1
        self.state = State.PREPARED

        try:
            self._submit(1)
        except Exception as e:
            error = e
        try:
            self._agree(error, "prepare its copies")
        except Exception:
            # Chains armed so far stay parked: close() flushes them
            self.state = State.FAILED
            raise

    def _check_round(self, round):
        # Counters are never reset and the copy engine compares 32 bits; the same on every rank
        if self._max_per_round * round > INT32_MAX:
            raise RuntimeError(
                f"a gate's growth per round (a chunk's writers, a group's chunks, or the lanes to a peer) x round "
                f"exceeds {INT32_MAX} at round {round}; close the plan and create a new one"
            )

    def _prepare_next(self):
        """Prepare the next round on this rank only. An error is kept for next_round() to raise."""
        if self._next_prepared or self._prepare_error is not None:
            return
        try:
            self._ledger.reserve(self, self.round + 1, self.rank, self._round_bytes)
            # From here close() must flush this round too (the completion signals follow even a failed submit)
            self._next_prepared = True
            self._submit(self.round + 1)
        except Exception as e:
            self._prepare_error = e

    def _build_chains(self):
        """Every (chunk, peer) chain of this rank, in arming order; the same every round."""
        heap = self.ctx.heap
        self._heap_bases = heap.heap_bases_cpu.copy()
        src, dst = self.buffer.data_ptr(), self.destination.data_ptr()
        word = self.gates.element_size()
        n, w = self.num_chunks, self.world_size
        chains = []
        for i, chunk in enumerate(self.schedules[self.rank]):
            chunk_id = self.chunk_id(self.rank, i)
            arrival = n + chunk_id if chunk.group is None else group_gate(n, w, chunk.group)
            for peer in chunk.recipients(self.rank, w):
                copies = [
                    armer.Copy(
                        src + r.offset_bytes,
                        heap.translate(dst + r.dst_offset, self.rank, peer),
                        r.row_bytes,
                        r.rows,
                        r.row_stride,
                        r.dst_row_stride,
                    )
                    for r in chunk.regions
                ]
                counter = self.counters.data_ptr() + word * chunk_id
                arrival_addr = heap.translate(self.gates.data_ptr() + word * arrival, self.rank, peer)
                devices = self.ctx._copy_engine_devices(peer)
                chain = armer.Chain(*devices, counter, copies, arrival_addr, chunk.is_2d, queue_index(chunk.lane))
                chains.append((chain, chunk.writers))
        self._chains = chains

    def _submit(self, round):
        error = None
        try:
            if self._chains is None:
                self._build_chains()
            for chain, writers in self._chains:
                armer.arm_chain(chain, writers * round)
        except Exception as e:
            error = e
        # Even after a failed arm, so a flush can tell when each queue has drained
        failed = self._signal_completion()
        if error is not None:
            raise error
        if failed:
            raise RuntimeError(f"could not queue the completion signal to rank(s) {failed}")

    def _signal_completion(self):
        """
        Queue the completion signal for this round on each queue to each peer, separately. Returns the
        peers with a signal that could not be queued: nothing will tell when their queues drain, so
        they are no longer waited for, and the plan's memory is kept for the life of the process
        instead of freed.
        """
        completion, word = self.completion.data_ptr(), self.gates.element_size()
        failed = []
        for peer, queue in self._queues:
            try:
                armer.signal_completion(*self.ctx._copy_engine_devices(peer), completion + word * peer, queue)
            except Exception:
                if peer not in failed:
                    failed.append(peer)
        for peer in failed:
            self._queues = [(p, q) for p, q in self._queues if p != peer]
            self.view.narrow(0, dv.HEADER_WORDS.value + self.num_chunks * dv.RECORD_WORDS.value + peer, 1).fill_(0)
            self._leaked = True
        return failed

    def _require(self, *states):
        if self.state not in states:
            names = " or ".join(s.value for s in states)
            raise RuntimeError(f"plan is {self.state.value if self.state else 'uninitialized'}, needs {names}")

    def _started(self, round):
        """True once ``round`` has started, so its gates will open."""
        return self.state in (State.RUNNING, State.DONE) and round == self.round

    def _set_round(self, round):
        # fill_ is queued on the stream; assigning an element copies from the host and waits for the stream
        self.view.narrow(0, dv.H_ROUND.value, 1).fill_(round)

    def start(self):
        """
        Start the prepared round and return the handle tensor its kernels take.

        The round is set in the handle by work queued on the current stream: launch the kernels that
        take it on that stream, or make their stream wait for it.
        """
        self._require(State.PREPARED)
        if not np.array_equal(self.ctx.heap.heap_bases_cpu, self._heap_bases):
            raise RuntimeError("symmetric heap was re-mapped since the plan was prepared")
        self._ledger.check_order(self, self.round, TriggeredPlan._started, "start")
        self._set_round(self.round)
        self.state = State.RUNNING
        return self.view

    def _launch_wait(self, own_only, stream, ticks):
        kernels.wait_kernel[(self.num_chunks + self.world_size,)](
            self.view, self._status, self._error, ticks, OWN_ONLY=own_only, STREAM=stream, num_warps=1
        )

    def _wait(self, own_only, timeout):
        deadline = time.monotonic() + (self.timeout if timeout is None else timeout)
        while True:
            self._launch_wait(own_only, False, _WAIT_SLICE)
            status = self._status.cpu()
            pending = status == 0
            if not pending.any():
                return
            if not own_only:
                # A peer that failed this round
                self._raise_recorded()
            if time.monotonic() > deadline:
                missing = pending[: self.num_chunks].nonzero().flatten().tolist()
                queues = pending[self.num_chunks :].nonzero().flatten().tolist()
                short = [
                    f"{self._first[self.rank] + i}: {m} of {w}"
                    for i, (m, w) in enumerate(zip(self._marks(), self._writers))
                    if m < w
                ]
                detail = f"; own chunks short of writers' marks: {', '.join(short[:16])}" if short else ""
                if queues:
                    detail = f"; copies to ranks {queues} not finished{detail}"
                raise TimeoutError(f"round {self.round}: chunks {missing[:16]} not complete{detail}")

    def _marks(self):
        """How many times each own chunk has been marked ready this round."""
        counters = self.counters[self._first[self.rank] : self._first[self.rank + 1]].tolist()
        return [c - w * (self.round - 1) for c, w in zip(counters, self._writers)]

    def _deferred_error(self, error):
        """The exception for an error word the GPU recorded (see kernels.wait_kernel), or None."""
        code, round, index, value = error
        if not code:
            return None
        if code == kernels.FENCE_TIMED_OUT.value:
            return TimeoutError(f"round {round}: rank {index} did not reach next_round()")
        if code == kernels.PEER_FAILED.value:
            return RuntimeError(
                f"round {round}: rank {index} failed this round or an earlier one; its own error says why"
            )
        own = self._first[self.rank] <= index < self._first[self.rank + 1]
        writers = self.schedules[self.rank][index - self._first[self.rank]].writers if own else 0
        if code == kernels.WRONG_MARKS.value:
            return RuntimeError(
                f"round {round}: chunk {index} was marked ready {value} times, but has {writers} writers"
            )
        # The same text as a blocking wait()'s timeout
        if index >= self.num_chunks:
            return TimeoutError(
                f"round {round}: chunks [] not complete; copies to ranks [{index - self.num_chunks}] not finished"
            )
        detail = f"; own chunks short of writers' marks: {index}: {value} of {writers}" if own else ""
        return TimeoutError(f"round {round}: chunks [{index}] not complete{detail}")

    def _raise_recorded(self):
        """Raise (once) an error the GPU recorded; the plan is FAILED."""
        error = self._deferred_error(self._error.tolist())
        if error:
            self._error.zero_()
            self.state = State.FAILED
            raise error

    def wait(self, timeout=None, blocking=True):
        """
        Wait until every chunk is delivered this round: own ones at every peer, others here.

        With ``blocking=False``, queue the wait on the current stream instead and return at once:
        kernels queued after it see the round's data, and the host never synchronizes. An error (a
        timeout, extra marks, a peer that failed the round, or a round fence that timed out) is then
        raised by the next blocking ``wait()``, or by ``close()``.
        """
        self._require(State.RUNNING)
        if not blocking:
            ticks = int((self.timeout if timeout is None else timeout) * REALTIME_HZ)
            self._launch_wait(False, True, ticks)
            self.state = State.DONE
            return
        self._raise_recorded()
        try:
            self._wait(own_only=False, timeout=timeout)
        except TimeoutError:
            self.state = State.FAILED
            raise
        # Extra marks open later rounds' gates early, so data would go out before it is written
        wrong = [(i, m, w) for i, (m, w) in enumerate(zip(self._marks(), self._writers)) if m != w]
        if wrong:
            self.state = State.FAILED
            i, m, w = wrong[0]
            raise RuntimeError(
                f"round {self.round}: chunk {self._first[self.rank] + i} was marked ready {m} times, but has {w} writers"
            )
        # Every copy of the round has landed: its ring space is free, and host put() / quiet() may run
        self._ledger.release(self, self.round)
        self.state = State.DONE

    def prepare_next(self):
        """
        Prepare the next round's copies now, so ``next_round()`` does not have to. Not collective.

        May be called once the current round has started, on a plan created with
        ``prepare_ahead=True``. The copies wait behind gates that no producer can open before every
        rank has passed ``next_round()``. If preparing fails, ``next_round()`` raises the error after
        its fence, so no rank is left waiting in it.
        """
        if not self._prepare_ahead:
            raise RuntimeError("prepare_next() needs a plan created with prepare_ahead=True")
        self._require(State.RUNNING, State.DONE)
        self._check_round(self.round + 1)
        self._prepare_next()

    def next_round(self):
        """
        Collective: move to the next round, preparing it first unless ``prepare_next()`` did.

        Calling it declares that this rank is done reading the current round's data: once every rank
        has called it, a peer may start the next round, and its copies can overwrite ``buffer`` at
        any time. The fence is a kernel on the current stream, so the host does not wait: work
        queued on that stream before the call is covered, and nothing issued after it may read
        ``buffer`` until the next ``wait()``. A peer that never reaches the fence is recorded as a
        timeout after the plan's timeout, and raised by the next blocking ``wait()`` or ``close()``.

        If this rank cannot prepare the next round, it still takes part in the fence, telling its
        peers, whose ``wait()`` for that round then raises; then this call raises. A rank whose plan
        has recorded an error (from a stream-ordered ``wait()``) does the same, without raising here.
        """
        self._require(State.DONE)
        self._check_round(self.round + 1)
        # The round's gates are open (its producer is queued), so its copies drain on their own and
        # its queue space can go to the next round
        self._ledger.release(self, self.round)
        self._prepare_next()
        error, self._prepare_error = self._prepare_error, None
        kernels.fence_kernel[(1,)](
            self.view,
            self.ctx.get_heap_bases(),
            self._error,
            int(self.timeout * REALTIME_HZ),
            ABORT=error is not None,
            num_warps=1,
        )
        if error is not None:
            # Chains may be left armed: close() flushes them
            self.state = State.FAILED
            raise error
        self.round += 1
        self._next_prepared = False
        self.state = State.PREPARED

    def run(self, timeout=None):
        """Run one round with no producer kernel: send every own chunk as it stands, then wait."""
        self.start()
        if len(self._own):
            kernels.release_kernel[(len(self._own),)](self.view, self._own)
        self.wait(timeout)

    def _release_own(self):
        """Open this rank's gates for every prepared round, so its chains send its chunks as they stand."""
        self._set_round(self.round + self._next_prepared)
        if len(self._own):
            kernels.release_kernel[(len(self._own),)](self.view, self._own)
        self.state = State.CLOSING

    def _blocked(self):
        # Chains run in order on each queue, so one parked behind a round that has not started cannot drain
        return not all(p._started(r) for p, r in self._ledger.earlier(self))

    def _drain_closing(self, timeout):
        """Drain every closing plan whose chains can now run, in queue order. Returns them."""
        drained = []
        progress = True
        while progress:
            progress = False
            for plan in self._ledger.plans():
                if plan.state is State.CLOSING and not plan._blocked():
                    plan._wait(own_only=True, timeout=timeout)
                    self._ledger.release(plan)
                    drained.append(plan)
                    progress = True
        return drained

    def close(self, timeout=None, check=True):
        """
        Collective: wait until every rank is done with the plan, then free it.

        Prepared copies cannot be withdrawn from the queue, so closing a plan whose round has not
        finished sends this rank's chunks as they stand (for every prepared round). A peer's copies
        and polls may still touch this rank's buffer and gates, so every rank must call close()
        before any of them is freed. Then raises any error a stream-ordered ``wait()`` recorded,
        unless ``check`` is false.

        Plans may be closed in any order. If a plan prepared earlier on the same queues still has
        copies waiting, this plan's copies are queued behind them and cannot run yet: close() then
        returns with the plan ``CLOSING``, and closing the earlier plan finishes this one too.
        """
        if self.state is State.CLOSED:
            return
        if self.state is not State.CLOSING:
            self._release_own()
        if self._blocked():
            return
        closed = self._drain_closing(timeout)
        if self not in closed:
            closed.append(self)
        self._ledger.release(self)
        self.ctx.barrier()
        error = self._deferred_error(self._error.tolist())
        for plan in closed:
            # Plans whose own close() returned while they were queued behind this one
            other = plan._deferred_error(plan._error.tolist()) if plan is not self else None
            if other:
                warnings.warn(f"a triggered plan finished by this close() recorded an error: {other}")
        _mark_closed(closed)
        if check and error:
            raise error
