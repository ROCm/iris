# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.

"""
RDMA peers of a triggered plan: dormant work in GPU-memory queues, released by a copy-engine kick.

For each chunk a rank sends and each peer on another host, the host posts the chunk's RDMA WRITEs
into a dedicated send queue and leaves them **dormant** (the NIC cannot see work whose doorbell was
not rung), followed by one more WRITE that puts the round into the chunk's arrival word on the peer.
It stages the two values that ring the doorbell (the doorbell record, the control header of the
newest work request) in GPU memory and arms a chain on the local GPU's host copy-engine queue:

    POLL_REGMEM(counter >= writers * round) -> copy the record to the doorbell record -> copy the header to the doorbell

so the producer's ``mark_ready`` opens the gate and the copy engine rings the NIC. A kick releases
every work request posted up to its chunk, so kicks must go out in posting order: chains on one
copy-engine queue run in arming order, and a lane is one (copy-engine queue, queue pair) pair.

A chunk's arrival word has exactly one writer (its owner), so the round is written, not added; a
chunk in a group shares its word with other senders and cannot go over RDMA. The sender learns that
its chunks reached a peer from the completion of the arrival writes, which the host polls during
``wait()``: then it sets this rank's completion counter for that peer to what the copy engine would
have counted (one per queue per round), so ``wait()`` and ``sent()`` read it as for any peer.

Needs ``engines=("sdma", "rdma")``; queue pairs and their send queues live in GPU memory (rocm-xio
``device_queues``).
"""

import time

import numpy as np
import torch
import torch.distributed as dist

from iris.device.sdma_utils import bind_sdma_ep
from iris.experimental.plan import armer
from iris.experimental.plan.queues import queue_index

# Work-queue sizes of a triggered queue pair (rocm-xio: sq_depth=256 -> 1024 slots, cq_depth=256 -> 512 entries)
SQ_SLOTS = 1024
CQ_SLOTS = 512
# rocm-xio passes a write's byte count as 32 bits, and an mlx5 message is at most 2 GiB
MAX_WRITE_BYTES = 1 << 31


def writes(region):
    """
    ``(source offset, destination offset, bytes)`` of the RDMA WRITEs a region needs: one if its rows
    are packed at both ends, else one per row.
    """
    if region.rows == 1 or (region.row_stride == region.row_bytes and region.dst_row_stride == region.row_bytes):
        return [(region.offset_bytes, region.dst_offset, region.row_bytes * region.rows)]
    return [
        (region.offset_bytes + r * region.row_stride, region.dst_offset + r * region.dst_row_stride, region.row_bytes)
        for r in range(region.rows)
    ]


def kicks(schedules, rank, world_size, rdma_peers, first):
    """
    This rank's ``(chunk id, chunk, peer, queue)`` for every chunk it sends to an RDMA peer, in
    posting order, and the ``(peer, queue)`` pairs it needs a queue pair for (as a sender or a receiver).
    """
    sends, keys = [], set()
    for owner, schedule in enumerate(schedules):
        for i, chunk in enumerate(schedule):
            queue = queue_index(chunk.lane)
            receivers = chunk.recipients(owner, world_size)
            if owner == rank:
                for peer in receivers:
                    if peer in rdma_peers:
                        sends.append((first[owner] + i, chunk, peer, queue))
                        keys.add((peer, queue))
            elif rank in receivers and owner in rdma_peers:
                keys.add((owner, queue))
    return sends, keys


def check(schedules, sends, uses_rdma):
    """Raise unless the plan can go over RDMA: no grouped chunk, and each queue pair's round fits its queues."""
    if uses_rdma and any(c.group is not None for s in schedules for c in s):
        raise ValueError("chunks in a group cannot go over RDMA: their arrival word has several writers")
    used = {}
    for chunk_id, chunk, peer, queue in sends:
        if any(n > MAX_WRITE_BYTES for r in chunk.regions for _, _, n in writes(r)):
            raise ValueError(f"chunk {chunk_id} has a write over {MAX_WRITE_BYTES} bytes, too large for RDMA; split it")
        slots, cqes = used.get((peer, queue), (0, 0))
        used[(peer, queue)] = (slots + sum(len(writes(r)) for r in chunk.regions) + 1, cqes + 1)
    for (peer, queue), (slots, cqes) in used.items():
        if slots > SQ_SLOTS or cqes > CQ_SLOTS:
            raise ValueError(
                f"the chunks to rank {peer} on queue {queue} need {slots} send-queue slots and {cqes} completions "
                f"per round; a triggered queue pair has {SQ_SLOTS} and {CQ_SLOTS}. Use fewer or larger chunks, "
                "or spread them over more lanes"
            )


def kick_bytes(sends, rank):
    """Ring bytes this rank's kick chains take per round, keyed like ``armer.round_bytes``: ``{(rank, queue): bytes}``."""
    per_kick = armer.POLL_REGMEM_BYTES + 2 * bind_sdma_ep().COPY_LINEAR_COMMAND_BYTES
    out = {}
    for _, _, _, queue in sends:
        out[(rank, queue)] = out.get((rank, queue), 0) + per_kick
    return out


class RdmaKicks:
    """
    The RDMA side of one ``TriggeredPlan`` on this rank. Collective to build (it connects queue pairs
    and allocates a symmetric word); afterwards ``arm(round)`` posts and arms a round, and
    ``progress()`` reaps completions and sets the completion counters of the peers that are done.

    The counters are written on a stream of their own, so a kernel waiting in ``sent()`` on the
    plan's stream does not hold them back, but only while the host polls: in ``wait()``. A send that
    fails, or a round that could not be armed, is recorded in ``error``: its peers are no longer
    waited for, and the plan's memory is kept, since the NIC may still write to and from it.
    """

    def __init__(self, plan, rdma_peers, sends, keys):
        from iris.device import rdma_utils

        ctx = plan.ctx
        self.plan = plan
        self.sends = sends
        self.armed = 0
        # Completion grows by one per queue to a peer per round, as copy-engine signals would
        self.queues = {}
        for _, _, peer, queue in sends:
            self.queues.setdefault(peer, set()).add(queue)
        self._done = set()
        self.error = None
        self.reported = False
        self._abandoned = False

        self.endpoints, mine, error = {}, {}, None
        try:
            rdma = rdma_utils.bind_rdma_ep()
            base, size = ctx.heap.rdma_region()
            for peer, queue in sorted(keys):
                ep = rdma.Endpoint("mlx5", ctx.rdma_device, ctx.gpu_id, device_queues=True)
                ep.register_buffer(base, size, True)
                self.endpoints[(peer, queue)] = ep
                mine[(peer, queue)] = ep.local_info()
        except Exception as e:
            error = e
        # Exchanged even by a rank that failed, which then raises with the others in the agreement
        everyone = [None] * plan.world_size
        dist.all_gather_object(everyone, mine)
        plan._agree(error, "create its queue pairs")
        try:
            for (peer, queue), ep in self.endpoints.items():
                ep.connect(everyone[peer][(plan.rank, queue)])
            self.addresses = {key: ep.host_addresses() for key, ep in self.endpoints.items()}
        except Exception as e:
            error = e
        plan._agree(error, "connect its queue pairs")
        self._stream = torch.cuda.Stream(device=plan.buffer.device)

        # The arrival writes read their value from here when they execute, so it holds the round (a
        # round's writes all carry the same value; the next round is posted after they completed)
        self.round_word = ctx.zeros(1, dtype=torch.int32)
        device = plan.buffer.device
        self.stage_rec = torch.zeros(max(1, len(sends)), dtype=torch.int32, device=device)
        self.stage_val = torch.zeros(max(1, len(sends)), dtype=torch.int64, device=device)

    def abandon(self, round):
        """
        Round ``round`` could not be (fully) armed: stop waiting for its sends. Kicks armed before the
        failure may still ring the NIC, so the plan's memory is kept.
        """
        self.armed, self._abandoned = round, True
        self.plan._leaked = True

    def arm(self, round):
        """Post the round's work dormant, stage the doorbell values, arm the kick chains."""
        if self.outstanding():
            raise RuntimeError("the previous round's RDMA sends have not completed")
        self.armed, self._done = round, set()
        if not self.sends:
            return
        try:
            self._arm(round)
        except Exception:
            self.abandon(round)
            raise

    def _arm(self, round):
        plan = self.plan
        self.round_word.fill_(round)
        heap = plan.ctx.heap
        src, dst = plan.buffer.data_ptr(), plan.destination.data_ptr()
        word = plan.gates.element_size()
        recs = np.zeros(len(self.sends), dtype=np.uint32)
        vals = np.zeros(len(self.sends), dtype=np.uint64)
        for i, (chunk_id, chunk, peer, queue) in enumerate(self.sends):
            ep = self.endpoints[(peer, queue)]
            for region in chunk.regions:
                for src_offset, dst_offset, nbytes in writes(region):
                    remote = heap.translate(dst + dst_offset, plan.rank, peer)
                    ep.host_post_write(remote, src + src_offset, nbytes, signaled=False)
            arrival = heap.translate(plan.arrivals.data_ptr() + word * chunk_id, plan.rank, peer)
            ep.host_post_write(arrival, self.round_word.data_ptr(), word, signaled=True)
            recs[i], vals[i] = ep.host_staged()
        # The CPU wrote the work requests into GPU-memory queues: drain them to the device before any
        # chain that will ring the doorbell is armed. The drain is per GPU, so one endpoint does it for all.
        next(iter(self.endpoints.values())).host_flush()
        self.stage_rec[: len(self.sends)].copy_(torch.from_numpy(recs.view(np.int32)))
        self.stage_val[: len(self.sends)].copy_(torch.from_numpy(vals.view(np.int64)))
        torch.cuda.synchronize()

        sdma = bind_sdma_ep()
        gpu = plan.ctx.gpu_id
        counters = plan.counters.data_ptr()
        for i, (chunk_id, chunk, peer, queue) in enumerate(self.sends):
            where = self.addresses[(peer, queue)]
            threshold = chunk.writers * round
            sdma.wait_flag_then_put(
                gpu,
                gpu,
                queue,
                counters + word * chunk_id,
                threshold,
                where["dbrec"],
                self.stage_rec.data_ptr() + 4 * i,
                4,
                32,
            )
            sdma.put(gpu, gpu, queue, where["gpu_doorbell"], self.stage_val.data_ptr() + 8 * i, 8)

    def outstanding(self):
        return sum(ep.host_outstanding() for ep in self.endpoints.values())

    def progress(self):
        """
        Reap completions; set the completion counter of each peer whose sends this round have all
        landed (or failed, or were abandoned). Returns True once every peer is done.
        """
        failed = set()
        for (peer, queue), ep in self.endpoints.items():
            try:
                if ep.host_outstanding():
                    ep.host_poll(64)
            except Exception as e:
                # An error completion: the queue pair is in error and will not finish the round
                if self.error is None:
                    self.error = e
                self.plan._leaked = True
                failed.add(peer)
        done = [
            peer
            for peer, queues in self.queues.items()
            if peer not in self._done
            and (
                self._abandoned
                or peer in failed
                or not any(self.endpoints[(peer, q)].host_outstanding() for q in queues)
            )
        ]
        if done:
            with torch.cuda.stream(self._stream):
                for peer in done:
                    self.plan.completion.narrow(0, peer, 1).fill_(len(self.queues[peer]) * self.armed)
            self._stream.synchronize()
            self._done.update(done)
        return len(self._done) == len(self.queues)

    def drain(self, deadline):
        """Poll until every peer is done this round, or ``deadline`` (``time.monotonic()``) passes."""
        while not self.progress() and time.monotonic() < deadline:
            time.sleep(0)
