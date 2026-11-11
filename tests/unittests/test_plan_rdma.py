# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.

from contextlib import nullcontext
from types import SimpleNamespace

import pytest
import torch

from iris.experimental.plan import Chunk, Region, rdma

FIRST = [0, 2, 3, 3]  # chunk ids: rank 0 has 2 chunks, rank 1 has 1, rank 2 none


def test_writes_split_a_region_only_when_it_is_not_packed():
    assert rdma.writes(Region(100, 64)) == [(100, 100, 64)]
    assert rdma.writes(Region(0, 32, 4, 32)) == [(0, 0, 128)]  # packed rows: one write
    assert rdma.writes(Region(8, 16, 3, 64)) == [(8, 8, 16), (72, 72, 16), (136, 136, 16)]
    # Packed at the source but not at the destination: one write per row
    assert rdma.writes(Region(0, 16, 2, dst_offset_bytes=512, dst_stride_bytes=64)) == [(0, 512, 16), (16, 576, 16)]
    assert rdma.writes(Region(0, 16, 1, dst_offset_bytes=512)) == [(0, 512, 16)]


def _schedules():
    return [
        [Chunk((Region(0, 64),)), Chunk((Region(64, 64),), peers=(1,), lane=1)],
        [Chunk((Region(128, 64),), peers=(0,))],
        [],
    ]


def test_kicks_are_this_rank_s_chunks_to_rdma_peers_in_posting_order():
    sends, keys = rdma.kicks(_schedules(), 0, 3, {1, 2}, FIRST)
    assert [(cid, peer, queue) for cid, _, peer, queue in sends] == [(0, 1, 0), (0, 2, 0), (1, 1, 1)]
    # Rank 1 sends chunk 2 to rank 0, so rank 0 needs a queue pair to rank 1 on queue 0 to receive it
    assert keys == {(1, 0), (2, 0), (1, 1)}


def test_a_receiver_needs_the_sender_s_queue_pair():
    sends, keys = rdma.kicks(_schedules(), 2, 3, {0, 1}, FIRST)
    assert sends == []
    assert keys == {(0, 0)}  # chunk 0 from rank 0 (chunk 1 goes to rank 1 only)


def test_groups_cannot_go_over_rdma():
    schedules = [[Chunk((Region(0, 64),), group=0)], []]
    with pytest.raises(ValueError, match="group"):
        rdma.check(schedules, [])


def test_a_round_must_fit_the_queue_pair():
    rows = Chunk((Region(0, 16, 600, 64),))  # 600 strided rows: 600 writes, then the arrival write
    sends = [(0, rows, 1, 0), (1, rows, 1, 0)]
    rdma.check([[rows]], sends[:1])
    with pytest.raises(ValueError, match="1202 WQEs"):
        rdma.check([[rows, rows]], sends)


class _Endpoint:
    def __init__(self, outstanding, fails=False):
        self.left = outstanding
        self.fails = fails

    def host_outstanding(self):
        return self.left

    def host_poll(self, n):
        if self.fails:
            raise RuntimeError("error completion")
        self.left = max(0, self.left - 1)


def _kicks(monkeypatch, sends, endpoints, armed):
    monkeypatch.setattr(rdma.torch.cuda, "stream", lambda stream: nullcontext())
    plan = SimpleNamespace(completion=torch.zeros(3, dtype=torch.int32), _leaked=False)
    k = rdma.RdmaKicks.__new__(rdma.RdmaKicks)
    k.plan, k.sends, k.endpoints, k.armed, k._done = plan, sends, endpoints, armed, set()
    k.error, k.reported, k._abandoned = None, False, False
    k._stream = SimpleNamespace(synchronize=lambda: None)
    k.queues = {}
    for _, _, peer, queue in sends:
        k.queues.setdefault(peer, set()).add(queue)
    return k, plan


def test_completion_is_set_once_every_queue_to_the_peer_is_done(monkeypatch):
    sends, _ = rdma.kicks(_schedules(), 0, 3, {1, 2}, FIRST)
    endpoints = {(1, 0): _Endpoint(2), (1, 1): _Endpoint(1), (2, 0): _Endpoint(1)}
    k, plan = _kicks(monkeypatch, sends, endpoints, armed=3)
    completion = plan.completion
    assert not k.progress()
    # Rank 2's one queue is done: one queue x round 3. Rank 1 still has a send on queue 0
    assert completion.tolist() == [0, 0, 3]
    assert k.progress()
    assert completion.tolist() == [0, 6, 3]  # two queues to rank 1
    completion[1] = 0
    k.progress()  # set once per round
    assert completion[1].item() == 0
    assert k.error is None and not plan._leaked


def test_an_error_completion_stops_waiting_for_that_peer_and_keeps_the_memory(monkeypatch):
    sends, _ = rdma.kicks(_schedules(), 0, 3, {1, 2}, FIRST)
    endpoints = {(1, 0): _Endpoint(1, fails=True), (1, 1): _Endpoint(0), (2, 0): _Endpoint(5)}
    k, plan = _kicks(monkeypatch, sends, endpoints, armed=2)
    assert not k.progress()
    assert str(k.error) == "error completion" and plan._leaked
    assert plan.completion.tolist() == [0, 4, 0]  # rank 1 is no longer waited for; rank 2 still is


def test_a_write_must_fit_32_bits():
    huge = Chunk((Region(0, rdma.MAX_WRITE_BYTES + 1, rows=1),))
    with pytest.raises(ValueError, match="too large for RDMA"):
        rdma.check([[huge]], [(0, huge, 1, 0)])


def test_round_bytes_kick_rdma_peers_on_this_rank_s_queues(monkeypatch):
    from iris.experimental.plan import armer
    from iris.experimental.plan.triggered import TriggeredPlan

    ep = SimpleNamespace(COPY_LINEAR_COMMAND_BYTES=32, COPY_LINEAR_SUB_WINDOW_COMMAND_BYTES=48, ATOMIC_COMMAND_BYTES=36)
    monkeypatch.setattr(armer, "bind_sdma_ep", lambda: ep)
    monkeypatch.setattr(rdma, "bind_sdma_ep", lambda: ep)
    schedules = _schedules()
    sends, _ = rdma.kicks(schedules, 0, 3, {2}, FIRST)
    plan = SimpleNamespace(schedules=schedules, rank=0, world_size=3, _rdma_peers={2}, _rdma_sends=sends)
    chain, signal, kick = 24 + 32 + 36, 36, 24 + 2 * 32
    # Rank 1 by copy engine (two lanes), rank 2 by RDMA: one kick on this rank's lane-0 queue to itself
    assert TriggeredPlan._bytes_per_round(plan) == {(1, 0): signal + chain, (1, 1): signal + chain, (0, 0): kick}
