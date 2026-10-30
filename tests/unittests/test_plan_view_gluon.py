# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.

# Gluon TriggeredView against an ordinary gate tensor: no copy engine involved.

import torch
from triton.experimental import gluon
from triton.experimental.gluon import language as gl

from iris.experimental.plan.gluon import TriggeredView
from iris.experimental.plan import triton as dv
from iris.experimental.plan import Chunk, Region
from iris.experimental.plan.triggered import group_counts, view_words


def _view(rank, schedules, round):
    """A handle and its gates, split into (counters, arrivals, completion, groups)."""
    n = sum(len(s) for s in schedules)
    groups = group_counts(schedules, len(schedules)).shape[1]
    gates = torch.zeros(2 * n + len(schedules) + groups, dtype=torch.int32, device="cuda")
    words = view_words(rank, len(schedules), gates.data_ptr(), schedules)
    words[dv.H_ROUND.value] = round
    w = len(schedules)
    parts = (gates[:n], gates[n : 2 * n], gates[2 * n : 2 * n + w], gates[2 * n + w :])
    return torch.tensor(words, dtype=torch.int64, device="cuda"), parts


@gluon.jit
def _mark_ready_kernel(handle, chunk):
    TriggeredView.initialize(handle).mark_ready(chunk)


@gluon.jit
def _arrived_kernel(handle, out, n):
    cv = TriggeredView.initialize(handle)
    for c in range(n):
        gl.store(out + c, cv.arrived(c).to(gl.int32) + 2 * cv.owns(c).to(gl.int32))


@gluon.jit
def _wait_kernel(handle, out, chunk):
    cv = TriggeredView.initialize(handle)
    cv.wait(chunk)
    gl.store(out, 1)


@gluon.jit
def _sent_kernel(handle, out):
    gl.store(out, TriggeredView.initialize(handle).sent().to(gl.int32))


@gluon.jit
def _wait_sent_kernel(handle, out):
    TriggeredView.initialize(handle).wait_sent()
    gl.store(out, 1)


def _arrived(view, n):
    out = torch.zeros(n, dtype=torch.int32, device="cuda")
    _arrived_kernel[(1,)](view, out, n)
    return out.tolist()


def test_mark_ready_counts_writers():
    k = 4
    view, (counters, _, _, _) = _view(0, [[Chunk((Region(0, 64),), writers=k)], []], round=1)
    _mark_ready_kernel[(k - 1,)](view, 0)
    assert counters[0].item() == k - 1 and _arrived(view, 1) == [2]
    _mark_ready_kernel[(1,)](view, 0)
    assert counters[0].item() == k and _arrived(view, 1) == [3]

    # The counter is never reset: the next round needs exactly k more marks
    view[dv.H_ROUND.value] = 2
    _mark_ready_kernel[(k - 1,)](view, 0)
    assert _arrived(view, 1) == [2]
    _mark_ready_kernel[(1,)](view, 0)
    assert counters[0].item() == 2 * k and _arrived(view, 1) == [3]


def test_arrived_reads_counter_for_own_chunks_and_arrival_for_others():
    view, (counters, arrivals, _, _) = _view(
        1, [[Chunk((Region(0, 64),))], [Chunk((Region(64, 64),), writers=2)]], round=3
    )
    counters[0] = 100  # another rank's counter: not this rank's to observe
    arrivals[1] = 3  # own chunk: arrivals happen at peers, not here
    counters[1] = 5  # one mark short of 2 writers x round 3
    assert _arrived(view, 2) == [0, 2]

    arrivals[0] = 3
    counters[1] = 6
    assert _arrived(view, 2) == [1, 3]


def test_wait_returns_once_arrived():
    one = [Chunk((Region(0, 64),))]
    view, (_, arrivals, _, _) = _view(1, [one, one], round=2)
    out = torch.zeros(1, dtype=torch.int32, device="cuda")
    arrivals[0] = 3  # a later round's arrival also counts (>=)
    _wait_kernel[(1,)](view, out, 0)
    assert out.item() == 1


def test_sent_waits_for_every_peer():
    one = [Chunk((Region(0, 64),))]
    out = torch.zeros(1, dtype=torch.int32, device="cuda")
    view, (_, _, completion, _) = _view(0, [one, [], []], round=2)
    completion[1] = 2
    _sent_kernel[(1,)](view, out)
    assert out.item() == 0
    completion[2] = 2
    _sent_kernel[(1,)](view, out)
    assert out.item() == 1
    out.zero_()
    _wait_sent_kernel[(1,)](view, out)
    assert out.item() == 1

    # A rank that sends nothing has always sent everything
    view, _ = _view(1, [one, [], []], round=2)
    _sent_kernel[(1,)](view, out)
    assert out.item() == 1


@gluon.jit
def _group_arrived_kernel(handle, out, group):
    gl.store(out, TriggeredView.initialize(handle).group_arrived(group).to(gl.int32))


def test_recipients_and_groups():
    """Rank 2 of 3: chunks not sent to it have always arrived; a group arrives once all its chunks sent here have."""
    schedules = [
        [Chunk((Region(0, 64),), peers=(1, 2), group=0), Chunk((Region(64, 64),), peers=(2,), group=0)],
        [Chunk((Region(128, 64),), peers=(0,), group=0), Chunk((Region(192, 64),), peers=(2,))],
        [],
    ]
    view, (_, arrivals, _, groups) = _view(2, schedules, round=2)
    out = torch.zeros(1, dtype=torch.int32, device="cuda")
    assert _arrived(view, 4) == [0, 0, 1, 0]  # chunk 2 goes to rank 0 only

    groups[0] = 3  # two chunks of group 0 come here, so round 2 needs 4
    _group_arrived_kernel[(1,)](view, out, 0)
    assert out.item() == 0 and _arrived(view, 4) == [0, 0, 1, 0]
    groups[0] = 4
    _group_arrived_kernel[(1,)](view, out, 0)
    assert out.item() == 1 and _arrived(view, 4) == [1, 1, 1, 0]
    arrivals[3] = 2
    assert _arrived(view, 4) == [1, 1, 1, 1]
