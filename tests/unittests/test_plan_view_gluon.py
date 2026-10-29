# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.

# Gluon TriggeredView against an ordinary gate table: no copy engine involved.

import torch
from triton.experimental import gluon
from triton.experimental.gluon import language as gl

from iris.experimental.plan.gluon import TriggeredView
from iris.experimental.plan import triton as dv
from iris.experimental.plan import Chunk, Region
from iris.experimental.plan.triggered import view_words


def _view(rank, schedules, round):
    num_chunks = sum(len(s) for s in schedules)
    gates = torch.zeros(num_chunks, dv.GATE_WORDS.value, dtype=torch.int64, device="cuda")
    words = view_words(rank, len(schedules), gates.data_ptr(), schedules)
    words[dv.H_ROUND.value] = round
    return torch.tensor(words, dtype=torch.int64, device="cuda"), gates


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


def test_mark_ready_opens_gate_on_last_writer():
    k = 4
    view, gates = _view(0, [[Chunk((Region(0, 64),), writers=k)]], round=1)
    _mark_ready_kernel[(k - 1,)](view, 0)
    assert gates[0, dv.G_READY.value].item() == 0
    _mark_ready_kernel[(1,)](view, 0)
    assert gates[0, dv.G_READY.value].item() == 1

    # The counter is never reset: the next round needs exactly k more marks
    view[dv.H_ROUND.value] = 2
    _mark_ready_kernel[(k - 1,)](view, 0)
    assert gates[0, dv.G_READY.value].item() == 1
    _mark_ready_kernel[(1,)](view, 0)
    assert gates[0, dv.G_READY.value].item() == 2
    assert gates[0, dv.G_COUNTER.value].item() == 2 * k


def test_arrived_reads_ready_for_own_chunks_and_arrival_for_others():
    one = [Chunk((Region(0, 64),))]
    view, gates = _view(1, [one, one], round=3)
    out = torch.zeros(2, dtype=torch.int32, device="cuda")

    gates[0, dv.G_READY.value] = 3  # another rank's gate: not this rank's to observe
    gates[1, dv.G_ARRIVAL.value] = 3  # own chunk: arrivals happen at peers, not here
    _arrived_kernel[(1,)](view, out, 2)
    assert out.tolist() == [0, 2]  # chunk 1 is owned, neither has arrived

    gates[0, dv.G_ARRIVAL.value] = 3
    gates[1, dv.G_READY.value] = 3
    _arrived_kernel[(1,)](view, out, 2)
    assert out.tolist() == [1, 3]


def test_wait_returns_once_arrived():
    one = [Chunk((Region(0, 64),))]
    view, gates = _view(1, [one, one], round=2)
    out = torch.zeros(1, dtype=torch.int32, device="cuda")
    gates[0, dv.G_ARRIVAL.value] = 2  # a later round's arrival also counts (>=)
    _wait_kernel[(1,)](view, out, 0)
    assert out.item() == 1
