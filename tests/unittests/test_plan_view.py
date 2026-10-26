# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.

# TriggeredView against an ordinary gate table: no copy engine involved.

import torch
import triton
import triton.language as tl

from iris.experimental.plan import triton as dv
from iris.experimental.plan import Chunk, Region, TriggeredView, kernels
from iris.experimental.plan.triggered import view_words


def _view(rank, schedules, round):
    num_chunks = sum(len(s) for s in schedules)
    gates = torch.zeros(num_chunks, dv.GATE_WORDS.value, dtype=torch.int64, device="cuda")
    words = view_words(rank, len(schedules), gates.data_ptr(), schedules)
    words[dv.H_ROUND.value] = round
    return torch.tensor(words, dtype=torch.int64, device="cuda"), gates


@triton.jit
def _mark_ready_kernel(handle, chunk):
    TriggeredView.initialize(handle).mark_ready(chunk)


@triton.jit
def _arrived_kernel(handle, out, n):
    cv = TriggeredView.initialize(handle)
    for c in range(n):
        tl.store(out + c, cv.arrived(c).to(tl.int32) + 2 * cv.owns(c).to(tl.int32))


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


def test_release_counts_every_writer():
    """run() and close() open gates directly; the counter must still advance by writers per round."""
    k = 3
    view, gates = _view(0, [[Chunk((Region(0, 64),), writers=k)]], round=1)
    chunks = torch.zeros(1, dtype=torch.int64, device="cuda")
    kernels.release_kernel[(1,)](view, chunks)
    assert gates[0, dv.G_READY.value].item() == 1
    assert gates[0, dv.G_COUNTER.value].item() == k

    # Writers' marks in the next round still open the gate on the k-th mark
    view[dv.H_ROUND.value] = 2
    _mark_ready_kernel[(k - 1,)](view, 0)
    assert gates[0, dv.G_READY.value].item() == 1
    _mark_ready_kernel[(1,)](view, 0)
    assert gates[0, dv.G_READY.value].item() == 2
