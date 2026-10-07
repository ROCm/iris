# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.

# TriggeredView against an ordinary gate table: no copy engine involved.

import torch
import triton
import triton.language as tl

from iris.mem.triton import triggered as dv
from iris.triggered import Batch, Segment, TriggeredView
from iris.triggered.plan import view_words


def _view(rank, schedules, epoch):
    num_batches = sum(len(s) for s in schedules)
    gates = torch.zeros(num_batches, dv.GATE_WORDS.value, dtype=torch.int64, device="cuda")
    words = view_words(rank, len(schedules), gates.data_ptr(), schedules)
    words[dv.H_EPOCH.value] = epoch
    return torch.tensor(words, dtype=torch.int64, device="cuda"), gates


@triton.jit
def _publish_kernel(view, batch):
    TriggeredView.initialize(view).publish(batch)


@triton.jit
def _arrived_kernel(view, out, n):
    tv = TriggeredView.initialize(view)
    for b in range(n):
        tl.store(out + b, tv.arrived(b).to(tl.int32))


def test_publish_opens_gate_on_last_contributor():
    k = 4
    view, gates = _view(0, [[Batch((Segment(0, 64),), contributors=k)]], epoch=1)
    _publish_kernel[(k - 1,)](view, 0)
    assert gates[0, dv.G_READY.value].item() == 0
    _publish_kernel[(1,)](view, 0)
    assert gates[0, dv.G_READY.value].item() == 1

    # The counter is never reset: the next epoch needs exactly k more publishes
    view[dv.H_EPOCH.value] = 2
    _publish_kernel[(k - 1,)](view, 0)
    assert gates[0, dv.G_READY.value].item() == 1
    _publish_kernel[(1,)](view, 0)
    assert gates[0, dv.G_READY.value].item() == 2
    assert gates[0, dv.G_COUNTER.value].item() == 2 * k


def test_arrived_reads_ready_for_own_batches_and_arrival_for_others():
    one = [Batch((Segment(0, 64),))]
    view, gates = _view(1, [one, one], epoch=3)
    out = torch.zeros(2, dtype=torch.int32, device="cuda")

    gates[0, dv.G_READY.value] = 3  # another rank's gate: not this rank's to observe
    gates[1, dv.G_ARRIVAL.value] = 3  # own batch: arrivals happen at peers, not here
    _arrived_kernel[(1,)](view, out, 2)
    assert out.tolist() == [0, 0]

    gates[0, dv.G_ARRIVAL.value] = 3
    gates[1, dv.G_READY.value] = 3
    _arrived_kernel[(1,)](view, out, 2)
    assert out.tolist() == [1, 1]
