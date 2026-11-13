# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.

import pytest
import torch

from iris.ccl.sdma.plans import _check, _parts
from iris.experimental.plan import whole

WORLD = 4


def _tensor(n=WORLD, rows=8, cols=16):
    return torch.zeros(n * rows, cols, dtype=torch.int32)


def test_parts_split_dim_0_evenly():
    tensor = _tensor()
    parts = _parts(tensor, WORLD)
    assert [p.data_ptr() - tensor.data_ptr() for p in parts] == [i * 8 * 16 * 4 for i in range(WORLD)]
    with pytest.raises(ValueError, match="multiple of 3"):
        _parts(tensor, 3)
    with pytest.raises(ValueError, match="contiguous"):
        _parts(tensor.t(), 2)


def test_default_schedule_is_one_whole_chunk_per_part():
    tensor = _tensor()
    parts = _parts(tensor, WORLD)
    sends = {(r,): parts[r] for r in (1, 2, 3)}  # a scatter from rank 0
    schedule = _check(None, tensor, sends, 0, WORLD)
    assert [c.recipients(0, WORLD) for c in schedule] == [(1,), (2,), (3,)]
    assert [c.regions[0].offset_bytes for c in schedule] == [512, 1024, 1536]


@pytest.mark.parametrize(
    "case",
    ["wrong peers", "outside the part", "partial", "lands elsewhere", "a rank that sends nothing", "missing a part"],
)
def test_bad_schedules(case):
    tensor = _tensor()
    parts = _parts(tensor, WORLD)
    sends = {(0,): parts[1]}  # rank 1's part of a gather to rank 0
    schedule = {
        "wrong peers": whole(parts[1], base=tensor, peers=(2,)),
        "outside the part": whole(tensor[:16], base=tensor, peers=(0,)),
        "partial": whole(parts[1][:-1], base=tensor, peers=(0,)),
        "lands elsewhere": whole(parts[1], base=tensor, peers=(0,), dst=parts[2], dst_base=tensor),
        "a rank that sends nothing": whole(parts[0], base=tensor, peers=(1,)),
        "missing a part": whole(parts[1], base=tensor, peers=(0,)),
    }[case]
    if case == "a rank that sends nothing":
        sends = {}
    if case == "missing a part":
        sends = {(0,): parts[1], (2,): parts[2]}
    with pytest.raises(ValueError):
        _check(schedule, tensor, sends, 1, WORLD)
