# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.

# Schedule builders and validation for triggered plans (host only).

import pytest
import torch

from iris.triggered import Batch, Segment, chunks, waves, whole
from iris.triggered.schedule import MAX_LINEAR_BYTES, validate


def test_whole():
    base = torch.zeros(4, 64, dtype=torch.float32)
    (batch,) = whole(base[2:], base=base, contributors=3)
    assert batch == Batch((Segment(2 * 256, 2 * 256),), 3)
    assert not batch.is_2d


def test_chunks_uneven():
    base = torch.zeros(10, 16, dtype=torch.int32)
    batches = chunks(base, 3)
    # 10 rows in chunks of 4: rows 0-3, 4-7, 8-9
    assert [b.segments for b in batches] == [
        (Segment(0, 4 * 64),),
        (Segment(4 * 64, 4 * 64),),
        (Segment(8 * 64, 2 * 64),),
    ]


def test_waves_clips_edge_tiles():
    base = torch.zeros(2 * 40, 72, dtype=torch.float32)
    view = base[40:]
    batches = waves(view, (16, 32), num_programs=4, base=base)
    # 3 x 3 tiles in waves of 4, 4, 1
    assert [b.contributors for b in batches] == [4, 4, 1]
    pitch = 72 * 4
    first = batches[0].segments
    assert first[0] == Segment(40 * pitch, 32 * 4, 16, pitch)
    assert first[2] == Segment(40 * pitch + 64 * 4, 8 * 4, 16, pitch)  # last column: 8 wide
    assert batches[2].segments == (Segment(40 * pitch + 32 * pitch + 64 * 4, 8 * 4, 8, pitch),)  # corner: 8 x 8
    assert all(b.is_2d for b in batches)


def test_waves_order():
    base = torch.zeros(32, 32, dtype=torch.float16)
    batches = waves(base, (16, 16), num_programs=2, order=[3, 2, 1, 0])
    assert batches[0].segments[0].offset == 16 * 64 + 32
    with pytest.raises(ValueError):
        waves(base, (16, 16), num_programs=2, order=[0, 1, 2])


@pytest.mark.parametrize(
    "batch, buffer_bytes",
    [
        (Batch(()), 1024),
        (Batch((Segment(0, 64),), contributors=0), 1024),
        (Batch((Segment(0, 0),)), 1024),
        (Batch((Segment(0, 64, 2, 32),)), 1024),  # pitch < width
        (Batch((Segment(1000, 64),)), 1024),  # past the end
        (Batch((Segment(0, MAX_LINEAR_BYTES + 1),)), 1 << 32),
    ],
)
def test_validate_rejects(batch, buffer_bytes):
    with pytest.raises(ValueError):
        validate([batch], buffer_bytes)
