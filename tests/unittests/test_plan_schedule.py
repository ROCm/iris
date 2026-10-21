# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.

# Schedule builders and validation for collective plans (host only).

import pytest
import torch

from iris.plan import Chunk, Region, chunks, tile_chunks, whole
from iris.plan.schedule import MAX_LINEAR_BYTES, validate


def test_whole():
    base = torch.zeros(4, 64, dtype=torch.float32)
    (chunk,) = whole(base[2:], base=base, writers=3)
    assert chunk == Chunk((Region(2 * 256, 2 * 256),), 3)
    assert not chunk.is_2d


def test_chunks_uneven():
    base = torch.zeros(10, 16, dtype=torch.int32)
    result = chunks(base, 3)
    # 10 rows in chunks of 4: rows 0-3, 4-7, 8-9
    assert [c.regions for c in result] == [
        (Region(0, 4 * 64),),
        (Region(4 * 64, 4 * 64),),
        (Region(8 * 64, 2 * 64),),
    ]


def test_tile_chunks_clips_edge_tiles():
    base = torch.zeros(2 * 40, 72, dtype=torch.float32)
    view = base[40:]
    result = tile_chunks(view, (16, 32), tiles_per_chunk=4, base=base)
    # 3 x 3 tiles in chunks of 4, 4, 1
    assert [c.writers for c in result] == [4, 4, 1]
    stride = 72 * 4
    first = result[0].regions
    assert first[0] == Region(40 * stride, 32 * 4, 16, stride)
    assert first[2] == Region(40 * stride + 64 * 4, 8 * 4, 16, stride)  # last column: 8 wide
    assert result[2].regions == (Region(40 * stride + 32 * stride + 64 * 4, 8 * 4, 8, stride),)  # corner: 8 x 8
    assert all(c.is_2d for c in result)


def test_tile_chunks_order():
    base = torch.zeros(32, 32, dtype=torch.float16)
    result = tile_chunks(base, (16, 16), tiles_per_chunk=2, order=[3, 2, 1, 0])
    assert result[0].regions[0].offset_bytes == 16 * 64 + 32
    with pytest.raises(ValueError):
        tile_chunks(base, (16, 16), tiles_per_chunk=2, order=[0, 1, 2])


def test_region_stride_defaults_to_packed():
    assert Region(0, 64, 2).end == 128
    assert Region(0, 64, 2, 256).end == 256 + 64


@pytest.mark.parametrize(
    "chunk, buffer_bytes",
    [
        (Chunk(()), 1024),
        (Chunk((Region(0, 64),), writers=0), 1024),
        (Chunk((Region(0, 0),)), 1024),
        (Chunk((Region(0, 64, 2, 32),)), 1024),  # stride_bytes < row_bytes
        (Chunk((Region(1000, 64),)), 1024),  # past the end
        (Chunk((Region(0, MAX_LINEAR_BYTES + 1),)), 1 << 32),
    ],
)
def test_validate_rejects(chunk, buffer_bytes):
    with pytest.raises(ValueError):
        validate([chunk], buffer_bytes)
