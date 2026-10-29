# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.

# Schedule builders and validation for triggered plans (host only).

import pytest
import torch
import triton
import triton.language as tl

from iris.experimental.plan import Chunk, Region, chunks, tile_chunks, whole
from iris.experimental.plan.schedule import MAX_LINEAR_BYTES, check_disjoint, validate


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


@triton.jit
def _grouped(i, tiles_m, tiles_n, GROUP_M: tl.constexpr):
    group = i // (GROUP_M * tiles_n)
    size_m = min(tiles_m - group * GROUP_M, GROUP_M)
    within = i % (GROUP_M * tiles_n)
    return (group * GROUP_M + within % size_m) * tiles_n + within // size_m


def test_tile_chunks_order_from_kernel_mapping():
    """The kernel's own @triton.jit tile mapping gives the same schedule as the list it produces."""
    base = torch.zeros(5 * 16, 3 * 16, dtype=torch.float16)
    order = [_grouped.fn(i, 5, 3, 2) for i in range(15)]
    assert order != list(range(15))
    by_fn = tile_chunks(base, (16, 16), tiles_per_chunk=4, order=lambda i: _grouped.fn(i, 5, 3, 2))
    assert by_fn == tile_chunks(base, (16, 16), tiles_per_chunk=4, order=order)


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


def test_check_disjoint_row_slices():
    out = torch.zeros(4 * 16, 32, dtype=torch.float32)
    check_disjoint([chunks(out[r * 16 : (r + 1) * 16], 2, base=out) for r in range(4)] + [[]])


def test_check_disjoint_rejects_whole_buffer_on_every_rank():
    """The usual slip: every rank describes all of the buffer instead of its own slice."""
    out = torch.zeros(4 * 16, 32, dtype=torch.float32)
    with pytest.raises(ValueError, match="ranks 0 and 1 both send bytes 0 to"):
        check_disjoint([chunks(out, 2) for _ in range(4)])


def test_check_disjoint_interleaved_columns():
    """Ranks owning column blocks of the same rows interleave in extent but not in bytes."""
    out = torch.zeros(48, 4 * 32, dtype=torch.float32)
    columns = [tile_chunks(out[:, r * 32 : (r + 1) * 32], (16, 32), 3, base=out) for r in range(4)]
    check_disjoint(columns)

    # Rank 1 one column too wide: its last column overlaps rank 2's first
    wide = tile_chunks(out[:, 32 : 2 * 32 + 1], (16, 33), 3, base=out)
    with pytest.raises(ValueError, match="ranks 1 and 2 both send bytes 256 to 260"):
        check_disjoint([columns[0], wide, columns[2], columns[3]])


def test_check_disjoint_allows_overlap_within_a_rank():
    one = [Chunk((Region(0, 64),)), Chunk((Region(32, 64, 2, 128),))]
    check_disjoint([one, [Chunk((Region(256, 64),))]])


def test_region_destination_defaults_to_source():
    r = Region(128, 64, 2, 256)
    assert (r.dst_offset, r.dst_row_stride, r.dst_end) == (128, 256, r.end)
    r = Region(128, 64, 2, 256, dst_offset_bytes=0, dst_stride_bytes=64)
    assert (r.dst_offset, r.dst_row_stride, r.dst_end) == (0, 64, 128)


def test_builders_land_in_dst():
    base = torch.zeros(4, 64, dtype=torch.float32)
    staging = torch.zeros(8, 64, dtype=torch.float32)
    (chunk,) = whole(base[2:], base=base, dst=staging[4:6], dst_base=staging, peers=[3, 1])
    assert chunk.regions == (Region(2 * 256, 2 * 256, dst_offset_bytes=4 * 256),)
    assert chunk.peers == (1, 3)
    assert [c.regions[0].dst_offset for c in chunks(base, 2, dst=staging[2:6], dst_base=staging)] == [512, 1024]
    with pytest.raises(ValueError, match="shape"):
        whole(base, dst=staging)


def test_tile_chunks_dst_pitch_and_group():
    """Tiles of a (32, 48) slice land in a staging buffer with a wider row pitch, at a column offset."""
    out = torch.zeros(32, 48, dtype=torch.float16)
    staging = torch.zeros(32, 2 * 64, dtype=torch.float16)
    dst = staging[:, 64 : 64 + 48]
    result = tile_chunks(out, (16, 16), tiles_per_chunk=3, dst=dst, dst_base=staging, group=5)
    assert all(c.group == 5 and c.peers is None for c in result)
    tile = result[1].regions[1]  # tile 4: row block 1, column block 1
    assert tile.offset_bytes == 16 * 96 + 32 and tile.row_stride == 96
    assert tile.dst_offset == 16 * 256 + 128 + 32 and tile.dst_row_stride == 256


@pytest.mark.parametrize(
    "chunk",
    [
        Chunk((Region(0, 64, dst_offset_bytes=1000),)),  # past the destination
        Chunk((Region(0, 64, 2, dst_stride_bytes=32),)),  # destination stride < row_bytes
        Chunk((Region(0, 64),), peers=()),
        Chunk((Region(0, 64),), peers=(0,)),  # the sending rank itself
        Chunk((Region(0, 64),), peers=(4,)),
        Chunk((Region(0, 64),), group=-1),
    ],
)
def test_validate_rejects_destinations(chunk):
    with pytest.raises(ValueError):
        validate([chunk], 1024, 1024, rank=0, world_size=4)


def _inbox_schedules(world, block, slot_of):
    """Rank s sends block p of its buffer to rank p only, into the inbox slot ``slot_of(s)``."""
    return [
        [
            Chunk((Region(p * block, block, dst_offset_bytes=slot_of(s) * block),), peers=(p,))
            for p in range(world)
            if p != s
        ]
        for s in range(world)
    ]


def test_check_disjoint_per_receiver():
    """A reduce-scatter's inboxes: every sender lands in its own slot of each receiver's destination."""
    check_disjoint(_inbox_schedules(4, 64, lambda s: s), same_buffer=False)
    # Ranks 1 and 2 share slot 1: they would race on ranks 0 and 3
    with pytest.raises(ValueError, match="ranks 1 and 2 both send bytes 64 to 128 of rank 0's destination"):
        check_disjoint(_inbox_schedules(4, 64, lambda s: min(s, 1)), same_buffer=False)


def test_check_disjoint_same_buffer_protects_own_data():
    """With the buffer as destination, what lands on a rank may not cover the bytes it sends."""
    check_disjoint([[Chunk((Region(0, 64),), peers=(1,))], [Chunk((Region(64, 64),), peers=(2,))], []])
    # Rank 0's bytes land on rank 1 over rank 1's own; rank 2 does not send, so landing there is fine
    clash = [[Chunk((Region(0, 64),), peers=(1,))], [Chunk((Region(32, 64),), peers=(2,))], []]
    with pytest.raises(ValueError, match="ranks 0 and 1 both send bytes 32 to 64 of the buffer"):
        check_disjoint(clash)
