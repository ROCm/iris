# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.

"""
Schedules for triggered plans.

A schedule is the list of chunks one rank sends, in the order its producer completes them. Once a
chunk has been marked ready by all of its ``writers``, the copy engine copies every region of the
chunk to each peer and signals the chunk's arrival there.

All of a rank's copies to one peer run in schedule order, so a chunk completed early waits for
every chunk before it. Order the schedule the way the producer finishes its work.
"""

from dataclasses import dataclass

import numpy as np

# COPY_LINEAR's count field is 30 bits on SDMA 4.4.x
MAX_LINEAR_BYTES = 1 << 30


@dataclass(frozen=True)
class Region:
    """
    One copy: ``rows`` rows of ``row_bytes`` bytes, ``stride_bytes`` apart (packed if ``None``),
    starting ``offset_bytes`` into the plan's buffer. The same bytes are written at the same offset
    in each peer's buffer.
    """

    offset_bytes: int
    row_bytes: int
    rows: int = 1
    stride_bytes: int | None = None

    @property
    def row_stride(self):
        """Bytes between the starts of consecutive rows (``row_bytes`` if packed)."""
        return self.row_bytes if self.stride_bytes is None else self.stride_bytes

    @property
    def end(self):
        """Byte offset one past the last byte this region copies."""
        return self.offset_bytes + (self.rows - 1) * self.row_stride + self.row_bytes


@dataclass(frozen=True)
class Chunk:
    """Regions sent together once all ``writers`` have marked the chunk ready, every round."""

    regions: tuple
    writers: int = 1

    @property
    def is_2d(self):
        """True if any region has more than one row."""
        return any(r.rows > 1 for r in self.regions)


def _byte_offset(view, base):
    base = view if base is None else base
    offset = view.data_ptr() - base.data_ptr()
    if offset < 0 or view.untyped_storage().data_ptr() != base.untyped_storage().data_ptr():
        raise ValueError("view must lie inside base")
    return offset


def _linear(offset, nbytes):
    return tuple(
        Region(offset + start, min(MAX_LINEAR_BYTES, nbytes - start)) for start in range(0, nbytes, MAX_LINEAR_BYTES)
    )


def whole(view, base=None, writers=1):
    """One chunk covering a contiguous ``view``, marked ready by ``writers`` programs per round."""
    if not view.is_contiguous():
        raise ValueError("whole() needs a contiguous view")
    return [Chunk(_linear(_byte_offset(view, base), view.numel() * view.element_size()), writers)]


def chunks(view, n, base=None, writers=1):
    """
    Split a contiguous ``view`` along dim 0 into ``n`` chunks, in order. Suits producers that
    finish row blocks in order: row-block, streaming and elementwise kernels.
    """
    if not view.is_contiguous():
        raise ValueError("chunks() needs a contiguous view")
    rows = view.shape[0] if view.dim() else 1
    row_bytes = view.numel() * view.element_size() // max(rows, 1)
    offset = _byte_offset(view, base)
    per_chunk = -(-rows // n)
    return [
        Chunk(_linear(offset + r * row_bytes, (min(r + per_chunk, rows) - r) * row_bytes), writers)
        for r in range(0, rows, per_chunk)
    ]


def tile_chunks(view, block_shape, tiles_per_chunk, order=None, base=None):
    """
    Group the tiles of a tiled producer into chunks of ``tiles_per_chunk``, in completion order.

    ``view`` is a 2D tensor with unit column stride, split into ``block_shape`` tiles (edge tiles
    clipped; tile ids are row-major). ``order`` gives the tile the producer makes at each step
    ``i``: a list of tile ids, or a function of ``i``. Chunk ``k`` holds the tiles of steps
    ``k * tiles_per_chunk`` up to the next chunk, one region per tile, and is marked ready once per
    tile. The producer must mark the tile of step ``i`` ready as part of chunk
    ``i // tiles_per_chunk``, so pass the mapping the kernel itself uses: a ``@triton.jit`` mapping
    written with plain integer arithmetic runs on the host as ``lambda i: mapping.fn(i, ...)``.

    For a persistent kernel of ``P`` programs where program ``p`` takes steps ``p``, ``p + P``, ...,
    use ``tiles_per_chunk=P``: chunk ``k`` is every program's ``k``-th tile, so chunks complete in
    order.
    """
    if view.dim() != 2 or view.stride(1) != 1:
        raise ValueError("tile_chunks() needs a 2D view with unit column stride")
    rows, cols = view.shape
    block_m, block_n = block_shape
    tiles_n = -(-cols // block_n)
    num_tiles = -(-rows // block_m) * tiles_n
    if order is None:
        order = range(num_tiles)
    elif callable(order):
        order = [int(order(i)) for i in range(num_tiles)]
    else:
        order = list(order)
    if sorted(order) != list(range(num_tiles)):
        raise ValueError(f"order must be a permutation of the {num_tiles} tile ids")

    elem = view.element_size()
    stride = view.stride(0) * elem
    offset = _byte_offset(view, base)
    result = []
    for start in range(0, num_tiles, tiles_per_chunk):
        regions = []
        for tile in order[start : start + tiles_per_chunk]:
            m0, n0 = (tile // tiles_n) * block_m, (tile % tiles_n) * block_n
            height, width = min(block_m, rows - m0), min(block_n, cols - n0)
            regions.append(Region(offset + m0 * stride + n0 * elem, width * elem, height, stride))
        result.append(Chunk(tuple(regions), len(regions)))
    return result


def validate(schedule, buffer_bytes):
    """Check every chunk of ``schedule`` fits a buffer of ``buffer_bytes`` and can be encoded."""
    for i, chunk in enumerate(schedule):
        if not chunk.regions:
            raise ValueError(f"chunk {i} has no regions")
        if chunk.writers < 1:
            raise ValueError(f"chunk {i} needs at least one writer")
        for r in chunk.regions:
            if r.row_bytes <= 0 or r.rows <= 0 or r.offset_bytes < 0:
                raise ValueError(f"chunk {i}: empty or negative region {r}")
            if r.stride_bytes is not None and r.stride_bytes < r.row_bytes:
                raise ValueError(f"chunk {i}: stride_bytes smaller than row_bytes in {r}")
            if r.end > buffer_bytes:
                raise ValueError(f"chunk {i}: region {r} runs past the buffer ({buffer_bytes} bytes)")
            if not chunk.is_2d and r.row_bytes > MAX_LINEAR_BYTES:
                raise ValueError(f"chunk {i}: linear region over {MAX_LINEAR_BYTES} bytes; split it")
    return schedule


def _rows(schedule):
    """Every row of every region in ``schedule``, merged into sorted disjoint [start, end) byte ranges."""
    regions = [r for chunk in schedule for r in chunk.regions]
    offset, width, rows, stride = (
        np.array(v, dtype=np.int64)
        for v in zip(*((r.offset_bytes, r.row_bytes, r.rows, r.row_stride) for r in regions))
    )
    region = np.repeat(np.arange(len(regions)), rows)
    row = np.arange(len(region)) - np.repeat(np.cumsum(rows) - rows, rows)
    starts = offset[region] + row * stride[region]
    order = np.argsort(starts, kind="stable")
    starts, ends = starts[order], starts[order] + width[region][order]
    # A row starts a new range unless it begins inside the ranges before it
    first = np.flatnonzero(np.r_[True, starts[1:] >= np.maximum.accumulate(ends)[:-1]])
    return starts[first], np.maximum.reduceat(ends, first)


def check_disjoint(schedules):
    """
    Check that no byte belongs to the schedules of two different ranks: every chunk is copied to the
    same offsets on every peer, so two senders of one byte race, and each overwrites the other's own
    data. Regions of one rank may overlap.
    """
    owners = [rank for rank, schedule in enumerate(schedules) if schedule]
    extents = sorted(
        (
            min(r.offset_bytes for c in schedules[rank] for r in c.regions),
            max(r.end for c in schedules[rank] for r in c.regions),
        )
        for rank in owners
    )
    if all(prev[1] <= cur[0] for prev, cur in zip(extents, extents[1:])):
        return
    # Ranks' extents interleave (e.g. they own column blocks): compare their rows exactly
    ranges = [_rows(schedules[rank]) for rank in owners]
    starts = np.concatenate([s for s, _ in ranges])
    ends = np.concatenate([e for _, e in ranges])
    rank = np.repeat(owners, [len(s) for s, _ in ranges])
    order = np.argsort(starts, kind="stable")
    starts, ends, rank = starts[order], ends[order], rank[order]
    # Each rank's ranges are disjoint, so any overlap shows up between neighbors
    clash = np.flatnonzero(starts[1:] < ends[:-1])
    if len(clash):
        i = clash[0]
        raise ValueError(
            f"ranks {rank[i]} and {rank[i + 1]} both send bytes {starts[i + 1]} to {min(ends[i], ends[i + 1])} of the "
            "buffer; each rank's schedule must cover only the part of the buffer it writes"
        )
