# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.

"""
Schedules for collective plans.

A schedule is the list of chunks one rank sends, in the order its producer completes them. Once a
chunk has been marked ready by all of its ``writers``, the copy engine copies every region of the
chunk to each peer and signals the chunk's arrival there.

All of a rank's copies to one peer run in schedule order, so a chunk completed early waits for
every chunk before it. Order the schedule the way the producer finishes its work.
"""

from dataclasses import dataclass

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
        return self.row_bytes if self.stride_bytes is None else self.stride_bytes

    @property
    def end(self):
        return self.offset_bytes + (self.rows - 1) * self.row_stride + self.row_bytes


@dataclass(frozen=True)
class Chunk:
    """Regions sent together once all ``writers`` have marked the chunk ready, every round."""

    regions: tuple
    writers: int = 1

    @property
    def is_2d(self):
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
    clipped). ``order`` lists the tile ids (row-major by default) in the order they complete; each
    chunk holds the next ``tiles_per_chunk`` of them, one region per tile, and is marked ready once
    per tile.

    For a persistent kernel of ``P`` programs that each work through their tiles in order, use
    ``tiles_per_chunk=P``: chunk ``k`` is every program's ``k``-th tile, so chunks complete in order.
    """
    if view.dim() != 2 or view.stride(1) != 1:
        raise ValueError("tile_chunks() needs a 2D view with unit column stride")
    rows, cols = view.shape
    block_m, block_n = block_shape
    tiles_n = -(-cols // block_n)
    num_tiles = -(-rows // block_m) * tiles_n
    order = range(num_tiles) if order is None else list(order)
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
