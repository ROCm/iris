# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.

"""
Schedules for triggered SDMA plans.

A schedule is the list of batches one rank sends, in the order its producer completes them. Each
batch is one gate: once it has received ``contributors`` publishes, the copy engine copies every
segment of the batch to each peer and signals the batch's arrival there.

Chains on one copy-engine queue run strictly in arming order, so a batch completed early waits for
every batch armed before it. Order the schedule the way the producer finishes its work.
"""

from dataclasses import dataclass

# COPY_LINEAR's count field is 30 bits on SDMA 4.4.x
MAX_LINEAR_BYTES = 1 << 30


@dataclass(frozen=True)
class Segment:
    """
    One copy: ``height`` rows of ``width`` bytes, ``pitch`` bytes apart, starting ``offset`` bytes
    into the plan's buffer. The same bytes are written at the same offset in each peer's buffer.
    """

    offset: int
    width: int
    height: int = 1
    pitch: int = 0

    @property
    def row_pitch(self):
        return self.pitch or self.width

    @property
    def end(self):
        return self.offset + (self.height - 1) * self.row_pitch + self.width


@dataclass(frozen=True)
class Batch:
    """A group of segments behind one gate, complete after ``contributors`` publishes per epoch."""

    segments: tuple
    contributors: int = 1

    @property
    def is_2d(self):
        return any(s.height > 1 for s in self.segments)


def _byte_offset(view, base):
    base = view if base is None else base
    offset = view.data_ptr() - base.data_ptr()
    if offset < 0 or view.untyped_storage().data_ptr() != base.untyped_storage().data_ptr():
        raise ValueError("view must lie inside base")
    return offset


def _linear(offset, nbytes):
    return tuple(
        Segment(offset + start, min(MAX_LINEAR_BYTES, nbytes - start)) for start in range(0, nbytes, MAX_LINEAR_BYTES)
    )


def whole(view, base=None, contributors=1):
    """One batch covering a contiguous ``view``, published ``contributors`` times per epoch."""
    if not view.is_contiguous():
        raise ValueError("whole() needs a contiguous view")
    return [Batch(_linear(_byte_offset(view, base), view.numel() * view.element_size()), contributors)]


def chunks(view, num_chunks, base=None, contributors=1):
    """
    Split a contiguous ``view`` along dim 0 into ``num_chunks`` batches, in order. Suits producers
    that finish row blocks in order: row-block, streaming and elementwise kernels.
    """
    if not view.is_contiguous():
        raise ValueError("chunks() needs a contiguous view")
    rows = view.shape[0] if view.dim() else 1
    row_bytes = view.numel() * view.element_size() // max(rows, 1)
    offset = _byte_offset(view, base)
    per_chunk = -(-rows // num_chunks)
    return [
        Batch(_linear(offset + r * row_bytes, (min(r + per_chunk, rows) - r) * row_bytes), contributors)
        for r in range(0, rows, per_chunk)
    ]


def waves(view, block_shape, num_programs, order=None, base=None):
    """
    One batch per wave of a persistent tiled producer.

    ``view`` is a 2D tensor with unit column stride, split into ``block_shape`` tiles (edge tiles
    clipped). Tile ``order[i]`` (row-major tile ids by default) is produced by program
    ``i % num_programs`` in wave ``i // num_programs``. Each wave's batch holds one sub-window per
    tile and completes after one publish per tile.

    A wave is complete only once every program has finished its tile in it, which each does after
    its tile in the previous wave, so waves always complete in order.
    """
    if view.dim() != 2 or view.stride(1) != 1:
        raise ValueError("waves() needs a 2D view with unit column stride")
    rows, cols = view.shape
    block_m, block_n = block_shape
    tiles_n = -(-cols // block_n)
    num_tiles = -(-rows // block_m) * tiles_n
    order = range(num_tiles) if order is None else list(order)
    if sorted(order) != list(range(num_tiles)):
        raise ValueError(f"order must be a permutation of the {num_tiles} tile ids")

    elem = view.element_size()
    pitch = view.stride(0) * elem
    offset = _byte_offset(view, base)
    batches = []
    for start in range(0, num_tiles, num_programs):
        segments = []
        for tile in order[start : start + num_programs]:
            m0, n0 = (tile // tiles_n) * block_m, (tile % tiles_n) * block_n
            height, width = min(block_m, rows - m0), min(block_n, cols - n0)
            segments.append(Segment(offset + m0 * pitch + n0 * elem, width * elem, height, pitch))
        batches.append(Batch(tuple(segments), len(segments)))
    return batches


def validate(schedule, buffer_bytes):
    """Check every batch of ``schedule`` fits a buffer of ``buffer_bytes`` and can be encoded."""
    for i, batch in enumerate(schedule):
        if not batch.segments:
            raise ValueError(f"batch {i} has no segments")
        if batch.contributors < 1:
            raise ValueError(f"batch {i} needs at least one contributor")
        for s in batch.segments:
            if s.width <= 0 or s.height <= 0 or s.offset < 0:
                raise ValueError(f"batch {i}: empty or negative segment {s}")
            if s.pitch and s.pitch < s.width:
                raise ValueError(f"batch {i}: pitch smaller than width in {s}")
            if s.end > buffer_bytes:
                raise ValueError(f"batch {i}: segment {s} runs past the buffer ({buffer_bytes} bytes)")
            if not batch.is_2d and s.width > MAX_LINEAR_BYTES:
                raise ValueError(f"batch {i}: linear segment over {MAX_LINEAR_BYTES} bytes; split it")
    return schedule
