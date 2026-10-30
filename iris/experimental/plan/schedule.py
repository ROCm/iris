# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.

"""
Schedules for triggered plans.

A schedule is the list of chunks one rank sends, in the order its producer completes them. Once a
chunk has been marked ready by all of its ``writers``, the copy engine copies every region of the
chunk to each of its peers and signals the chunk's arrival there.

All of a rank's copies to one peer run in schedule order, so a chunk completed early waits for
every chunk before it. Order the schedule the way the producer finishes its work.
"""

from dataclasses import dataclass, replace

import numpy as np

# COPY_LINEAR's count field is 30 bits on SDMA 4.4.x
MAX_LINEAR_BYTES = 1 << 30


@dataclass(frozen=True)
class Region:
    """
    One copy: ``rows`` rows of ``row_bytes`` bytes, ``stride_bytes`` apart (packed if ``None``),
    starting ``offset_bytes`` into the plan's buffer. Each peer receives them ``dst_offset_bytes``
    into its destination (the plan's ``destination``, by default the buffer itself), rows
    ``dst_stride_bytes`` apart; both default to the source's.
    """

    offset_bytes: int
    row_bytes: int
    rows: int = 1
    stride_bytes: int | None = None
    dst_offset_bytes: int | None = None
    dst_stride_bytes: int | None = None

    @property
    def row_stride(self):
        """Bytes between the starts of consecutive rows (``row_bytes`` if packed)."""
        return self.row_bytes if self.stride_bytes is None else self.stride_bytes

    @property
    def end(self):
        """Byte offset one past the last byte this region copies."""
        return self.offset_bytes + (self.rows - 1) * self.row_stride + self.row_bytes

    @property
    def dst_offset(self):
        """Where the region lands in each peer's destination."""
        return self.offset_bytes if self.dst_offset_bytes is None else self.dst_offset_bytes

    @property
    def dst_row_stride(self):
        """Bytes between the starts of consecutive rows at the destination."""
        return self.row_stride if self.dst_stride_bytes is None else self.dst_stride_bytes

    @property
    def dst_end(self):
        """Destination byte offset one past the last byte this region writes."""
        return self.dst_offset + (self.rows - 1) * self.dst_row_stride + self.row_bytes


@dataclass(frozen=True)
class Chunk:
    """
    Regions sent together once all ``writers`` have marked the chunk ready, every round.

    ``peers`` are the ranks it goes to (every other rank if ``None``). Chunks with the same
    ``group`` (a small integer, shared across ranks) signal one arrival counter at each receiver,
    so a consumer can wait for all of them at once with ``wait_group``.
    """

    regions: tuple
    writers: int = 1
    peers: tuple | None = None
    group: int | None = None

    def __post_init__(self):
        if self.peers is not None:
            object.__setattr__(self, "peers", tuple(sorted(set(self.peers))))

    def recipients(self, sender, world_size):
        """The ranks this chunk goes to, when ``sender`` sends it."""
        return self.peers if self.peers is not None else tuple(p for p in range(world_size) if p != sender)

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


def _linear(offset, nbytes, dst_offset=None):
    return tuple(
        Region(
            offset + start,
            min(MAX_LINEAR_BYTES, nbytes - start),
            dst_offset_bytes=None if dst_offset is None else dst_offset + start,
        )
        for start in range(0, nbytes, MAX_LINEAR_BYTES)
    )


def _dst_offset(view, dst, dst_base, what):
    """Byte offset of ``dst`` in ``dst_base``, after checking it has ``view``'s shape; None without ``dst``."""
    if dst is None:
        return None
    if dst.shape != view.shape or dst.element_size() != view.element_size():
        raise ValueError(f"{what}: dst must have the view's shape and element size")
    if dst_base is None:
        raise ValueError(f"{what}: pass dst_base, the plan's destination, with dst")
    return _byte_offset(dst, dst_base)


def whole(view, base=None, writers=1, dst=None, dst_base=None, peers=None):
    """
    One chunk covering a contiguous ``view``, marked ready by ``writers`` programs per round.

    By default it lands where it is, in each peer's buffer. With ``dst`` (a contiguous view of the
    same shape inside ``dst_base``, the plan's destination), it lands there instead. ``peers``
    limits the ranks it goes to.
    """
    if not view.is_contiguous() or (dst is not None and not dst.is_contiguous()):
        raise ValueError("whole() needs contiguous views")
    dst_offset = _dst_offset(view, dst, dst_base, "whole()")
    nbytes = view.numel() * view.element_size()
    return [Chunk(_linear(_byte_offset(view, base), nbytes, dst_offset), writers, peers)]


def chunks(view, n, base=None, writers=1, dst=None, dst_base=None, peers=None):
    """
    Split a contiguous ``view`` along dim 0 into ``n`` chunks, in order. Suits producers that
    finish row blocks in order: row-block, streaming and elementwise kernels. ``dst``, ``dst_base``
    and ``peers`` are as for ``whole()``.
    """
    if not view.is_contiguous() or (dst is not None and not dst.is_contiguous()):
        raise ValueError("chunks() needs contiguous views")
    dst_offset = _dst_offset(view, dst, dst_base, "chunks()")
    rows = view.shape[0] if view.dim() else 1
    row_bytes = view.numel() * view.element_size() // max(rows, 1)
    offset = _byte_offset(view, base)
    per_chunk = -(-rows // n)
    return [
        Chunk(
            _linear(
                offset + r * row_bytes,
                (min(r + per_chunk, rows) - r) * row_bytes,
                None if dst_offset is None else dst_offset + r * row_bytes,
            ),
            writers,
            peers,
        )
        for r in range(0, rows, per_chunk)
    ]


def tile_chunks(
    view, block_shape, tiles_per_chunk, order=None, base=None, dst=None, dst_base=None, peers=None, group=None
):
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

    With ``dst`` (a 2D view of the same shape and unit column stride inside ``dst_base``, the plan's
    destination), each tile lands at its place in ``dst`` instead, with ``dst``'s row pitch.
    ``peers`` limits the ranks every chunk goes to, and every chunk joins ``group``.
    """
    if view.dim() != 2 or view.stride(1) != 1 or (dst is not None and (dst.dim() != 2 or dst.stride(1) != 1)):
        raise ValueError("tile_chunks() needs 2D views with unit column stride")
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
    dst_offset = _dst_offset(view, dst, dst_base, "tile_chunks()")
    dst_stride = None if dst is None else dst.stride(0) * elem
    result = []
    for start in range(0, num_tiles, tiles_per_chunk):
        regions = []
        for tile in order[start : start + tiles_per_chunk]:
            m0, n0 = (tile // tiles_n) * block_m, (tile % tiles_n) * block_n
            height, width = min(block_m, rows - m0), min(block_n, cols - n0)
            region = Region(offset + m0 * stride + n0 * elem, width * elem, height, stride)
            if dst is not None:
                region = replace(
                    region, dst_offset_bytes=dst_offset + m0 * dst_stride + n0 * elem, dst_stride_bytes=dst_stride
                )
            regions.append(region)
        result.append(Chunk(tuple(regions), len(regions), peers, group))
    return result


def validate(schedule, buffer_bytes, destination_bytes=None, rank=None, world_size=None):
    """
    Check every chunk of ``schedule`` fits a buffer of ``buffer_bytes`` (and a destination of
    ``destination_bytes``, by default the buffer) and can be encoded. With ``rank`` and
    ``world_size``, also check each chunk's ``peers``.
    """
    destination_bytes = buffer_bytes if destination_bytes is None else destination_bytes
    for i, chunk in enumerate(schedule):
        if not chunk.regions:
            raise ValueError(f"chunk {i} has no regions")
        if chunk.writers < 1:
            raise ValueError(f"chunk {i} needs at least one writer")
        if chunk.group is not None and (not isinstance(chunk.group, int) or chunk.group < 0):
            raise ValueError(f"chunk {i}: group must be a non-negative integer, not {chunk.group!r}")
        if chunk.peers is not None:
            if not chunk.peers:
                raise ValueError(f"chunk {i} has no peers")
            if world_size is not None and not all(0 <= p < world_size and p != rank for p in chunk.peers):
                raise ValueError(f"chunk {i}: peers {chunk.peers} must be other ranks, below {world_size}")
        for r in chunk.regions:
            if r.row_bytes <= 0 or r.rows <= 0 or r.offset_bytes < 0 or r.dst_offset < 0:
                raise ValueError(f"chunk {i}: empty or negative region {r}")
            if r.row_stride < r.row_bytes or r.dst_row_stride < r.row_bytes:
                raise ValueError(f"chunk {i}: a stride smaller than row_bytes in {r}")
            if r.end > buffer_bytes:
                raise ValueError(f"chunk {i}: region {r} runs past the buffer ({buffer_bytes} bytes)")
            if r.dst_end > destination_bytes:
                raise ValueError(f"chunk {i}: region {r} runs past the destination ({destination_bytes} bytes)")
            if not chunk.is_2d and r.row_bytes > MAX_LINEAR_BYTES:
                raise ValueError(f"chunk {i}: linear region over {MAX_LINEAR_BYTES} bytes; split it")
    return schedule


def _region_rows(regions, dst):
    """Every row of ``regions`` (at the destination if ``dst``), merged into sorted disjoint [start, end) byte ranges."""
    offset, width, rows, stride = (
        np.array(v, dtype=np.int64)
        for v in zip(
            *(
                (r.dst_offset, r.row_bytes, r.rows, r.dst_row_stride)
                if dst
                else (r.offset_bytes, r.row_bytes, r.rows, r.row_stride)
                for r in regions
            )
        )
    )
    region = np.repeat(np.arange(len(regions)), rows)
    row = np.arange(len(region)) - np.repeat(np.cumsum(rows) - rows, rows)
    starts = offset[region] + row * stride[region]
    order = np.argsort(starts, kind="stable")
    starts, ends = starts[order], starts[order] + width[region][order]
    # A row starts a new range unless it begins inside the ranges before it
    first = np.flatnonzero(np.r_[True, starts[1:] >= np.maximum.accumulate(ends)[:-1]])
    return starts[first], np.maximum.reduceat(ends, first)


def _rows(schedule):
    """Every row of every region in ``schedule``, at the source, merged into sorted disjoint [start, end) byte ranges."""
    return _region_rows([r for chunk in schedule for r in chunk.regions], dst=False)


def _first_clash(parts, rows):
    """
    The first byte range two different ranks both cover, as ``(rank, rank, start, end)``, or None.
    ``parts`` are ``(rank, key)``; ``rows(key)`` gives that part's merged byte ranges.
    """
    ranges = {(rank, key): rows(key) for rank, key in parts}
    extents = sorted((s[0], e.max(), rank) for (rank, _), (s, e) in ranges.items() if len(s))
    if all(prev[1] <= cur[0] for prev, cur in zip(extents, extents[1:])):
        return None
    # Extents interleave (e.g. ranks own column blocks): compare rows exactly
    starts = np.concatenate([s for s, _ in ranges.values()])
    ends = np.concatenate([e for _, e in ranges.values()])
    rank = np.repeat([r for r, _ in ranges], [len(s) for s, _ in ranges.values()])
    order = np.argsort(starts, kind="stable")
    starts, ends, rank = starts[order], ends[order], rank[order]
    # Each part's ranges are disjoint, so any overlap shows up between neighbors
    clash = np.flatnonzero((starts[1:] < np.maximum.accumulate(ends)[:-1]) & (rank[1:] != rank[:-1]))
    if not len(clash):
        return None
    i = clash[0]
    return min(rank[i], rank[i + 1]), max(rank[i], rank[i + 1]), starts[i + 1], min(ends[i], ends[i + 1])


def check_disjoint(schedules, same_buffer=True):
    """
    Check that no two ranks write the same bytes of any rank's destination: two senders of one byte
    race. If the destination is the buffer itself (``same_buffer``), what lands on a rank must also
    miss the bytes that rank sends, or its own data would be overwritten. Regions of one rank may
    overlap.
    """
    world_size = len(schedules)
    cache = {}

    def rows(key):
        if key not in cache:
            sender, dst, indices = key
            cache[key] = _region_rows([r for i in indices for r in schedules[sender][i].regions], dst)
        return cache[key]

    for receiver in range(world_size):
        parts = []
        if same_buffer and schedules[receiver]:
            parts.append((receiver, (receiver, False, tuple(range(len(schedules[receiver]))))))
        for sender, schedule in enumerate(schedules):
            addressed = tuple(
                i for i, c in enumerate(schedule) if sender != receiver and receiver in c.recipients(sender, world_size)
            )
            if addressed:
                parts.append((sender, (sender, True, addressed)))
        clash = _first_clash(parts, rows)
        if clash:
            a, b, start, end = clash
            where = "the buffer" if same_buffer else f"rank {receiver}'s destination"
            raise ValueError(
                f"ranks {a} and {b} both send bytes {start} to {end} of {where} (on rank {receiver}); each rank's "
                "schedule must cover only the part of the buffer it writes"
            )
