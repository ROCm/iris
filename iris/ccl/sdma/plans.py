# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.

"""Scatter, gather, broadcast and all-to-all as in-place triggered plans on the SDMA copy engines."""

import numpy as np

from iris.host.distributed.helpers import distributed_allgather


def _parts(tensor, n):
    """``tensor`` split along dim 0 into ``n`` equal views."""
    if not tensor.is_contiguous() or tensor.dim() == 0:
        raise ValueError("the tensor must be contiguous with at least one dimension")
    rows, rem = divmod(tensor.shape[0], n)
    if rem or rows == 0:
        raise ValueError(f"the tensor's first dimension {tensor.shape[0]} is not a positive multiple of {n}")
    return [tensor[i * rows : (i + 1) * rows] for i in range(n)]


def _check(schedule, tensor, sends, rank, world_size):
    """
    Check that ``schedule`` sends exactly ``sends``: ``{recipients: view}``, each view covered by the
    chunks that go to those recipients, in place. Without a schedule, build one ``whole()`` chunk per view.
    """
    from iris.experimental.plan import whole
    from iris.experimental.plan.schedule import _rows

    if schedule is None:
        return [c for peers, view in sends.items() for c in whole(view, base=tensor, peers=peers)]
    schedule = list(schedule)
    if not sends:
        if schedule:
            raise ValueError(f"rank {rank} sends nothing in this collective; pass an empty schedule")
        return schedule
    elem = tensor.element_size()
    ranges = {peers: (view.data_ptr() - tensor.data_ptr(), view.numel() * elem) for peers, view in sends.items()}
    ranges = {peers: (lo, lo + n) for peers, (lo, n) in ranges.items()}
    covered = {peers: [] for peers in sends}
    for i, chunk in enumerate(schedule):
        peers = chunk.recipients(rank, world_size)
        if peers not in ranges:
            raise ValueError(f"chunk {i} goes to ranks {list(peers)}, expected one of {[list(p) for p in ranges]}")
        lo, hi = ranges[peers]
        for region in chunk.regions:
            if region.dst_offset_bytes is not None or region.dst_stride_bytes is not None:
                raise ValueError(f"chunk {i}: region {region} lands elsewhere; the collective copies in place")
            if region.offset_bytes < lo or region.end > hi:
                raise ValueError(f"chunk {i}: region {region} lies outside bytes {lo} to {hi}")
        covered[peers].append(chunk)
    for peers, chunks in covered.items():
        lo, hi = ranges[peers]
        if not chunks:
            raise ValueError(f"no chunk sends bytes {lo} to {hi} to ranks {list(peers)}")
        starts, ends = _rows(chunks)
        if starts[0] != lo or ends[-1] != hi or (starts[1:] != ends[:-1]).any():
            raise ValueError(f"the schedule leaves part of bytes {lo} to {hi} (to ranks {list(peers)}) out")
    return schedule


def _prepare(ctx, tensor, schedule, what, root, sends_of, timeout, prepare_ahead):
    from iris.experimental.plan import TriggeredPlan

    rank, world_size = ctx.get_rank(), ctx.get_num_ranks()
    error = None
    try:
        ctx._require_copy_engine()
        if root is not None and not 0 <= root < world_size:
            raise ValueError(f"root must be a rank in 0..{world_size - 1}, got {root}")
        schedule = _check(schedule, tensor, sends_of(rank, world_size), rank, world_size)
    except Exception as e:
        error = e
    # Agree before the plan's own collectives, so a rank that raised leaves none waiting
    failed = distributed_allgather(np.array([error is not None], dtype=np.int64))[:, 0].nonzero()[0].tolist()
    if error is not None:
        raise error
    if failed:
        raise RuntimeError(f"rank(s) {failed} could not prepare the {what}")
    return TriggeredPlan(ctx, tensor, schedule, timeout=timeout, prepare_ahead=prepare_ahead)


def prepare_scatter(tensor, root, ctx, schedule=None, timeout=30.0, prepare_ahead=False):
    """Collective: part ``r`` of the root's ``tensor`` lands in rank ``r``'s. See ``CCL.prepare_scatter``."""

    def sends(rank, world_size):
        if rank != root:
            return {}
        return {(r,): part for r, part in enumerate(_parts(tensor, world_size)) if r != root}

    return _prepare(ctx, tensor, schedule, "scatter", root, sends, timeout, prepare_ahead)


def prepare_gather(tensor, root, ctx, schedule=None, timeout=30.0, prepare_ahead=False):
    """Collective: part ``r`` of rank ``r``'s ``tensor`` lands in the root's. See ``CCL.prepare_gather``."""

    def sends(rank, world_size):
        if rank == root:
            return {}
        return {(root,): _parts(tensor, world_size)[rank]}

    return _prepare(ctx, tensor, schedule, "gather", root, sends, timeout, prepare_ahead)


def prepare_broadcast(tensor, root, ctx, schedule=None, timeout=30.0, prepare_ahead=False):
    """Collective: the root's whole ``tensor`` lands in every other rank's. See ``CCL.prepare_broadcast``."""

    def sends(rank, world_size):
        if rank != root:
            return {}
        return {tuple(p for p in range(world_size) if p != root): _parts(tensor, 1)[0]}

    return _prepare(ctx, tensor, schedule, "broadcast", root, sends, timeout, prepare_ahead)


def prepare_all_to_all(tensor, ctx, schedule=None, timeout=30.0, prepare_ahead=False):
    """Collective: block ``(r, j)`` of rank ``r``'s ``tensor`` lands in rank ``j``'s. See ``CCL.prepare_all_to_all``."""

    def sends(rank, world_size):
        blocks = _parts(tensor, world_size * world_size)
        return {(j,): blocks[rank * world_size + j] for j in range(world_size) if j != rank}

    return _prepare(ctx, tensor, schedule, "all-to-all", None, sends, timeout, prepare_ahead)
