# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.

"""
Scatter, gather, broadcast and all-to-all as triggered plans on the SDMA copy engines.

They back ``CCL.prepare_scatter``, ``prepare_gather``, ``prepare_broadcast`` and
``prepare_all_to_all`` (``prepare_all_gather`` is in ``all_gather``). Each is direct (one hop) and in
place: a part of the tensor lands at the same offset on the ranks it goes to.

| collective | sends | to |
|---|---|---|
| scatter | the root, part ``r`` of the tensor | rank ``r`` |
| gather | every other rank, its own part | the root |
| broadcast | the root, the whole tensor | every other rank |
| all-to-all | every rank ``r``, block ``(r, j)`` | rank ``j`` |

A tensor of ``world_size * M`` rows has one part of ``M`` rows per rank. The all-to-all's has
``world_size ** 2`` blocks of ``M`` rows, laid out ``[source][destination]``: rank ``r`` holds
block ``(r, j)`` for every ``j``, and afterwards rank ``j``'s blocks ``(r, j)`` hold what every rank
sent it. Bytes a rank does not receive are not touched.
"""

import numpy as np

from iris.host.distributed.helpers import distributed_allgather


def _agree(error, root, what):
    """
    Collective: raise on every rank if any rank has an error or a different root. ``root`` is a
    checked rank, or -1 (none, or one that failed the check, which ``error`` then holds).
    """
    flags = distributed_allgather(np.array([error is not None, root], dtype=np.int64))
    if error is not None:
        raise error
    failed = flags[:, 0].nonzero()[0].tolist()
    if failed:
        raise RuntimeError(f"rank(s) {failed} could not prepare the {what}")
    if len(set(flags[:, 1].tolist())) > 1:
        raise ValueError(f"ranks passed different roots to the {what}: {flags[:, 1].tolist()}")


def _root(root, world_size):
    if isinstance(root, bool) or not isinstance(root, (int, np.integer)) or not 0 <= root < world_size:
        raise ValueError(f"root must be a rank in 0..{world_size - 1}, got {root!r}")
    return int(root)


def _parts(tensor, n):
    """``tensor`` split along dim 0 into ``n`` equal views."""
    if not tensor.is_contiguous() or tensor.dim() == 0:
        raise ValueError("the tensor must be contiguous with at least one dimension")
    rows, rem = divmod(tensor.shape[0], n)
    if rem or rows == 0:
        raise ValueError(f"the tensor's first dimension {tensor.shape[0]} is not a positive multiple of {n}")
    return [tensor[i * rows : (i + 1) * rows] for i in range(n)]


def _check(schedule, tensor, sends, rank, world_size, hint):
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
            raise ValueError(f"chunk {i} goes to ranks {list(peers)}; {hint}")
        lo, hi = ranges[peers]
        for region in chunk.regions:
            if region.dst_offset_bytes is not None or region.dst_stride_bytes is not None:
                raise ValueError(f"chunk {i}: region {region} lands elsewhere; the collective copies in place")
            if region.offset_bytes < lo or region.end > hi:
                raise ValueError(f"chunk {i}: region {region} lies outside bytes {lo} to {hi}; {hint}")
        covered[peers].append(chunk)
    for peers, chunks in covered.items():
        lo, hi = ranges[peers]
        if not chunks:
            raise ValueError(f"no chunk sends bytes {lo} to {hi} to ranks {list(peers)}; {hint}")
        starts, ends = _rows(chunks)
        if starts[0] != lo or ends[-1] != hi or (starts[1:] != ends[:-1]).any():
            raise ValueError(f"the schedule leaves part of bytes {lo} to {hi} (to ranks {list(peers)}) out; {hint}")
    return schedule


def _prepare(ctx, tensor, schedule, what, root, sends_of, hint, timeout, prepare_ahead, has_root=True):
    from iris.experimental.plan import TriggeredPlan

    rank, world_size = ctx.get_rank(), ctx.get_num_ranks()
    error, checked = None, -1
    try:
        ctx._require_copy_engine()
        if has_root:
            root = checked = _root(root, world_size)
        schedule = _check(schedule, tensor, sends_of(rank, world_size, root), rank, world_size, hint)
    except Exception as e:
        error = e
    # Agree before the plan's own collectives, so a rank that raised leaves none waiting
    _agree(error, checked, what)
    return TriggeredPlan(ctx, tensor, schedule, timeout=timeout, prepare_ahead=prepare_ahead)


def prepare_scatter(tensor, root, ctx, schedule=None, timeout=30.0, prepare_ahead=False):
    """Collective: part ``r`` of the root's ``tensor`` lands in rank ``r``'s. See ``CCL.prepare_scatter``."""

    def sends(rank, world_size, root):
        if rank != root:
            return {}
        return {(r,): part for r, part in enumerate(_parts(tensor, world_size)) if r != root}

    hint = "the root sends part r of the tensor to rank r only (peers=(r,)), and no other rank sends"
    return _prepare(ctx, tensor, schedule, "scatter", root, sends, hint, timeout, prepare_ahead)


def prepare_gather(tensor, root, ctx, schedule=None, timeout=30.0, prepare_ahead=False):
    """Collective: part ``r`` of rank ``r``'s ``tensor`` lands in the root's. See ``CCL.prepare_gather``."""

    def sends(rank, world_size, root):
        if rank == root:
            return {}
        return {(root,): _parts(tensor, world_size)[rank]}

    hint = "every rank but the root sends its own part of the tensor to the root only (peers=(root,))"
    return _prepare(ctx, tensor, schedule, "gather", root, sends, hint, timeout, prepare_ahead)


def prepare_broadcast(tensor, root, ctx, schedule=None, timeout=30.0, prepare_ahead=False):
    """Collective: the root's whole ``tensor`` lands in every other rank's. See ``CCL.prepare_broadcast``."""

    def sends(rank, world_size, root):
        if rank != root:
            return {}
        return {tuple(p for p in range(world_size) if p != root): _parts(tensor, 1)[0]}

    hint = "the root sends the whole tensor to every other rank, and no other rank sends"
    return _prepare(ctx, tensor, schedule, "broadcast", root, sends, hint, timeout, prepare_ahead)


def prepare_all_to_all(tensor, ctx, schedule=None, timeout=30.0, prepare_ahead=False):
    """Collective: block ``(r, j)`` of rank ``r``'s ``tensor`` lands in rank ``j``'s. See ``CCL.prepare_all_to_all``."""

    def sends(rank, world_size, root):
        blocks = _parts(tensor, world_size * world_size)
        return {(j,): blocks[rank * world_size + j] for j in range(world_size) if j != rank}

    hint = "rank r sends block (r, j) of the tensor to rank j only (peers=(j,))"
    return _prepare(ctx, tensor, schedule, "all-to-all", None, sends, hint, timeout, prepare_ahead, has_root=False)
