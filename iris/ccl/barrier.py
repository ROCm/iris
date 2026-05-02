# SPDX-License-Identifier: MIT
# Copyright (c) 2025 Advanced Micro Devices, Inc. All rights reserved.

"""
Barrier collective operation — public API.

GPU-side barrier using atomic signaling on the symmetric heap.
Triton only (no Gluon support).
"""

import torch

from iris.ccl.utils import extract_group_info


def barrier(ctx, group=None):
    """
    Global barrier: all ranks block until every rank has arrived.

    Uses device-side atomic flag signaling on the symmetric heap.
    A flags workspace (one int32 per rank) is allocated on first call
    and cached on the ctx for reuse.

    Args:
        ctx: Iris instance.
        group: ProcessGroup or None. If None, uses all ranks.
    """
    rank_in_group, rank_global, world_size, rank_start, rank_stride = extract_group_info(group, ctx)

    # Cache flags tensor per group on the ctx instance.
    # Use a dict keyed by group (None for default group).
    if not hasattr(ctx, "_ccl_barrier_flags"):
        ctx._ccl_barrier_flags = {}

    if group not in ctx._ccl_barrier_flags:
        # Allocate num_ranks elements so global rank indexing is always in-bounds,
        # even when operating on a subset group.
        ctx._ccl_barrier_flags[group] = ctx.zeros((ctx.get_num_ranks(),), dtype=torch.int32)

    flags = ctx._ccl_barrier_flags[group]

    from iris.ccl.triton.barrier import launch

    launch(flags, ctx, rank_global, world_size, rank_start, rank_stride)
