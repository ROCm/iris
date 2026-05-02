# SPDX-License-Identifier: MIT
# Copyright (c) 2025 Advanced Micro Devices, Inc. All rights reserved.

"""
Barrier collective operation — public API.

GPU-side barrier using atomic signaling on the symmetric heap.
Triton only (no Gluon support).
"""


def barrier(ctx, group=None):
    """
    Global barrier: all ranks block until every rank has arrived.

    Delegates to ``ctx.device_barrier()`` which uses a single-kernel
    epoch-counter approach (atomic_add + spin-wait) on the symmetric heap.
    Stateless w.r.t. host-side epoch tracking — no zeroing or preamble
    host barrier needed.

    Args:
        ctx: Iris instance.
        group: ProcessGroup or None. If None, uses all ranks.
    """
    ctx.device_barrier(group)
