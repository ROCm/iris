# SPDX-License-Identifier: MIT
# Copyright (c) 2025 Advanced Micro Devices, Inc. All rights reserved.

"""
Broadcast collective operation — public API.

Single-kernel all-ranks push: root loads each tile once and iris.store's
it to every other rank.  One kernel launch, one barrier.
"""

from iris.ccl.utils import extract_group_info


def broadcast(tensor, ctx, src=0, group=None, async_op=False, config=None):
    """
    In-place broadcast collective operation.

    The rank identified by ``src`` broadcasts its data to all other ranks.
    After the operation, all ranks hold a copy of the source rank's data.

    Uses a single persistent kernel: only the root rank does work, pushing
    every tile to all other ranks via iris.store.  Non-root ranks exit the
    kernel immediately.

    Args:
        tensor: Tensor on the symmetric heap. Modified in-place.
        ctx: Iris instance.
        src: Source rank within the group (default: 0).
        group: ProcessGroup or None. If None, uses all ranks.
        async_op: If True, skip trailing barrier.
        config: Config with kernel parameters. Default: None.
    """
    from iris.ccl.config import Config

    if config is None:
        config = Config()
    if config.use_gluon:
        raise ValueError(
            "broadcast does not support use_gluon=True. Gluon implementation is not available for broadcast."
        )

    rank_in_group, rank_global, world_size, rank_start, rank_stride = extract_group_info(group, ctx)

    if src < 0 or src >= world_size:
        raise ValueError(f"src rank {src} out of range [0, {world_size})")

    numel = tensor.numel()
    if numel == 0:
        if not async_op:
            ctx.barrier(group=group)
        return

    from iris.ccl.triton.broadcast import launch

    # Flatten to 1-D — broadcast is a plain data copy so shape doesn't
    # matter, only the total number of elements.
    tensor = tensor.contiguous().view(-1)

    launch(
        tensor,
        ctx,
        rank_in_group,
        rank_global,
        world_size,
        rank_start,
        rank_stride,
        src,
        config,
    )

    if not async_op:
        ctx.barrier(group=group)
