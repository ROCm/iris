# SPDX-License-Identifier: MIT
# Copyright (c) 2025 Advanced Micro Devices, Inc. All rights reserved.

"""
Broadcast collective operation — public API.

Routes to triton/ or gluon/ based on config.use_gluon.
"""

from iris.ccl.utils import extract_group_info


def broadcast(tensor, ctx, src=0, group=None, async_op=False, config=None):
    """
    In-place broadcast collective operation.

    The rank identified by ``src`` broadcasts its data to all other ranks.
    After the operation, all ranks hold a copy of the source rank's data.

    Args:
        tensor: Tensor on the symmetric heap. Modified in-place.
        ctx: Iris instance.
        src: Source rank within the group (default: 0).
        group: ProcessGroup or None. If None, uses all ranks.
        async_op: If False, performs a barrier at the end.
        config: Config with kernel parameters. Default: None.
    """
    from iris.ccl.config import Config

    if config is None:
        config = Config()
    if config.use_gluon:
        raise ValueError(
            "broadcast does not support use_gluon=True. "
            "Gluon implementation is not available for broadcast."
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
