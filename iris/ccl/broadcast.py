# SPDX-License-Identifier: MIT
# Copyright (c) 2025 Advanced Micro Devices, Inc. All rights reserved.

"""
Broadcast collective operation — public API.

Routes to triton/ or gluon/ based on config.use_gluon.
"""

from iris.ccl.utils import extract_group_info


def broadcast(tensor, ctx, src=0, group=None, async_op=False, config=None):
    """
    Broadcast: root rank sends its data to all other ranks in-place.

    After the operation, every rank holds a copy of src's original data.

    Args:
        tensor: Tensor of shape (M, N) -- root's data is broadcast to all ranks in-place.
        ctx: Iris instance.
        src: Source rank (within the group) that sends the data.
             Default: 0.
        group: ProcessGroup or None. If None, uses all ranks.
        async_op: If True, skip trailing barrier. Default: False.
        config: Config with kernel parameters. Default: None.
    """
    from iris.ccl.config import Config

    if config is None:
        config = Config(block_size_m=32, block_size_n=64)

    rank_in_group, rank_global, world_size, rank_start, rank_stride = extract_group_info(group, ctx)

    if src < 0 or src >= world_size:
        raise ValueError(
            f"src must be in [0, world_size), got src={src}, world_size={world_size}."
        )

    if config.use_gluon:
        from iris.ccl.gluon.broadcast import launch
    else:
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
        ctx.barrier()
