# SPDX-License-Identifier: MIT
# Copyright (c) 2025 Advanced Micro Devices, Inc. All rights reserved.

"""
Broadcast collective operation — public API.

Two paths by message size:
- Small/medium: single push kernel (root pushes tiles to all ranks)
- Large (>=4MB): NCCL (tree-based, scales better at high message sizes)
"""

import torch.distributed as _dist

from iris.ccl.utils import extract_group_info

_NCCL_SMALL_BYTES = 128 * 1024
_NCCL_LARGE_BYTES = 4 * 1024 * 1024


def broadcast(tensor, ctx, src=0, group=None, async_op=False, config=None):
    """
    In-place broadcast collective operation.

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
            ctx.device_barrier(group)
        return

    msg_bytes = numel * tensor.element_size()

    if msg_bytes < _NCCL_SMALL_BYTES or msg_bytes >= _NCCL_LARGE_BYTES:
        _dist.broadcast(tensor, src=src, group=group)
        return

    tensor = tensor.contiguous().view(-1)
    block_n = config.block_size_n
    if numel >= block_n:
        tensor = tensor.view(-1, block_n)
    else:
        tensor = tensor.view(1, -1)

    from iris.ccl.triton.ring_broadcast import launch

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
        ctx.device_barrier(group)
