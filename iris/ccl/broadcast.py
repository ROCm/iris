# SPDX-License-Identifier: MIT
# Copyright (c) 2025 Advanced Micro Devices, Inc. All rights reserved.

"""
Broadcast collective operation — public API.

Routes to triton/ based on config.
"""

from iris.ccl.utils import extract_group_info


def broadcast(output_tensor, input_tensor, ctx, src=0, group=None, async_op=False, config=None):
    """
    Broadcast: root rank sends its data to all ranks.

    The source rank's input_tensor is copied to output_tensor on ALL ranks.
    Non-root ranks' input_tensor values are ignored.

    Pull-based broadcast adapted for
    iris's symmetric heap. Instead of passing data around a ring, we
    leverage direct XGMI writes for better performance.

    Args:
        output_tensor: Shape (M, N) — will contain src's data on all ranks
        input_tensor: Shape (M, N) — only src rank's data is used
        ctx: Iris instance
        src: Source rank within the group (default: 0)
        group: ProcessGroup or None
        async_op: If True, skip trailing barrier
        config: Config with kernel parameters
    """
    from iris.ccl.config import Config

    if config is None:
        config = Config(block_size_m=32, block_size_n=64)

    rank_in_group, rank_global, world_size, rank_start, rank_stride = extract_group_info(group, ctx)

    M, N = input_tensor.shape[:2]
    if output_tensor.shape[:2] != (M, N):
        raise ValueError(
            f"Output tensor shape {output_tensor.shape[:2]} does not match input shape "
            f"({M}, {N}). Broadcast requires input and output to have the same shape."
        )

    if src < 0 or src >= world_size:
        raise ValueError(f"src rank {src} is out of range [0, {world_size})")

    from iris.ccl.triton.broadcast import launch

    launch(
        input_tensor,
        output_tensor,
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
