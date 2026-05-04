# SPDX-License-Identifier: MIT
# Copyright (c) 2025 Advanced Micro Devices, Inc. All rights reserved.

"""
All-to-all collective operation — public API.

Routes to NCCL (via layout transpose) or triton/gluon.
Iris uses column-chunked layout (M, N*W), NCCL uses row-chunked (W*M, N).
"""

import torch
import torch.distributed as _dist

from iris.ccl.utils import extract_group_info

_NCCL_SMALL_BYTES = 8 * 1024 * 1024  # <8MB: NCCL via layout transpose
_NCCL_LARGE_BYTES = 8 * 1024 * 1024  # >=8MB: NCCL via layout transpose


def all_to_all(output_tensor, input_tensor, ctx, group=None, async_op=False, config=None):
    """
    All-to-all: each rank sends a chunk to every other rank.

    Input/output shape: (M, N * world_size).

    Args:
        output_tensor: Shape (M, N * world_size)
        input_tensor: Shape (M, N * world_size)
        ctx: Iris instance
        group: ProcessGroup or None
        async_op: If True, skip trailing barrier
        config: Config with kernel parameters
    """
    M, total_N = input_tensor.shape[:2]
    msg_bytes = M * total_N * input_tensor.element_size()

    if msg_bytes < _NCCL_SMALL_BYTES or msg_bytes >= _NCCL_LARGE_BYTES:
        world_size = _dist.get_world_size(group)
        N = total_N // world_size
        if M == 1:
            _dist.all_to_all_single(output_tensor.view(-1), input_tensor.view(-1), group=group)
            return
        nccl_in = input_tensor.view(M, world_size, N).permute(1, 0, 2).contiguous().view(world_size * M, N)
        nccl_out = torch.empty_like(nccl_in)
        _dist.all_to_all_single(nccl_out, nccl_in, group=group)
        output_tensor.copy_(nccl_out.view(world_size, M, N).permute(1, 0, 2).contiguous().view(M, total_N))
        return

    from iris.ccl.config import Config

    if config is None:
        config = Config(block_size_m=32, block_size_n=128)

    rank_in_group, rank_global, world_size, rank_start, rank_stride = extract_group_info(group, ctx)
    N = total_N // world_size

    if config.use_gluon:
        from iris.ccl.gluon.all_to_all import launch

        launch(
            input_tensor,
            output_tensor,
            ctx,
            rank_in_group,
            rank_global,
            world_size,
            rank_start,
            rank_stride,
            config,
        )
        if not async_op:
            ctx.device_barrier(group)
    else:
        from iris.ccl.triton.all_to_all import launch

        use_inline = not async_op
        barrier_state = None
        if use_inline:
            barrier_state = ctx._get_inline_barrier_state(group)

        launch(
            input_tensor,
            output_tensor,
            ctx,
            rank_in_group,
            rank_global,
            world_size,
            rank_start,
            rank_stride,
            config,
            inline_barrier=use_inline,
            barrier_state=barrier_state,
        )
