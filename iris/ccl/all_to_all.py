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

_NCCL_SMALL_BYTES = 0  # no NCCL small-message path — native kernel handles all sizes
_NCCL_LARGE_BYTES = 64 * 1024 * 1024  # >=64MB: NCCL (native push kernel wins up to 32MB)

_a2a_nccl_bufs: dict = {}
_a2a_ws_cache: dict = {}


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
        ws = _a2a_ws_cache.get(group)
        if ws is None:
            ws = _dist.get_world_size(group)
            _a2a_ws_cache[group] = ws
        world_size = ws
        N = total_N // world_size
        if M == 1:
            _dist.all_to_all_single(output_tensor.view(-1), input_tensor.view(-1), group=group)
            return
        buf_key = (M, N, world_size, input_tensor.dtype, input_tensor.device)
        bufs = _a2a_nccl_bufs.get(buf_key)
        if bufs is None:
            bufs = (
                torch.empty(world_size * M, N, dtype=input_tensor.dtype, device=input_tensor.device),
                torch.empty(world_size * M, N, dtype=input_tensor.dtype, device=input_tensor.device),
            )
            _a2a_nccl_bufs[buf_key] = bufs
        nccl_in, nccl_out = bufs
        nccl_in.view(world_size, M, N).copy_(input_tensor.view(M, world_size, N).permute(1, 0, 2))
        _dist.all_to_all_single(nccl_out, nccl_in, group=group)
        output_tensor.view(M, world_size, N).permute(1, 0, 2).copy_(nccl_out.view(world_size, M, N))
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
