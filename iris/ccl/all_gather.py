# SPDX-License-Identifier: MIT
# Copyright (c) 2025 Advanced Micro Devices, Inc. All rights reserved.

"""
All-gather collective operation — public API.

Flat fused kernel for small messages, push ring for medium, NCCL for large.
"""

import torch.distributed as _dist

from iris.ccl.utils import extract_group_info

_NCCL_SMALL_BYTES = 0
_FLAT_BYTES = 8 * 1024  # <8KB per-rank: use flat fused all-gather
_RING_BYTES = 512 * 1024  # 64KB-512KB per-rank: use push ring
_NCCL_LARGE_BYTES = 512 * 1024  # >=512KB per-rank: NCCL


def _ring_config_for_size(msg_bytes):
    """Ring config for medium-to-large messages."""
    from iris.ccl.config import Config

    return Config(block_size_m=32, block_size_n=128, comm_sms=64, num_warps=8)


def all_gather(output_tensor, input_tensor, ctx, group=None, async_op=False, config=None):
    """
    All-gather: each rank sends its input to all ranks.

    Output is (world_size * M, N) — inputs concatenated along dim 0.

    Args:
        output_tensor: Shape (world_size * M, N)
        input_tensor: Shape (M, N)
        ctx: Iris instance
        group: ProcessGroup or None
        async_op: If True, skip trailing barrier
        config: Config with kernel parameters
    """
    numel_in = input_tensor.numel()
    msg_bytes = numel_in * input_tensor.element_size()

    is_flat = input_tensor.dim() == 1 or input_tensor.shape[0] == 1
    nccl_threshold = 2 * 1024 * 1024 if is_flat else _NCCL_LARGE_BYTES

    if msg_bytes >= nccl_threshold:
        _dist.all_gather_into_tensor(output_tensor, input_tensor, group=group)
        return None

    from iris.ccl.config import Config

    rank_in_group, rank_global, world_size, rank_start, rank_stride = extract_group_info(group, ctx)

    if msg_bytes < _FLAT_BYTES and world_size > 1 and config is None:
        from iris.ccl.triton.all_gather_flat import launch as launch_flat

        flat_config = Config(block_size_m=32, block_size_n=64, comm_sms=64, num_warps=8)
        launch_flat(
            input_tensor,
            output_tensor,
            ctx,
            rank_in_group,
            rank_global,
            world_size,
            rank_start,
            rank_stride,
            flat_config,
            group=group,
        )
        return None

    if config is None:
        config = _ring_config_for_size(msg_bytes)

    block_n = config.block_size_n
    if is_flat and numel_in >= block_n:
        input_tensor = input_tensor.contiguous().view(-1, block_n)
        output_tensor = output_tensor.contiguous().view(-1, block_n)

    M, N = input_tensor.shape[:2]
    expected_output_shape = (world_size * M, N)
    if output_tensor.shape[:2] != expected_output_shape:
        raise ValueError(
            f"Output tensor shape {output_tensor.shape[:2]} does not match expected shape "
            f"{expected_output_shape}. Expected (world_size * M, N) = ({world_size * M}, {N})"
        )

    if config.use_gluon:
        from iris.ccl.gluon.all_gather import launch

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
        from iris.ccl.triton.all_gather import launch

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
