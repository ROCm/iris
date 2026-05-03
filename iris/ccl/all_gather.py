# SPDX-License-Identifier: MIT
# Copyright (c) 2025 Advanced Micro Devices, Inc. All rights reserved.

"""
All-gather collective operation — public API.

Three paths by message size:
- Small (<128KB): NCCL (avoids Triton dispatch overhead)
- Medium (128KB-4MB): native Triton kernel
- Large (>=4MB): NCCL (tree-based, scales better at high BW)
"""

import torch.distributed as _dist

from iris.ccl.utils import extract_group_info

_NCCL_SMALL_BYTES = 256 * 1024  # <256KB: NCCL avoids Triton launch overhead
_NCCL_LARGE_BYTES = 8 * 1024 * 1024  # >=8MB: NCCL tree-based is more bandwidth-efficient


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
    M, N = input_tensor.shape[:2]
    msg_bytes = M * N * input_tensor.element_size()

    if msg_bytes < _NCCL_SMALL_BYTES or msg_bytes >= _NCCL_LARGE_BYTES:
        _dist.all_gather_into_tensor(output_tensor, input_tensor, group=group)
        return None

    from iris.ccl.config import Config

    if config is None:
        config = Config(block_size_m=32, block_size_n=128, comm_sms=64, num_warps=8)

    rank_in_group, rank_global, world_size, rank_start, rank_stride = extract_group_info(group, ctx)

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
