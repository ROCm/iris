# SPDX-License-Identifier: MIT
# Copyright (c) 2025 Advanced Micro Devices, Inc. All rights reserved.

"""
Gather collective operation -- public API.

Routes to triton/ or gluon/ based on config.use_gluon.
"""

from iris.ccl.utils import extract_group_info


def gather(output_tensor, input_tensor, ctx, dst=0, group=None, async_op=False, config=None):
    """
    Gather: each rank sends its input to the root rank.

    Root concatenates all inputs along dim 0, producing output of shape
    (world_size * M, N). Output is only valid on the root rank.

    Args:
        output_tensor: Shape (world_size * M, N). Must be allocated on all ranks
                       (needed for heap address translation), but only root's
                       contents are meaningful after the operation.
        input_tensor: Shape (M, N)
        ctx: Iris instance
        dst: Destination (root) rank within the group (default: 0)
        group: ProcessGroup or None
        async_op: If True, skip trailing barrier
        config: Config with kernel parameters
    """
    from iris.ccl.config import Config

    if config is None:
        config = Config(block_size_m=32, block_size_n=64)

    rank_in_group, rank_global, world_size, rank_start, rank_stride = extract_group_info(group, ctx)

    if dst < 0 or dst >= world_size:
        raise ValueError(
            f"dst rank {dst} is out of range for world_size {world_size}. Expected 0 <= dst < {world_size}."
        )

    M, N = input_tensor.shape[:2]
    expected_output_shape = (world_size * M, N)
    if output_tensor.shape[:2] != expected_output_shape:
        raise ValueError(
            f"Output tensor shape {output_tensor.shape[:2]} does not match expected shape "
            f"{expected_output_shape}. Expected (world_size * M, N) = ({world_size * M}, {N})"
        )

    if config.use_gluon:
        from iris.ccl.gluon.gather import launch

        launch(
            input_tensor,
            output_tensor,
            ctx,
            rank_in_group,
            rank_global,
            world_size,
            rank_start,
            rank_stride,
            dst,
            config,
        )
        if not async_op:
            ctx.device_barrier(group)
    else:
        from iris.ccl.triton.gather import launch

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
            dst,
            config,
            inline_barrier=use_inline,
            barrier_state=barrier_state,
        )
