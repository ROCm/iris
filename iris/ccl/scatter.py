# SPDX-License-Identifier: MIT
# Copyright (c) 2025 Advanced Micro Devices, Inc. All rights reserved.

"""
Scatter collective operation — public API.

Routes to triton/ or gluon/ based on config.use_gluon.
"""

from iris.ccl.utils import extract_group_info


def scatter(output_tensor, input_tensor, ctx, src=0, group=None, async_op=False, config=None):
    """
    Scatter: root rank distributes equal-sized chunks of its data to all ranks.

    Input on root is (world_size * M, N), each rank receives (M, N).
    On non-root ranks, input_tensor is ignored.

    Args:
        output_tensor: Shape (M, N) — receives this rank's chunk
        input_tensor: Shape (world_size * M, N) on root, ignored on non-root
        ctx: Iris instance
        src: Source (root) rank within the group (default: 0)
        group: ProcessGroup or None
        async_op: If True, skip trailing barrier
        config: Config with kernel parameters
    """
    from iris.ccl.config import Config

    if config is None:
        config = Config(block_size_m=32, block_size_n=64)

    rank_in_group, rank_global, world_size, rank_start, rank_stride = extract_group_info(group, ctx)

    M, N = output_tensor.shape[:2]

    # Validate input shape on root
    if rank_in_group == src:
        expected_input_shape = (world_size * M, N)
        if input_tensor.shape[:2] != expected_input_shape:
            raise ValueError(
                f"Input tensor shape {input_tensor.shape[:2]} on root does not match expected shape "
                f"{expected_input_shape}. Expected (world_size * M, N) = ({world_size * M}, {N})"
            )

    if config.use_gluon:
        from iris.ccl.gluon.scatter import launch
    else:
        from iris.ccl.triton.scatter import launch

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
