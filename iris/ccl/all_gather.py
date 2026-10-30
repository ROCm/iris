# SPDX-License-Identifier: MIT
# Copyright (c) 2025 Advanced Micro Devices, Inc. All rights reserved.

"""
All-gather collective operation — public API.

Routes to sdma/ for config.engine == "sdma", otherwise to triton/ or gluon/ based on config.use_gluon.
"""

from iris.ccl.utils import extract_group_info


def all_gather(output_tensor, input_tensor, ctx, group=None, async_op=False, config=None):
    """
    All-gather: each rank sends its input to all ranks.

    Output is (world_size * M, N) — inputs concatenated along dim 0.

    Args:
        output_tensor: Shape (world_size * M, N)
        input_tensor: Shape (M, N)
        ctx: Iris instance
        group: ProcessGroup or None
        async_op: If False (default), end with a device-side barrier on the current
            stream: work queued after this call runs only once every rank's part of
            the collective is done. The host does not wait. If True, skip the
            barrier; the caller must synchronize the ranks, e.g. with
            ``ctx.device_barrier(group=group)``, before reading the output or
            reusing the buffers
        config: Config with kernel parameters
    """
    from iris.ccl.config import Config

    if config is None:
        config = Config(block_size_m=32, block_size_n=64)

    rank_in_group, rank_global, world_size, rank_start, rank_stride = extract_group_info(group, ctx)

    M, N = input_tensor.shape[:2]
    expected_output_shape = (world_size * M, N)
    if output_tensor.shape[:2] != expected_output_shape:
        raise ValueError(
            f"Output tensor shape {output_tensor.shape[:2]} does not match expected shape "
            f"{expected_output_shape}. Expected (world_size * M, N) = ({world_size * M}, {N})"
        )

    if config.engine == "sdma":
        from iris.ccl.sdma.all_gather import launch

        # The copy engines have no algorithm variants or Gluon kernel
        if config.use_gluon or config.all_gather_variant is not None:
            raise ValueError("engine='sdma' does not take use_gluon or all_gather_variant; leave them unset")

        launch(input_tensor, output_tensor, ctx, rank_in_group, rank_global, world_size, rank_start, rank_stride)
        if not async_op:
            ctx.barrier()
        return

    if config.use_gluon and config.all_gather_variant == "tdm":
        # gfx1250 tensor data movement engine; see gluon/all_gather_tdm.py
        from iris.ccl.gluon.all_gather_tdm import launch
    elif config.use_gluon:
        from iris.ccl.gluon.all_gather import launch
    else:
        from iris.ccl.triton.all_gather import launch

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
        ctx.device_barrier(group=group)
