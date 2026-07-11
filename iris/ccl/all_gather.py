# SPDX-License-Identifier: MIT
# Copyright (c) 2025 Advanced Micro Devices, Inc. All rights reserved.

"""
All-gather collective operation — public API.

Drop-in replacement for torch.distributed.all_gather_into_tensor.
Accepts regular CUDA tensors, handles heap copy via as_symmetric.
"""

from iris.ccl.utils import extract_group_info


def all_gather(output_tensor, input_tensor, ctx, group=None, async_op=False, config=None, workspace=None):
    """
    All-gather: each rank sends its input to all ranks.

    Output is (world_size * M, N) — inputs concatenated along dim 0.
    Accepts regular CUDA tensors — copies to/from symmetric heap via as_symmetric.
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

    heap_in = ctx.as_symmetric(input_tensor)
    heap_out = ctx.as_symmetric(output_tensor)

    if config.use_gluon:
        from iris.ccl.gluon.all_gather import launch
    else:
        from iris.ccl.triton.all_gather import launch

    workspace = launch(
        heap_in,
        heap_out,
        ctx,
        rank_in_group,
        rank_global,
        world_size,
        rank_start,
        rank_stride,
        config,
        workspace=workspace,
        group=group,
    )

    if not ctx.is_symmetric(output_tensor):
        output_tensor.copy_(heap_out)

    if not async_op:
        ctx.barrier()

    return workspace
