# SPDX-License-Identifier: MIT
# Copyright (c) 2025 Advanced Micro Devices, Inc. All rights reserved.

"""
All-gather collective operation — public API.

Drop-in replacement for torch.distributed.all_gather_into_tensor.
Graph-capture safe with in-kernel barriers and cached workspace.
"""

from iris.ccl.utils import extract_group_info

_cached_workspace = None


def all_gather(output_tensor, input_tensor, ctx, group=None, async_op=False, config=None, workspace=None):
    """
    All-gather: each rank sends its input to all ranks.

    Output is (world_size * M, N) — inputs concatenated along dim 0.
    Accepts regular CUDA tensors. Graph-capture safe.
    """
    global _cached_workspace
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

    heap_in = ctx.as_symmetric(input_tensor, tag="ag_inp")
    heap_out = ctx.as_symmetric(output_tensor, tag="ag_out")

    if workspace is None:
        workspace = _cached_workspace

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

    _cached_workspace = workspace

    if not ctx.is_symmetric(output_tensor):
        output_tensor.copy_(heap_out)

    return workspace
