# SPDX-License-Identifier: MIT
# Copyright (c) 2025 Advanced Micro Devices, Inc. All rights reserved.

"""
All-gather collective operation — public API.

Accepts regular CUDA tensors. Internally copies to/from symmetric heap.
"""

from iris.ccl.utils import extract_group_info
from iris.ccl.dispatch import get_heap_buffer


def all_gather(output_tensor, input_tensor, ctx, group=None, async_op=False, config=None, workspace=None):
    """
    All-gather: each rank sends its input to all ranks.

    Output is (world_size * M, N) — inputs concatenated along dim 0.
    Accepts regular CUDA tensors — copies to heap internally.

    Args:
        output_tensor: Shape (world_size * M, N)
        input_tensor: Shape (M, N)
        ctx: Iris instance
        group: ProcessGroup or None
        async_op: If True, skip trailing barrier
        config: Config with kernel parameters
        workspace: Optional workspace for ring variant
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

    heap_inp = get_heap_buffer(ctx, (M, N), input_tensor.dtype, "ag_inp")
    heap_out = get_heap_buffer(ctx, (world_size * M, N), input_tensor.dtype, "ag_out")

    heap_inp.copy_(input_tensor)

    if config.use_gluon:
        from iris.ccl.gluon.all_gather import launch
    else:
        from iris.ccl.triton.all_gather import launch

    workspace = launch(
        heap_inp,
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

    if not async_op:
        ctx.barrier()

    output_tensor.copy_(heap_out)

    return workspace
