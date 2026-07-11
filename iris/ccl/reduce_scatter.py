# SPDX-License-Identifier: MIT
# Copyright (c) 2025 Advanced Micro Devices, Inc. All rights reserved.

"""
Reduce-scatter collective operation — public API.

Drop-in replacement for torch.distributed.reduce_scatter_tensor.
Accepts regular CUDA tensors, handles heap copy via as_symmetric.
"""

from iris.ccl.utils import extract_group_info


def reduce_scatter(output_tensor, input_tensor, ctx, op=None, group=None, async_op=False, config=None, workspace=None):
    """
    Reduce-scatter: each rank reduces its assigned tiles, stores locally.

    Accepts regular CUDA tensors — copies to/from symmetric heap via as_symmetric.
    """
    from iris.ccl.config import Config
    from iris.ccl.utils import ReduceOp

    if op is None:
        op = ReduceOp.SUM
    if op != ReduceOp.SUM:
        raise ValueError(
            f"Only ReduceOp.SUM is currently supported, got {op}. "
            "Support for other operations will be added in a future release."
        )
    if config is None:
        config = Config(block_size_m=32, block_size_n=64, all_reduce_distribution=1)

    rank_in_group, rank_global, world_size, rank_start, rank_stride = extract_group_info(group, ctx)

    heap_in = ctx.as_symmetric(input_tensor, tag="rs_inp")
    heap_out = ctx.as_symmetric(output_tensor, tag="rs_out")

    from iris.ccl.triton.reduce_scatter import launch

    workspace = launch(
        heap_out,
        heap_in,
        ctx,
        rank_in_group,
        rank_global,
        world_size,
        rank_start,
        rank_stride,
        config,
        workspace=workspace,
    )

    if not ctx.is_symmetric(output_tensor):
        output_tensor.copy_(heap_out)

    return workspace
