# SPDX-License-Identifier: MIT
# Copyright (c) 2025 Advanced Micro Devices, Inc. All rights reserved.

"""
Reduce collective operation — public API.

Accepts regular CUDA tensors. Internally copies to/from symmetric heap.
"""

from iris.ccl.utils import extract_group_info
from iris.ccl.dispatch import get_heap_buffer


def reduce_preamble(output_tensor, input_tensor, ctx, root=0, config=None, workspace=None):
    """Prepare reusable workspace for reduce."""
    from iris.ccl.triton.reduce import reduce_preamble as _preamble

    return _preamble(output_tensor, input_tensor, ctx, root=root, config=config, workspace=workspace)


def reduce(output_tensor, input_tensor, ctx, root=0, op=None, group=None, async_op=False, config=None, workspace=None):
    """
    Reduce: sum inputs across all ranks, result only on root rank.

    Accepts regular CUDA tensors — copies to heap internally.
    """
    from iris.ccl.config import Config
    from iris.ccl.utils import ReduceOp

    if op is None:
        op = ReduceOp.SUM
    if op != ReduceOp.SUM:
        raise ValueError(
            f"Only ReduceOp.SUM is currently supported, got {op}."
        )
    if config is None:
        config = Config(block_size_m=32, block_size_n=64, all_reduce_distribution=1)

    if not hasattr(config, "reduce_variant"):
        config.reduce_variant = "two_shot"

    rank_in_group, rank_global, world_size, rank_start, rank_stride = extract_group_info(group, ctx)

    if root < 0 or root >= world_size:
        raise ValueError(f"root must be in [0, {world_size}), got {root}")

    M, N = input_tensor.shape[:2]
    heap_inp = get_heap_buffer(ctx, (M, N), input_tensor.dtype, "red_inp")
    heap_out = get_heap_buffer(ctx, (M, N), input_tensor.dtype, "red_out")

    heap_inp.copy_(input_tensor)

    from iris.ccl.triton.reduce import launch

    workspace = launch(
        heap_out,
        heap_inp,
        ctx,
        rank_in_group,
        rank_global,
        world_size,
        rank_start,
        rank_stride,
        root,
        config,
        workspace,
        group=group,
    )

    if workspace is not None:
        workspace.prepared = False

    if not async_op:
        ctx.barrier()

    output_tensor.copy_(heap_out)

    return workspace
