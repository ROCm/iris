# SPDX-License-Identifier: MIT
# Copyright (c) 2025 Advanced Micro Devices, Inc. All rights reserved.

"""
All-reduce collective operation — public API.

Drop-in replacement for torch.distributed.all_reduce.
Accepts regular CUDA tensors, handles heap copy via as_symmetric.
Graph-capture safe with in-kernel barriers (one_shot variant).
"""

from iris.ccl.utils import extract_group_info


def all_reduce_preamble(output_tensor, input_tensor, ctx, config=None, workspace=None):
    """Prepare reusable workspace for all-reduce."""
    from iris.ccl.triton.all_reduce import all_reduce_preamble as _preamble

    heap_in = ctx.as_symmetric(input_tensor)
    heap_out = ctx.as_symmetric(output_tensor)
    return _preamble(heap_out, heap_in, ctx, config=config, workspace=workspace)


def all_reduce(output_tensor, input_tensor, ctx, op=None, group=None, async_op=False, config=None, workspace=None):
    """
    All-reduce: sum inputs across all ranks, result on every rank.

    Accepts regular CUDA tensors — copies to/from symmetric heap via as_symmetric.
    Graph-capture safe when using one_shot variant (in-kernel barriers).
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
        config = Config(all_reduce_variant="one_shot")

    variant = config.all_reduce_variant.lower()
    valid_variants = ["atomic", "spinlock", "ring", "two_shot", "one_shot", "one_shot_legacy"]
    if variant not in valid_variants:
        raise ValueError(f"Invalid all_reduce_variant: {variant}. Must be one of: {', '.join(valid_variants)}")

    rank_in_group, rank_global, world_size, rank_start, rank_stride = extract_group_info(group, ctx)

    heap_in = ctx.as_symmetric(input_tensor)
    heap_out = ctx.as_symmetric(output_tensor)

    from iris.ccl.triton.all_reduce import launch

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
        workspace,
        group=group,
    )

    if not ctx.is_symmetric(output_tensor):
        output_tensor.copy_(heap_out)

    if workspace is not None:
        if variant != "one_shot":
            workspace.prepared = False

    if not async_op and variant != "one_shot":
        ctx.barrier()

    return workspace
