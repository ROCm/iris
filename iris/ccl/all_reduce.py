# SPDX-License-Identifier: MIT
# Copyright (c) 2025 Advanced Micro Devices, Inc. All rights reserved.

"""
All-reduce collective operation — public API.

Supports Triton variants (atomic, spinlock, ring, two_shot, one_shot)
and a Gluon variant (one_shot_gluon) optimized for small messages.
"""

from iris.ccl.utils import extract_group_info


def all_reduce_preamble(output_tensor, input_tensor, ctx, config=None, workspace=None):
    """Prepare reusable workspace for all-reduce."""
    from iris.ccl.config import Config

    if config is None:
        config = Config()

    variant = config.all_reduce_variant.lower()
    if variant == "one_shot_gluon" or config.use_gluon:
        from iris.ccl.gluon.all_reduce import all_reduce_preamble as _gluon_preamble

        return _gluon_preamble(output_tensor, input_tensor, ctx, config=config, workspace=workspace)

    from iris.ccl.triton.all_reduce import all_reduce_preamble as _preamble

    return _preamble(output_tensor, input_tensor, ctx, config=config, workspace=workspace)


def all_reduce(output_tensor, input_tensor, ctx, op=None, group=None, async_op=False, config=None, workspace=None):
    """
    All-reduce: sum inputs across all ranks, result on every rank.

    Args:
        output_tensor: Shape (M, N)
        input_tensor: Shape (M, N)
        ctx: Iris instance
        op: ReduceOp (only SUM supported)
        group: ProcessGroup or None
        async_op: If True, skip trailing barrier
        config: Config with kernel parameters
        workspace: Reusable workspace from all_reduce_preamble
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
        config = Config(all_reduce_variant="one_shot_gluon", use_gluon=True)
    if config.use_gluon and config.all_reduce_variant != "one_shot_gluon":
        config.all_reduce_variant = "one_shot_gluon"

    variant = config.all_reduce_variant.lower()
    valid_variants = ["atomic", "spinlock", "ring", "two_shot", "one_shot", "one_shot_gluon"]
    if variant not in valid_variants:
        raise ValueError(f"Invalid all_reduce_variant: {variant}. Must be one of: {', '.join(valid_variants)}")

    rank_in_group, rank_global, world_size, rank_start, rank_stride = extract_group_info(group, ctx)

    if variant == "one_shot_gluon":
        from iris.ccl.gluon.all_reduce import launch as gluon_launch

        inplace = output_tensor.data_ptr() == input_tensor.data_ptr()
        if inplace:
            import torch
            actual_output = torch.empty_like(output_tensor)
        else:
            actual_output = output_tensor

        workspace = gluon_launch(
            actual_output,
            input_tensor,
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

        if inplace:
            ctx.barrier()
            output_tensor.copy_(actual_output)
    else:
        from iris.ccl.triton.all_reduce import launch

        workspace = launch(
            output_tensor,
            input_tensor,
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

    if workspace is not None:
        if variant not in ("one_shot", "one_shot_gluon"):
            workspace.prepared = False

    if not async_op and variant not in ("one_shot", "one_shot_gluon"):
        ctx.barrier()

    return workspace
