# SPDX-License-Identifier: MIT
# Copyright (c) 2025 Advanced Micro Devices, Inc. All rights reserved.

"""
Reduce collective operation — public API.

Triton only (no gluon support).
"""

from iris.ccl.utils import extract_group_info


def reduce_preamble(output_tensor, input_tensor, ctx, root=0, config=None, workspace=None):
    """Prepare reusable workspace for reduce."""
    from iris.ccl.triton.reduce import reduce_preamble as _preamble

    return _preamble(output_tensor, input_tensor, ctx, root=root, config=config, workspace=workspace)


def reduce(output_tensor, input_tensor, ctx, root=0, op=None, group=None, async_op=False, config=None, workspace=None):
    """
    Reduce: sum inputs across all ranks, result only on root rank.

    Unlike AllReduce, only the root rank receives the reduced result.

    Args:
        output_tensor: Shape (M, N) — on root, receives the reduced result
        input_tensor: Shape (M, N) — local rank's partial data
        ctx: Iris instance
        root: Root rank (receives the result). Default: 0.
        op: ReduceOp (only SUM supported)
        group: ProcessGroup or None
        async_op: If True, skip trailing barrier
        config: Config with kernel parameters
        workspace: Reusable workspace from reduce_preamble
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

    # Set default reduce variant if not set
    if not hasattr(config, "reduce_variant"):
        config.reduce_variant = "two_shot"

    rank_in_group, rank_global, world_size, rank_start, rank_stride = extract_group_info(group, ctx)

    if root < 0 or root >= world_size:
        raise ValueError(f"root must be in [0, {world_size}), got {root}")

    from iris.ccl.triton.reduce import launch

    workspace = launch(
        output_tensor,
        input_tensor,
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

    return workspace
