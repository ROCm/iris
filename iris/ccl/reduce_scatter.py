# SPDX-License-Identifier: MIT
# Copyright (c) 2025 Advanced Micro Devices, Inc. All rights reserved.

"""
Reduce-scatter collective operation -- public API.

Triton only (no gluon support).

Variants:
- auto: Select best variant based on message size (default)
- two_shot: Serial accumulation chain, general-purpose
- inreg: In-register reduction, best at <=2MB (8-GPU)
- twophase: Two-phase decomposition, best at 2-8MB (8-GPU)
"""

from iris.ccl.utils import extract_group_info


def reduce_scatter(output_tensor, input_tensor, ctx, op=None, group=None, async_op=False, config=None, workspace=None):
    """
    Reduce-scatter: each rank reduces its assigned tiles, stores locally.

    Args:
        output_tensor: Shape (M, N)
        input_tensor: Shape (M, N)
        ctx: Iris instance
        op: ReduceOp (only SUM supported)
        group: ProcessGroup or None
        async_op: If True, skip trailing barrier
        config: Config with kernel parameters
        workspace: Reusable workspace (for twophase variant scratch buffer)

    Returns:
        workspace if variant needs it (twophase), else None
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
    if config.use_gluon:
        raise ValueError(
            "reduce_scatter does not support use_gluon=True. "
            "Gluon implementation is not available for reduce_scatter. "
            "Use default config (use_gluon=False)."
        )

    rank_in_group, rank_global, world_size, rank_start, rank_stride = extract_group_info(group, ctx)
    M, N = input_tensor.shape[:2]

    if output_tensor.shape[:2] != (M, N):
        raise ValueError(
            f"Output tensor shape {output_tensor.shape[:2]} does not match input shape {(M, N)}. "
            f"For reduce-scatter, output should have the same shape as input."
        )

    from iris.ccl.triton.reduce_scatter import launch

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
        workspace=workspace,
    )

    if not async_op:
        ctx.barrier()

    return workspace
