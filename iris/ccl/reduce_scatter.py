# SPDX-License-Identifier: MIT
# Copyright (c) 2025 Advanced Micro Devices, Inc. All rights reserved.

"""
Reduce-scatter collective operation — public API.

Routes to the Triton two-shot kernel, or with use_gluon and
reduce_scatter_variant="tdm" to the gfx1250 TDM kernel in gluon/.
"""

from iris.ccl.utils import extract_group_info


def reduce_scatter(output_tensor, input_tensor, ctx, op=None, group=None, async_op=False, config=None):
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
    variant = getattr(config, "reduce_scatter_variant", "two_shot")
    if variant not in ("two_shot", "tdm"):
        raise ValueError(f"reduce_scatter supports variant='two_shot' or 'tdm', got '{variant}'.")
    if config.use_gluon and variant != "tdm":
        raise ValueError(
            "reduce_scatter with use_gluon=True needs reduce_scatter_variant='tdm'; "
            "the two-shot kernel is Triton only. Use use_gluon=False."
        )

    rank_in_group, rank_global, world_size, rank_start, rank_stride = extract_group_info(group, ctx)
    M, N = input_tensor.shape[:2]

    if output_tensor.shape[:2] != (M, N):
        raise ValueError(
            f"Output tensor shape {output_tensor.shape[:2]} does not match input shape {(M, N)}. "
            f"For reduce-scatter, output should have the same shape as input."
        )

    if variant == "tdm":
        # gfx1250 tensor data movement engine; see gluon/reduce_scatter_tdm.py
        from iris.ccl.gluon.reduce_scatter_tdm import launch
    else:
        from iris.ccl.triton.reduce_scatter import launch

    launch(
        output_tensor,
        input_tensor,
        ctx,
        rank_in_group,
        rank_global,
        world_size,
        rank_start,
        rank_stride,
        config,
    )

    if not async_op:
        ctx.barrier()
