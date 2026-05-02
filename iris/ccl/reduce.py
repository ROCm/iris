# SPDX-License-Identifier: MIT
# Copyright (c) 2025 Advanced Micro Devices, Inc. All rights reserved.

"""
Reduce collective operation — public API.

Routes to triton/ or gluon/ based on config.use_gluon.
"""

from iris.ccl.utils import extract_group_info


def reduce(output_tensor, input_tensor, ctx, dst=0, op=None, group=None, async_op=False, config=None):
    """
    Reduce: sum inputs across all ranks, result stored only on root (dst).

    All ranks contribute their input_tensor. The element-wise sum is
    computed and written to output_tensor on the root rank only.
    Non-root ranks' output_tensor contents are undefined (MPI semantics).

    Args:
        output_tensor: Shape (M, N) — receives the reduced result on root.
        input_tensor: Shape (M, N) — local rank's partial data.
        ctx: Iris instance.
        dst: Destination rank (within the group) that receives the result.
             Default: 0.
        op: ReduceOp (only SUM supported). Default: ReduceOp.SUM.
        group: ProcessGroup or None. If None, uses all ranks.
        async_op: If True, skip trailing barrier. Default: False.
        config: Config with kernel parameters. Default: None.
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
        config = Config(block_size_m=32, block_size_n=64)

    rank_in_group, rank_global, world_size, rank_start, rank_stride = extract_group_info(group, ctx)

    if dst < 0 or dst >= world_size:
        raise ValueError(
            f"dst must be in [0, world_size), got dst={dst}, world_size={world_size}."
        )

    if config.use_gluon:
        from iris.ccl.gluon.reduce import launch
    else:
        from iris.ccl.triton.reduce import launch

    launch(
        output_tensor,
        input_tensor,
        ctx,
        rank_in_group,
        rank_global,
        world_size,
        rank_start,
        rank_stride,
        dst,
        config,
    )

    if not async_op:
        ctx.barrier()
