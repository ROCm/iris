# SPDX-License-Identifier: MIT
# Copyright (c) 2025 Advanced Micro Devices, Inc. All rights reserved.

"""
Reduce-scatter collective operation — public API.

Two-shot kernel for medium messages, NCCL for small/large.
"""

import torch.distributed as _dist

from iris.ccl.utils import extract_group_info

_NCCL_SMALL_BYTES = 0
_NCCL_LARGE_BYTES = 8 * 1024 * 1024  # <8MB: native two_shot, >=8MB: NCCL


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
    numel = input_tensor.numel()
    msg_bytes = numel * input_tensor.element_size()

    if msg_bytes < _NCCL_SMALL_BYTES or msg_bytes >= _NCCL_LARGE_BYTES:
        world_size = _dist.get_world_size(group)
        chunk_size = numel // world_size
        _dist.reduce_scatter_tensor(output_tensor.view(-1)[:chunk_size], input_tensor.view(-1), group=group)
        return

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
        config = Config(block_size_m=32, block_size_n=64, all_reduce_distribution=1, comm_sms=64)

    rank_in_group, rank_global, world_size, rank_start, rank_stride = extract_group_info(group, ctx)

    if config.use_gluon:
        raise ValueError(
            "reduce_scatter does not support use_gluon=True. "
            "Gluon implementation is not available for reduce_scatter. "
            "Use default config (use_gluon=False)."
        )

    variant = getattr(config, "reduce_scatter_variant", "two_shot")
    if variant not in ("two_shot", "fused"):
        raise ValueError(f"reduce_scatter variant must be 'two_shot' or 'fused', got '{variant}'.")

    block_n = config.block_size_n
    if numel >= block_n:
        input_tensor = input_tensor.contiguous().view(-1, block_n)
        output_tensor = output_tensor.contiguous().view(-1, block_n)
    else:
        input_tensor = input_tensor.contiguous().view(1, -1)
        output_tensor = output_tensor.contiguous().view(1, -1)

    from iris.ccl.triton.reduce_scatter import launch

    if variant == "fused":
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
            variant="fused",
        )
    else:
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
            inline_barrier=False,
            barrier_state=None,
        )
