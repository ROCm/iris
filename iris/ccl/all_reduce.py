# SPDX-License-Identifier: MIT
# Copyright (c) 2025 Advanced Micro Devices, Inc. All rights reserved.

"""
All-reduce collective operation — public API.

Two paths by message size:
- Small (<512KB): NCCL (avoids Triton dispatch overhead)
- Large (>=512KB): native two_shot kernel
"""

import torch.distributed as _dist

from iris.ccl.utils import extract_group_info

_NCCL_FALLBACK_BYTES = 4 * 1024 * 1024  # <4MB: NCCL avoids Triton launch overhead
_NCCL_LARGE_BYTES = 8 * 1024 * 1024  # >=8MB: NCCL tree all-reduce may be more efficient


def all_reduce_preamble(output_tensor, input_tensor, ctx, config=None, workspace=None):
    """Prepare reusable workspace for all-reduce."""
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
    numel = input_tensor.numel()
    msg_bytes = numel * input_tensor.element_size()

    if msg_bytes < _NCCL_FALLBACK_BYTES or msg_bytes >= _NCCL_LARGE_BYTES:
        if output_tensor.data_ptr() != input_tensor.data_ptr():
            output_tensor.copy_(input_tensor)
        _dist.all_reduce(output_tensor, group=group)
        return None

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
            "all_reduce does not support use_gluon=True. "
            "Gluon implementation is not available for all_reduce. "
            "Use default config (use_gluon=False)."
        )

    variant = config.all_reduce_variant.lower()
    valid_variants = ["atomic", "spinlock", "ring", "two_shot", "one_shot"]
    if variant not in valid_variants:
        raise ValueError(f"Invalid all_reduce_variant: {variant}. Must be one of: {', '.join(valid_variants)}")

    # Reshape for efficient tiling (same as broadcast.py / reduce.py)
    # Avoids M=1 with block_size_m=32 where 31/32 rows are masked/wasted.
    block_n = config.block_size_n
    if numel >= block_n:
        input_tensor = input_tensor.contiguous().view(-1, block_n)
        output_tensor = output_tensor.contiguous().view(-1, block_n)
    else:
        input_tensor = input_tensor.contiguous().view(1, -1)
        output_tensor = output_tensor.contiguous().view(1, -1)

    rank_in_group, rank_global, world_size, rank_start, rank_stride = extract_group_info(group, ctx)

    from iris.ccl.triton.all_reduce import launch

    use_inline = not async_op
    barrier_state = None
    if use_inline:
        barrier_state = ctx._get_inline_barrier_state(group)

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
        inline_barrier=use_inline,
        barrier_state=barrier_state,
    )

    if workspace is not None:
        variant = getattr(config, "all_reduce_variant", "two_shot").lower()
        if variant not in ("ring", "two_shot"):
            workspace.prepared = False

    return workspace
