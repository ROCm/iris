# SPDX-License-Identifier: MIT
# Copyright (c) 2025 Advanced Micro Devices, Inc. All rights reserved.

"""
Reduce collective operation — public API.

Three paths by message size:
- Small (<512KB): NCCL (avoids Triton launch overhead)
- Small/medium (512KB-64KB): Direct-read (root reads all peers)
- Medium (64KB-8MB): Two-phase (reduce-scatter + push to root)
- Large (>=8MB): NCCL tree reduce
"""

import torch.distributed as _dist

from iris.ccl.utils import extract_group_info

_NCCL_SMALL_BYTES = 0  # native kernel wins at all sizes (25us vs NCCL 27-61us)
_TWOPHASE_BYTES = 64 * 1024
_NCCL_LARGE_BYTES = 8 * 1024 * 1024  # >=8MB: NCCL tree reduce


def reduce(output_tensor, input_tensor, ctx, dst=0, op=None, group=None, async_op=False, config=None):
    """
    Reduce: sum inputs across all ranks, result stored only on root (dst).

    Args:
        output_tensor: Shape (M, N) — receives the reduced result on root.
        input_tensor: Shape (M, N) — local rank's partial data.
        ctx: Iris instance.
        dst: Destination rank (within the group) that receives the result.
        op: ReduceOp (only SUM supported).
        group: ProcessGroup or None.
        async_op: If True, skip trailing barrier.
        config: Config with kernel parameters.
    """
    # Compute message size early for fast NCCL dispatch
    numel = input_tensor.numel()
    msg_bytes = numel * input_tensor.element_size()

    # Validate dst early (before NCCL dispatch) to give a clean error
    world_size = _dist.get_world_size(group)
    if dst < 0 or dst >= world_size:
        raise ValueError(f"dst must be in [0, world_size), got dst={dst}, world_size={world_size}.")

    # Small and large messages: use NCCL (avoids Triton launch overhead at small
    # sizes, and NCCL tree reduce is more bandwidth-efficient at large sizes).
    if msg_bytes < _NCCL_SMALL_BYTES or msg_bytes >= _NCCL_LARGE_BYTES:
        # torch.distributed.reduce is in-place. Avoid full copy on all ranks —
        # reduce in-place on input, then copy result to output only on root.
        if output_tensor.data_ptr() == input_tensor.data_ptr():
            _dist.reduce(output_tensor, dst=dst, group=group)
        else:
            _dist.reduce(input_tensor, dst=dst, group=group)
            rank_in_group = _dist.get_rank(group)
            if rank_in_group == dst:
                output_tensor.copy_(input_tensor)
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
        config = Config(block_size_m=32, block_size_n=64)

    rank_in_group, rank_global, world_size, rank_start, rank_stride = extract_group_info(group, ctx)

    # Reshape for efficient tiling (same as broadcast.py)
    # Avoids M=1 with block_size_m=32 where 31/32 rows are masked/wasted.
    block_n = config.block_size_n
    if numel >= block_n:
        input_tensor = input_tensor.contiguous().view(-1, block_n)
        output_tensor = output_tensor.contiguous().view(-1, block_n)
    else:
        input_tensor = input_tensor.contiguous().view(1, -1)
        output_tensor = output_tensor.contiguous().view(1, -1)

    if config.use_gluon:
        from iris.ccl.gluon.reduce import launch

        launch(
            output_tensor,
            input_tensor,
            ctx,
            rank_in_group,
            rank_global,
            dst,
            world_size,
            rank_start,
            rank_stride,
            config,
        )
        if not async_op:
            ctx.device_barrier(group)
    elif msg_bytes >= _TWOPHASE_BYTES and world_size > 1:
        from iris.ccl.triton.reduce_twophase import launch as launch_twophase

        use_inline = not async_op
        barrier_state = None
        if use_inline:
            barrier_state = ctx._get_inline_barrier_state(group)

        launch_twophase(
            output_tensor,
            input_tensor,
            ctx,
            rank_in_group,
            rank_global,
            dst,
            world_size,
            rank_start,
            rank_stride,
            config,
            inline_barrier=use_inline,
            barrier_state=barrier_state,
        )
    else:
        from iris.ccl.triton.reduce import launch

        use_inline = not async_op
        barrier_state = None
        if use_inline:
            barrier_state = ctx._get_inline_barrier_state(group)

        launch(
            output_tensor,
            input_tensor,
            ctx,
            rank_in_group,
            rank_global,
            dst,
            world_size,
            rank_start,
            rank_stride,
            config,
            inline_barrier=use_inline,
            barrier_state=barrier_state,
        )
