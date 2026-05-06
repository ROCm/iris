# SPDX-License-Identifier: MIT
# Copyright (c) 2025 Advanced Micro Devices, Inc. All rights reserved.

"""
Reduce collective operation — public API.

NCCL at all sizes (Triton launch overhead ~65-80us vs NCCL ~37-55us on MI300X).
"""

import torch.distributed as _dist

from iris.ccl.utils import extract_group_info

_NCCL_SMALL_BYTES = 0
_RING_BYTES = 512 * 1024     # >=512KB: ring reduce with p2p flags
_NCCL_LARGE_BYTES = 0        # NCCL at all sizes (ring reduce doesn't pipeline)

_reduce_group_cache: dict = {}


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

    if msg_bytes < _NCCL_SMALL_BYTES or msg_bytes >= _NCCL_LARGE_BYTES:
        cached = _reduce_group_cache.get(group)
        if cached is None:
            cached = (_dist.get_world_size(group), _dist.get_rank(group))
            _reduce_group_cache[group] = cached
        world_size, rank_in_group = cached
        if dst < 0 or dst >= world_size:
            raise ValueError(f"dst must be in [0, world_size), got dst={dst}, world_size={world_size}.")
        if output_tensor.data_ptr() == input_tensor.data_ptr():
            _dist.reduce(output_tensor, dst=dst, group=group)
        else:
            _dist.reduce(input_tensor, dst=dst, group=group)
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

    use_inline = not async_op
    barrier_state = None
    if use_inline:
        barrier_state = ctx._get_inline_barrier_state(group)

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
    elif msg_bytes >= _RING_BYTES and world_size > 1:
        from iris.ccl.triton.reduce_ring import launch as launch_ring

        launch_ring(
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
            group=group,
            use_p2p=True,
        )
    else:
        from iris.ccl.triton.reduce import launch

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
