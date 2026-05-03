# SPDX-License-Identifier: MIT
# Copyright (c) 2025 Advanced Micro Devices, Inc. All rights reserved.

"""
Reduce collective operation — public API.

Two paths by message size:
- Small/medium: lock-based single kernel
- Large (>=4MB): NCCL (tree-based, scales better at high message sizes)
"""

import torch.distributed as _dist

from iris.ccl.utils import extract_group_info

_NCCL_SMALL_BYTES = 128 * 1024
_NCCL_LARGE_BYTES = 4 * 1024 * 1024


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
        raise ValueError(f"dst must be in [0, world_size), got dst={dst}, world_size={world_size}.")

    M, N = input_tensor.shape[:2]
    msg_bytes = M * N * input_tensor.element_size()

    if msg_bytes < _NCCL_SMALL_BYTES or msg_bytes >= _NCCL_LARGE_BYTES:
        if output_tensor.data_ptr() != input_tensor.data_ptr():
            output_tensor.copy_(input_tensor)
        _dist.reduce(output_tensor, dst=dst, group=group)
        return

    if config.use_gluon:
        from iris.ccl.gluon.reduce import launch
    else:
        from iris.ccl.triton.reduce import launch

    launch(
        output_tensor, input_tensor, ctx,
        rank_in_group, rank_global, dst,
        world_size, rank_start, rank_stride, config,
    )

    if not async_op:
        ctx.device_barrier(group)
