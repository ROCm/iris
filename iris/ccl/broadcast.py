# SPDX-License-Identifier: MIT
# Copyright (c) 2025 Advanced Micro Devices, Inc. All rights reserved.

"""
Broadcast collective operation — public API.

Drop-in replacement for torch.distributed.broadcast.
Accepts regular CUDA tensors, handles heap copy via as_symmetric.
"""

from iris.ccl.utils import extract_group_info


def broadcast(output_tensor, input_tensor, ctx, src=0, group=None, async_op=False, config=None):
    """
    Broadcast: root rank sends its data to all ranks.

    Accepts regular CUDA tensors — copies to/from symmetric heap via as_symmetric.
    """
    from iris.ccl.config import Config

    if config is None:
        config = Config(block_size_m=32, block_size_n=64)

    rank_in_group, rank_global, world_size, rank_start, rank_stride = extract_group_info(group, ctx)

    M, N = input_tensor.shape[:2]
    if output_tensor.shape[:2] != (M, N):
        raise ValueError(
            f"Output tensor shape {output_tensor.shape[:2]} does not match input shape "
            f"({M}, {N}). Broadcast requires input and output to have the same shape."
        )

    if src < 0 or src >= world_size:
        raise ValueError(f"src rank {src} is out of range [0, {world_size})")

    heap_in = ctx.as_symmetric(input_tensor, tag="bc_inp")
    heap_out = ctx.as_symmetric(output_tensor, tag="bc_out")

    from iris.ccl.triton.broadcast import launch

    launch(
        heap_in,
        heap_out,
        ctx,
        rank_in_group,
        rank_global,
        world_size,
        rank_start,
        rank_stride,
        src,
        config,
    )

    if not ctx.is_symmetric(output_tensor):
        output_tensor.copy_(heap_out)

    if not async_op:
        ctx.barrier()
