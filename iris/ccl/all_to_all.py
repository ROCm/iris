# SPDX-License-Identifier: MIT
# Copyright (c) 2025 Advanced Micro Devices, Inc. All rights reserved.

"""
All-to-all collective operations — public API.

Drop-in replacement for torch.distributed.all_to_all_single.
Accepts regular CUDA tensors, handles heap copy via as_symmetric.
"""

from iris.ccl.utils import extract_group_info


def all_to_all(output_tensor, input_tensor, ctx, group=None, async_op=False, config=None):
    """
    All-to-all: each rank sends a chunk to every other rank.

    Input/output shape: (M, N * world_size).
    Accepts regular CUDA tensors — copies to/from symmetric heap via as_symmetric.
    """
    from iris.ccl.config import Config

    if config is None:
        config = Config(block_size_m=32, block_size_n=128)

    rank_in_group, rank_global, world_size, rank_start, rank_stride = extract_group_info(group, ctx)

    heap_in = ctx.as_symmetric(input_tensor)
    heap_out = ctx.as_symmetric(output_tensor)

    if config.use_gluon:
        from iris.ccl.gluon.all_to_all import launch
    else:
        from iris.ccl.triton.all_to_all import launch

    launch(
        heap_in,
        heap_out,
        ctx,
        rank_in_group,
        rank_global,
        world_size,
        rank_start,
        rank_stride,
        config,
    )

    if not ctx.is_symmetric(output_tensor):
        output_tensor.copy_(heap_out)

    if not async_op:
        ctx.barrier()
