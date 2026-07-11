# SPDX-License-Identifier: MIT
# Copyright (c) 2025 Advanced Micro Devices, Inc. All rights reserved.

"""
All-to-all collective operations — public API.

Accepts regular CUDA tensors. Internally copies to/from symmetric heap.
"""

import torch
from iris.ccl.utils import extract_group_info
from iris.ccl.dispatch import get_heap_buffer


def all_to_all(output_tensor, input_tensor, ctx, group=None, async_op=False, config=None):
    """
    All-to-all: each rank sends a chunk to every other rank.

    Input/output shape: (M, N * world_size).
    Accepts regular CUDA tensors — copies to heap internally.
    """
    from iris.ccl.config import Config

    if config is None:
        config = Config(block_size_m=32, block_size_n=128)

    rank_in_group, rank_global, world_size, rank_start, rank_stride = extract_group_info(group, ctx)

    shape = input_tensor.shape[:2]
    heap_inp = get_heap_buffer(ctx, shape, input_tensor.dtype, "a2a_inp")
    heap_out = get_heap_buffer(ctx, shape, input_tensor.dtype, "a2a_out")

    heap_inp.copy_(input_tensor)

    if config.use_gluon:
        from iris.ccl.gluon.all_to_all import launch
    else:
        from iris.ccl.triton.all_to_all import launch

    launch(
        heap_inp,
        heap_out,
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

    output_tensor.copy_(heap_out)
