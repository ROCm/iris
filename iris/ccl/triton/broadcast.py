# SPDX-License-Identifier: MIT
# Copyright (c) 2025 Advanced Micro Devices, Inc. All rights reserved.

"""
Broadcast — Triton kernel.

All-pairs-read pattern: non-root ranks read root's data via iris.load.
Root is a no-op (data already in place on the symmetric heap).
"""

import triton
import triton.language as tl
import iris


@triton.jit()
def _broadcast_kernel(
    tensor_ptr,
    numel,
    heap_bases: tl.tensor,
    group_rank: tl.constexpr,
    iris_rank: tl.constexpr,
    world_size: tl.constexpr,
    src_rank_in_group: tl.constexpr,
    src_rank_global: tl.constexpr,
    rank_start: tl.constexpr,
    rank_stride: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    """
    Broadcast kernel: non-root ranks read root's data via iris.load.

    Grid: (cdiv(numel, BLOCK_SIZE),)
    """
    if group_rank != src_rank_in_group:
        pid = tl.program_id(0)
        offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
        mask = offsets < numel

        data = iris.load(
            tensor_ptr + offsets,
            iris_rank,
            src_rank_global,
            heap_bases,
            mask=mask,
        )
        tl.store(tensor_ptr + offsets, data, mask=mask)


def launch(
    tensor,
    ctx,
    rank_in_group,
    rank_global,
    world_size,
    rank_start,
    rank_stride,
    src,
    config,
):
    """Launch broadcast Triton kernel."""
    numel = tensor.numel()
    tensor_flat = tensor.view(-1)

    src_rank_global = rank_start + src * rank_stride
    heap_bases = ctx.get_heap_bases()

    block_size = config.block_size_m * config.block_size_n
    grid = ((numel + block_size - 1) // block_size,)

    _broadcast_kernel[grid](
        tensor_flat,
        numel,
        heap_bases,
        rank_in_group,
        rank_global,
        world_size,
        src,
        src_rank_global,
        rank_start,
        rank_stride,
        block_size,
    )
