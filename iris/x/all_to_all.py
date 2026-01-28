# SPDX-License-Identifier: MIT
# Copyright (c) 2025 Advanced Micro Devices, Inc. All rights reserved.

"""
Tile-level all-to-all primitive for Iris.

Performs all-to-all communication where each rank sends and receives data to/from all other ranks.
"""

import triton
import triton.language as tl
import iris
from .core import Tile, TensorView, DeviceContext


@triton.jit()
def all_to_all(
    tile: Tile,
    src_view: TensorView,
    dst_view: TensorView,
    N_per_rank: tl.constexpr,
    ctx: DeviceContext,
):
    """
    Tile-level all-to-all for iris.x.

    Each rank sends a portion of its data to every other rank and receives data
    from every other rank. The data is organized by columns (N dimension).

    Args:
        tile: Tile object with position and dimensions.
        src_view: TensorView for input tensor.
        dst_view: TensorView for output tensor.
        N_per_rank: Number of columns each rank sends/receives.
        ctx: DeviceContext with rank, world_size, and heap_bases.

    Layout:
        Input: Each rank has (M, world_size * N_per_rank)
        Output: Each rank has (M, world_size * N_per_rank)
        Rank i sends columns [j*N_per_rank:(j+1)*N_per_rank] to rank j
        Rank i receives into columns [j*N_per_rank:(j+1)*N_per_rank] from rank j
    """
    # For each rank, read the appropriate slice and write to output
    for r in range(ctx.world_size):
        # Source column offset for data destined to rank r
        src_col_offset = r * N_per_rank + tile.pid_n * tile.block_n
        src_indices_m = tile.pid_m * tile.block_m + tl.arange(0, tile.block_m)
        src_indices_n = src_col_offset + tl.arange(0, tile.block_n)

        # Compute mask
        mask_m = src_indices_m < src_view.M
        mask_n = src_indices_n < src_view.N
        mask = mask_m[:, None] & mask_n[None, :]

        # Compute source offset
        src_offsets = src_indices_m[:, None] * src_view.stride_m + src_indices_n[None, :] * src_view.stride_n

        # Destination column offset for data coming from rank r
        dst_col_offset = r * N_per_rank + tile.pid_n * tile.block_n
        dst_indices_m = tile.pid_m * tile.block_m + tl.arange(0, tile.block_m)
        dst_indices_n = dst_col_offset + tl.arange(0, tile.block_n)
        dst_offsets = dst_indices_m[:, None] * dst_view.stride_m + dst_indices_n[None, :] * dst_view.stride_n

        # Read from appropriate rank and write to output
        if r == ctx.rank:
            # Local data: direct copy
            data = tl.load(src_view.ptr + src_offsets, mask=mask, other=0.0)
            tl.store(dst_view.ptr + dst_offsets, data, mask=mask)
        else:
            # Remote data: read from rank r
            data = iris.load(
                src_view.ptr + src_offsets,
                ctx.rank,  # to_rank (current rank)
                r,  # from_rank (remote rank)
                ctx.heap_bases,
                mask=mask,
            )
            tl.store(dst_view.ptr + dst_offsets, data, mask=mask)
