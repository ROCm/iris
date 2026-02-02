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
    # For each remote rank, read the data that rank sent to us
    for r in range(ctx.world_size):
        # Read from rank r's column for current rank (ctx.rank)
        # Rank r has data destined for us in column [ctx.rank * N_per_rank]
        src_ptr, mask = src_view.offset_tile_ptr(tile, offset_n=ctx.rank * N_per_rank)

        # Write to our output column r (data from rank r)
        dst_ptr, _ = dst_view.offset_tile_ptr(tile, offset_n=r * N_per_rank)

        # Read from appropriate rank and write to output
        if r == ctx.rank:
            # Local data: direct copy from our own column
            data = tl.load(src_ptr, mask=mask, other=0.0)
            tl.store(dst_ptr, data, mask=mask)
        else:
            # Remote data: read from rank r's memory
            data = iris.load(
                src_ptr,
                ctx.rank,  # to_rank (current rank doing the read)
                r,  # from_rank (remote rank we're reading from)
                ctx.heap_bases,
                mask=mask,
            )
            tl.store(dst_ptr, data, mask=mask)
