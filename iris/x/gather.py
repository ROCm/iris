# SPDX-License-Identifier: MIT
# Copyright (c) 2025-2026 Advanced Micro Devices, Inc. All rights reserved.

"""
Tile-level gather primitive for Iris.

This is a simpler, lower-level primitive than all_gather:
- gather: Load a tile from a SPECIFIC rank and return it (no store)
- all_gather: Load tiles from ALL ranks and store them to a buffer

Use `gather` when you want to consume the tile immediately without materializing it.
"""

import triton
import triton.language as tl
import iris
from iris.iris import DeviceContext
from .core import Tile, TensorView


@triton.jit()
def gather(
    tile: Tile,
    src_view: TensorView,
    source_rank: tl.constexpr,
    ctx: DeviceContext,
):
    """
    Tile-level gather from a specific rank.

    Loads a tile from source_rank's memory and returns it directly.
    Unlike all_gather, this does NOT store the tile - it's meant to be
    immediately consumed by the caller (e.g., in a GEMM dot product).

    Args:
        tile: Tile object with position and dimensions.
        src_view: TensorView for source tensor on source_rank.
        source_rank: Specific rank to load from (constexpr).
        ctx: DeviceContext with rank, world_size, and heap_bases.

    Returns:
        Loaded tile data as a tensor.

    Example usage in fused GEMM:
        for source_rank in range(world_size):
            a = gather(tile, src_view, source_rank, ctx)
            b = tl.load(...)
            acc += tl.dot(a, b)  # Consume immediately, no materialization
    """
    # Get tile pointer and mask
    src_tile_ptr, mask = src_view.tile_ptr(tile)

    if source_rank == ctx.rank:
        # Local load
        tile_data = tl.load(src_tile_ptr, mask=mask, other=0.0)
    else:
        # Remote load using RMA
        tile_data = iris.load(
            src_tile_ptr,
            ctx.rank,  # to_rank (current rank)
            source_rank,  # from_rank (source rank)
            ctx.heap_bases,
            mask=mask,
        )

    return tile_data
