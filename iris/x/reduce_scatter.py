# SPDX-License-Identifier: MIT
# Copyright (c) 2025 Advanced Micro Devices, Inc. All rights reserved.

"""
Tile-level reduce-scatter primitive for Iris.

Reduces tiles from all ranks and stores the result only to the assigned rank.
"""

import triton
import triton.language as tl
import iris
from .core import Tile, TensorView, DeviceContext


@triton.jit()
def reduce_scatter(
    tile: Tile,
    src_view: TensorView,
    dst_view: TensorView,
    ctx: DeviceContext,
):
    """
    Tile-level reduce-scatter for iris.x.

    Reduces data from all ranks and each rank stores only its assigned portion.

    Args:
        tile: Tile object with position and dimensions.
        src_view: TensorView for input tensor.
        dst_view: TensorView for output tensor.
        ctx: DeviceContext with rank, world_size, and heap_bases.
    """
    # Get tile pointer and mask
    src_tile_ptr, mask = src_view.tile_ptr(tile)
    dst_tile_ptr, _ = dst_view.tile_ptr(tile)

    # Load local tile
    local_tile = tl.load(src_tile_ptr, mask=mask, other=0.0)

    # Initialize accumulator with proper dtype
    acc_dtype = tl.float32 if local_tile.dtype == tl.float16 else local_tile.dtype
    acc = tl.zeros((tile.block_m, tile.block_n), dtype=acc_dtype)
    acc += local_tile.to(acc_dtype)

    # Accumulate from all other ranks
    for r in range(ctx.world_size):
        if r != ctx.rank:
            # Read tile from remote rank
            remote_tile = iris.load(
                src_tile_ptr,
                ctx.heap_bases,
                r,
                mask=mask,
                other=0.0,
            )
            acc += remote_tile.to(acc_dtype)

    # Convert back to original dtype and store
    result = acc.to(local_tile.dtype)
    tl.store(dst_tile_ptr, result, mask=mask)
