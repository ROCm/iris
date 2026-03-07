# SPDX-License-Identifier: MIT
# Copyright (c) 2025-2026 Advanced Micro Devices, Inc. All rights reserved.

"""
Tile-level all-gather primitive for Iris.

Gathers tiles from all ranks and concatenates them along the output dimension.
"""

import triton
import triton.language as tl
import iris
from iris.iris import DeviceContext
from .core import Tile, TensorView


@triton.jit()
def all_gather(
    tile: Tile,
    dst_view: TensorView,
    dim: tl.constexpr,
    ctx: DeviceContext,
):
    """
    Tile-level all-gather operation (scatter pre-computed data mode).

    Scatters a pre-computed tile to all ranks at correct offsets.
    Automatically computes local dimensions from dst_view and world_size.

    Args:
        tile: Tile object with position, dimensions, and computed data in tile.data.
        dst_view: TensorView for destination tensor after gather (full gathered size).
        dim: Dimension to gather along (0 for rows, 1 for columns).
        ctx: DeviceContext with rank, world_size, and heap_bases.

    Layout:
        - dim=0: Current rank's rows go to output[ctx.rank * M_local : (ctx.rank+1) * M_local, :]
                 where M_local = dst_view.M / world_size
        - dim=1: Current rank's cols go to output[:, ctx.rank * N_local : (ctx.rank+1) * N_local]
                 where N_local = dst_view.N / world_size

    Example:
        tile_obj = iris.x.Tile(pid_m, pid_n, BLOCK_SIZE_M, BLOCK_SIZE_N, c)
        # dst_view has shape (M_total, N) where M_total = M_local * world_size
        iris.x.all_gather(tile_obj, dst_view, dim=0, ctx)

    Note:
        The dst_view should represent the FULL gathered tensor size, not the local size.
        For dim=0: dst_view.M = M_local * world_size
        For dim=1: dst_view.N = N_local * world_size
    """
    # Compute local dimensions from dst_view and world_size
    if dim == 0:
        M_local = dst_view.M // ctx.world_size
    else:
        N_local = dst_view.N // ctx.world_size

    # Scatter to all ranks
    for dest_rank in range(ctx.world_size):
        if dim == 0:
            # Scatter along M dimension: write to [ctx.rank * M_local : (ctx.rank+1) * M_local, :]
            dst_ptr, combined_mask = dst_view.offset_tile_ptr(tile, offset_m=ctx.rank * M_local, src_mask=None)
        else:
            # Scatter along N dimension: write to [:, ctx.rank * N_local : (ctx.rank+1) * N_local]
            dst_ptr, combined_mask = dst_view.offset_tile_ptr(tile, offset_n=ctx.rank * N_local, src_mask=None)

        # Use iris.store to write to dest_rank's memory
        iris.store(
            dst_ptr,
            tile.data,
            ctx.rank,  # from_rank (current rank)
            dest_rank,  # to_rank (destination rank)
            ctx.heap_bases,
            mask=combined_mask,
        )
