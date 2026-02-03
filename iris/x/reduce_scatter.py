# SPDX-License-Identifier: MIT
# Copyright (c) 2025-2026 Advanced Micro Devices, Inc. All rights reserved.

"""
Tile-level reduce-scatter primitive for Iris.

Reduces tiles from all ranks and stores the result only to the assigned rank.
"""

import triton
import triton.language as tl
import iris
from iris.iris import DeviceContext
from .core import Tile, TensorView


@triton.jit()
def reduce_scatter(
    tile: Tile,
    src_view: TensorView,
    dst_view: TensorView,
    locks,
    ctx: DeviceContext,
):
    """
    Tile-level reduce-scatter using two-shot algorithm with contiguous work distribution.

    Each rank reduces only its assigned contiguous block of tiles, then stores
    the result locally (no scatter to other ranks).

    Uses locks as ready flags: before loading, wait for remote tiles to be ready (lock == 1).

    Args:
        tile: Tile object with position, dimensions, and local data (tile.data).
        src_view: TensorView for source tensor (to load remote data).
        dst_view: TensorView for output tensor where reduced result will be written.
        locks: Pointer to lock array (one per tile) used as ready flags.
        ctx: DeviceContext with rank, world_size, and heap_bases.

    Example:
        # With 4 ranks and 12 tiles:
        # Rank 0 handles tiles 0, 1, 2
        # Rank 1 handles tiles 3, 4, 5
        tile = iris.x.Tile(pid_m, pid_n, BLOCK_SIZE_M, BLOCK_SIZE_N, local_result)
        src_view = iris.x.TensorView(temp_buffer, M, N, stride_m, stride_n)
        dst_view = iris.x.TensorView(output_ptr, M, N, stride_m, stride_n)
        iris.x.reduce_scatter(tile, src_view, dst_view, locks, ctx)
    """
    num_tiles_n = tl.cdiv(dst_view.N, tile.block_n)
    num_tiles_m = tl.cdiv(dst_view.M, tile.block_m)
    total_tiles = num_tiles_m * num_tiles_n
    tile_id = tile.pid_m * num_tiles_n + tile.pid_n

    tiles_per_rank = total_tiles // ctx.world_size
    start_tile = ctx.rank * tiles_per_rank
    end_tile = start_tile + tiles_per_rank

    if ctx.rank == ctx.world_size - 1:
        end_tile = total_tiles

    is_responsible = (tile_id >= start_tile) and (tile_id < end_tile)

    if is_responsible:
        src_tile_ptr, mask = src_view.tile_ptr(tile)
        dst_tile_ptr, _ = dst_view.tile_ptr(tile)

        acc_dtype = tl.float32 if tile.data.dtype == tl.float16 else tile.data.dtype
        acc = tile.data.to(acc_dtype)

        # Skip current rank - tile.data already contains local contribution
        for remote_rank in range(ctx.world_size):
            if remote_rank != ctx.rank:
                lock_ptr = locks + tile_id
                while (
                    iris.atomic_add(lock_ptr, 0, ctx.rank, remote_rank, ctx.heap_bases, sem="acquire", scope="gpu") != 1
                ):
                    pass

                partial = iris.load(src_tile_ptr, ctx.rank, remote_rank, ctx.heap_bases, mask=mask)
                acc += partial.to(acc_dtype)

        result = acc.to(tile.data.dtype)
        tl.store(dst_tile_ptr, result, mask=mask)
