# SPDX-License-Identifier: MIT
# Copyright (c) 2025 Advanced Micro Devices, Inc. All rights reserved.

"""
Tile-level all-reduce primitives for Iris.

These functions operate on a single tile (BLOCK_SIZE_M x BLOCK_SIZE_N) given tile coordinates.
Users manage tile iteration themselves and call these functions from their own kernels.
"""

import triton
import triton.language as tl
import iris
from .core import Tile, TensorView, DeviceContext


@triton.jit()
def all_reduce_atomic(
    tile: Tile,
    dst_view: TensorView,
    ctx: DeviceContext,
):
    """
    Tile-level all-reduce using atomic operations.
    
    Takes a tile with pre-computed data (tile.data) and atomically adds it
    to the destination on all ranks.

    Args:
        tile: Tile object with position, dimensions, and data to reduce (tile.data).
        dst_view: TensorView for output tensor where reduced result will be written.
        ctx: DeviceContext with rank, world_size, and heap_bases.
        
    Example:
        # After computing a local tile result
        tile = iris.x.Tile(pid_m, pid_n, BLOCK_SIZE_M, BLOCK_SIZE_N, local_result)
        dst_view = iris.x.TensorView(output_ptr, M, N, stride_m, stride_n)
        iris.x.all_reduce_atomic(tile, dst_view, ctx)
    """
    # Get destination tile pointer and mask for this tile position
    dst_tile_ptr, mask = dst_view.tile_ptr(tile)
    
    # Atomically add local tile.data to all ranks' destination
    for dest_rank in range(ctx.world_size):
        iris.atomic_add(
            dst_tile_ptr,
            tile.data,
            ctx.rank,      # from_rank (current rank)
            dest_rank,     # to_rank (destination rank)
            ctx.heap_bases,
            mask=mask,
        )


@triton.jit()
def all_reduce_spinlock(
    tile: Tile,
    dst_view: TensorView,
    locks,
    ctx: DeviceContext,
):
    """
    Tile-level all-reduce using spinlock synchronization.
    
    Similar to atomic-add based all-reduce but uses spinlocks for exclusive
    access. For each rank's tile, acquires the lock, reads current value,
    adds local contribution (tile.data), writes back, and releases the lock.

    Args:
        tile: Tile object with position, dimensions, and local data (tile.data).
        dst_view: TensorView for output tensor where reduced result will be written.
        locks: Pointer to locks array (one lock per tile).
        ctx: DeviceContext with rank, world_size, and heap_bases.
        
    Example:
        # After computing a local tile result
        tile = iris.x.Tile(pid_m, pid_n, BLOCK_SIZE_M, BLOCK_SIZE_N, local_result)
        dst_view = iris.x.TensorView(output_ptr, M, N, stride_m, stride_n)
        iris.x.all_reduce_spinlock(tile, dst_view, locks_ptr, ctx)
    """
    # Compute tile ID for lock indexing
    num_tiles_n = tl.cdiv(dst_view.N, tile.block_n)
    tile_id = tile.pid_m * num_tiles_n + tile.pid_n
    
    # Get destination tile pointer and mask
    dst_tile_ptr, mask = dst_view.tile_ptr(tile)
    
    # For each rank, do spinlock-protected read-modify-write using iris RMA
    for dest_rank in range(ctx.world_size):
        # Acquire lock for this tile at dest_rank (spin until we swap 0 -> 1)
        # iris.atomic_cas handles remote rank access automatically
        while iris.atomic_cas(locks + tile_id, 0, 1, ctx.rank, dest_rank, ctx.heap_bases) != 0:
            pass
        
        # Load current value from dest_rank's tile using iris.load
        current_value = iris.load(dst_tile_ptr, ctx.rank, dest_rank, ctx.heap_bases, mask=mask)
        
        # Add our local contribution
        acc_dtype = tl.float32 if tile.data.dtype == tl.float16 else tile.data.dtype
        acc = current_value.to(acc_dtype) + tile.data.to(acc_dtype)
        
        # Store accumulated result back to dest_rank (overwriting) using iris.store
        result = acc.to(tile.data.dtype)
        iris.store(dst_tile_ptr, result, ctx.rank, dest_rank, ctx.heap_bases, mask=mask)
        
        # Release lock for this tile at dest_rank using iris.atomic_xchg
        iris.atomic_xchg(locks + tile_id, 0, ctx.rank, dest_rank, ctx.heap_bases)


@triton.jit()
def all_reduce_one_shot(
    tile: Tile,
    src_view: TensorView,
    dst_view: TensorView,
    ctx: DeviceContext,
):
    """
    Tile-level all-reduce using one-shot algorithm (all ranks gather and reduce locally).

    Each rank reads from all other ranks in one shot and computes the reduction locally.

    Args:
        tile: Tile object with position and dimensions.
        src_view: TensorView for input tensor.
        dst_view: TensorView for output tensor.
        ctx: DeviceContext with rank, world_size, and heap_bases.
    """
    # Get tile pointer and mask
    src_tile_ptr, mask = src_view.tile_ptr(tile)
    dst_tile_ptr, _ = dst_view.tile_ptr(tile)

    # Load local tile to get dtype
    local_tile = tl.load(src_tile_ptr, mask=mask, other=0.0)

    # Initialize accumulator
    acc_dtype = tl.float32 if local_tile.dtype == tl.float16 else local_tile.dtype
    acc = tl.zeros((tile.block_m, tile.block_n), dtype=acc_dtype)

    # Gather all partials from all ranks (including self) and accumulate
    for remote_rank in range(ctx.world_size):
        partial = iris.load(src_tile_ptr, ctx.rank, remote_rank, ctx.heap_bases, mask=mask)
        acc += partial.to(acc_dtype)

    # Store result
    result = acc.to(local_tile.dtype)
    tl.store(dst_tile_ptr, result, mask=mask)


@triton.jit()
def all_reduce_ring(
    tile: Tile,
    src_view: TensorView,
    dst_view: TensorView,
    ctx: DeviceContext,
):
    """
    Tile-level all-reduce using ring algorithm.

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

    # Initialize accumulator
    acc_dtype = tl.float32 if local_tile.dtype == tl.float16 else local_tile.dtype
    acc = tl.zeros((tile.block_m, tile.block_n), dtype=acc_dtype)
    acc += local_tile.to(acc_dtype)

    # Ring reduce-scatter phase
    for step in range(ctx.world_size - 1):
        send_rank = (ctx.rank - step) % ctx.world_size
        recv_rank = (ctx.rank - step - 1) % ctx.world_size

        # Compute chunk for this step
        chunk_id = (ctx.rank - step - 1) % ctx.world_size

        # Receive and accumulate from previous rank in ring
        if recv_rank != ctx.rank:
            remote_tile = iris.load(src_tile_ptr, ctx.rank, recv_rank, ctx.heap_bases, mask=mask, other=0.0)
            acc += remote_tile.to(acc_dtype)

    # Ring all-gather phase
    result = acc.to(local_tile.dtype)
    tl.store(dst_tile_ptr, result, mask=mask)

    for step in range(ctx.world_size - 1):
        send_rank = (ctx.rank + step) % ctx.world_size
        recv_rank = (ctx.rank + step + 1) % ctx.world_size

        if recv_rank != ctx.rank:
            remote_result = iris.load(dst_tile_ptr, ctx.rank, recv_rank, ctx.heap_bases, mask=mask)
            tl.store(dst_tile_ptr, remote_result, mask=mask)


@triton.jit()
def all_reduce_two_shot(
    tile: Tile,
    src_view: TensorView,
    dst_view: TensorView,
    ctx: DeviceContext,
):
    """
    Tile-level all-reduce using two-shot algorithm (all ranks gather, compute, then sync).

    Similar to one_shot but may use different synchronization strategy.

    Args:
        tile: Tile object with position and dimensions.
        src_view: TensorView for input tensor.
        dst_view: TensorView for output tensor.
        ctx: DeviceContext with rank, world_size, and heap_bases.
    """
    # Get tile pointer and mask
    src_tile_ptr, mask = src_view.tile_ptr(tile)
    dst_tile_ptr, _ = dst_view.tile_ptr(tile)

    # Load local tile to get dtype
    local_tile = tl.load(src_tile_ptr, mask=mask, other=0.0)

    # Initialize accumulator
    acc_dtype = tl.float32 if local_tile.dtype == tl.float16 else local_tile.dtype
    acc = tl.zeros((tile.block_m, tile.block_n), dtype=acc_dtype)

    # Gather all partials from all ranks and accumulate
    for remote_rank in range(ctx.world_size):
        partial = iris.load(src_tile_ptr, ctx.rank, remote_rank, ctx.heap_bases, mask=mask)
        acc += partial.to(acc_dtype)

    # Store result
    result = acc.to(local_tile.dtype)
    tl.store(dst_tile_ptr, result, mask=mask)


# Convenience alias for default all_reduce
all_reduce = all_reduce_atomic
