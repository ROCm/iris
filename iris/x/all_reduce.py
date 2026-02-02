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
            ctx.rank,  # from_rank (current rank)
            dest_rank,  # to_rank (destination rank)
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
        while (
            iris.atomic_cas(locks + tile_id, 0, 1, ctx.rank, dest_rank, ctx.heap_bases, sem="acquire", scope="sys") != 0
        ):
            pass

        # Load current value from dest_rank's tile using iris.load
        current_value = iris.load(dst_tile_ptr, ctx.rank, dest_rank, ctx.heap_bases, mask=mask)

        # Add our local contribution
        acc_dtype = tl.float32 if tile.data.dtype == tl.float16 else tile.data.dtype
        acc = current_value.to(acc_dtype) + tile.data.to(acc_dtype)

        # Store accumulated result back to dest_rank (overwriting) using iris.store
        result = acc.to(tile.data.dtype)
        iris.store(
            dst_tile_ptr, result, ctx.rank, dest_rank, ctx.heap_bases, mask=mask
        )  # Should be cache-modifier ".wt"
        tl.debug_barrier()

        # Release lock for this tile at dest_rank using iris.atomic_xchg
        iris.atomic_xchg(locks + tile_id, 0, ctx.rank, dest_rank, ctx.heap_bases, sem="release", scope="sys")


@triton.jit()
def all_reduce_one_shot(
    tile: Tile,
    src_view: TensorView,
    dst_view: TensorView,
    locks,
    ctx: DeviceContext,
):
    """
    Tile-level all-reduce using one-shot algorithm (all ranks gather and reduce locally).

    Each rank reads from all ranks (including itself) and computes the reduction locally.
    All ranks do all tiles (duplicated work), but no remote stores needed.

    Uses locks as ready flags (producer-consumer): each rank waits for remote tiles
    to be ready (lock == 1) before loading.

    Args:
        tile: Tile object with position, dimensions, and local data (tile.data).
        src_view: TensorView for source tensor (to load remote data).
        dst_view: TensorView for output tensor where reduced result will be written locally.
        locks: Pointer to lock array (one per tile) used as ready flags.
        ctx: DeviceContext with rank, world_size, and heap_bases.

    Example:
        # After computing and storing a local tile result and signaling ready
        tile = iris.x.Tile(pid_m, pid_n, BLOCK_SIZE_M, BLOCK_SIZE_N, local_result)
        src_view = iris.x.TensorView(input_ptr, M, N, stride_m, stride_n)
        dst_view = iris.x.TensorView(output_ptr, M, N, stride_m, stride_n)
        iris.x.all_reduce_one_shot(tile, src_view, dst_view, locks, ctx)
    """
    # Get tile pointers and mask
    src_tile_ptr, mask = src_view.tile_ptr(tile)
    dst_tile_ptr, _ = dst_view.tile_ptr(tile)

    # Compute tile ID for lock indexing
    num_tiles_n = tl.cdiv(dst_view.N, tile.block_n)
    tile_id = tile.pid_m * num_tiles_n + tile.pid_n

    # Initialize accumulator with local tile data (already in registers)
    acc_dtype = tl.float32 if tile.data.dtype == tl.float16 else tile.data.dtype
    acc = tile.data.to(acc_dtype)

    # Gather partials from all remote ranks and accumulate
    # Note: Skip current rank - tile.data already contains local contribution
    #       (avoids race condition where we might load our own final result instead of partial)
    for remote_rank in range(ctx.world_size):
        if remote_rank != ctx.rank:
            # Wait for remote tile to be ready (spin on lock == 1)
            # Use atomic_add with 0 to check readiness (consumer uses acquire semantics on read)
            lock_ptr = locks + tile_id
            # Spin wait until ready
            while iris.atomic_add(lock_ptr, 0, ctx.rank, remote_rank, ctx.heap_bases, sem="acquire", scope="sys") != 1:
                pass  # Spin wait until ready

            # Load remote tile data from temp buffer
            partial = iris.load(src_tile_ptr, ctx.rank, remote_rank, ctx.heap_bases, mask=mask)
            acc += partial.to(acc_dtype)

    # Store result to local rank only (no remote stores)
    result = acc.to(tile.data.dtype)
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
            remote_tile = iris.load(src_tile_ptr, ctx.rank, recv_rank, ctx.heap_bases, mask=mask)
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
    locks,
    start_tile: tl.constexpr,
    stride: tl.constexpr,
    ctx: DeviceContext,
):
    """
    Tile-level all-reduce using two-shot algorithm with work distribution.

    Each rank reduces only its assigned tiles (no duplicated work), then scatters
    the result to all other ranks.

    Uses locks as ready flags: before loading, wait for remote tiles to be ready (lock == 1).

    Phase 1: If this tile is rank's responsibility, load from all ranks and reduce locally
    Phase 2: Scatter reduced tile to all ranks using iris.store

    Args:
        tile: Tile object with position, dimensions, and local data (tile.data).
        src_view: TensorView for source tensor (to load remote data).
        dst_view: TensorView for output tensor where reduced result will be written.
        locks: Pointer to lock array (one per tile) used as ready flags.
        start_tile: Starting tile ID for this rank's responsibility.
        stride: Stride between tiles this rank is responsible for.
        ctx: DeviceContext with rank, world_size, and heap_bases.

    Example (interleaved distribution):
        # Rank 0 handles tiles 0, 2, 4, ... (start_tile=0, stride=2)
        # Rank 1 handles tiles 1, 3, 5, ... (start_tile=1, stride=2)
        tile = iris.x.Tile(pid_m, pid_n, BLOCK_SIZE_M, BLOCK_SIZE_N, local_result)
        src_view = iris.x.TensorView(input_ptr, M, N, stride_m, stride_n)
        dst_view = iris.x.TensorView(output_ptr, M, N, stride_m, stride_n)
        iris.x.all_reduce_two_shot(tile, src_view, dst_view, locks, rank, world_size, ctx)
    """
    # Compute tile ID
    num_tiles_n = tl.cdiv(dst_view.N, tile.block_n)
    tile_id = tile.pid_m * num_tiles_n + tile.pid_n

    # Check if this tile is this rank's responsibility
    # Tile is responsible if: (tile_id - start_tile) % stride == 0 and tile_id >= start_tile
    is_responsible = (tile_id >= start_tile) and ((tile_id - start_tile) % stride == 0)

    if is_responsible:
        # Phase 1: Reduce - load from all ranks and accumulate locally
        src_tile_ptr, mask = src_view.tile_ptr(tile)
        dst_tile_ptr, _ = dst_view.tile_ptr(tile)

        # Initialize accumulator with local tile data (already in registers)
        acc_dtype = tl.float32 if tile.data.dtype == tl.float16 else tile.data.dtype
        acc = tile.data.to(acc_dtype)

        # Gather partials from all remote ranks and accumulate
        # Note: Skip current rank - tile.data already contains local contribution
        #       (avoids race condition where we might load our own final result instead of partial)
        for remote_rank in range(ctx.world_size):
            if remote_rank != ctx.rank:
                # Wait for remote tile to be ready (spin on lock == 1)
                # Use atomic_add with 0 to check readiness (consumer uses acquire semantics on read)
                lock_ptr = locks + tile_id
                # Spin wait until ready
                while (
                    iris.atomic_add(lock_ptr, 0, ctx.rank, remote_rank, ctx.heap_bases, sem="acquire", scope="sys") != 1
                ):
                    pass  # Spin wait until ready

                # Load remote tile data from temp buffer
                partial = iris.load(src_tile_ptr, ctx.rank, remote_rank, ctx.heap_bases, mask=mask)
                acc += partial.to(acc_dtype)

        # Store reduced result locally
        result = acc.to(tile.data.dtype)
        tl.store(dst_tile_ptr, result, mask=mask)

        # Phase 2: Scatter - broadcast reduced tile to all ranks
        for dest_rank in range(ctx.world_size):
            if dest_rank != ctx.rank:
                iris.store(
                    dst_tile_ptr,
                    result,
                    ctx.rank,  # from_rank (current rank with reduced result)
                    dest_rank,  # to_rank (destination rank)
                    ctx.heap_bases,
                    mask=mask,
                )


# Convenience alias for default all_reduce
all_reduce = all_reduce_atomic
