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
    src_view: TensorView,
    dst_view: TensorView,
    ctx: DeviceContext,
):
    """
    Tile-level all-reduce using atomic operations.

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

    # Initialize output with local data
    tl.store(dst_tile_ptr, local_tile, mask=mask)

    # Accumulate from all remote ranks using atomics
    for r in range(ctx.world_size):
        if r != ctx.rank:
            remote_tile = iris.load(src_tile_ptr, ctx.rank, r, ctx.heap_bases, mask=mask)
            iris.atomic_add(dst_tile_ptr, remote_tile, ctx.rank, ctx.rank, ctx.heap_bases, mask=mask)


@triton.jit()
def all_reduce_spinlock(
    tile: Tile,
    src_view: TensorView,
    dst_view: TensorView,
    locks_ptr,
    tile_id,
    ctx: DeviceContext,
):
    """
    Tile-level all-reduce using spinlock synchronization.

    Uses atomic locks to ensure only one rank computes the reduction at a time.

    Args:
        tile: Tile object with position and dimensions.
        src_view: TensorView for input tensor.
        dst_view: TensorView for output tensor.
        locks_ptr: Pointer to locks array (one lock per tile).
        tile_id: Unique identifier for this tile for lock indexing.
        ctx: DeviceContext with rank, world_size, and heap_bases.
    """
    lock_ptr = locks_ptr + tile_id

    # Acquire lock (spin until we swap 0 -> 1)
    while tl.atomic_cas(lock_ptr, 0, 1, sem="acquire", scope="sys") != 0:
        pass

    # Get tile pointer and mask
    src_tile_ptr, mask = src_view.tile_ptr(tile)
    dst_tile_ptr, _ = dst_view.tile_ptr(tile)

    # Load local tile to get dtype
    local_tile = tl.load(src_tile_ptr, mask=mask, other=0.0)

    # Initialize accumulator
    acc_dtype = tl.float32 if local_tile.dtype == tl.float16 else local_tile.dtype
    acc = tl.zeros((tile.block_m, tile.block_n), dtype=acc_dtype)

    # Accumulate from all ranks
    for remote_rank in range(ctx.world_size):
        partial = iris.load(src_tile_ptr, ctx.rank, remote_rank, ctx.heap_bases, mask=mask)
        acc += partial.to(acc_dtype)

    # Store result and release lock
    result = acc.to(local_tile.dtype)
    tl.store(dst_tile_ptr, result, mask=mask)
    tl.atomic_xchg(lock_ptr, 0, sem="release", scope="sys")


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
            remote_tile = iris.load(src_view.ptr + ring_offset, ctx.heap_bases, recv_rank, mask=mask, other=0.0)
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
