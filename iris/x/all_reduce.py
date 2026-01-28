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
            remote_tile = iris.load(src_tile_ptr, ctx.heap_bases, r, mask=mask, other=0.0)
            iris.atomic_add(dst_tile_ptr, remote_tile, ctx.heap_bases, ctx.rank, mask=mask)


@triton.jit()
def all_reduce_spinlock(
    tile: Tile,
    src_view: TensorView,
    dst_view: TensorView,
    ctx: DeviceContext,
):
    """
    Tile-level all-reduce using spinlock synchronization.

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

    # Accumulate from remote ranks
    for r in range(ctx.world_size):
        if r != ctx.rank:
            remote_tile = iris.load(src_tile_ptr, ctx.heap_bases, r, mask=mask, other=0.0)
            acc += remote_tile.to(acc_dtype)

    # Store result
    result = acc.to(local_tile.dtype)
    tl.store(dst_tile_ptr, result, mask=mask)

    # Spinlock: wait for all ranks to write their results
    for r in range(ctx.world_size):
        if r != ctx.rank:
            expected = result
            while True:
                remote_result = iris.load(dst_tile_ptr, ctx.heap_bases, r, mask=mask, other=0.0)
                if tl.sum(tl.abs(remote_result - expected)) < 1e-6:
                    break


@triton.jit()
def all_reduce_one_shot(
    tile: Tile,
    src_view: TensorView,
    dst_view: TensorView,
    ctx: DeviceContext,
):
    """
    Tile-level all-reduce using one-shot algorithm (rank 0 aggregates and broadcasts).

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

    # Rank 0 aggregates
    if ctx.rank == 0:
        acc_dtype = tl.float32 if local_tile.dtype == tl.float16 else local_tile.dtype
        acc = tl.zeros((tile.block_m, tile.block_n), dtype=acc_dtype)
        acc += local_tile.to(acc_dtype)

        for r in range(1, ctx.world_size):
            remote_tile = iris.load(src_tile_ptr, ctx.heap_bases, r, mask=mask, other=0.0)
            acc += remote_tile.to(acc_dtype)

        result = acc.to(local_tile.dtype)
        tl.store(dst_tile_ptr, result, mask=mask)
    else:
        # Non-zero ranks wait and read from rank 0
        result = iris.load(dst_tile_ptr, ctx.heap_bases, 0, mask=mask, other=0.0)
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

        # Calculate offset for ring algorithm
        indices_m, indices_n = tile.layout(src_view.M, src_view.N)
        ring_offset = indices_m[:, None] * src_view.stride_m + indices_n[None, :] * src_view.stride_n

        # Receive and accumulate from previous rank in ring
        if recv_rank != ctx.rank:
            remote_tile = iris.load(
                src_view.ptr + ring_offset, ctx.heap_bases, recv_rank, mask=mask, other=0.0
            )
            acc += remote_tile.to(acc_dtype)

    # Ring all-gather phase
    result = acc.to(local_tile.dtype)
    tl.store(dst_tile_ptr, result, mask=mask)

    for step in range(ctx.world_size - 1):
        send_rank = (ctx.rank + step) % ctx.world_size
        recv_rank = (ctx.rank + step + 1) % ctx.world_size

        if recv_rank != ctx.rank:
            remote_result = iris.load(dst_tile_ptr, ctx.heap_bases, recv_rank, mask=mask, other=0.0)
            tl.store(dst_tile_ptr, remote_result, mask=mask)


@triton.jit()
def all_reduce_two_shot(
    tile: Tile,
    src_view: TensorView,
    dst_view: TensorView,
    ctx: DeviceContext,
):
    """
    Tile-level all-reduce using two-shot algorithm (reduce to rank 0, then broadcast).

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

    # Phase 1: Reduce to rank 0
    if ctx.rank == 0:
        acc_dtype = tl.float32 if local_tile.dtype == tl.float16 else local_tile.dtype
        acc = tl.zeros((tile.block_m, tile.block_n), dtype=acc_dtype)
        acc += local_tile.to(acc_dtype)

        for r in range(1, ctx.world_size):
            remote_tile = iris.load(src_tile_ptr, ctx.heap_bases, r, mask=mask, other=0.0)
            acc += remote_tile.to(acc_dtype)

        result = acc.to(local_tile.dtype)
        tl.store(dst_tile_ptr, result, mask=mask)

    # Phase 2: Broadcast from rank 0
    if ctx.rank != 0:
        result = iris.load(dst_tile_ptr, ctx.heap_bases, 0, mask=mask, other=0.0)
        tl.store(dst_tile_ptr, result, mask=mask)


# Convenience alias for default all_reduce
all_reduce = all_reduce_atomic
