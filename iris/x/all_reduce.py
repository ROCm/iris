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
from .common import compute_tile_indices, compute_tile_offsets


@triton.jit()
def all_reduce_atomic(
    input_ptr,
    output_ptr,
    pid_m,
    pid_n,
    M,
    N,
    stride_in_m,
    stride_in_n,
    stride_out_m,
    stride_out_n,
    heap_bases: tl.tensor,
    cur_rank: tl.constexpr,
    world_size: tl.constexpr,
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
):
    """
    Atomic-based all-reduce for a single tile.

    Each rank atomically adds its local partial result to the global output buffer.
    All ranks write to all locations using atomic operations.

    Args:
        input_ptr: Pointer to input tensor (local rank's partial data)
        output_ptr: Pointer to output tensor (will contain sum of all ranks)
        pid_m: Tile coordinate in M dimension
        pid_n: Tile coordinate in N dimension
        M: Number of rows in full tensor
        N: Number of columns in full tensor
        stride_in_m, stride_in_n: Strides for input tensor
        stride_out_m, stride_out_n: Strides for output tensor
        heap_bases: Heap base pointers for all ranks
        cur_rank: Current rank
        world_size: Total number of ranks
        BLOCK_SIZE_M: Block size for M dimension
        BLOCK_SIZE_N: Block size for N dimension
    """
    tl.assume(pid_m >= 0)
    tl.assume(pid_n >= 0)

    # Compute tile indices and mask
    rm, rn, mask = compute_tile_indices(pid_m, pid_n, M, N, BLOCK_SIZE_M, BLOCK_SIZE_N)
    input_offset, output_offset = compute_tile_offsets(
        rm, rn, stride_in_m, stride_in_n, stride_out_m, stride_out_n
    )

    input_ptr_local = input_ptr + input_offset
    input_ptr_local = tl.multiple_of(input_ptr_local, (BLOCK_SIZE_M, BLOCK_SIZE_N))

    # Load local partial result
    data = tl.load(input_ptr_local, mask=mask)

    # Atomically add to output buffer on all ranks
    for target_rank in range(world_size):
        if target_rank == cur_rank:
            # For the current rank, use local atomic add
            tl.atomic_add(output_ptr + output_offset, data, mask=mask)
        else:
            # For remote ranks, use iris.atomic_add to translate pointer
            iris.atomic_add(
                output_ptr + output_offset,
                data,
                cur_rank,
                target_rank,
                heap_bases,
                mask=mask,
            )
    # Ensure all atomic operations complete
    tl.debug_barrier()


@triton.jit()
def all_reduce_spinlock(
    input_ptr,
    output_ptr,
    locks_ptr,
    tile_id,
    pid_m,
    pid_n,
    M,
    N,
    stride_in_m,
    stride_in_n,
    stride_out_m,
    stride_out_n,
    heap_bases: tl.tensor,
    cur_rank: tl.constexpr,
    world_size: tl.constexpr,
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
):
    """
    Spinlock-based all-reduce for a single tile.

    Each tile acquires its lock across the entire system before accumulating remote
    partials locally, then writes the reduced result once and releases the lock.
    Atomics are used only for CAS/XCHG (lock/unlock); the accumulation itself is done
    with ordinary loads/stores.

    Args:
        input_ptr: Pointer to input tensor (local rank's partial data)
        output_ptr: Pointer to output tensor (will contain sum of all ranks)
        locks_ptr: Pointer to locks array (one lock per tile)
        tile_id: Unique tile identifier for lock indexing
        pid_m: Tile coordinate in M dimension
        pid_n: Tile coordinate in N dimension
        M: Number of rows in full tensor
        N: Number of columns in full tensor
        stride_in_m, stride_in_n: Strides for input tensor
        stride_out_m, stride_out_n: Strides for output tensor
        heap_bases: Heap base pointers for all ranks
        cur_rank: Current rank
        world_size: Total number of ranks
        BLOCK_SIZE_M: Block size for M dimension
        BLOCK_SIZE_N: Block size for N dimension
    """
    tl.assume(pid_m >= 0)
    tl.assume(pid_n >= 0)

    lock_ptr = locks_ptr + tile_id

    # Acquire lock (spin until we swap 0 -> 1)
    while tl.atomic_cas(lock_ptr, 0, 1, sem="acquire", scope="sys") != 0:
        pass

    # Compute tile indices and mask
    rm, rn, mask = compute_tile_indices(pid_m, pid_n, M, N, BLOCK_SIZE_M, BLOCK_SIZE_N)
    input_offset, output_offset = compute_tile_offsets(
        rm, rn, stride_in_m, stride_in_n, stride_out_m, stride_out_n
    )

    acc_dtype = tl.float32 if output_ptr.type.element_ty != tl.int8 else tl.int32
    acc = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=acc_dtype)

    # Accumulate from all ranks
    for remote_rank in range(world_size):
        partial = iris.load(
            input_ptr + input_offset,
            cur_rank,
            remote_rank,
            heap_bases,
            mask=mask,
        )
        acc += partial.to(acc_dtype)

    # Store result and release lock
    tl.store(output_ptr + output_offset, acc.to(output_ptr.type.element_ty), mask=mask)
    tl.atomic_xchg(lock_ptr, 0, sem="release", scope="sys")


@triton.jit()
def all_reduce_one_shot(
    input_ptr,
    output_ptr,
    pid_m,
    pid_n,
    M,
    N,
    stride_in_m,
    stride_in_n,
    stride_out_m,
    stride_out_n,
    heap_bases: tl.tensor,
    cur_rank: tl.constexpr,
    world_size: tl.constexpr,
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
):
    """
    One-shot all-reduce for a single tile.

    Gathers all partials directly using iris.load and writes the final result once.
    Suitable for small/latency-bound buffers.

    Args:
        input_ptr: Pointer to input tensor (local rank's partial data)
        output_ptr: Pointer to output tensor (will contain sum of all ranks)
        pid_m: Tile coordinate in M dimension
        pid_n: Tile coordinate in N dimension
        M: Number of rows in full tensor
        N: Number of columns in full tensor
        stride_in_m, stride_in_n: Strides for input tensor
        stride_out_m, stride_out_n: Strides for output tensor
        heap_bases: Heap base pointers for all ranks
        cur_rank: Current rank
        world_size: Total number of ranks
        BLOCK_SIZE_M: Block size for M dimension
        BLOCK_SIZE_N: Block size for N dimension
    """
    tl.assume(pid_m >= 0)
    tl.assume(pid_n >= 0)

    # Compute tile indices and mask
    rm, rn, mask = compute_tile_indices(pid_m, pid_n, M, N, BLOCK_SIZE_M, BLOCK_SIZE_N)
    input_offset, output_offset = compute_tile_offsets(
        rm, rn, stride_in_m, stride_in_n, stride_out_m, stride_out_n
    )

    acc_dtype = tl.float32 if output_ptr.type.element_ty != tl.int8 else tl.int32
    acc = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=acc_dtype)

    # Gather all partials and accumulate
    for remote_rank in range(world_size):
        partial = iris.load(
            input_ptr + input_offset,
            cur_rank,
            remote_rank,
            heap_bases,
            mask=mask,
        )
        acc += partial.to(acc_dtype)

    # Store result
    tl.store(
        output_ptr + output_offset,
        acc.to(output_ptr.type.element_ty),
        mask=mask,
    )


@triton.jit()
def all_reduce_ring(
    input_ptr,
    output_ptr,
    ring_buffer,
    flags,
    tile_id,
    pid_m,
    pid_n,
    M,
    N,
    stride_in_m,
    stride_in_n,
    stride_out_m,
    stride_out_n,
    heap_bases: tl.tensor,
    cur_rank: tl.constexpr,
    world_size: tl.constexpr,
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
    FLAGS_PER_TILE: tl.constexpr,
):
    """
    Ring-based all-reduce for a single tile.

    Streams the tile around the ring using a single-buffer, producer/consumer handshake.
    Each rank keeps a running accumulator, forwards the tile to its successor, and
    consumes the predecessor's contribution. After (world_size - 1) hops, every rank
    has seen all partial tiles.

    Args:
        input_ptr: Pointer to input tensor (local rank's partial data)
        output_ptr: Pointer to output tensor (will contain sum of all ranks)
        ring_buffer: Temporary buffer for ring communication
        flags: Synchronization flags for ring communication
        tile_id: Unique tile identifier for flag indexing
        pid_m: Tile coordinate in M dimension
        pid_n: Tile coordinate in N dimension
        M: Number of rows in full tensor
        N: Number of columns in full tensor
        stride_in_m, stride_in_n: Strides for input tensor
        stride_out_m, stride_out_n: Strides for output tensor
        heap_bases: Heap base pointers for all ranks
        cur_rank: Current rank
        world_size: Total number of ranks
        BLOCK_SIZE_M: Block size for M dimension
        BLOCK_SIZE_N: Block size for N dimension
        FLAGS_PER_TILE: Number of flags per tile
    """
    tl.assume(pid_m >= 0)
    tl.assume(pid_n >= 0)
    tl.static_assert(FLAGS_PER_TILE >= 1, "FLAGS_PER_TILE must be at least 1")

    # Ring topology
    next_rank = (cur_rank + 1) % world_size

    # Compute tile indices and mask
    rm, rn, mask = compute_tile_indices(pid_m, pid_n, M, N, BLOCK_SIZE_M, BLOCK_SIZE_N)
    input_offset, output_offset = compute_tile_offsets(
        rm, rn, stride_in_m, stride_in_n, stride_out_m, stride_out_n
    )

    acc_dtype = tl.float32 if output_ptr.type.element_ty != tl.int8 else tl.int32
    local_tile = tl.load(input_ptr + input_offset, mask=mask, other=0)
    acc = local_tile.to(acc_dtype)
    send_data = local_tile

    flag_offset = tile_id * FLAGS_PER_TILE
    remote_flag_ptr = flags + flag_offset
    local_flag_ptr = flags + flag_offset

    # Ring communication: (world_size - 1) hops
    for _step in range(0, world_size - 1):
        # Wait for remote flag to be ready (0)
        while (
            iris.atomic_cas(
                remote_flag_ptr,
                0,
                0,
                cur_rank,
                next_rank,
                heap_bases,
                sem="acquire",
                scope="sys",
            )
            != 0
        ):
            pass

        # Send data to next rank
        iris.store(
            ring_buffer + input_offset,
            send_data,
            cur_rank,
            next_rank,
            heap_bases,
            mask=mask,
        )
        tl.debug_barrier()
        # Signal that data is ready
        iris.atomic_xchg(
            remote_flag_ptr,
            1,
            cur_rank,
            next_rank,
            heap_bases,
            sem="release",
            scope="sys",
        )

        # Wait for local flag to indicate data is ready (1)
        while tl.atomic_cas(local_flag_ptr, 0, 0, sem="acquire", scope="sys") != 1:
            pass

        # Receive data from previous rank
        recv_tile = tl.load(ring_buffer + input_offset, mask=mask, other=0)
        acc += recv_tile.to(acc_dtype)
        send_data = recv_tile
        tl.debug_barrier()
        # Reset local flag
        tl.atomic_xchg(local_flag_ptr, 0, sem="release", scope="sys")

    # Store final result
    tl.store(
        output_ptr + output_offset,
        acc.to(output_ptr.type.element_ty),
        mask=mask,
    )


@triton.jit()
def all_reduce_two_shot(
    input_ptr,
    output_ptr,
    pid_m,
    pid_n,
    M,
    N,
    stride_in_m,
    stride_in_n,
    stride_out_m,
    stride_out_n,
    heap_bases: tl.tensor,
    cur_rank: tl.constexpr,
    world_size: tl.constexpr,
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
):
    """
    Two-shot all-reduce for a single tile.

    First phase: reduce assigned tiles from all ranks.
    Second phase: broadcast the result to all peers.

    Args:
        input_ptr: Pointer to input tensor (local rank's partial data)
        output_ptr: Pointer to output tensor (will contain sum of all ranks)
        pid_m: Tile coordinate in M dimension
        pid_n: Tile coordinate in N dimension
        M: Number of rows in full tensor
        N: Number of columns in full tensor
        stride_in_m, stride_in_n: Strides for input tensor
        stride_out_m, stride_out_n: Strides for output tensor
        heap_bases: Heap base pointers for all ranks
        cur_rank: Current rank
        world_size: Total number of ranks
        BLOCK_SIZE_M: Block size for M dimension
        BLOCK_SIZE_N: Block size for N dimension
    """
    tl.assume(pid_m >= 0)
    tl.assume(pid_n >= 0)

    # Compute tile indices and mask
    rm, rn, mask = compute_tile_indices(pid_m, pid_n, M, N, BLOCK_SIZE_M, BLOCK_SIZE_N)
    input_offset, output_offset = compute_tile_offsets(
        rm, rn, stride_in_m, stride_in_n, stride_out_m, stride_out_n
    )

    acc_dtype = tl.float32 if output_ptr.type.element_ty != tl.int8 else tl.int32
    acc = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=acc_dtype)

    # Phase 1: Reduce from all ranks
    for remote_rank in range(world_size):
        partial = iris.load(
            input_ptr + input_offset,
            cur_rank,
            remote_rank,
            heap_bases,
            mask=mask,
        )
        acc += partial.to(acc_dtype)

    reduced = acc.to(output_ptr.type.element_ty)

    # Phase 2: Broadcast to all ranks
    for remote_rank in range(world_size):
        if remote_rank == cur_rank:
            tl.store(output_ptr + output_offset, reduced, mask=mask, cache_modifier=".wt")
        else:
            iris.store(
                output_ptr + output_offset,
                reduced,
                cur_rank,
                remote_rank,
                heap_bases,
                mask=mask,
            )

