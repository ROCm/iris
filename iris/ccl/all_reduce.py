# SPDX-License-Identifier: MIT
# Copyright (c) 2025 Advanced Micro Devices, Inc. All rights reserved.

"""
All-reduce collective communication primitive for Iris.
Supports multiple variants: atomic, spinlock, ring, two-shot, and one-shot.
"""

from dataclasses import dataclass
from typing import Optional, Tuple

import triton
import triton.language as tl
import torch
import iris
from .config import Config

# Variant types
VARIANT_ATOMIC = "atomic"
VARIANT_RING = "ring"
VARIANT_TWO_SHOT = "two_shot"
VARIANT_ONE_SHOT = "one_shot"
VARIANT_SPINLOCK = "spinlock"


@dataclass
class AllReduceWorkspace:
    """
    Holds reusable workspace allocations for all-reduce variants.

    Attributes:
        variant: Selected all-reduce variant.
        shape: Tuple of (M, N) for tensor shape.
        dtype: Torch dtype of buffers.
        ring_buffer: Temporary buffer used by ring-based algorithm.
        flags: Synchronization flags for ring-based algorithm.
        num_rings: Number of concurrent rings prepared for ring-based variant.
        prepared: Indicates whether preamble has been executed since last use.
    """

    variant: str = ""
    shape: Tuple[int, int] = ()
    dtype: Optional[torch.dtype] = None
    ring_buffer: Optional[torch.Tensor] = None
    flags: Optional[torch.Tensor] = None
    locks: Optional[torch.Tensor] = None
    num_rings: int = 1
    flags_per_tile: int = 0
    prepared: bool = False


def all_reduce_preamble(
    output_tensor,
    input_tensor,
    shmem,
    config: Optional[Config] = None,
    workspace: Optional[AllReduceWorkspace] = None,
):
    """
    Allocate and reset temporary buffers for the chosen variant.

    Returns:
        AllReduceWorkspace instance ready for the next call to all_reduce.
    """
    if config is None:
        config = Config()

    variant = config.all_reduce_variant.lower()
    if variant not in [VARIANT_ATOMIC, VARIANT_RING, VARIANT_TWO_SHOT, VARIANT_ONE_SHOT, VARIANT_SPINLOCK]:
        raise ValueError(
            f"Invalid all_reduce_variant: {variant}. Must be one of: {VARIANT_ATOMIC}, {VARIANT_RING}, {VARIANT_TWO_SHOT}, {VARIANT_ONE_SHOT}, {VARIANT_SPINLOCK}"
        )

    M, N = input_tensor.shape[:2]
    dtype = input_tensor.dtype

    if workspace is None:
        workspace = AllReduceWorkspace()

    workspace.variant = variant
    workspace.shape = (M, N)
    workspace.dtype = dtype
    workspace.num_rings = getattr(config, "all_reduce_num_rings", 1)
    workspace.prepared = False

    if variant in (VARIANT_ATOMIC, VARIANT_SPINLOCK, VARIANT_ONE_SHOT):
        output_tensor.zero_()
        shmem.barrier()

    elif variant == VARIANT_RING:
        num_pid_m = (M + config.block_size_m - 1) // config.block_size_m
        num_pid_n = (N + config.block_size_n - 1) // config.block_size_n
        total_tiles = num_pid_m * num_pid_n
        workspace.flags_per_tile = 1
        total_flags = total_tiles * workspace.flags_per_tile
        if (
            workspace.ring_buffer is None
            or workspace.ring_buffer.shape != (M, N)
            or workspace.ring_buffer.dtype != dtype
        ):
            workspace.ring_buffer = shmem.zeros((M, N), dtype=dtype)
        else:
            workspace.ring_buffer.zero_()

        if workspace.flags is None or workspace.flags.numel() != total_flags:
            workspace.flags = shmem.zeros((total_flags,), dtype=torch.int32)
        else:
            workspace.flags.zero_()

        output_tensor.zero_()
        shmem.barrier()

    elif variant == VARIANT_TWO_SHOT:
        pass

    if variant == VARIANT_SPINLOCK:
        num_pid_m = (M + config.block_size_m - 1) // config.block_size_m
        num_pid_n = (N + config.block_size_n - 1) // config.block_size_n
        total_tiles = num_pid_m * num_pid_n
        if workspace.locks is None or workspace.locks.numel() != total_tiles:
            workspace.locks = shmem.zeros((total_tiles,), dtype=torch.int32)
        else:
            workspace.locks.zero_()

    workspace.prepared = True
    return workspace


@triton.jit()
def chiplet_transform_chunked(pid, num_workgroups: tl.constexpr, num_xcds: tl.constexpr, chunk_size: tl.constexpr):
    if pid > (num_workgroups // (num_xcds * chunk_size)) * (num_xcds * chunk_size):
        return pid

    local_pid = pid // num_xcds
    chunk_idx = local_pid // chunk_size
    pos_in_chunk = local_pid % chunk_size

    xcd = pid % num_xcds
    new_pid = chunk_idx * num_xcds * chunk_size + xcd * chunk_size + pos_in_chunk
    return new_pid


@triton.jit()
def persistent_all_reduce_atomic(
    input_ptr,
    output_ptr,
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
    GROUP_SIZE_M: tl.constexpr,
    COMM_SMS: tl.constexpr,
    NUM_XCDS: tl.constexpr,
    CHUNK_SIZE: tl.constexpr,
):
    """
    Atomic-based all-reduce kernel.

    Each rank atomically adds its local partial result to the global output buffer.
    All ranks write to all locations using atomic operations.

    Args:
        input_ptr: Pointer to input tensor (local rank's partial data)
        output_ptr: Pointer to output tensor (will contain sum of all ranks)
        M: Number of rows
        N: Number of columns
        heap_bases: Heap base pointers for all ranks
        cur_rank: Current rank
        world_size: Total number of ranks
    """
    pid = tl.program_id(0)

    # Use chiplet transform to distribute program IDs across XCDs
    if NUM_XCDS != 1:
        pid = chiplet_transform_chunked(pid, COMM_SMS, NUM_XCDS, CHUNK_SIZE)

    num_pid_m = tl.cdiv(M, BLOCK_SIZE_M)
    num_pid_n = tl.cdiv(N, BLOCK_SIZE_N)
    total_tiles = num_pid_m * num_pid_n

    for tile_id in range(pid, total_tiles, COMM_SMS):
        num_pid_in_group = GROUP_SIZE_M * num_pid_n
        group_id = tile_id // num_pid_in_group
        first_pid_m = group_id * GROUP_SIZE_M
        group_size_m = min(num_pid_m - first_pid_m, GROUP_SIZE_M)
        pid_m = first_pid_m + ((tile_id % num_pid_in_group) % group_size_m)
        pid_n = (tile_id % num_pid_in_group) // group_size_m

        tl.assume(pid_m >= 0)
        tl.assume(pid_n >= 0)

        # Compute row and column indices
        # Calculate base indices without modulo to avoid double-counting when blocks are larger than dimensions
        rm_base = pid_m * BLOCK_SIZE_M
        rn_base = pid_n * BLOCK_SIZE_N
        rm = rm_base + tl.arange(0, BLOCK_SIZE_M)
        rn = rn_base + tl.arange(0, BLOCK_SIZE_N)
        rm = tl.max_contiguous(tl.multiple_of(rm, BLOCK_SIZE_M), BLOCK_SIZE_M)
        rn = tl.max_contiguous(tl.multiple_of(rn, BLOCK_SIZE_N), BLOCK_SIZE_N)
        # Create mask to prevent out-of-bounds access
        mask = (rm[:, None] < M) & (rn[None, :] < N)

        # Use the original rm/rn for offsets (mask will prevent out-of-bounds access)
        # This avoids double-counting that occurs with modulo when block_size > dimension
        input_offset = rm[:, None] * stride_in_m + rn[None, :] * stride_in_n
        output_offset = rm[:, None] * stride_out_m + rn[None, :] * stride_out_n

        input_ptr_local = input_ptr + input_offset
        input_ptr_local = tl.multiple_of(input_ptr_local, (BLOCK_SIZE_M, BLOCK_SIZE_N))

        # Load local partial result
        data = tl.load(input_ptr_local, mask=mask)

        # Atomically add to output buffer on all ranks
        # Each rank's output tensor is in its own heap, accessible via IPC
        for target_rank in range(world_size):
            if target_rank == cur_rank:
                # For the current rank, use local atomic add
                # output_ptr is already in current rank's address space
                tl.atomic_add(output_ptr + output_offset, data, mask=mask)
            else:
                # For remote ranks, use iris.atomic_add to translate pointer
                # This accesses the remote rank's heap via IPC
                iris.atomic_add(
                    output_ptr + output_offset,
                    data,
                    cur_rank,
                    target_rank,
                    heap_bases,
                    mask=mask,
                )
        # Ensure all atomic operations complete before moving to next tile
        tl.debug_barrier()


@triton.jit()
def persistent_all_reduce_spinlock(
    input_ptr,
    output_ptr,
    locks_ptr,
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
    GROUP_SIZE_M: tl.constexpr,
    COMM_SMS: tl.constexpr,
    NUM_XCDS: tl.constexpr,
    CHUNK_SIZE: tl.constexpr,
):
    """
    Spinlock-based all-reduce kernel that mimics an “atomic add” by using a lock per tile.

    Each tile acquires its lock across the entire system before accumulating remote
    partials locally, then writes the reduced result once and releases the lock.
    Atomics are used only for CAS/XCHG (lock/unlock); the accumulation itself is done
    with ordinary loads/stores.
    """
    pid = tl.program_id(0)

    if NUM_XCDS != 1:
        pid = chiplet_transform_chunked(pid, COMM_SMS, NUM_XCDS, CHUNK_SIZE)

    num_pid_m = tl.cdiv(M, BLOCK_SIZE_M)
    num_pid_n = tl.cdiv(N, BLOCK_SIZE_N)
    total_tiles = num_pid_m * num_pid_n

    acc_dtype = tl.float32 if output_ptr.type.element_ty != tl.int8 else tl.int32

    for tile_id in range(pid, total_tiles, COMM_SMS):
        lock_ptr = locks_ptr + tile_id

        # Acquire lock (spin until we swap 0 -> 1)
        while tl.atomic_cas(lock_ptr, 0, 1, sem="acquire", scope="sys") != 0:
            pass

        num_pid_in_group = GROUP_SIZE_M * num_pid_n
        group_id = tile_id // num_pid_in_group
        first_pid_m = group_id * GROUP_SIZE_M
        group_size_m = min(num_pid_m - first_pid_m, GROUP_SIZE_M)
        pid_m = first_pid_m + ((tile_id % num_pid_in_group) % group_size_m)
        pid_n = (tile_id % num_pid_in_group) // group_size_m

        tl.assume(pid_m >= 0)
        tl.assume(pid_n >= 0)

        rm_base = pid_m * BLOCK_SIZE_M
        rn_base = pid_n * BLOCK_SIZE_N
        rm = rm_base + tl.arange(0, BLOCK_SIZE_M)
        rn = rn_base + tl.arange(0, BLOCK_SIZE_N)
        rm = tl.max_contiguous(tl.multiple_of(rm, BLOCK_SIZE_M), BLOCK_SIZE_M)
        rn = tl.max_contiguous(tl.multiple_of(rn, BLOCK_SIZE_N), BLOCK_SIZE_N)
        mask = (rm[:, None] < M) & (rn[None, :] < N)

        input_offset = rm[:, None] * stride_in_m + rn[None, :] * stride_in_n
        output_offset = rm[:, None] * stride_out_m + rn[None, :] * stride_out_n

        acc = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=acc_dtype)

        for remote_rank in range(world_size):
            partial = iris.load(
                input_ptr + input_offset,
                cur_rank,
                remote_rank,
                heap_bases,
                mask=mask,
            )
            acc += partial.to(acc_dtype)

        tl.store(output_ptr + output_offset, acc.to(output_ptr.type.element_ty), mask=mask)

        tl.atomic_xchg(lock_ptr, 0, sem="release", scope="sys")


@triton.jit()
def persistent_all_reduce_one_shot(
    input_ptr,
    output_ptr,
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
    GROUP_SIZE_M: tl.constexpr,
    COMM_SMS: tl.constexpr,
    NUM_XCDS: tl.constexpr,
    CHUNK_SIZE: tl.constexpr,
):
    """
    One-shot all-reduce for small/latency-bound buffers.

    Each CTA gathers all partials directly using iris.load and writes the final result once.
    """
    pid = tl.program_id(0)

    if NUM_XCDS != 1:
        pid = chiplet_transform_chunked(pid, COMM_SMS, NUM_XCDS, CHUNK_SIZE)

    num_pid_m = tl.cdiv(M, BLOCK_SIZE_M)
    num_pid_n = tl.cdiv(N, BLOCK_SIZE_N)
    total_tiles = num_pid_m * num_pid_n

    acc_dtype = tl.float32 if output_ptr.type.element_ty != tl.int8 else tl.int32

    for tile_id in range(pid, total_tiles, COMM_SMS):
        num_pid_in_group = GROUP_SIZE_M * num_pid_n
        group_id = tile_id // num_pid_in_group
        first_pid_m = group_id * GROUP_SIZE_M
        group_size_m = min(num_pid_m - first_pid_m, GROUP_SIZE_M)
        pid_m = first_pid_m + ((tile_id % num_pid_in_group) % group_size_m)
        pid_n = (tile_id % num_pid_in_group) // group_size_m

        tl.assume(pid_m >= 0)
        tl.assume(pid_n >= 0)

        rm_base = pid_m * BLOCK_SIZE_M
        rn_base = pid_n * BLOCK_SIZE_N
        rm = rm_base + tl.arange(0, BLOCK_SIZE_M)
        rn = rn_base + tl.arange(0, BLOCK_SIZE_N)
        rm = tl.max_contiguous(tl.multiple_of(rm, BLOCK_SIZE_M), BLOCK_SIZE_M)
        rn = tl.max_contiguous(tl.multiple_of(rn, BLOCK_SIZE_N), BLOCK_SIZE_N)
        mask = (rm[:, None] < M) & (rn[None, :] < N)

        input_offset = rm[:, None] * stride_in_m + rn[None, :] * stride_in_n
        output_offset = rm[:, None] * stride_out_m + rn[None, :] * stride_out_n

        acc = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=acc_dtype)

        for remote_rank in range(world_size):
            partial = iris.load(
                input_ptr + input_offset,
                cur_rank,
                remote_rank,
                heap_bases,
                mask=mask,
            )
            acc += partial.to(acc_dtype)

        tl.store(
            output_ptr + output_offset,
            acc.to(output_ptr.type.element_ty),
            mask=mask,
        )


@triton.jit()
def persistent_all_reduce_ring(
    input_ptr,
    output_ptr,
    ring_buffer,
    flags,
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
    GROUP_SIZE_M: tl.constexpr,
    COMM_SMS: tl.constexpr,
    NUM_XCDS: tl.constexpr,
    CHUNK_SIZE: tl.constexpr,
    NUM_RINGS: tl.constexpr,
    SLICE_SIZE_N: tl.constexpr,
    FLAGS_PER_TILE: tl.constexpr,
):
    """
    Ring-based all-reduce kernel that streams whole tiles around the ring using a
    single-buffer, producer/consumer handshake.

    Each rank keeps a running accumulator for its local tile, forwards the tile it
    just received to its successor, and consumes the predecessor's contribution in
    lock-step.  After (world_size - 1) hops every rank has seen all partial tiles,
    so the accumulator holds the fully reduced result which is written back locally.
    """
    pid_raw = tl.program_id(0)

    # Use chiplet transform to distribute program IDs across XCDs
    pid = pid_raw
    if NUM_XCDS != 1:
        pid = chiplet_transform_chunked(pid_raw, COMM_SMS, NUM_XCDS, CHUNK_SIZE)

    tl.static_assert(NUM_RINGS > 0, "NUM_RINGS must be >= 1")
    tl.static_assert(FLAGS_PER_TILE >= 1, "FLAGS_PER_TILE must be at least 1")

    num_pid_m = tl.cdiv(M, BLOCK_SIZE_M)
    num_pid_n = tl.cdiv(N, BLOCK_SIZE_N)
    total_tiles = num_pid_m * num_pid_n

    # Ring topology
    next_rank = (cur_rank + 1) % world_size

    acc_dtype = tl.float32 if output_ptr.type.element_ty != tl.int8 else tl.int32
    elem_ty = input_ptr.type.element_ty

    # Partition CTAs across rings to form NUM_RINGS concurrent rings.
    ctas_per_ring = (COMM_SMS + NUM_RINGS - 1) // NUM_RINGS
    ring_id = pid % NUM_RINGS
    cta_in_ring = pid // NUM_RINGS

    if (cta_in_ring < ctas_per_ring) and (total_tiles > 0) and (total_tiles > ring_id):
        tiles_per_ring = (total_tiles - ring_id + NUM_RINGS - 1) // NUM_RINGS
        for tile_index_in_ring in range(cta_in_ring, tiles_per_ring, ctas_per_ring):
            tile_id = ring_id + tile_index_in_ring * NUM_RINGS
            if tile_id < total_tiles:
                num_pid_in_group = GROUP_SIZE_M * num_pid_n
                group_id = tile_id // num_pid_in_group
                first_pid_m = group_id * GROUP_SIZE_M
                group_size_m = min(num_pid_m - first_pid_m, GROUP_SIZE_M)
                pid_m = first_pid_m + ((tile_id % num_pid_in_group) % group_size_m)
                pid_n = (tile_id % num_pid_in_group) // group_size_m

                tl.assume(pid_m >= 0)
                tl.assume(pid_n >= 0)

                rm_base = pid_m * BLOCK_SIZE_M
                rn_base = pid_n * BLOCK_SIZE_N
                rm = rm_base + tl.arange(0, BLOCK_SIZE_M)
                rm = tl.max_contiguous(tl.multiple_of(rm, BLOCK_SIZE_M), BLOCK_SIZE_M)

                rn = rn_base + tl.arange(0, BLOCK_SIZE_N)
                mask = (rm[:, None] < M) & (rn[None, :] < N)
                tile_offset = rm[:, None] * stride_in_m + rn[None, :] * stride_in_n

                local_tile = tl.load(input_ptr + tile_offset, mask=mask, other=0)
                acc = local_tile.to(acc_dtype)
                send_data = local_tile

                flag_offset = tile_id * FLAGS_PER_TILE
                remote_flag_ptr = flags + flag_offset
                local_flag_ptr = flags + flag_offset

                for _step in range(0, world_size - 1):
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

                    iris.store(
                        ring_buffer + tile_offset,
                        send_data,
                        cur_rank,
                        next_rank,
                        heap_bases,
                        mask=mask,
                    )
                    tl.debug_barrier()
                    iris.atomic_xchg(
                        remote_flag_ptr,
                        1,
                        cur_rank,
                        next_rank,
                        heap_bases,
                        sem="release",
                        scope="sys",
                    )

                    while tl.atomic_cas(local_flag_ptr, 0, 0, sem="acquire", scope="sys") != 1:
                        pass

                    recv_tile = tl.load(ring_buffer + tile_offset, mask=mask, other=0)
                    acc += recv_tile.to(acc_dtype)
                    send_data = recv_tile
                    tl.debug_barrier()
                    tl.atomic_xchg(local_flag_ptr, 0, sem="release", scope="sys")

                tl.store(
                    output_ptr + tile_offset,
                    acc.to(output_ptr.type.element_ty),
                    mask=mask,
                )


@triton.jit()
def persistent_all_reduce_two_shot(
    input_ptr,
    output_ptr,
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
    GROUP_SIZE_M: tl.constexpr,
    COMM_SMS: tl.constexpr,
    NUM_XCDS: tl.constexpr,
    CHUNK_SIZE: tl.constexpr,
    DISTRIBUTION: tl.constexpr,
):
    """Reduce assigned tiles for a rank and broadcast the result to all peers."""
    pid = tl.program_id(0)

    if NUM_XCDS != 1:
        pid = chiplet_transform_chunked(pid, COMM_SMS, NUM_XCDS, CHUNK_SIZE)

    num_pid_m = tl.cdiv(M, BLOCK_SIZE_M)
    num_pid_n = tl.cdiv(N, BLOCK_SIZE_N)
    total_tiles = num_pid_m * num_pid_n

    acc_dtype = tl.float32 if output_ptr.type.element_ty != tl.int8 else tl.int32

    tiles_per_rank = tl.cdiv(total_tiles, world_size)
    if DISTRIBUTION == 0:
        start_tile = cur_rank
        stride = world_size
        remaining = total_tiles - start_tile
        remaining = tl.maximum(remaining, 0)
        max_tile_offset = tl.cdiv(remaining, stride)
    else:
        start_tile = cur_rank * tiles_per_rank
        stride = 1
        remaining = total_tiles - start_tile
        remaining = tl.maximum(remaining, 0)
        max_tile_offset = tl.minimum(tiles_per_rank, remaining)

    for tile_offset in range(pid, max_tile_offset, COMM_SMS):
        tile_id = start_tile + tile_offset * stride

        num_pid_in_group = GROUP_SIZE_M * num_pid_n
        group_id = tile_id // num_pid_in_group
        first_pid_m = group_id * GROUP_SIZE_M
        group_size_m = min(num_pid_m - first_pid_m, GROUP_SIZE_M)
        pid_m = first_pid_m + ((tile_id % num_pid_in_group) % group_size_m)
        pid_n = (tile_id % num_pid_in_group) // group_size_m

        tl.assume(pid_m >= 0)
        tl.assume(pid_n >= 0)

        rm_base = pid_m * BLOCK_SIZE_M
        rn_base = pid_n * BLOCK_SIZE_N
        rm = rm_base + tl.arange(0, BLOCK_SIZE_M)
        rn = rn_base + tl.arange(0, BLOCK_SIZE_N)
        rm = tl.max_contiguous(tl.multiple_of(rm, BLOCK_SIZE_M), BLOCK_SIZE_M)
        rn = tl.max_contiguous(tl.multiple_of(rn, BLOCK_SIZE_N), BLOCK_SIZE_N)
        mask = (rm[:, None] < M) & (rn[None, :] < N)

        input_offset = rm[:, None] * stride_in_m + rn[None, :] * stride_in_n
        output_offset = rm[:, None] * stride_out_m + rn[None, :] * stride_out_n

        acc = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=acc_dtype)
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


def all_reduce(
    output_tensor, input_tensor, shmem, config=None, async_op=False, workspace: Optional[AllReduceWorkspace] = None
):
    """
    Internal all-reduce collective operation implementation.

    This function is called internally by shmem.ccl.all_reduce().
    Users should use the Iris instance method instead:
        >>> shmem.ccl.all_reduce(output_tensor, input_tensor)

    Each rank has a local input tensor, and all ranks compute the sum of all
    input tensors. The result is written to output_tensor on all ranks.

    Args:
        output_tensor: Output tensor of shape (M, N) - will contain sum of all inputs
        input_tensor: Input tensor of shape (M, N) - local rank's partial data
        shmem: Iris shmem context
        config: Config instance with kernel parameters (default: None).
                If None, uses default Config values.
                Set config.all_reduce_variant to choose variant: "atomic", "spinlock", "ring", "two_shot", or "one_shot"
        async_op: If False, performs a barrier at the end. If True, returns immediately.
                  Default: False.
        workspace: Optional AllReduceWorkspace instance prepared via all_reduce_preamble.
    """
    if config is None:
        config = Config()

    rank = shmem.get_rank()
    world_size = shmem.get_num_ranks()
    M, N = input_tensor.shape[:2]

    stride_in_m, stride_in_n = input_tensor.stride(0), input_tensor.stride(1)
    stride_out_m, stride_out_n = output_tensor.stride(0), output_tensor.stride(1)

    variant = config.all_reduce_variant.lower()
    if variant not in [
        VARIANT_ATOMIC,
        VARIANT_SPINLOCK,
        VARIANT_RING,
        VARIANT_TWO_SHOT,
        VARIANT_ONE_SHOT,
    ]:
        raise ValueError(
            f"Invalid all_reduce_variant: {variant}. Must be one of: {VARIANT_ATOMIC}, {VARIANT_SPINLOCK}, {VARIANT_RING}, {VARIANT_TWO_SHOT}, {VARIANT_ONE_SHOT}"
        )

    slice_n = config.all_reduce_ring_slice_n
    if variant == VARIANT_RING:
        if config.block_size_n % world_size != 0:
            raise ValueError(
                f"block_size_n ({config.block_size_n}) must be divisible by world_size ({world_size}) for ring variant"
            )
        expected_slice = config.block_size_n // world_size
        if slice_n is None or slice_n * world_size != config.block_size_n:
            slice_n = expected_slice
        config.all_reduce_ring_slice_n = slice_n

    needs_prepare = (
        workspace is None
        or not getattr(workspace, "prepared", False)
        or workspace.variant != variant
        or workspace.shape != (M, N)
        or workspace.dtype != input_tensor.dtype
        or (variant == VARIANT_RING and workspace.num_rings != config.all_reduce_num_rings)
        or (variant == VARIANT_RING and workspace.flags_per_tile != 1)
        or (variant == VARIANT_SPINLOCK and (workspace.locks is None))
    )

    if needs_prepare:
        workspace = all_reduce_preamble(
            output_tensor,
            input_tensor,
            shmem,
            config=config,
            workspace=workspace,
        )

    heap_bases = shmem.get_heap_bases()

    if variant == VARIANT_ATOMIC:
        persistent_all_reduce_atomic[(config.comm_sms,)](
            input_tensor,
            output_tensor,
            M,
            N,
            stride_in_m,
            stride_in_n,
            stride_out_m,
            stride_out_n,
            heap_bases,
            rank,
            world_size,
            config.block_size_m,
            config.block_size_n,
            config.swizzle_size,
            config.comm_sms,
            config.num_xcds,
            config.chunk_size,
        )

    elif variant == VARIANT_SPINLOCK:
        if workspace is None or workspace.locks is None:
            raise RuntimeError(
                "Spinlock variant requires workspace preparation. Call all_reduce_preamble before all_reduce."
            )

        persistent_all_reduce_spinlock[(config.comm_sms,)](
            input_tensor,
            output_tensor,
            workspace.locks,
            M,
            N,
            stride_in_m,
            stride_in_n,
            stride_out_m,
            stride_out_n,
            heap_bases,
            rank,
            world_size,
            config.block_size_m,
            config.block_size_n,
            config.swizzle_size,
            config.comm_sms,
            config.num_xcds,
            config.chunk_size,
        )

    elif variant == VARIANT_RING:
        if workspace is None or workspace.ring_buffer is None or workspace.flags is None:
            raise RuntimeError(
                "Ring variant requires workspace preparation. Call all_reduce_preamble before all_reduce."
            )

        persistent_all_reduce_ring[(config.comm_sms,)](
            input_tensor,
            output_tensor,
            workspace.ring_buffer,
            workspace.flags,
            M,
            N,
            stride_in_m,
            stride_in_n,
            stride_out_m,
            stride_out_n,
            heap_bases,
            rank,
            world_size,
            config.block_size_m,
            config.block_size_n,
            config.swizzle_size,
            config.comm_sms,
            config.num_xcds,
            config.chunk_size,
            config.all_reduce_num_rings,
            slice_n,
            workspace.flags_per_tile,
        )

    elif variant == VARIANT_TWO_SHOT:
        persistent_all_reduce_two_shot[(config.comm_sms,)](
            input_tensor,
            output_tensor,
            M,
            N,
            stride_in_m,
            stride_in_n,
            stride_out_m,
            stride_out_n,
            heap_bases,
            rank,
            world_size,
            config.block_size_m,
            config.block_size_n,
            config.swizzle_size,
            config.comm_sms,
            config.num_xcds,
            config.chunk_size,
            config.all_reduce_distribution,
        )
    elif variant == VARIANT_ONE_SHOT:
        persistent_all_reduce_one_shot[(config.comm_sms,)](
            input_tensor,
            output_tensor,
            M,
            N,
            stride_in_m,
            stride_in_n,
            stride_out_m,
            stride_out_n,
            heap_bases,
            rank,
            world_size,
            config.block_size_m,
            config.block_size_n,
            config.swizzle_size,
            config.comm_sms,
            config.num_xcds,
            config.chunk_size,
        )

    if workspace is not None:
        workspace.prepared = False

    if not async_op:
        shmem.barrier()

    return workspace
