# SPDX-License-Identifier: MIT
# Copyright (c) 2025 Advanced Micro Devices, Inc. All rights reserved.

"""
All-reduce collective communication primitive for Iris.
Supports three variants: atomic-based, ring-based, and two-shot-based.
"""

import triton
import triton.language as tl
import torch
import iris
from .config import Config

# Variant types
VARIANT_ATOMIC = "atomic"
VARIANT_RING = "ring"
VARIANT_TWO_SHOT = "two_shot"


@triton.jit()
def chiplet_transform_chunked(
    pid, 
    num_workgroups: tl.constexpr, 
    num_xcds: tl.constexpr, 
    chunk_size: tl.constexpr
):
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

    # Use same transform as example 08 for consistency
    if NUM_XCDS != 1:
        pid = (pid % NUM_XCDS) * (COMM_SMS // NUM_XCDS) + (pid // NUM_XCDS)
    
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
        # Following example 08 pattern: each rank adds its contribution to all ranks' outputs
        # Each rank's output tensor is in its own heap, accessible via IPC
        # We write to each rank's output separately
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
):
    """
    Ring-based all-reduce kernel.
    
    Data is passed around in a ring topology. Each rank receives data from the
    previous rank, accumulates it with its local partial, and forwards it to the next rank.
    After (world_size - 1) steps, the data is fully reduced.
    
    Args:
        input_ptr: Pointer to input tensor (local rank's partial data)
        output_ptr: Pointer to output tensor (will contain sum of all ranks)
        ring_buffer: Temporary buffer for ring communication
        flags: Synchronization flags for ring communication
        M: Number of rows
        N: Number of columns
        heap_bases: Heap base pointers for all ranks
        cur_rank: Current rank
        world_size: Total number of ranks
    """
    pid = tl.program_id(0)

    # Use same transform as example 08/16 for consistency
    if NUM_XCDS != 1:
        pid = (pid % NUM_XCDS) * (COMM_SMS // NUM_XCDS) + (pid // NUM_XCDS)
    
    num_pid_m = tl.cdiv(M, BLOCK_SIZE_M)
    num_pid_n = tl.cdiv(N, BLOCK_SIZE_N)
    total_tiles = num_pid_m * num_pid_n

    # Ring topology
    next_rank = (cur_rank + 1) % world_size
    prev_rank = (cur_rank + world_size - 1) % world_size
    
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
        offset = rm[:, None] * stride_in_m + rn[None, :] * stride_in_n
        input_ptr_local = input_ptr + offset
        input_ptr_local = tl.multiple_of(input_ptr_local, (BLOCK_SIZE_M, BLOCK_SIZE_N))
        
        # Initialize accumulator with local partial result
        acc = tl.load(input_ptr_local, mask=mask).to(acc_dtype)

        # Initialize: First, write our partial result to ring_buffer for sending
        # Convert to input dtype for sending (to match what we'll receive)
        send_data = acc.to(input_ptr.type.element_ty)

        # Ring algorithm: send to next, wait/recv from prev, add
        for _step in range(0, world_size - 1):
            # 1a) Wait for NEXT rank to be ready (its flag should be 0)
            while (
                iris.atomic_cas(flags + tile_id, 0, 0, cur_rank, next_rank, heap_bases, sem="acquire", scope="sys") != 0
            ):
                pass

            # 1b) Send our current accumulator tile to NEXT rank's ring buffer
            # Ring buffer has same shape and strides as input
            ring_offset = rm[:, None] * stride_in_m + rn[None, :] * stride_in_n
            iris.store(ring_buffer + ring_offset, send_data, cur_rank, next_rank, heap_bases, mask=mask)

            tl.debug_barrier()
            # Signal "ready" by setting NEXT rank's flag for this tile to 1
            iris.atomic_xchg(flags + tile_id, 1, cur_rank, next_rank, heap_bases, sem="release", scope="sys")

            # 2) Wait for PREV rank to signal our local flag for this tile
            while tl.atomic_cas(flags + tile_id, 0, 0, sem="acquire", scope="sys") != 1:
                pass

            # 3) Consume the received tile from our LOCAL ring buffer (prev wrote here)
            recv_tile = tl.load(ring_buffer + ring_offset, mask=mask, other=tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=input_ptr.type.element_ty))

            # 4) Accumulate received data and prepare to forward it in next iteration
            acc += recv_tile.to(acc_dtype)
            send_data = recv_tile  # Forward what we just received (not the accumulated sum)

            # 5) Reset our local flag to 0 (done consuming this step)
            tl.atomic_xchg(flags + tile_id, 0, sem="release", scope="sys")

        # Write fully-reduced tile to output
        output_offset = rm[:, None] * stride_out_m + rn[None, :] * stride_out_n
        o = acc.to(output_ptr.type.element_ty)
        tl.store(output_ptr + output_offset, o, mask=mask)


@triton.jit()
def persistent_all_reduce_two_shot_producer(
    input_ptr,
    local_buffer,
    locks,
    tile_ready,
    M,
    N,
    stride_in_m,
    stride_in_n,
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
    Producer kernel for two-shot all-reduce: stores local partials to local_buffer.
    """
    pid = tl.program_id(0)

    # Use same transform as example 17 for consistency
    if NUM_XCDS != 1:
        pid = (pid % NUM_XCDS) * (COMM_SMS // NUM_XCDS) + (pid // NUM_XCDS)
    
    num_pid_m = tl.cdiv(M, BLOCK_SIZE_M)
    num_pid_n = tl.cdiv(N, BLOCK_SIZE_N)
    total_tiles = num_pid_m * num_pid_n

    # Phase 1: Producer - store local partials
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
        offset = rm[:, None] * stride_in_m + rn[None, :] * stride_in_n
        input_ptr_local = input_ptr + offset
        input_ptr_local = tl.multiple_of(input_ptr_local, (BLOCK_SIZE_M, BLOCK_SIZE_N))
        
        # Load and store local partial
        data = tl.load(input_ptr_local, mask=mask)
        local_offset = rm[:, None] * stride_in_m + rn[None, :] * stride_in_n
        tl.store(local_buffer + local_offset, data, mask=mask, cache_modifier=".wt")

        # Ensure local write completes before signaling
        tl.debug_barrier()

        # Signal that this tile is ready
        tl.store(locks + tile_id, 1, cache_modifier=".wt")

        # Signal to all remote ranks that this tile is ready
        for remote_rank in range(world_size):
            if remote_rank != cur_rank:
                iris.atomic_add(tile_ready + tile_id, 1, cur_rank, remote_rank, heap_bases, sem="release", scope="sys")


@triton.jit()
def persistent_all_reduce_two_shot_consumer(
    output_ptr,
    local_buffer,
    locks,
    tile_ready,
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
    DISTRIBUTION: tl.constexpr,  # 0 for striding, 1 for block
):
    """
    Consumer kernel for two-shot all-reduce: reduces assigned tiles and scatters results.
    """
    pid = tl.program_id(0)

    # Use same transform as example 17 for consistency
    if NUM_XCDS != 1:
        pid = (pid % NUM_XCDS) * (COMM_SMS // NUM_XCDS) + (pid // NUM_XCDS)
    
    num_pid_m = tl.cdiv(M, BLOCK_SIZE_M)
    num_pid_n = tl.cdiv(N, BLOCK_SIZE_N)
    total_tiles = num_pid_m * num_pid_n

    acc_dtype = tl.float32 if output_ptr.type.element_ty != tl.int8 else tl.int32

    # Phase 2: Consumer - reduce assigned tiles and scatter
    # Determine which tiles this rank is responsible for reducing
    if DISTRIBUTION == 0:
        # Striding: rank reduces tiles cur_rank, cur_rank + world_size, ...
        tiles_per_rank = tl.cdiv(total_tiles, world_size)
        start_tile = cur_rank
        stride = world_size
    else:
        # Block: rank reduces continuous block of tiles
        tiles_per_rank = tl.cdiv(total_tiles, world_size)
        start_tile = cur_rank * tiles_per_rank
        stride = 1

    # Calculate max tile_offset to avoid boundary issues
    max_tile_offset = tiles_per_rank
    if DISTRIBUTION == 0:  # striding
        max_tile_offset = tl.minimum(tiles_per_rank, tl.cdiv(total_tiles - cur_rank, world_size))
    else:  # block
        max_tile_offset = tl.minimum(tiles_per_rank, total_tiles - cur_rank * tiles_per_rank)

    for tile_offset in range(pid, max_tile_offset, COMM_SMS):
        tile_id = start_tile + tile_offset * stride

        # Wait for all ranks to produce this tile
        # Local tile
        while tl.load(locks + tile_id, cache_modifier=".cv", volatile=True) != 1:
            pass

        # Ensure local producer's writes are visible
        tl.debug_barrier()

        # Wait for remote ranks - each remote rank increments tile_ready when done
        # We expect (world_size - 1) increments from all other ranks
        while iris.atomic_cas(
            tile_ready + tile_id,
            0,  # Never matches when ready, so acts as atomic read
            0,
            cur_rank,
            cur_rank,
            heap_bases,
            sem="acquire",
            scope="sys",
        ) < (world_size - 1):
            pass

        # Map tile_id to (pid_m, pid_n)
        num_pid_in_group = GROUP_SIZE_M * num_pid_n
        group_id = tile_id // num_pid_in_group
        first_pid_m = group_id * GROUP_SIZE_M
        group_size_m = min(num_pid_m - first_pid_m, GROUP_SIZE_M)
        pid_m = first_pid_m + ((tile_id % num_pid_in_group) % group_size_m)
        pid_n = (tile_id % num_pid_in_group) // group_size_m

        # Compute offsets
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
        local_offset = rm[:, None] * stride_in_m + rn[None, :] * stride_in_n

        # Accumulate from all ranks
        acc = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=acc_dtype)
        for remote_rank in range(world_size):
            partial = iris.load(local_buffer + local_offset, cur_rank, remote_rank, heap_bases, mask=mask)
            acc += partial.to(acc_dtype)

        # Convert to output type
        c_out = acc.to(output_ptr.type.element_ty)

        # Scatter to all ranks
        output_offset = rm[:, None] * stride_out_m + rn[None, :] * stride_out_n
        for remote_rank in range(world_size):
            if remote_rank == cur_rank:
                tl.store(output_ptr + output_offset, c_out, mask=mask)
            else:
                iris.store(output_ptr + output_offset, c_out, cur_rank, remote_rank, heap_bases, mask=mask)


def all_reduce(output_tensor, input_tensor, shmem, config=None, async_op=False):
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
                Set config.all_reduce_variant to choose variant: "atomic", "ring", or "two_shot"
        async_op: If False, performs a barrier at the end. If True, returns immediately.
                  Default: False.
    """
    # Use provided config or create default one
    if config is None:
        config = Config()
    
    rank = shmem.get_rank()
    world_size = shmem.get_num_ranks()
    
    M, N = input_tensor.shape[:2]
    
    stride_in_m, stride_in_n = input_tensor.stride(0), input_tensor.stride(1)
    stride_out_m, stride_out_n = output_tensor.stride(0), output_tensor.stride(1)
    
    # Determine variant
    variant = config.all_reduce_variant.lower()
    if variant not in [VARIANT_ATOMIC, VARIANT_RING, VARIANT_TWO_SHOT]:
        raise ValueError(f"Invalid all_reduce_variant: {variant}. Must be one of: {VARIANT_ATOMIC}, {VARIANT_RING}, {VARIANT_TWO_SHOT}")
    
    heap_bases = shmem.get_heap_bases()
    
    if variant == VARIANT_ATOMIC:
        # Initialize output to zero on all ranks
        # Use barrier to ensure all ranks see the zeroed output before starting
        output_tensor.zero_()
        shmem.barrier()
        
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
        
        # Synchronize GPU to ensure all atomic operations complete
        torch.cuda.synchronize()
    
    elif variant == VARIANT_RING:
        # Initialize output to zero
        output_tensor.zero_()
        
        # Allocate temporary buffers for ring algorithm
        num_pid_m = (M + config.block_size_m - 1) // config.block_size_m
        num_pid_n = (N + config.block_size_n - 1) // config.block_size_n
        total_tiles = num_pid_m * num_pid_n
        
        ring_buffer = shmem.zeros((M, N), dtype=input_tensor.dtype)
        flags = shmem.zeros((total_tiles,), dtype=torch.int32)
        flags.zero_()
        
        # Ensure all ranks see zeroed flags before starting
        shmem.barrier()
        
        persistent_all_reduce_ring[(config.comm_sms,)](
            input_tensor,
            output_tensor,
            ring_buffer,
            flags,
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
        
        # Synchronize GPU to ensure all operations complete
        torch.cuda.synchronize()
    
    elif variant == VARIANT_TWO_SHOT:
        # Initialize output to zero
        output_tensor.zero_()
        
        # Allocate temporary buffers for two-shot algorithm
        num_pid_m = (M + config.block_size_m - 1) // config.block_size_m
        num_pid_n = (N + config.block_size_n - 1) // config.block_size_n
        total_tiles = num_pid_m * num_pid_n
        
        local_buffer = shmem.zeros((M, N), dtype=input_tensor.dtype)
        locks = shmem.zeros((total_tiles,), dtype=torch.int32)
        tile_ready = shmem.zeros((total_tiles,), dtype=torch.int32)
        locks.zero_()
        tile_ready.zero_()
        
        # Phase 1: Producer - all ranks store their local partials
        persistent_all_reduce_two_shot_producer[(config.comm_sms,)](
            input_tensor,
            local_buffer,
            locks,
            tile_ready,
            M,
            N,
            stride_in_m,
            stride_in_n,
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
        
        # Synchronize before consumer phase
        shmem.barrier()
        
        # Phase 2: Consumer - each rank reduces assigned tiles and scatters
        persistent_all_reduce_two_shot_consumer[(config.comm_sms,)](
            output_tensor,
            local_buffer,
            locks,
            tile_ready,
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
            config.all_reduce_distribution,  # 0 for striding, 1 for block
        )
    
    if not async_op:
        shmem.barrier()

