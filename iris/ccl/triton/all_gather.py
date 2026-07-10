# SPDX-License-Identifier: MIT
# Copyright (c) 2025 Advanced Micro Devices, Inc. All rights reserved.

"""
Triton kernels for all-gather collective communication.
Supports multiple variants: persistent, partitioned, and ring.

The ring variant is ported from RCCL's ring AllGather algorithm
(see rccl/src/device/all_gather.h) and leverages iris's symmetric heap
for zero-copy remote stores.
"""

from dataclasses import dataclass
from typing import Optional, Tuple

import triton
import triton.language as tl
import torch
import iris
from iris.host.tracing.kernel_artifacts import iris_launch
from ..utils import chiplet_transform_chunked


@dataclass
class AllGatherWorkspace:
    """
    Holds reusable workspace allocations for ring-based all-gather.

    Attributes:
        variant: Selected all-gather variant.
        shape: Tuple of (M, N) for per-rank tensor shape.
        dtype: Torch dtype of buffers.
        flags: Synchronization flags for ring-based algorithm.
        prepared: Indicates whether preamble has been executed.
    """

    variant: str = ""
    shape: Tuple[int, int] = ()
    dtype: Optional[torch.dtype] = None
    flags: Optional[torch.Tensor] = None
    prepared: bool = False


def all_gather_preamble(
    output_tensor,
    input_tensor,
    ctx,
    config=None,
    workspace=None,
):
    """
    Allocate and reset temporary buffers for the ring all-gather variant.

    Returns:
        AllGatherWorkspace instance ready for the next call to all_gather.
    """
    from ..config import Config

    if config is None:
        config = Config()

    M, N = input_tensor.shape[:2]
    dtype = input_tensor.dtype

    if workspace is None:
        workspace = AllGatherWorkspace()

    workspace.variant = config.all_gather_variant
    workspace.shape = (M, N)
    workspace.dtype = dtype
    workspace.prepared = False

    if config.all_gather_variant == "ring":
        num_pid_m = (M + config.block_size_m - 1) // config.block_size_m
        num_pid_n = (N + config.block_size_n - 1) // config.block_size_n
        total_tiles = num_pid_m * num_pid_n

        if workspace.flags is None or workspace.flags.numel() < total_tiles:
            workspace.flags = ctx.zeros((total_tiles,), dtype=torch.int32)
        else:
            workspace.flags.zero_()

        ctx.barrier()

    workspace.prepared = True
    return workspace


@triton.jit()
def persistent_all_gather_ring(
    input_ptr,
    output_ptr,
    flags,
    M,
    N,
    stride_in_m,
    stride_in_n,
    stride_out_m,
    stride_out_n,
    heap_bases: tl.tensor,
    group_rank: tl.constexpr,
    iris_rank: tl.constexpr,
    world_size: tl.constexpr,
    next_rank: tl.constexpr,
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
    GROUP_SIZE_M: tl.constexpr,
    COMM_SMS: tl.constexpr,
    NUM_XCDS: tl.constexpr,
    CHUNK_SIZE: tl.constexpr,
):
    """
    Ring-based all-gather kernel ported from RCCL's ring AllGather algorithm
    (rccl/src/device/all_gather.h).

    Algorithm (from RCCL):
        For N ranks in a ring, AllGather completes in N-1 steps:
        - Step 0: Each rank sends its own data chunk to the next rank in the ring
                  (RCCL: directCopySend). The data is written directly to the
                  correct slot in the next rank's output buffer.
        - Steps 1..N-2: Each rank receives data (written by prev rank into its
                        output buffer), reads it, and forwards to next rank.
                        (RCCL: directRecvCopyDirectSend).
        - Step N-1: Final receive — no forwarding needed (RCCL: directRecv).

    Iris advantage over RCCL:
        With iris's symmetric heap, we perform zero-copy remote stores directly
        into the destination rank's output buffer. RCCL uses intermediate ring
        buffers that require extra copies; iris eliminates these entirely.

    Synchronization (producer/consumer handshake):
        Uses per-tile flags following iris's ring all-reduce pattern:
        - Before writing: check next rank's flag is 0 (ready) via remote atomic CAS
        - After writing: set next rank's flag to 1 via remote atomic XCHG
        - To receive: wait for local flag == 1 via local atomic CAS
        - After receiving: reset local flag to 0 via local atomic XCHG
    """
    pid_raw = tl.program_id(0)

    # Use chiplet transform to distribute PIDs across XCDs
    pid = pid_raw
    if NUM_XCDS != 1:
        pid = chiplet_transform_chunked(pid_raw, COMM_SMS, NUM_XCDS, CHUNK_SIZE)

    num_pid_m = tl.cdiv(M, BLOCK_SIZE_M)
    num_pid_n = tl.cdiv(N, BLOCK_SIZE_N)
    total_tiles = num_pid_m * num_pid_n
    tl.assume(total_tiles > 0)

    for tile_id in range(pid, total_tiles, COMM_SMS):
        # Tile coordinate computation (swizzled)
        num_pid_in_group = GROUP_SIZE_M * num_pid_n
        group_id = tile_id // num_pid_in_group
        first_pid_m = group_id * GROUP_SIZE_M
        group_size_m = min(num_pid_m - first_pid_m, GROUP_SIZE_M)
        pid_m = first_pid_m + ((tile_id % num_pid_in_group) % group_size_m)
        pid_n = (tile_id % num_pid_in_group) // group_size_m

        tl.assume(pid_m >= 0)
        tl.assume(pid_n >= 0)
        tl.assume(stride_in_m >= 0)
        tl.assume(stride_in_n >= 0)
        tl.assume(stride_out_m >= 0)
        tl.assume(stride_out_n >= 0)

        # Compute tile row/col indices within the per-rank (M, N) block
        rm_base = pid_m * BLOCK_SIZE_M
        rn_base = pid_n * BLOCK_SIZE_N
        rm = rm_base + tl.arange(0, BLOCK_SIZE_M)
        rn = rn_base + tl.arange(0, BLOCK_SIZE_N)
        rm = tl.max_contiguous(tl.multiple_of(rm, BLOCK_SIZE_M), BLOCK_SIZE_M)
        rn = tl.max_contiguous(tl.multiple_of(rn, BLOCK_SIZE_N), BLOCK_SIZE_N)

        tile_mask = (rm[:, None] < M) & (rn[None, :] < N)
        input_offset = rm[:, None] * stride_in_m + rn[None, :] * stride_in_n

        # Flag pointer for this tile (on local memory)
        local_flag_ptr = flags + tile_id
        # Remote flag pointer (same offset but on next rank's memory)
        remote_flag_ptr = flags + tile_id

        # ---------------------------------------------------------------
        # RCCL Ring AllGather (N-1 steps)
        # Output layout: [Rank0_data | Rank1_data | ... | RankN-1_data]
        # At each step, data for one rank propagates one hop around the ring.
        # ---------------------------------------------------------------

        # Load my input data
        data = tl.load(input_ptr + input_offset, mask=tile_mask, other=0.0)

        # Write to my own output buffer (local copy to my slot)
        rm_my_slot = rm + group_rank * M
        my_slot_offset = rm_my_slot[:, None] * stride_out_m + rn[None, :] * stride_out_n
        tl.store(output_ptr + my_slot_offset, data, mask=tile_mask, cache_modifier=".wt")

        # For world_size == 1, AllGather is just a local copy (already done above).
        # Skip all ring communication to avoid self-targeted remote atomics which
        # may hang on some symmetric heap implementations.
        if world_size > 1:
            # --- Step 0: Send my data to next rank (RCCL directCopySend) ---
            # Check next rank is ready (flag == 0)
            while (
                iris.atomic_cas(
                    remote_flag_ptr,
                    0,
                    0,
                    iris_rank,
                    next_rank,
                    heap_bases,
                    sem="acquire",
                    scope="sys",
                )
                != 0
            ):
                pass

            # Write my data to next rank's output buffer at my slot position
            iris.store(
                output_ptr + my_slot_offset,
                data,
                iris_rank,
                next_rank,
                heap_bases,
                mask=tile_mask,
                hint=(1, BLOCK_SIZE_N),
            )

            # Signal next rank: data for step 0 is ready
            tl.debug_barrier()
            iris.atomic_xchg(
                remote_flag_ptr,
                1,
                iris_rank,
                next_rank,
                heap_bases,
                sem="release",
                scope="sys",
            )

            # --- Steps 1 to world_size-2: Receive, read, forward ---
            # (RCCL directRecvCopyDirectSend)
            for _step in range(1, world_size - 1):
                # Wait for prev rank to deliver data (local flag becomes 1)
                while tl.atomic_cas(local_flag_ptr, 0, 0, sem="acquire", scope="sys") != 1:
                    pass

                # Identify which rank's data was just written to us.
                # At step s, data originated from (group_rank - s) % world_size.
                # The prev rank wrote it at the source rank's slot in our buffer.
                source_rank_idx = (group_rank + world_size - _step) % world_size
                rm_slot = rm + source_rank_idx * M
                slot_offset = rm_slot[:, None] * stride_out_m + rn[None, :] * stride_out_n

                # Read the received data from our output buffer.
                # Use cache-volatile (.cv) to bypass L2 cache and ensure we see
                # the most recent remote store, not a stale cached copy. This is
                # necessary because we read from the same output buffer that
                # remote ranks write into (unlike all-reduce which uses a
                # separate ring_buffer for data transfer).
                recv_data = tl.load(output_ptr + slot_offset, mask=tile_mask, other=0.0, cache_modifier=".cv")

                # Forward to next rank: check it's ready
                while (
                    iris.atomic_cas(
                        remote_flag_ptr,
                        0,
                        0,
                        iris_rank,
                        next_rank,
                        heap_bases,
                        sem="acquire",
                        scope="sys",
                    )
                    != 0
                ):
                    pass

                # Write to next rank's output buffer at the same slot
                iris.store(
                    output_ptr + slot_offset,
                    recv_data,
                    iris_rank,
                    next_rank,
                    heap_bases,
                    mask=tile_mask,
                    hint=(1, BLOCK_SIZE_N),
                )

                # Signal next rank
                tl.debug_barrier()
                iris.atomic_xchg(
                    remote_flag_ptr,
                    1,
                    iris_rank,
                    next_rank,
                    heap_bases,
                    sem="release",
                    scope="sys",
                )

                # Reset our flag to 0 AFTER forwarding data, so prev rank
                # cannot overwrite data we still need. This closes the race
                # window where resetting the flag early would unblock the prev
                # rank to write new data for the next step.
                tl.debug_barrier()
                tl.atomic_xchg(local_flag_ptr, 0, sem="release", scope="sys")

            # --- Step world_size-1: Final receive (RCCL directRecv) ---
            # Wait for last piece of data
            while tl.atomic_cas(local_flag_ptr, 0, 0, sem="acquire", scope="sys") != 1:
                pass

            # Reset flag for next invocation
            tl.debug_barrier()
            tl.atomic_xchg(local_flag_ptr, 0, sem="release", scope="sys")


@triton.jit()
def persistent_all_gather(
    input_ptr,
    output_ptr,
    M,
    N,
    stride_in_m,
    stride_in_n,
    stride_out_m,
    stride_out_n,
    heap_bases: tl.tensor,
    group_rank: tl.constexpr,
    iris_rank: tl.constexpr,
    world_size: tl.constexpr,
    rank_start: tl.constexpr,
    rank_stride: tl.constexpr,
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
    GROUP_SIZE_M: tl.constexpr,
    COMM_SMS: tl.constexpr,
    NUM_XCDS: tl.constexpr,
    CHUNK_SIZE: tl.constexpr,
):
    """
    Persistent all-gather kernel.

    Each rank sends its input tensor to all ranks, and all ranks receive
    and concatenate all input tensors along dimension 0 (rows), matching
    torch.distributed.all_gather_into_tensor behavior.

    Args:
        input_ptr: Pointer to input tensor (local rank's data to send) of shape (M, N)
        output_ptr: Pointer to output tensor (will receive from all ranks) of shape (world_size * M, N)
        M: Number of rows per rank (output will be world_size * M rows)
        N: Number of columns
        stride_in_m, stride_in_n: Strides for input tensor
        stride_out_m, stride_out_n: Strides for output tensor
        heap_bases: Heap base pointers for all ranks
        group_rank: Rank within the ProcessGroup (0 to group_size-1), used for tile assignment and comparisons
        iris_rank: Rank in the iris context, used for iris RMA operations (heap_bases indexing)
        world_size: Total number of ranks in the group
        BLOCK_SIZE_M, BLOCK_SIZE_N: Block sizes for tiling
        GROUP_SIZE_M: Group size for M dimension tiling
        COMM_SMS: Number of SMs for communication
        NUM_XCDS: Number of XCDs
        CHUNK_SIZE: Chunk size for chiplet transform
    """
    pid = tl.program_id(0)

    num_pid_m = tl.cdiv(M, BLOCK_SIZE_M)
    num_pid_n = tl.cdiv(N, BLOCK_SIZE_N)
    total_tiles = num_pid_m * num_pid_n
    tl.assume(total_tiles > 0)
    for tile_id in range(pid, total_tiles, COMM_SMS):
        num_pid_in_group = GROUP_SIZE_M * num_pid_n
        group_id = tile_id // num_pid_in_group
        first_pid_m = group_id * GROUP_SIZE_M
        group_size_m = min(num_pid_m - first_pid_m, GROUP_SIZE_M)
        pid_m = first_pid_m + ((tile_id % num_pid_in_group) % group_size_m)
        pid_n = (tile_id % num_pid_in_group) // group_size_m

        tl.assume(pid_m >= 0)
        tl.assume(pid_n >= 0)
        tl.assume(tile_id >= 0)
        tl.assume(stride_in_m >= 0)
        tl.assume(stride_in_n >= 0)
        tl.assume(stride_out_m >= 0)
        tl.assume(stride_out_n >= 0)

        # Compute local row and column indices for input tensor
        rm_base = pid_m * BLOCK_SIZE_M
        rn_base = pid_n * BLOCK_SIZE_N
        rm_input = rm_base + tl.arange(0, BLOCK_SIZE_M)
        rn = rn_base + tl.arange(0, BLOCK_SIZE_N)
        rm_input = tl.max_contiguous(tl.multiple_of(rm_input, BLOCK_SIZE_M), BLOCK_SIZE_M)
        rn = tl.max_contiguous(tl.multiple_of(rn, BLOCK_SIZE_N), BLOCK_SIZE_N)

        # Mask for local input bounds
        input_mask = (rm_input[:, None] < M) & (rn[None, :] < N)

        # Compute input offset and load local shard data once
        input_base_m = rm_input[:, None] * stride_in_m
        input_base_n = rn[None, :] * stride_in_n
        input_offset = input_base_m + input_base_n
        input_ptr_source = input_ptr + input_offset
        input_ptr_source = tl.multiple_of(input_ptr_source, (BLOCK_SIZE_M, BLOCK_SIZE_N))

        # Load local input data once for this tile
        data = tl.load(input_ptr_source, mask=input_mask, other=0.0)

        # Send local shard data to all destination ranks
        # Each rank's input goes to output[group_rank * M : (group_rank + 1) * M, :] on all ranks
        for i in tl.static_range(world_size):
            target_rank = rank_start + i * rank_stride

            # Compute global output row indices: offset by group_rank * M
            rm_output = rm_input + group_rank * M

            # Output mask: only write where input was valid
            output_mask = (rm_output[:, None] < (group_rank + 1) * M) & (rn[None, :] < N)

            # Combine masks: must be valid in both input and output
            combined_mask = input_mask & output_mask

            # Compute output offset
            output_base_m = rm_output[:, None] * stride_out_m
            output_base_n = rn[None, :] * stride_out_n
            output_offset = output_base_m + output_base_n
            output_ptr_target = output_ptr + output_offset
            output_ptr_target = tl.multiple_of(output_ptr_target, (BLOCK_SIZE_M, BLOCK_SIZE_N))

            if i == group_rank:
                # Local destination (i == group_rank): use direct store
                tl.store(output_ptr_target, data, mask=combined_mask, cache_modifier=".wt")
            else:
                # Remote destination: use iris.store to send data to remote destination
                # Use iris_rank for iris RMA operations (heap_bases indexing)
                iris.store(
                    output_ptr_target,
                    data,
                    iris_rank,
                    target_rank,
                    heap_bases,
                    mask=combined_mask,
                    hint=(1, BLOCK_SIZE_N),
                )


@triton.jit()
def persistent_all_gather_partitioned(
    input_ptr,
    output_ptr,
    M,
    N,
    stride_in_m,
    stride_in_n,
    stride_out_m,
    stride_out_n,
    heap_bases: tl.tensor,
    group_rank: tl.constexpr,
    iris_rank: tl.constexpr,
    world_size: tl.constexpr,
    rank_start: tl.constexpr,
    rank_stride: tl.constexpr,
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
    GROUP_SIZE_M: tl.constexpr,
    COMM_SMS: tl.constexpr,
    NUM_XCDS: tl.constexpr,
    CHUNK_SIZE: tl.constexpr,
):
    """
    Persistent all-gather kernel with rank-partitioned work distribution.

    Each PID is assigned to work on a specific destination rank, and multiple PIDs
    partition the tiles for that rank. This avoids the inner loop over world_size.

    Work distribution:
    - PIDs are partitioned across destination ranks
    - PIDs_per_rank = COMM_SMS // world_size
    - Each group of PIDs handles all tiles for one destination rank
    - Within each rank group, PIDs partition the tiles

    Args:
        input_ptr: Pointer to input tensor (local rank's data to send) of shape (M, N)
        output_ptr: Pointer to output tensor (will receive from all ranks) of shape (world_size * M, N)
        M: Number of rows per rank (output will be world_size * M rows)
        N: Number of columns
        stride_in_m, stride_in_n: Strides for input tensor
        stride_out_m, stride_out_n: Strides for output tensor
        heap_bases: Heap base pointers for all ranks
        group_rank: Rank within the ProcessGroup (0 to group_size-1), used for tile assignment and comparisons
        iris_rank: Rank in the iris context, used for iris RMA operations (heap_bases indexing)
        world_size: Total number of ranks in the group
        BLOCK_SIZE_M, BLOCK_SIZE_N: Block sizes for tiling
        GROUP_SIZE_M: Group size for M dimension tiling
        COMM_SMS: Number of SMs for communication (must be divisible by world_size)
        NUM_XCDS: Number of XCDs
        CHUNK_SIZE: Chunk size for chiplet transform
    """
    pid = tl.program_id(0)

    # Partition PIDs across destination ranks
    pids_per_rank = COMM_SMS // world_size
    dest_rank_idx = pid // pids_per_rank  # Which destination rank this PID works on (0 to world_size-1)
    pid_in_rank_group = pid % pids_per_rank  # Which PID within the rank group

    # Compute the actual target rank
    target_rank = rank_start + dest_rank_idx * rank_stride

    num_pid_m = tl.cdiv(M, BLOCK_SIZE_M)
    num_pid_n = tl.cdiv(N, BLOCK_SIZE_N)
    total_tiles = num_pid_m * num_pid_n
    tl.assume(total_tiles > 0)

    # Iterate over tiles with this PID's offset and stride within the rank group
    for tile_id in range(pid_in_rank_group, total_tiles, pids_per_rank):
        num_pid_in_group = GROUP_SIZE_M * num_pid_n
        group_id = tile_id // num_pid_in_group
        first_pid_m = group_id * GROUP_SIZE_M
        group_size_m = min(num_pid_m - first_pid_m, GROUP_SIZE_M)
        pid_m = first_pid_m + ((tile_id % num_pid_in_group) % group_size_m)
        pid_n = (tile_id % num_pid_in_group) // group_size_m

        tl.assume(pid_m >= 0)
        tl.assume(pid_n >= 0)
        tl.assume(tile_id >= 0)
        tl.assume(stride_in_m >= 0)
        tl.assume(stride_in_n >= 0)
        tl.assume(stride_out_m >= 0)
        tl.assume(stride_out_n >= 0)

        # Compute local row and column indices for input tensor
        rm_base = pid_m * BLOCK_SIZE_M
        rn_base = pid_n * BLOCK_SIZE_N
        rm_input = rm_base + tl.arange(0, BLOCK_SIZE_M)
        rn = rn_base + tl.arange(0, BLOCK_SIZE_N)
        rm_input = tl.max_contiguous(tl.multiple_of(rm_input, BLOCK_SIZE_M), BLOCK_SIZE_M)
        rn = tl.max_contiguous(tl.multiple_of(rn, BLOCK_SIZE_N), BLOCK_SIZE_N)

        # Mask for local input bounds
        input_mask = (rm_input[:, None] < M) & (rn[None, :] < N)

        # Compute input offset and load local shard data once
        input_base_m = rm_input[:, None] * stride_in_m
        input_base_n = rn[None, :] * stride_in_n
        input_offset = input_base_m + input_base_n
        input_ptr_source = input_ptr + input_offset
        input_ptr_source = tl.multiple_of(input_ptr_source, (BLOCK_SIZE_M, BLOCK_SIZE_N))

        # Load local input data once for this tile
        data = tl.load(input_ptr_source, mask=input_mask, other=0.0)

        # Compute global output row indices: offset by group_rank * M
        rm_output = rm_input + group_rank * M

        # Output mask: only write where input was valid
        output_mask = (rm_output[:, None] < (group_rank + 1) * M) & (rn[None, :] < N)

        # Combine masks: must be valid in both input and output
        combined_mask = input_mask & output_mask

        # Compute output offset
        output_base_m = rm_output[:, None] * stride_out_m
        output_base_n = rn[None, :] * stride_out_n
        output_offset = output_base_m + output_base_n
        output_ptr_target = output_ptr + output_offset
        output_ptr_target = tl.multiple_of(output_ptr_target, (BLOCK_SIZE_M, BLOCK_SIZE_N))

        # Send to the assigned destination rank
        if dest_rank_idx == group_rank:
            # Local destination: use direct store
            tl.store(output_ptr_target, data, mask=combined_mask, cache_modifier=".wt")
        else:
            # Remote destination: use iris.store to send data to remote destination
            iris.store(
                output_ptr_target,
                data,
                iris_rank,
                target_rank,
                heap_bases,
                mask=combined_mask,
                hint=(1, BLOCK_SIZE_N),
            )


def launch(
    input_tensor,
    output_tensor,
    ctx,
    rank_in_group,
    rank_global,
    world_size,
    rank_start,
    rank_stride,
    config,
    workspace=None,
    group=None,
):
    """Launch the Triton all-gather kernel."""
    M, N = input_tensor.shape[:2]
    stride_in_m, stride_in_n = input_tensor.stride(0), input_tensor.stride(1)
    stride_out_m, stride_out_n = output_tensor.stride(0), output_tensor.stride(1)

    variant = config.all_gather_variant

    # Validate COMM_SMS divisibility for partitioned variant
    if variant == "partitioned" and config.comm_sms % world_size != 0:
        raise ValueError(
            f"For all_gather_variant='partitioned', COMM_SMS ({config.comm_sms}) must be divisible by world_size ({world_size}). "
            f"Please adjust config.comm_sms to be a multiple of {world_size}."
        )

    heap_bases = ctx.get_heap_bases()

    if variant == "ring":
        # Ring variant requires workspace with flags
        needs_prepare = (
            workspace is None
            or not getattr(workspace, "prepared", False)
            or workspace.variant != variant
            or workspace.shape != (M, N)
            or workspace.dtype != input_tensor.dtype
        )
        if needs_prepare:
            workspace = all_gather_preamble(
                output_tensor,
                input_tensor,
                ctx,
                config=config,
                workspace=workspace,
            )

        # Calculate ring neighbor (only next_rank needed by kernel)
        if group is None:
            next_group_rank = (rank_in_group + 1) % world_size
            next_rank = rank_start + next_group_rank * rank_stride
        else:
            import torch.distributed as dist

            group_ranks = dist.get_process_group_ranks(group)
            next_group_rank = (rank_in_group + 1) % world_size
            next_rank = group_ranks[next_group_rank]

        iris_launch(
            persistent_all_gather_ring,
            (config.comm_sms,),
            input_tensor,
            output_tensor,
            workspace.flags,
            M,
            N,
            stride_in_m,
            stride_in_n,
            stride_out_m,
            stride_out_n,
            heap_bases,
            rank_in_group,
            rank_global,
            world_size,
            next_rank,
            config.block_size_m,
            config.block_size_n,
            config.swizzle_size,
            config.comm_sms,
            config.num_xcds,
            config.chunk_size,
            num_stages=config.num_stages,
            num_warps=config.num_warps,
            waves_per_eu=config.waves_per_eu,
            algorithm="all_gather",
            rank=rank_global,
            dtype=input_tensor.dtype,
        )
        # Reset prepared so the next call re-runs the preamble (flag-zeroing +
        # barrier). This is critical for async_op=True where the trailing barrier
        # is skipped — without this, a subsequent call would skip the preamble
        # and launch with flags in an inconsistent state, risking deadlock.
        workspace.prepared = False
        return workspace

    # Non-ring variants
    if variant == "persistent":
        kernel_fn = persistent_all_gather
    elif variant == "partitioned":
        kernel_fn = persistent_all_gather_partitioned
    else:
        raise ValueError(f"Unknown all_gather_variant: {variant}")

    iris_launch(
        kernel_fn,
        (config.comm_sms,),
        input_tensor,
        output_tensor,
        M,
        N,
        stride_in_m,
        stride_in_n,
        stride_out_m,
        stride_out_n,
        heap_bases,
        rank_in_group,
        rank_global,
        world_size,
        rank_start,
        rank_stride,
        config.block_size_m,
        config.block_size_n,
        config.swizzle_size,
        config.comm_sms,
        config.num_xcds,
        config.chunk_size,
        num_stages=config.num_stages,
        num_warps=config.num_warps,
        waves_per_eu=config.waves_per_eu,
        algorithm="all_gather",
        rank=rank_global,
        dtype=input_tensor.dtype,
    )
    return workspace
