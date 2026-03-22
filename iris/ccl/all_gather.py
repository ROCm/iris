# SPDX-License-Identifier: MIT
# Copyright (c) 2025 Advanced Micro Devices, Inc. All rights reserved.

"""
All-gather collective communication primitive for Iris.
Gathers tensors from all ranks and concatenates them along the last dimension.
"""

import triton
import triton.language as tl
import iris
from .config import Config
from .utils import extract_group_info
from iris.tracing.events import TraceEvent

# Conditional import for Gluon
try:
    from triton.experimental import gluon
    from triton.experimental.gluon import language as gl
    from iris.experimental.iris_gluon import IrisDeviceCtx

    GLUON_AVAILABLE = True
except ImportError:
    GLUON_AVAILABLE = False


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


# Gluon implementation
if GLUON_AVAILABLE:

    @gluon.jit
    def persistent_all_gather_gluon(
        IrisDeviceCtx: gl.constexpr,
        context_tensor,
        input_ptr,
        output_ptr,
        M,
        N,
        stride_in_m,
        stride_in_n,
        stride_out_m,
        stride_out_n,
        group_rank: gl.constexpr,
        iris_rank: gl.constexpr,
        world_size: gl.constexpr,
        rank_start: gl.constexpr,
        rank_stride: gl.constexpr,
        BLOCK_SIZE_M: gl.constexpr,
        BLOCK_SIZE_N: gl.constexpr,
        GROUP_SIZE_M: gl.constexpr,
        COMM_SMS: gl.constexpr,
        NUM_XCDS: gl.constexpr,
        CHUNK_SIZE: gl.constexpr,
        THREADS_PER_WARP: gl.constexpr,
        WARPS_PER_CTA: gl.constexpr,
        TRACING: gl.constexpr = False,
    ):
        """
        Persistent all-gather kernel using Gluon with explicit memory layout control.

        Each rank loads its local input once per row and writes it to the
        corresponding output slice on ALL ranks (local + remote), avoiding
        redundant loads. Column indices use an explicit BlockedLayout to
        control vectorization width.

        Memory layout (BlockedLayout):
            The column dimension is distributed across the GPU thread hierarchy
            using gl.BlockedLayout([ELEMS_PER_THREAD], [THREADS_PER_WARP], [WARPS_PER_CTA], [order]).

            - ELEMS_PER_THREAD: number of contiguous elements each thread loads/stores.
              Controls the vector width of memory instructions. For fp16:
                1 -> 2-byte scalar load
                2 -> 4-byte dword load
                4 -> 8-byte dwordx4 load (optimal on AMD GFX9+)
            - THREADS_PER_WARP: threads per warp/wavefront (64 on AMD, 32 on NVIDIA).
            - WARPS_PER_CTA: number of warps in the cooperative thread array (workgroup).

            The product ELEMS_PER_THREAD * THREADS_PER_WARP * WARPS_PER_CTA must
            equal BLOCK_SIZE_N. ELEMS_PER_THREAD is derived as:
                ELEMS_PER_THREAD = BLOCK_SIZE_N // (THREADS_PER_WARP * WARPS_PER_CTA)

        Constraints (validated by host wrapper before launch):
            - BLOCK_SIZE_N must be a multiple of (THREADS_PER_WARP * WARPS_PER_CTA).
            - BLOCK_SIZE_N must be >= (THREADS_PER_WARP * WARPS_PER_CTA) so that
              ELEMS_PER_THREAD >= 1.
            - WARPS_PER_CTA must match the num_warps kernel launch parameter.
            - THREADS_PER_WARP must match the hardware wavefront size (64 for AMD).

        Args:
            IrisDeviceCtx: Gluon device context class for remote memory operations.
            context_tensor: Opaque tensor holding IrisDeviceCtx state.
            input_ptr: Pointer to local input tensor of shape (M, N).
            output_ptr: Pointer to output tensor of shape (world_size * M, N).
            M: Number of rows in the input tensor (per rank).
            N: Number of columns.
            stride_in_m, stride_in_n: Row and column strides for input tensor.
            stride_out_m, stride_out_n: Row and column strides for output tensor.
            group_rank: This rank's index within the ProcessGroup (0..world_size-1).
            iris_rank: This rank's global index in the iris context (for RMA addressing).
            world_size: Total number of ranks in the group.
            rank_start: First iris rank in the group (for RMA target computation).
            rank_stride: Stride between consecutive iris ranks in the group.
            BLOCK_SIZE_M: Number of rows per tile.
            BLOCK_SIZE_N: Number of columns per tile. Must be a multiple of
                          (THREADS_PER_WARP * WARPS_PER_CTA).
            GROUP_SIZE_M: Swizzle group size for M-dimension tiling.
            COMM_SMS: Number of SMs used for persistent scheduling.
            NUM_XCDS: Number of XCDs (chiplet count).
            CHUNK_SIZE: Chunk size for XCD-aware tile mapping.
            THREADS_PER_WARP: Threads per warp/wavefront (64 for AMD, 32 for NVIDIA).
            WARPS_PER_CTA: Number of warps per workgroup. Must match num_warps.
            TRACING: If True, record load/store events into trace buffers.
        """
        ctx = IrisDeviceCtx.initialize(context_tensor, tracing=TRACING)
        events = TraceEvent()

        pid = gl.program_id(0)

        num_pid_m = gl.cdiv(M, BLOCK_SIZE_M)
        num_pid_n = gl.cdiv(N, BLOCK_SIZE_N)
        total_tiles = num_pid_m * num_pid_n

        # Build the 1D BlockedLayout for the column dimension.
        # ELEMS_PER_THREAD controls how many contiguous elements each thread
        # handles, which directly maps to the vector load/store width:
        #   elems=1 -> scalar, elems=2 -> dword, elems=4 -> dwordx4 (optimal)
        ELEMS_PER_THREAD: gl.constexpr = BLOCK_SIZE_N // (THREADS_PER_WARP * WARPS_PER_CTA)
        col_layout: gl.constexpr = gl.BlockedLayout([ELEMS_PER_THREAD], [THREADS_PER_WARP], [WARPS_PER_CTA], [0])

        for tile_id in range(pid, total_tiles, COMM_SMS):
            # Swizzled tile index computation for better L2 locality
            num_pid_in_group = GROUP_SIZE_M * num_pid_n
            group_id = tile_id // num_pid_in_group
            first_pid_m = group_id * GROUP_SIZE_M
            group_size_m = min(num_pid_m - first_pid_m, GROUP_SIZE_M)
            pid_m = first_pid_m + ((tile_id % num_pid_in_group) % group_size_m)
            pid_n = (tile_id % num_pid_in_group) // group_size_m

            # Build column index vector with explicit layout for vectorized access
            rn = (pid_n * BLOCK_SIZE_N + gl.arange(0, BLOCK_SIZE_N, layout=col_layout)) % N
            rn = gl.max_contiguous(gl.multiple_of(rn, BLOCK_SIZE_N), BLOCK_SIZE_N)

            col_offsets_in = rn * stride_in_n
            col_offsets_out = rn * stride_out_n
            col_mask = rn < N

            rm_base = pid_m * BLOCK_SIZE_M

            # Iterate row-by-row: load each row once, then write to all ranks.
            # This avoids reloading the same data for each destination rank.
            for i in range(BLOCK_SIZE_M):
                row_idx = (rm_base + i) % M

                if row_idx < M:
                    # Single load from local input
                    input_addr = input_ptr + row_idx * stride_in_m + col_offsets_in
                    if TRACING:
                        h_load = ctx.tracing.record_event_start(
                            event_id=events.load,
                            target_rank=group_rank,
                            address=input_addr,
                            pid_m=pid_m,
                            pid_n=pid_n,
                            mask=col_mask,
                        )
                    data = gl.load(input_addr, mask=col_mask)
                    if TRACING:
                        ctx.tracing.record_event_end(h_load)

                    # Output row position: this rank's slice starts at group_rank * M
                    output_offset = (group_rank * M + row_idx) * stride_out_m + col_offsets_out

                    # Traffic shaping: stagger the write order per rank so that
                    # at any given moment, each rank is writing to a different
                    # target. Without this, all ranks write to rank 0 first,
                    # then rank 1, etc., causing memory controller contention.
                    #
                    # With offset = group_rank:
                    #   Rank 0: writes to 0(local), 1, 2, 3
                    #   Rank 1: writes to 1(local), 2, 3, 0
                    #   Rank 2: writes to 2(local), 3, 0, 1
                    #   Rank 3: writes to 3(local), 0, 1, 2
                    for rank_idx in range(world_size):
                        dest_idx = (group_rank + rank_idx) % world_size
                        target_rank = rank_start + dest_idx * rank_stride
                        output_ptr_target = output_ptr + output_offset

                        if TRACING:
                            h_store = ctx.tracing.record_event_start(
                                event_id=events.store,
                                target_rank=target_rank,
                                address=output_ptr_target,
                                pid_m=pid_m,
                                pid_n=pid_n,
                                mask=col_mask,
                            )
                        if dest_idx == group_rank:
                            gl.store(output_ptr_target, data, mask=col_mask, cache_modifier=".wt")
                        else:
                            ctx.store(output_ptr_target, data, target_rank, mask=col_mask)
                        if TRACING:
                            ctx.tracing.record_event_end(h_store)


if GLUON_AVAILABLE:

    @gluon.jit
    def persistent_all_gather_gluon_hoisted(
        IrisDeviceCtx: gl.constexpr,
        context_tensor,
        input_ptr,
        output_ptr,
        M,
        N,
        stride_in_m,
        stride_in_n,
        stride_out_m,
        stride_out_n,
        group_rank: gl.constexpr,
        iris_rank: gl.constexpr,
        world_size: gl.constexpr,
        rank_start: gl.constexpr,
        rank_stride: gl.constexpr,
        BLOCK_SIZE_M: gl.constexpr,
        BLOCK_SIZE_N: gl.constexpr,
        GROUP_SIZE_M: gl.constexpr,
        COMM_SMS: gl.constexpr,
        NUM_XCDS: gl.constexpr,
        CHUNK_SIZE: gl.constexpr,
        THREADS_PER_WARP: gl.constexpr,
        WARPS_PER_CTA: gl.constexpr,
        TRACING: gl.constexpr = False,
    ):
        """
        Persistent gluon all-gather with hoisted pointer translation.

        Same structure as persistent_all_gather_gluon (load row once, store to
        all ranks) but pre-computes the pointer translation delta for each
        destination rank ONCE, outside the tile loop. This eliminates
        2 * BLOCK_SIZE_M * (world_size-1) gl.load(heap_bases) calls per tile.

        The delta approach: for any pointer ``p`` in the local address space,
        the translated pointer in rank ``r``'s address space is simply
        ``p + delta[r]``, where ``delta[r] = heap_base[r] - heap_base[local]``.
        """
        ctx = IrisDeviceCtx.initialize(context_tensor, tracing=TRACING)
        events = TraceEvent()

        pid = gl.program_id(0)

        num_pid_m = gl.cdiv(M, BLOCK_SIZE_M)
        num_pid_n = gl.cdiv(N, BLOCK_SIZE_N)
        total_tiles = num_pid_m * num_pid_n

        ELEMS_PER_THREAD: gl.constexpr = BLOCK_SIZE_N // (THREADS_PER_WARP * WARPS_PER_CTA)
        col_layout: gl.constexpr = gl.BlockedLayout([ELEMS_PER_THREAD], [THREADS_PER_WARP], [WARPS_PER_CTA], [0])

        # Pre-compute pointer translation deltas for ALL ranks.
        # delta[i] = heap_base[target_iris_rank_i] - heap_base[local_iris_rank]
        # Then translated_ptr = local_ptr + delta[i]
        local_base = gl.load(ctx.heap_bases + iris_rank)

        for tile_id in range(pid, total_tiles, COMM_SMS):
            # Swizzled tile index computation for better L2 locality
            num_pid_in_group = GROUP_SIZE_M * num_pid_n
            group_id = tile_id // num_pid_in_group
            first_pid_m = group_id * GROUP_SIZE_M
            group_size_m = min(num_pid_m - first_pid_m, GROUP_SIZE_M)
            pid_m = first_pid_m + ((tile_id % num_pid_in_group) % group_size_m)
            pid_n = (tile_id % num_pid_in_group) // group_size_m

            rn = (pid_n * BLOCK_SIZE_N + gl.arange(0, BLOCK_SIZE_N, layout=col_layout)) % N
            rn = gl.max_contiguous(gl.multiple_of(rn, BLOCK_SIZE_N), BLOCK_SIZE_N)

            col_offsets_in = rn * stride_in_n
            col_offsets_out = rn * stride_out_n
            col_mask = rn < N

            rm_base = pid_m * BLOCK_SIZE_M

            for i in range(BLOCK_SIZE_M):
                row_idx = (rm_base + i) % M

                if row_idx < M:
                    input_addr = input_ptr + row_idx * stride_in_m + col_offsets_in
                    if TRACING:
                        h_load = ctx.tracing.record_event_start(
                            event_id=events.load,
                            target_rank=group_rank,
                            address=input_addr,
                            pid_m=pid_m,
                            pid_n=pid_n,
                            mask=col_mask,
                        )
                    data = gl.load(input_addr, mask=col_mask)
                    if TRACING:
                        ctx.tracing.record_event_end(h_load)

                    output_offset = (group_rank * M + row_idx) * stride_out_m + col_offsets_out

                    # Traffic shaping + hoisted translation
                    for rank_idx in range(world_size):
                        dest_idx = (group_rank + rank_idx) % world_size
                        target_iris_rank = rank_start + dest_idx * rank_stride
                        output_addr = output_ptr + output_offset

                        if TRACING:
                            h_store = ctx.tracing.record_event_start(
                                event_id=events.store,
                                target_rank=target_iris_rank,
                                address=output_addr,
                                pid_m=pid_m,
                                pid_n=pid_n,
                                mask=col_mask,
                            )

                        if dest_idx == group_rank:
                            gl.store(output_addr, data, mask=col_mask, cache_modifier=".wt")
                        else:
                            # Hoisted translation: compute delta on the fly
                            # but only load target_base once (compiler should
                            # hoist this out of the BLOCK_SIZE_M loop since
                            # target_iris_rank is loop-invariant w.r.t. i)
                            target_base = gl.load(ctx.heap_bases + target_iris_rank)
                            ptr_delta = target_base - local_base
                            output_addr_int = tl.cast(output_addr, gl.uint64)
                            remote_addr_int = output_addr_int + ptr_delta
                            remote_addr = tl.cast(remote_addr_int, output_addr.dtype)
                            gl.store(remote_addr, data, mask=col_mask)

                        if TRACING:
                            ctx.tracing.record_event_end(h_store)



if GLUON_AVAILABLE:

    @gluon.jit
    def persistent_all_gather_gluon_partitioned(
        IrisDeviceCtx: gl.constexpr,
        context_tensor,
        input_ptr,
        output_ptr,
        M,
        N,
        stride_in_m,
        stride_in_n,
        stride_out_m,
        stride_out_n,
        group_rank: gl.constexpr,
        iris_rank: gl.constexpr,
        world_size: gl.constexpr,
        rank_start: gl.constexpr,
        rank_stride: gl.constexpr,
        BLOCK_SIZE_M: gl.constexpr,
        BLOCK_SIZE_N: gl.constexpr,
        GROUP_SIZE_M: gl.constexpr,
        COMM_SMS: gl.constexpr,
        NUM_XCDS: gl.constexpr,
        CHUNK_SIZE: gl.constexpr,
        THREADS_PER_WARP: gl.constexpr,
        WARPS_PER_CTA: gl.constexpr,
        TRACING: gl.constexpr = False,
    ):
        """
        CU-partitioned gluon all-gather with hoisted pointer translation.

        Each CU is pre-assigned to one destination rank, eliminating the inner
        loop over world_size. Pointer translation (heap base lookup) is computed
        once per CU rather than once per store, reducing instruction count.

        Work distribution:
          - PIDS_PER_RANK = COMM_SMS // world_size
          - CU ``pid`` is assigned to destination rank ``pid // PIDS_PER_RANK``
          - Within each rank group, CUs partition the tiles

        This provides natural traffic shaping: at any given moment, different CUs
        target different ranks, avoiding memory controller contention.
        """
        ctx = IrisDeviceCtx.initialize(context_tensor, tracing=TRACING)
        events = TraceEvent()

        pid = gl.program_id(0)

        # CU partitioning: each PID is assigned to one destination rank
        PIDS_PER_RANK: gl.constexpr = COMM_SMS // world_size
        dest_rank_idx = pid // PIDS_PER_RANK
        pid_in_group = pid % PIDS_PER_RANK
        target_rank = rank_start + dest_rank_idx * rank_stride
        is_local = dest_rank_idx == group_rank

        # Pre-compute pointer translation ONCE per CU.
        # ctx.store() calls _translate() which does 2x gl.load(heap_bases) per
        # call. With row-by-row iteration that's 2 * BLOCK_SIZE_M loads per tile.
        # By hoisting, we do 2 loads total for all tiles on this CU.
        local_base = gl.load(ctx.heap_bases + iris_rank)
        target_base = gl.load(ctx.heap_bases + target_rank)
        # delta: add this to any local pointer to get the translated remote pointer
        ptr_delta = target_base - local_base

        num_pid_m = gl.cdiv(M, BLOCK_SIZE_M)
        num_pid_n = gl.cdiv(N, BLOCK_SIZE_N)
        total_tiles = num_pid_m * num_pid_n

        ELEMS_PER_THREAD: gl.constexpr = BLOCK_SIZE_N // (THREADS_PER_WARP * WARPS_PER_CTA)
        col_layout: gl.constexpr = gl.BlockedLayout([ELEMS_PER_THREAD], [THREADS_PER_WARP], [WARPS_PER_CTA], [0])

        for tile_id in range(pid_in_group, total_tiles, PIDS_PER_RANK):
            # Swizzled tile index computation for better L2 locality
            num_pid_in_group = GROUP_SIZE_M * num_pid_n
            group_id = tile_id // num_pid_in_group
            first_pid_m = group_id * GROUP_SIZE_M
            group_size_m = min(num_pid_m - first_pid_m, GROUP_SIZE_M)
            pid_m = first_pid_m + ((tile_id % num_pid_in_group) % group_size_m)
            pid_n = (tile_id % num_pid_in_group) // group_size_m

            rn = (pid_n * BLOCK_SIZE_N + gl.arange(0, BLOCK_SIZE_N, layout=col_layout)) % N
            rn = gl.max_contiguous(gl.multiple_of(rn, BLOCK_SIZE_N), BLOCK_SIZE_N)

            col_offsets_in = rn * stride_in_n
            col_offsets_out = rn * stride_out_n
            col_mask = rn < N

            rm_base = pid_m * BLOCK_SIZE_M

            for i in range(BLOCK_SIZE_M):
                row_idx = (rm_base + i) % M

                if row_idx < M:
                    # Load from local input
                    input_addr = input_ptr + row_idx * stride_in_m + col_offsets_in
                    if TRACING:
                        h_load = ctx.tracing.record_event_start(
                            event_id=events.load,
                            target_rank=group_rank,
                            address=input_addr,
                            pid_m=pid_m,
                            pid_n=pid_n,
                            mask=col_mask,
                        )
                    data = gl.load(input_addr, mask=col_mask)
                    if TRACING:
                        ctx.tracing.record_event_end(h_load)

                    # Output row: this rank's slice at group_rank * M
                    output_addr = output_ptr + (group_rank * M + row_idx) * stride_out_m + col_offsets_out

                    if TRACING:
                        h_store = ctx.tracing.record_event_start(
                            event_id=events.store,
                            target_rank=target_rank,
                            address=output_addr,
                            pid_m=pid_m,
                            pid_n=pid_n,
                            mask=col_mask,
                        )

                    if is_local:
                        # Local store: direct write
                        gl.store(output_addr, data, mask=col_mask, cache_modifier=".wt")
                    else:
                        # Remote store: use pre-computed delta instead of ctx.store()
                        # This avoids 2x gl.load(heap_bases) per store call
                        output_addr_int = tl.cast(output_addr, gl.uint64)
                        remote_addr_int = output_addr_int + ptr_delta
                        remote_addr = tl.cast(remote_addr_int, output_addr.dtype)
                        gl.store(remote_addr, data, mask=col_mask)

                    if TRACING:
                        ctx.tracing.record_event_end(h_store)



def all_gather(
    output_tensor,
    input_tensor,
    shmem,
    group=None,
    async_op=False,
    config=None,
):
    """
    Internal all-gather collective operation implementation.

    This function is called internally by shmem.ccl.all_gather().
    Users should use the Iris instance method instead:
        >>> shmem.ccl.all_gather(output_tensor, input_tensor)

    Each rank sends its input tensor to all ranks, and all ranks receive
    and concatenate all input tensors along dimension 0 (rows), matching
    torch.distributed.all_gather_into_tensor behavior.

    Args:
        output_tensor: Output tensor of shape (world_size * M, N) - will contain concatenated inputs
        input_tensor: Input tensor of shape (M, N) - local rank's data to send
        shmem: Iris shmem context
        group: ProcessGroup or None. If None, uses all ranks in `iris` context.
               Default: None.
        async_op: If False, performs a barrier at the end. If True, returns immediately.
                  Default: False.
        config: Config instance with kernel parameters (default: None).
                If None, uses default Config values.
                Set config.all_gather_variant to choose variant: "persistent" or "partitioned"
    """
    # Use provided config or create default one
    if config is None:
        config = Config(block_size_m=32, block_size_n=64)

    # Extract group information
    # rank_in_group: position within the ProcessGroup (0, 1, 2, ...) - passed as group_rank to kernel
    # rank_global: global rank in iris context - passed as iris_rank to kernel for RMA operations
    rank_in_group, rank_global, world_size, rank_start, rank_stride = extract_group_info(group, shmem)

    M, N = input_tensor.shape[:2]
    expected_output_shape = (world_size * M, N)

    if output_tensor.shape[:2] != expected_output_shape:
        raise ValueError(
            f"Output tensor shape {output_tensor.shape[:2]} does not match expected shape {expected_output_shape}. "
            f"Expected (world_size * M, N) = ({world_size * M}, {N})"
        )

    stride_in_m, stride_in_n = input_tensor.stride(0), input_tensor.stride(1)
    stride_out_m, stride_out_n = output_tensor.stride(0), output_tensor.stride(1)

    # Choose between Triton and Gluon implementation
    if config.use_gluon and GLUON_AVAILABLE:
        # Check if shmem is Iris Gluon (has get_device_context method)
        if not hasattr(shmem, "get_device_context"):
            raise ValueError("use_gluon=True requires Iris Gluon context. Use iris.experimental.iris_gluon.iris()")

        # Validate BlockedLayout constraints.
        # The gluon kernel distributes BLOCK_SIZE_N elements across the thread
        # hierarchy: ELEMS_PER_THREAD * THREADS_PER_WARP * WARPS_PER_CTA = BLOCK_SIZE_N.
        # ELEMS_PER_THREAD controls vector load width (4 = dwordx4 for fp16, optimal).
        threads_per_cta = config.threads_per_warp * config.num_warps
        if config.block_size_n < threads_per_cta:
            raise ValueError(
                f"Gluon all-gather requires block_size_n >= threads_per_warp * num_warps "
                f"({config.threads_per_warp} * {config.num_warps} = {threads_per_cta}), "
                f"got block_size_n={config.block_size_n}."
            )
        if config.block_size_n % threads_per_cta != 0:
            raise ValueError(
                f"Gluon all-gather requires block_size_n to be a multiple of "
                f"threads_per_warp * num_warps ({threads_per_cta}), "
                f"got block_size_n={config.block_size_n}. "
                f"This ensures each thread handles a whole number of elements. "
                f"Recommended: block_size_n=1024 with threads_per_warp=64, num_warps=4 "
                f"for dwordx4 vectorization (elems_per_thread=4)."
            )

        tracing_enabled = hasattr(shmem, "tracing") and shmem.tracing.enabled
        context_tensor = shmem.get_device_context()

        # Dispatch gluon variant
        if config.all_gather_variant == "partitioned":
            if config.comm_sms % world_size != 0:
                raise ValueError(
                    f"For gluon partitioned variant, COMM_SMS ({config.comm_sms}) must be "
                    f"divisible by world_size ({world_size})."
                )
            gluon_kernel = persistent_all_gather_gluon_partitioned
        elif config.all_gather_variant == "hoisted":
            gluon_kernel = persistent_all_gather_gluon_hoisted
        else:
            gluon_kernel = persistent_all_gather_gluon

        gluon_kernel[(config.comm_sms,)](
            IrisDeviceCtx,
            context_tensor,
            input_tensor,
            output_tensor,
            M,
            N,
            stride_in_m,
            stride_in_n,
            stride_out_m,
            stride_out_n,
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
            config.threads_per_warp,
            config.num_warps,
            tracing_enabled,
            num_stages=config.num_stages,
            num_warps=config.num_warps,
            waves_per_eu=config.waves_per_eu,
        )
    else:
        if config.use_gluon and not GLUON_AVAILABLE:
            raise ValueError("Gluon is not available. Install Triton with Gluon support or set use_gluon=False")

        # Validate COMM_SMS divisibility for partitioned variant
        if config.all_gather_variant == "partitioned" and config.comm_sms % world_size != 0:
            raise ValueError(
                f"For all_gather_variant='partitioned', COMM_SMS ({config.comm_sms}) must be divisible by world_size ({world_size}). "
                f"Please adjust config.comm_sms to be a multiple of {world_size}."
            )

        heap_bases = shmem.get_heap_bases()

        # Dispatch to the appropriate kernel based on variant
        if config.all_gather_variant == "persistent":
            kernel_fn = persistent_all_gather
        elif config.all_gather_variant == "partitioned":
            kernel_fn = persistent_all_gather_partitioned
        else:
            raise ValueError(f"Unknown all_gather_variant: {config.all_gather_variant}")

        kernel_fn[(config.comm_sms,)](
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
        )

    if not async_op:
        shmem.barrier()
