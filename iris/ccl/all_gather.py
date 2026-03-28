# SPDX-License-Identifier: MIT
# Copyright (c) 2025 Advanced Micro Devices, Inc. All rights reserved.

"""
All-gather collective communication primitive for Iris.
Gathers tensors from all ranks and concatenates them along the last dimension.
"""

from dataclasses import dataclass
from typing import Optional, Tuple

import torch
import triton
import triton.language as tl
import iris
from .config import Config
from .utils import chiplet_transform_chunked, extract_group_info


@dataclass
class AllGatherWorkspace:
    """
    Holds reusable workspace allocations for ring-based all-gather.

    Pre-allocate via ``all_gather_preamble`` and pass to ``all_gather``
    to avoid per-call heap allocation overhead.
    """

    shape: Tuple[int, int] = ()
    dtype: Optional[torch.dtype] = None
    flags: Optional[torch.Tensor] = None
    flags_per_tile: int = 0
    prepared: bool = False


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

    # Chiplet transform for XCD-aware scheduling
    if NUM_XCDS != 1:
        pid = chiplet_transform_chunked(pid, COMM_SMS, NUM_XCDS, CHUNK_SIZE)

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

        # Compute global output row indices: offset by group_rank * M
        rm_output = rm_input + group_rank * M
        output_mask = (rm_output[:, None] < (group_rank + 1) * M) & (rn[None, :] < N)
        combined_mask = input_mask & output_mask

        output_base_m = rm_output[:, None] * stride_out_m
        output_base_n = rn[None, :] * stride_out_n
        output_offset = output_base_m + output_base_n
        output_ptr_target = output_ptr + output_offset
        output_ptr_target = tl.multiple_of(output_ptr_target, (BLOCK_SIZE_M, BLOCK_SIZE_N))

        # Traffic-shaped stores: stagger write order per rank so each rank
        # writes to a different target at any given moment, avoiding memory
        # controller contention on the receiver side.
        for rank_idx in tl.static_range(world_size):
            dest_idx = (group_rank + rank_idx) % world_size
            target_rank = rank_start + dest_idx * rank_stride

            if dest_idx == group_rank:
                tl.store(output_ptr_target, data, mask=combined_mask, cache_modifier=".wt")
            else:
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
def persistent_all_gather_pull(
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
    Pull-based all-gather: each rank gathers from all others via iris.load.

    Instead of each rank pushing its data to all others (broadcast), each rank
    pulls data from every other rank's input buffer. This avoids receiver-side
    memory controller contention and is better for small-to-medium messages
    where latency dominates.
    """
    pid = tl.program_id(0)

    # Chiplet transform for XCD-aware scheduling
    if NUM_XCDS != 1:
        pid = chiplet_transform_chunked(pid, COMM_SMS, NUM_XCDS, CHUNK_SIZE)

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

        rm_base = pid_m * BLOCK_SIZE_M
        rn_base = pid_n * BLOCK_SIZE_N
        rm = rm_base + tl.arange(0, BLOCK_SIZE_M)
        rn = rn_base + tl.arange(0, BLOCK_SIZE_N)
        rm = tl.max_contiguous(tl.multiple_of(rm, BLOCK_SIZE_M), BLOCK_SIZE_M)
        rn = tl.max_contiguous(tl.multiple_of(rn, BLOCK_SIZE_N), BLOCK_SIZE_N)

        mask = (rm[:, None] < M) & (rn[None, :] < N)
        input_offset = rm[:, None] * stride_in_m + rn[None, :] * stride_in_n

        # Gather from all ranks: load each rank's input tile and write to
        # the corresponding slot in the local output buffer.
        for rank_idx in tl.static_range(world_size):
            # Stagger source rank order to distribute XGMI traffic
            src_idx = (group_rank + rank_idx) % world_size
            src_rank = rank_start + src_idx * rank_stride

            if src_idx == group_rank:
                # Local: load from own input buffer
                data = tl.load(input_ptr + input_offset, mask=mask, other=0.0)
            else:
                # Remote: pull from src_rank's input buffer via iris.load
                data = iris.load(
                    input_ptr + input_offset,
                    iris_rank,
                    src_rank,
                    heap_bases,
                    mask=mask,
                )

            # Write to output[src_idx * M + rm, rn]
            rm_out = rm + src_idx * M
            out_offset = rm_out[:, None] * stride_out_m + rn[None, :] * stride_out_n
            tl.store(output_ptr + out_offset, data, mask=mask, cache_modifier=".wt")


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

    # Chiplet transform for XCD-aware scheduling
    if NUM_XCDS != 1:
        pid = chiplet_transform_chunked(pid, COMM_SMS, NUM_XCDS, CHUNK_SIZE)

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


# Gluon implementation: flat-2D tiling approach
#
# Uses a single 1D arange over BLOCK_SIZE_M * BLOCK_SIZE_N elements with
# div/mod to compute 2D row/col indices. This gives one load + world_size
# stores per tile (matching Triton's 2D load/store structure) while staying
# within gluon's 1D BlockedLayout framework.
#
# Key optimizations:
#   - Flat-2D tiling: eliminates the inner BLOCK_SIZE_M row loop
#   - Hoisted pointer translation: local_base loaded once outside tile loop
#   - Traffic shaping: staggered write order avoids memory controller contention
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
        THREADS_PER_WARP: gl.constexpr,
        WARPS_PER_CTA: gl.constexpr,
    ):
        """
        Persistent all-gather kernel using Gluon with flat-2D tiling.

        Uses a flat 1D index space of BLOCK_SIZE_M * BLOCK_SIZE_N elements,
        computing 2D row/col via integer div/mod. This produces one vectorized
        load and world_size vectorized stores per tile, matching Triton's 2D
        load/store instruction structure while staying within gluon's 1D
        BlockedLayout framework.

        Memory layout (BlockedLayout):
            A 1D BlockedLayout distributes TOTAL_ELEMS = BLOCK_SIZE_M * BLOCK_SIZE_N
            elements across the thread hierarchy:
                ELEMS_PER_THREAD = TOTAL_ELEMS // (THREADS_PER_WARP * WARPS_PER_CTA)

            Each thread handles ELEMS_PER_THREAD contiguous elements in the
            flattened row-major order. Row/col are recovered via:
                row = flat_idx // BLOCK_SIZE_N
                col = flat_idx %  BLOCK_SIZE_N

        Constraints:
            - BLOCK_SIZE_M * BLOCK_SIZE_N must be a multiple of
              (THREADS_PER_WARP * WARPS_PER_CTA).
            - Optimal tile: 2048-4096 total elements (8-16 per thread).
              Larger tiles cause register spilling and performance collapse.
            - Recommended: BLOCK_SIZE_M=8, BLOCK_SIZE_N=256 (2048 elems, 8/thread).

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
            BLOCK_SIZE_N: Number of columns per tile.
            GROUP_SIZE_M: Swizzle group size for M-dimension tiling.
            COMM_SMS: Number of CUs used for persistent scheduling.
            THREADS_PER_WARP: Threads per warp/wavefront (64 for AMD, 32 for NVIDIA).
            WARPS_PER_CTA: Number of warps per workgroup. Must match num_warps.
        """
        ctx = IrisDeviceCtx.initialize(context_tensor, tracing=False)

        pid = gl.program_id(0)

        num_pid_m = gl.cdiv(M, BLOCK_SIZE_M)
        num_pid_n = gl.cdiv(N, BLOCK_SIZE_N)
        total_tiles = num_pid_m * num_pid_n

        # Flat 1D layout covering BLOCK_SIZE_M * BLOCK_SIZE_N elements
        TOTAL_ELEMS: gl.constexpr = BLOCK_SIZE_M * BLOCK_SIZE_N
        ELEMS_PER_THREAD: gl.constexpr = TOTAL_ELEMS // (THREADS_PER_WARP * WARPS_PER_CTA)
        flat_layout: gl.constexpr = gl.BlockedLayout([ELEMS_PER_THREAD], [THREADS_PER_WARP], [WARPS_PER_CTA], [0])

        # Hoist local heap base outside the tile loop: eliminates redundant
        # gl.load(heap_bases) calls in the inner store loop.
        local_base = gl.load(ctx.heap_bases + iris_rank)

        for tile_id in range(pid, total_tiles, COMM_SMS):
            # Swizzled tile index computation for better L2 locality
            num_pid_in_group = GROUP_SIZE_M * num_pid_n
            group_id = tile_id // num_pid_in_group
            first_pid_m = group_id * GROUP_SIZE_M
            group_size_m = min(num_pid_m - first_pid_m, GROUP_SIZE_M)
            pid_m = first_pid_m + ((tile_id % num_pid_in_group) % group_size_m)
            pid_n = (tile_id % num_pid_in_group) // group_size_m

            # Flat index -> 2D row/col within tile
            flat_idx = gl.arange(0, TOTAL_ELEMS, layout=flat_layout)
            row_local = flat_idx // BLOCK_SIZE_N
            col_local = flat_idx % BLOCK_SIZE_N

            # Global row/col
            row = pid_m * BLOCK_SIZE_M + row_local
            col = pid_n * BLOCK_SIZE_N + col_local

            mask = (row < M) & (col < N)

            # Single flat load of the entire tile
            input_offsets = row * stride_in_m + col * stride_in_n
            input_addr = input_ptr + input_offsets
            data = gl.load(input_addr, mask=mask, other=0.0)

            # Output: this rank's data goes to output[group_rank * M + row, col]
            output_row = group_rank * M + row
            output_offsets = output_row * stride_out_m + col * stride_out_n

            # Traffic-shaped stores to all ranks: stagger write order per rank
            # so each rank writes to a different target at any given moment,
            # avoiding memory controller contention on the receiver side.
            for rank_idx in range(world_size):
                dest_idx = (group_rank + rank_idx) % world_size
                target_iris_rank = rank_start + dest_idx * rank_stride
                output_ptrs = output_ptr + output_offsets

                if dest_idx == group_rank:
                    gl.store(output_ptrs, data, mask=mask, cache_modifier=".wt")
                else:
                    # Hoisted translation: compute ptr_delta from pre-loaded
                    # local_base rather than calling ctx.store() which would
                    # do 2x gl.load(heap_bases) per call.
                    target_base = gl.load(ctx.heap_bases + target_iris_rank)
                    ptr_delta = target_base - local_base
                    output_ptrs_int = tl.cast(output_ptrs, gl.uint64)
                    remote_ptrs_int = output_ptrs_int + ptr_delta
                    remote_ptrs = tl.cast(remote_ptrs_int, output_ptrs.dtype)
                    gl.store(remote_ptrs, data, mask=mask)


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
    rank_start: tl.constexpr,
    rank_stride: tl.constexpr,
    next_rank: tl.constexpr,
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
    GROUP_SIZE_M: tl.constexpr,
    COMM_SMS: tl.constexpr,
    NUM_XCDS: tl.constexpr,
    CHUNK_SIZE: tl.constexpr,
    NUM_RINGS: tl.constexpr,
    FLAGS_PER_TILE: tl.constexpr,
):
    """
    Ring-based all-gather with direct output writes.

    Key optimization: writes directly to the next rank's output buffer
    (no intermediate ring_buffer), matching RCCL's directRecvCopyDirectSend
    pattern. Each step writes the shard to output[src_rank * M] on the
    next rank, and reads from the local output buffer to forward.

    Supports NUM_RINGS concurrent rings and chiplet-aware scheduling.

    Flags layout: one int32 per tile on symmetric heap.
      flag=0 means output tile slot is free (producer can write)
      flag=1 means output tile slot has data (consumer can read)
    """
    pid_raw = tl.program_id(0)

    # Chiplet transform for XCD-aware scheduling
    pid = pid_raw
    if NUM_XCDS != 1:
        pid = chiplet_transform_chunked(pid_raw, COMM_SMS, NUM_XCDS, CHUNK_SIZE)

    tl.static_assert(NUM_RINGS > 0, "NUM_RINGS must be >= 1")
    tl.static_assert(FLAGS_PER_TILE >= 1, "FLAGS_PER_TILE must be at least 1")

    num_pid_m = tl.cdiv(M, BLOCK_SIZE_M)
    num_pid_n = tl.cdiv(N, BLOCK_SIZE_N)
    total_tiles = num_pid_m * num_pid_n

    # Partition CTAs across rings
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
                rn = rn_base + tl.arange(0, BLOCK_SIZE_N)
                rm = tl.max_contiguous(tl.multiple_of(rm, BLOCK_SIZE_M), BLOCK_SIZE_M)
                rn = tl.max_contiguous(tl.multiple_of(rn, BLOCK_SIZE_N), BLOCK_SIZE_N)

                mask = (rm[:, None] < M) & (rn[None, :] < N)

                # Load own shard from input
                input_tile_offset = rm[:, None] * stride_in_m + rn[None, :] * stride_in_n
                send_data = tl.load(input_ptr + input_tile_offset, mask=mask, other=0)

                # Write own shard to local output[group_rank * M]
                rm_local = rm + group_rank * M
                local_out_offset = rm_local[:, None] * stride_out_m + rn[None, :] * stride_out_n
                tl.store(output_ptr + local_out_offset, send_data, mask=mask, cache_modifier=".wt")

                # The data source rank for each step:
                # step 0: we send our own shard (group_rank)
                # step 1: we forward the shard we received (group_rank - 1)
                # step s: we forward shard from (group_rank - s) mod world_size

                for _step in range(0, world_size - 1):
                    # Compute which rank's shard we're sending this step
                    src_rank_idx = (group_rank + world_size - _step) % world_size

                    # === SEND PHASE ===
                    # Wait for next rank's flag to be 0 (output slot is free)
                    flag_offset = tile_id * FLAGS_PER_TILE
                    remote_flag_ptr = flags + flag_offset
                    local_flag_ptr = flags + flag_offset

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

                    # Write data directly to next rank's output[src_rank * M]
                    rm_remote = rm + src_rank_idx * M
                    remote_out_offset = rm_remote[:, None] * stride_out_m + rn[None, :] * stride_out_n
                    iris.store(
                        output_ptr + remote_out_offset,
                        send_data,
                        iris_rank,
                        next_rank,
                        heap_bases,
                        mask=mask,
                        cache_modifier=".wt",
                        hint=(1, BLOCK_SIZE_N),
                    )
                    tl.debug_barrier()
                    # Signal next rank: data is ready
                    iris.atomic_xchg(
                        remote_flag_ptr,
                        1,
                        iris_rank,
                        next_rank,
                        heap_bases,
                        sem="release",
                        scope="sys",
                    )

                    # === RECEIVE PHASE ===
                    # Wait for predecessor to write data to our output buffer
                    while tl.atomic_cas(local_flag_ptr, 0, 0, sem="acquire", scope="sys") != 1:
                        pass

                    # Read the received shard from local output buffer
                    # Predecessor wrote to output[recv_rank * M] on our rank
                    recv_rank_idx = (group_rank + world_size - _step - 1) % world_size
                    rm_recv = rm + recv_rank_idx * M
                    recv_out_offset = rm_recv[:, None] * stride_out_m + rn[None, :] * stride_out_n
                    recv_data = tl.load(output_ptr + recv_out_offset, mask=mask, other=0)

                    # Forward this data in the next step
                    send_data = recv_data

                    tl.debug_barrier()
                    # Reset local flag (slot is free for predecessor)
                    tl.atomic_xchg(local_flag_ptr, 0, sem="release", scope="sys")


def all_gather_preamble(
    output_tensor,
    input_tensor,
    ctx,
    config=None,
    workspace=None,
):
    """
    Pre-allocate reusable workspace for ring-based all-gather.

    Call once, then pass the returned workspace to ``all_gather`` on
    every iteration to avoid per-call symmetric-heap allocation.

    Args:
        output_tensor: Output tensor of shape (world_size * M, N).
        input_tensor: Input tensor of shape (M, N).
        ctx: Iris context.
        config: Config instance (default: None → default Config).
        workspace: Existing workspace to reuse (default: None → create new).

    Returns:
        AllGatherWorkspace ready for the next ``all_gather`` call.
    """
    if config is None:
        config = Config(block_size_m=32, block_size_n=64)

    M, N = input_tensor.shape[:2]
    dtype = input_tensor.dtype

    if workspace is None:
        workspace = AllGatherWorkspace()

    workspace.shape = (M, N)
    workspace.dtype = dtype
    workspace.prepared = False

    if config.all_gather_variant == "ring":
        # Direct-write ring: no ring_buffer needed, writes go to output buffer
        # Only need flags for per-tile producer/consumer handshake
        num_pid_m = (M + config.block_size_m - 1) // config.block_size_m
        num_pid_n = (N + config.block_size_n - 1) // config.block_size_n
        total_tiles = num_pid_m * num_pid_n
        workspace.flags_per_tile = 1
        total_flags = total_tiles * workspace.flags_per_tile
        if workspace.flags is None or workspace.flags.numel() != total_flags:
            workspace.flags = ctx.zeros((total_flags,), dtype=torch.int32)
        else:
            workspace.flags.zero_()

        ctx.barrier()

    workspace.prepared = True
    return workspace


def all_gather(
    output_tensor,
    input_tensor,
    ctx,
    group=None,
    async_op=False,
    config=None,
    workspace=None,
):
    """
    Internal all-gather collective operation implementation.

    This function is called internally by ctx.ccl.all_gather().
    Users should use the Iris instance method instead:
        >>> ctx.ccl.all_gather(output_tensor, input_tensor)

    Each rank sends its input tensor to all ranks, and all ranks receive
    and concatenate all input tensors along dimension 0 (rows), matching
    torch.distributed.all_gather_into_tensor behavior.

    Args:
        output_tensor: Output tensor of shape (world_size * M, N) - will contain concatenated inputs
        input_tensor: Input tensor of shape (M, N) - local rank's data to send
        ctx: Iris context
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
    rank_in_group, rank_global, world_size, rank_start, rank_stride = extract_group_info(group, ctx)

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
        # Check if ctx is Iris Gluon (has get_device_context method)
        if not hasattr(ctx, "get_device_context"):
            raise ValueError("use_gluon=True requires Iris Gluon context. Use iris.experimental.iris_gluon.iris()")

        # Gluon only supports the persistent variant
        if config.all_gather_variant not in ("persistent",):
            raise ValueError(
                f"Gluon all_gather only supports all_gather_variant='persistent', got '{config.all_gather_variant}'."
            )

        # Apply optimal defaults for gluon flat-2D kernel when user hasn't
        # overridden block sizes from the Config defaults (32x64).
        block_size_m = config.block_size_m
        block_size_n = config.block_size_n
        if block_size_m == 32 and block_size_n == 64:
            # User didn't override — use optimal flat-2D tile: 8x256
            block_size_m = 8
            block_size_n = 256

        # Validate flat-2D layout constraints.
        # TOTAL_ELEMS = BLOCK_SIZE_M * BLOCK_SIZE_N must be a multiple of
        # THREADS_PER_WARP * WARPS_PER_CTA so each thread gets a whole
        # number of elements.
        total_elems = block_size_m * block_size_n
        threads_per_cta = config.threads_per_warp * config.num_warps
        if total_elems < threads_per_cta:
            raise ValueError(
                f"Gluon all-gather requires block_size_m * block_size_n >= "
                f"threads_per_warp * num_warps ({threads_per_cta}), "
                f"got {block_size_m} * {block_size_n} = {total_elems}."
            )
        if total_elems % threads_per_cta != 0:
            raise ValueError(
                f"Gluon all-gather requires block_size_m * block_size_n to be a "
                f"multiple of threads_per_warp * num_warps ({threads_per_cta}), "
                f"got {block_size_m} * {block_size_n} = {total_elems}. "
                f"Recommended: block_size_m=8, block_size_n=256."
            )

        context_tensor = ctx.get_device_context()

        persistent_all_gather_gluon[(config.comm_sms,)](
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
            block_size_m,
            block_size_n,
            config.swizzle_size,
            config.comm_sms,
            config.threads_per_warp,
            config.num_warps,
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

        heap_bases = ctx.get_heap_bases()

        if config.all_gather_variant == "ring":
            # Ring variant: direct-write to output (no ring_buffer needed)
            if workspace is not None and workspace.prepared:
                flags = workspace.flags
                flags_per_tile = workspace.flags_per_tile
                workspace.prepared = False
            else:
                num_pid_m = (M + config.block_size_m - 1) // config.block_size_m
                num_pid_n = (N + config.block_size_n - 1) // config.block_size_n
                total_tiles = num_pid_m * num_pid_n
                flags_per_tile = 1
                total_flags = total_tiles * flags_per_tile
                flags = ctx.zeros((total_flags,), dtype=torch.int32)
                ctx.barrier()

            # Calculate next rank in the ring
            if group is None:
                next_rank = (rank_in_group + 1) % world_size
            else:
                import torch.distributed as dist

                group_ranks = dist.get_process_group_ranks(group)
                next_rank_in_group = (rank_in_group + 1) % world_size
                next_rank = group_ranks[next_rank_in_group]

            num_rings = config.all_gather_num_rings

            persistent_all_gather_ring[(config.comm_sms,)](
                input_tensor,
                output_tensor,
                flags,
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
                next_rank,
                config.block_size_m,
                config.block_size_n,
                config.swizzle_size,
                config.comm_sms,
                config.num_xcds,
                config.chunk_size,
                num_rings,
                flags_per_tile,
                num_stages=config.num_stages,
                num_warps=config.num_warps,
                waves_per_eu=config.waves_per_eu,
            )
        else:
            # Dispatch to the appropriate kernel based on variant
            if config.all_gather_variant == "persistent":
                kernel_fn = persistent_all_gather
            elif config.all_gather_variant == "partitioned":
                kernel_fn = persistent_all_gather_partitioned
            elif config.all_gather_variant == "pull":
                kernel_fn = persistent_all_gather_pull
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
        ctx.device_barrier(group=group)
