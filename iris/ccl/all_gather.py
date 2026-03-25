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

        # Gluon only supports the persistent variant
        if config.all_gather_variant != "persistent":
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

        context_tensor = shmem.get_device_context()

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
