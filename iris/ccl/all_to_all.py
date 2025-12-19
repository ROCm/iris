# SPDX-License-Identifier: MIT
# Copyright (c) 2025 Advanced Micro Devices, Inc. All rights reserved.

"""
All-to-all collective communication primitive for Iris.
Supports both Triton and Gluon implementations based on config.
"""

import triton
import triton.language as tl
import iris
from .config import Config
from .utils import chiplet_transform_chunked

# Conditional import for Gluon
try:
    from triton.experimental import gluon
    from triton.experimental.gluon import language as gl
    from iris.experimental.iris_gluon import IrisDeviceCtx

    GLUON_AVAILABLE = True
except ImportError:
    GLUON_AVAILABLE = False


@triton.jit()
def persistent_all_to_all(
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
    Persistent all-to-all kernel.

    Each rank sends input data to all ranks and receives data from all ranks.
    Similar to all-scatter but bidirectional.

    Args:
        input_ptr: Pointer to input tensor (local rank's data to send)
        output_ptr: Pointer to output tensor (will receive from all ranks)
        M: Number of rows
        N: Number of columns per rank (output will be N * world_size)
        stride_in_m, stride_in_n: Strides for input tensor
        stride_out_m, stride_out_n: Strides for output tensor
        heap_bases: Heap base pointers for all ranks
        cur_rank: Current rank
        world_size: Total number of ranks
        BLOCK_SIZE_M, BLOCK_SIZE_N: Block sizes for tiling
        GROUP_SIZE_M: Group size for M dimension tiling
        COMM_SMS: Number of SMs for communication
        NUM_XCDS: Number of XCDs
        CHUNK_SIZE: Chunk size for chiplet transform
    """
    pid = tl.program_id(0)

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

        # Compute base indices for this tile
        rm_base = pid_m * BLOCK_SIZE_M
        rn_base = pid_n * BLOCK_SIZE_N

        # Check if this tile is fully within bounds (no edge cases)
        is_full = (rm_base + BLOCK_SIZE_M <= M) & (rn_base + BLOCK_SIZE_N <= N)

        # Build indices (used by both paths)
        rm = rm_base + tl.arange(0, BLOCK_SIZE_M)
        rn = rn_base + tl.arange(0, BLOCK_SIZE_N)
        rm = tl.max_contiguous(tl.multiple_of(rm, BLOCK_SIZE_M), BLOCK_SIZE_M)
        rn = tl.max_contiguous(tl.multiple_of(rn, BLOCK_SIZE_N), BLOCK_SIZE_N)

        # Pre-compute base offsets for better memory access patterns and vectorization
        input_base_m = rm[:, None] * stride_in_m
        output_base_m = rm[:, None] * stride_out_m
        input_base_n = rn[None, :] * stride_in_n
        output_base_n = rn[None, :] * stride_out_n

        # Fast path: NO MASKS (full tiles)
        # The masking is problem size dependent, and the compiler does not recognize it can have two paths
        # (one with masks and one without). Separate unmasked paths allow the compiler to generate
        # more efficient vectorized instructions.
        if is_full:
            # Process local rank first for better cache locality
            input_offset_local = input_base_m + (input_base_n + cur_rank * N * stride_in_n)
            output_offset_local = output_base_m + (output_base_n + cur_rank * N * stride_out_n)
            input_ptr_local = input_ptr + input_offset_local
            output_ptr_local = output_ptr + output_offset_local
            input_ptr_local = tl.multiple_of(input_ptr_local, (BLOCK_SIZE_M, BLOCK_SIZE_N))
            output_ptr_local = tl.multiple_of(output_ptr_local, (BLOCK_SIZE_M, BLOCK_SIZE_N))

            data = tl.load(input_ptr_local)
            tl.store(output_ptr_local, data, cache_modifier=".wt")

            # Process all remote ranks
            for target_rank in range(world_size):
                if target_rank != cur_rank:
                    input_offset_remote = input_base_m + (input_base_n + target_rank * N * stride_in_n)
                    output_offset_remote = output_base_m + (output_base_n + cur_rank * N * stride_out_n)
                    input_ptr_remote = input_ptr + input_offset_remote
                    output_ptr_remote = output_ptr + output_offset_remote
                    input_ptr_remote = tl.multiple_of(input_ptr_remote, (BLOCK_SIZE_M, BLOCK_SIZE_N))
                    output_ptr_remote = tl.multiple_of(output_ptr_remote, (BLOCK_SIZE_M, BLOCK_SIZE_N))

                    remote_data = tl.load(input_ptr_remote)
                    iris.store(
                        output_ptr_remote,
                        remote_data,
                        cur_rank,
                        target_rank,
                        heap_bases,
                    )

        # Slow path: MASKED (only boundary tiles land here)
        # This path handles tiles at tensor boundaries where not all elements are valid.
        else:
            mask = (rm[:, None] < M) & (rn[None, :] < N)

            # Process local rank first for better cache locality
            input_offset_local = input_base_m + (input_base_n + cur_rank * N * stride_in_n)
            output_offset_local = output_base_m + (output_base_n + cur_rank * N * stride_out_n)
            input_ptr_local = input_ptr + input_offset_local
            output_ptr_local = output_ptr + output_offset_local
            input_ptr_local = tl.multiple_of(input_ptr_local, (BLOCK_SIZE_M, BLOCK_SIZE_N))
            output_ptr_local = tl.multiple_of(output_ptr_local, (BLOCK_SIZE_M, BLOCK_SIZE_N))

            data = tl.load(input_ptr_local, mask=mask)
            tl.store(output_ptr_local, data, mask=mask, cache_modifier=".wt")

            # Process all remote ranks
            for target_rank in range(world_size):
                if target_rank != cur_rank:
                    input_offset_remote = input_base_m + (input_base_n + target_rank * N * stride_in_n)
                    output_offset_remote = output_base_m + (output_base_n + cur_rank * N * stride_out_n)
                    input_ptr_remote = input_ptr + input_offset_remote
                    output_ptr_remote = output_ptr + output_offset_remote
                    input_ptr_remote = tl.multiple_of(input_ptr_remote, (BLOCK_SIZE_M, BLOCK_SIZE_N))
                    output_ptr_remote = tl.multiple_of(output_ptr_remote, (BLOCK_SIZE_M, BLOCK_SIZE_N))

                    remote_data = tl.load(input_ptr_remote, mask=mask)
                    iris.store(
                        output_ptr_remote,
                        remote_data,
                        cur_rank,
                        target_rank,
                        heap_bases,
                        mask=mask,
                    )


# Gluon implementation with traffic shaping based on micro-benchmark algorithm
if GLUON_AVAILABLE:

    @gluon.jit
    def chiplet_transform_chunked_gluon(
        pid, num_xcds: gl.constexpr, num_workgroups: gl.constexpr, chunk_size: gl.constexpr
    ):
        if pid > (num_workgroups // (num_xcds * chunk_size)) * (num_xcds * chunk_size):
            return pid

        local_pid = pid // num_xcds
        chunk_idx = local_pid // chunk_size
        pos_in_chunk = local_pid % chunk_size

        xcd = pid % num_xcds
        new_pid = chunk_idx * num_xcds * chunk_size + xcd * chunk_size + pos_in_chunk
        return new_pid

    @gluon.jit
    def persistent_all_to_all_gluon(
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
        cur_rank: gl.constexpr,
        world_size: gl.constexpr,
        BLOCK_SIZE_M: gl.constexpr,
        BLOCK_SIZE_N: gl.constexpr,
        GROUP_SIZE_M: gl.constexpr,
        COMM_SMS: gl.constexpr,
        NUM_XCDS: gl.constexpr,
        CHUNK_SIZE: gl.constexpr,
    ):
        """
        Persistent all-to-all kernel using Gluon.

        Each rank sends input data to all ranks and receives data from all ranks.
        Simplified version that mirrors the Triton implementation.
        """
        ctx = IrisDeviceCtx.initialize(context_tensor)

        pid = gl.program_id(0)

        if NUM_XCDS != 1:
            pid = chiplet_transform_chunked_gluon(pid, NUM_XCDS, COMM_SMS, CHUNK_SIZE)

        num_pid_m = gl.cdiv(M, BLOCK_SIZE_M)
        num_pid_n = gl.cdiv(N, BLOCK_SIZE_N)
        total_tiles = num_pid_m * num_pid_n

        for tile_id in range(pid, total_tiles, COMM_SMS):
            num_pid_in_group = GROUP_SIZE_M * num_pid_n
            group_id = tile_id // num_pid_in_group
            first_pid_m = group_id * GROUP_SIZE_M
            group_size_m = min(num_pid_m - first_pid_m, GROUP_SIZE_M)
            pid_m = first_pid_m + ((tile_id % num_pid_in_group) % group_size_m)
            pid_n = (tile_id % num_pid_in_group) // group_size_m

            # Optimized layout for maximum VGPR usage and dwordx4 vectorization
            # Use layout that maximizes register utilization and enables wider loads
            # For AMD: 64 threads/warp, 4 warps = 256 threads total
            # BlockedLayout: [size_per_thread], [threads_per_warp], [warps_per_cta], [order]
            layout_col: gl.constexpr = gl.BlockedLayout([1], [64], [4], [0])  # Column access
            layout_row: gl.constexpr = gl.BlockedLayout([1], [64], [4], [0])  # Row indices

            rm = (pid_m * BLOCK_SIZE_M + gl.arange(0, BLOCK_SIZE_M, layout=layout_row)) % M
            rn = (pid_n * BLOCK_SIZE_N + gl.arange(0, BLOCK_SIZE_N, layout=layout_col)) % N
            # Strong hints for coalesced access and dwordx4
            rm = gl.max_contiguous(gl.multiple_of(rm, BLOCK_SIZE_M), BLOCK_SIZE_M)
            rn = gl.max_contiguous(gl.multiple_of(rn, BLOCK_SIZE_N), BLOCK_SIZE_N)

            # Pre-compute base offsets - maximize VGPR usage by keeping all offsets in registers
            row_offsets_m = rm * stride_in_m
            row_offsets_out_m = rm * stride_out_m
            col_offsets_n = rn * stride_in_n
            col_offsets_out_n = rn * stride_out_n

            # Process local rank - optimized access pattern for dwordx4
            # Process rows to maximize VGPR usage (BLOCK_SIZE_N elements per row)
            for i in range(BLOCK_SIZE_M):
                row_idx = (pid_m * BLOCK_SIZE_M + i) % M

                if row_idx < M:
                    row_offset_m = row_idx * stride_in_m
                    row_offset_out_m = row_idx * stride_out_m
                    col_mask = rn < N

                    # Compute offsets - compiler should see contiguous access pattern
                    input_offset_local = row_offset_m + (col_offsets_n + cur_rank * N * stride_in_n)
                    output_offset_local = row_offset_out_m + (col_offsets_out_n + cur_rank * N * stride_out_n)
                    input_ptr_local = input_ptr + input_offset_local
                    output_ptr_local = output_ptr + output_offset_local
                    # Critical: multiple_of(4) enables dwordx4 for aligned fp16 access
                    # This tells compiler that addresses are aligned to 4-element boundaries
                    input_ptr_local = gl.multiple_of(input_ptr_local, 4)
                    output_ptr_local = gl.multiple_of(output_ptr_local, 4)

                    # Load/store - should generate dwordx4 for 4 consecutive fp16 elements
                    data = gl.load(input_ptr_local, mask=col_mask)
                    gl.store(output_ptr_local, data, mask=col_mask, cache_modifier=".wt")

            # Process remote ranks - same optimized pattern
            for target_rank in range(world_size):
                if target_rank != cur_rank:
                    for i in range(BLOCK_SIZE_M):
                        row_idx = (pid_m * BLOCK_SIZE_M + i) % M

                        if row_idx < M:
                            row_offset_m = row_idx * stride_in_m
                            row_offset_out_m = row_idx * stride_out_m
                            col_mask = rn < N

                            input_offset_remote = row_offset_m + (col_offsets_n + target_rank * N * stride_in_n)
                            output_offset_remote = row_offset_out_m + (col_offsets_out_n + cur_rank * N * stride_out_n)
                            input_ptr_remote = input_ptr + input_offset_remote
                            output_ptr_remote = output_ptr + output_offset_remote
                            # Strong hints for dwordx4
                            input_ptr_remote = gl.multiple_of(input_ptr_remote, 4)
                            output_ptr_remote = gl.multiple_of(output_ptr_remote, 4)

                            remote_data = gl.load(input_ptr_remote, mask=col_mask)
                            ctx.store(output_ptr_remote, remote_data, target_rank, mask=col_mask)


def all_to_all(output_tensor, input_tensor, shmem, config=None, async_op=False):
    """
    Internal all-to-all collective operation implementation.

    This function is called internally by shmem.ccl.all_to_all().
    Users should use the Iris instance method instead:
        >>> shmem.ccl.all_to_all(output_tensor, input_tensor)

    Each rank sends a tensor chunk to each other rank and receives
    a tensor chunk from each other rank. Input/output tensors should have
    shape (M, N * world_size) where each chunk of N columns corresponds to one rank.

    Args:
        output_tensor: Output tensor of shape (M, N * world_size)
        input_tensor: Input tensor of shape (M, N * world_size)
        shmem: Iris shmem context (regular Iris or Iris Gluon)
        config: Config instance with kernel parameters (default: None).
                If None, uses default Config values.
                Set config.use_gluon=True to use Gluon implementation with traffic shaping.
        async_op: If False, performs a barrier at the end. If True, returns immediately.
                  Default: False.
    """
    # Use provided config or create default one
    if config is None:
        config = Config(block_size_m=32, block_size_n=128)

    rank = shmem.get_rank()
    world_size = shmem.get_num_ranks()

    M, total_N = input_tensor.shape[:2]
    N = total_N // world_size

    stride_in_m, stride_in_n = input_tensor.stride(0), input_tensor.stride(1)
    stride_out_m, stride_out_n = output_tensor.stride(0), output_tensor.stride(1)

    # Choose between Triton and Gluon implementation
    if config.use_gluon and GLUON_AVAILABLE:
        # Check if shmem is Iris Gluon (has get_device_context method)
        if not hasattr(shmem, "get_device_context"):
            raise ValueError("use_gluon=True requires Iris Gluon context. Use iris.experimental.iris_gluon.iris()")

        context_tensor = shmem.get_device_context()

        persistent_all_to_all_gluon[(config.comm_sms,)](
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
            rank,
            world_size,
            config.block_size_m,
            config.block_size_n,
            config.swizzle_size,
            config.comm_sms,
            config.num_xcds,
            config.chunk_size,
        )
    else:
        # Use Triton implementation
        if config.use_gluon and not GLUON_AVAILABLE:
            raise ValueError("Gluon is not available. Install Triton with Gluon support or set use_gluon=False")

        persistent_all_to_all[(config.comm_sms,)](
            input_tensor,
            output_tensor,
            M,
            N,
            stride_in_m,
            stride_in_n,
            stride_out_m,
            stride_out_n,
            shmem.get_heap_bases(),
            rank,
            world_size,
            config.block_size_m,
            config.block_size_n,
            config.swizzle_size,
            config.comm_sms,
            config.num_xcds,
            config.chunk_size,
        )

    if not async_op:
        shmem.barrier()
