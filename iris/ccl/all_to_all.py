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
from .utils import chiplet_transform_chunked, extract_group_info

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
            input_offset_local = input_base_m + (input_base_n + group_rank * N * stride_in_n)
            output_offset_local = output_base_m + (output_base_n + group_rank * N * stride_out_n)
            input_ptr_local = input_ptr + input_offset_local
            output_ptr_local = output_ptr + output_offset_local
            input_ptr_local = tl.multiple_of(input_ptr_local, (BLOCK_SIZE_M, BLOCK_SIZE_N))
            output_ptr_local = tl.multiple_of(output_ptr_local, (BLOCK_SIZE_M, BLOCK_SIZE_N))

            data = tl.load(input_ptr_local)
            tl.store(output_ptr_local, data, cache_modifier=".wt")

            # Process remote ranks with staggered ordering to avoid write contention
            for step in range(world_size):
                i = (group_rank + step) % world_size
                target_rank = rank_start + i * rank_stride
                if i != group_rank:
                    input_offset_remote = input_base_m + (input_base_n + i * N * stride_in_n)
                    output_offset_remote = output_base_m + (output_base_n + group_rank * N * stride_out_n)
                    input_ptr_remote = input_ptr + input_offset_remote
                    output_ptr_remote = output_ptr + output_offset_remote
                    input_ptr_remote = tl.multiple_of(input_ptr_remote, (BLOCK_SIZE_M, BLOCK_SIZE_N))
                    output_ptr_remote = tl.multiple_of(output_ptr_remote, (BLOCK_SIZE_M, BLOCK_SIZE_N))

                    remote_data = tl.load(input_ptr_remote)
                    iris.store(
                        output_ptr_remote,
                        remote_data,
                        iris_rank,
                        target_rank,
                        heap_bases,
                        hint=(1, BLOCK_SIZE_N),
                    )

        # Slow path: MASKED (only boundary tiles land here)
        # This path handles tiles at tensor boundaries where not all elements are valid.
        else:
            mask = (rm[:, None] < M) & (rn[None, :] < N)

            # Process local rank first for better cache locality
            input_offset_local = input_base_m + (input_base_n + group_rank * N * stride_in_n)
            output_offset_local = output_base_m + (output_base_n + group_rank * N * stride_out_n)
            input_ptr_local = input_ptr + input_offset_local
            output_ptr_local = output_ptr + output_offset_local
            input_ptr_local = tl.multiple_of(input_ptr_local, (BLOCK_SIZE_M, BLOCK_SIZE_N))
            output_ptr_local = tl.multiple_of(output_ptr_local, (BLOCK_SIZE_M, BLOCK_SIZE_N))

            data = tl.load(input_ptr_local, mask=mask)
            tl.store(output_ptr_local, data, mask=mask, cache_modifier=".wt")

            # Process remote ranks with staggered ordering to avoid write contention
            for step in range(world_size):
                i = (group_rank + step) % world_size
                target_rank = rank_start + i * rank_stride
                if i != group_rank:
                    input_offset_remote = input_base_m + (input_base_n + i * N * stride_in_n)
                    output_offset_remote = output_base_m + (output_base_n + group_rank * N * stride_out_n)
                    input_ptr_remote = input_ptr + input_offset_remote
                    output_ptr_remote = output_ptr + output_offset_remote
                    input_ptr_remote = tl.multiple_of(input_ptr_remote, (BLOCK_SIZE_M, BLOCK_SIZE_N))
                    output_ptr_remote = tl.multiple_of(output_ptr_remote, (BLOCK_SIZE_M, BLOCK_SIZE_N))

                    remote_data = tl.load(input_ptr_remote, mask=mask)
                    iris.store(
                        output_ptr_remote,
                        remote_data,
                        iris_rank,
                        target_rank,
                        heap_bases,
                        mask=mask,
                        hint=(1, BLOCK_SIZE_N),
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
                    input_offset_local = row_offset_m + (col_offsets_n + group_rank * N * stride_in_n)
                    output_offset_local = row_offset_out_m + (col_offsets_out_n + group_rank * N * stride_out_n)
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
            for rank_idx in range(world_size):
                target_rank = rank_start + rank_idx * rank_stride
                if rank_idx != group_rank:
                    for i in range(BLOCK_SIZE_M):
                        row_idx = (pid_m * BLOCK_SIZE_M + i) % M

                        if row_idx < M:
                            row_offset_m = row_idx * stride_in_m
                            row_offset_out_m = row_idx * stride_out_m
                            col_mask = rn < N

                            # Use rank_idx for input chunk offset (based on position in group)
                            input_offset_remote = row_offset_m + (col_offsets_n + rank_idx * N * stride_in_n)
                            output_offset_remote = row_offset_out_m + (
                                col_offsets_out_n + group_rank * N * stride_out_n
                            )
                            input_ptr_remote = input_ptr + input_offset_remote
                            output_ptr_remote = output_ptr + output_offset_remote
                            # Strong hints for dwordx4
                            input_ptr_remote = gl.multiple_of(input_ptr_remote, 4)
                            output_ptr_remote = gl.multiple_of(output_ptr_remote, 4)

                            remote_data = gl.load(input_ptr_remote, mask=col_mask)
                            ctx.store(output_ptr_remote, remote_data, target_rank, mask=col_mask)


def all_to_all(
    output_tensor,
    input_tensor,
    ctx,
    group=None,
    async_op=False,
    config=None,
):
    """
    Internal all-to-all collective operation implementation.

    This function is called internally by ctx.ccl.all_to_all().
    Users should use the Iris instance method instead:
        >>> ctx.ccl.all_to_all(output_tensor, input_tensor)

    Each rank sends a tensor chunk to each other rank and receives
    a tensor chunk from each other rank. Input/output tensors should have
    shape (M, N * world_size) where each chunk of N columns corresponds to one rank.

    Args:
        output_tensor: Output tensor of shape (M, N * world_size)
        input_tensor: Input tensor of shape (M, N * world_size)
        ctx: Iris context (regular Iris or Iris Gluon)
        group: ProcessGroup or None. If None, uses all ranks in ctx.
               Default: None.
        async_op: If False, performs a barrier at the end. If True, returns immediately.
                  Default: False.
        config: Config instance with kernel parameters (default: None).
                If None, uses default Config values.
                Set config.use_gluon=True to use Gluon implementation with traffic shaping.
    """
    # Use provided config or create default one
    if config is None:
        config = Config(block_size_m=32, block_size_n=128)

    # Extract group information
    # rank_in_group: position within the ProcessGroup (0, 1, 2, ...) - passed as group_rank to kernel
    # rank_global: global rank in iris context - passed as iris_rank to kernel for RMA operations
    rank_in_group, rank_global, world_size, rank_start, rank_stride = extract_group_info(group, ctx)

    M, total_N = input_tensor.shape[:2]
    N = total_N // world_size

    stride_in_m, stride_in_n = input_tensor.stride(0), input_tensor.stride(1)
    stride_out_m, stride_out_n = output_tensor.stride(0), output_tensor.stride(1)

    # Choose between Triton and Gluon implementation
    if config.use_gluon and GLUON_AVAILABLE:
        # Check if ctx is Iris Gluon (has get_device_context method)
        if not hasattr(ctx, "get_device_context"):
            raise ValueError("use_gluon=True requires Iris Gluon context. Use iris.experimental.iris_gluon.iris()")

        context_tensor = ctx.get_device_context()

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
            ctx.get_heap_bases(),
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
        ctx.barrier()


@triton.jit()
def persistent_all_to_all_v(
    input_ptr,
    output_ptr,
    send_counts_ptr,
    send_displs_ptr,
    recv_counts_ptr,
    recv_displs_ptr,
    heap_bases: tl.tensor,
    group_rank: tl.constexpr,
    iris_rank: tl.constexpr,
    world_size: tl.constexpr,
    rank_start: tl.constexpr,
    rank_stride: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
    COMM_SMS: tl.constexpr,
    NUM_XCDS: tl.constexpr,
    CHUNK_SIZE: tl.constexpr,
):
    """
    Persistent all-to-all-v kernel for variable-sized exchanges.

    1D layout: input and output are flat buffers. send_counts[i] elements
    starting at send_displs[i] go to rank i. recv_counts[i] elements from
    rank i land at recv_displs[i] in the output.

    Each SM iterates over (rank, tile_within_chunk) pairs using staggered
    rank ordering to avoid write contention.
    """
    pid = tl.program_id(0)

    if NUM_XCDS != 1:
        pid = chiplet_transform_chunked(pid, COMM_SMS, NUM_XCDS, CHUNK_SIZE)

    # Process each peer rank with staggered ordering
    for step in range(world_size):
        i = (group_rank + step) % world_size
        target_rank = rank_start + i * rank_stride

        # Load counts and displacements for this peer
        send_count = tl.load(send_counts_ptr + i)
        send_displ = tl.load(send_displs_ptr + i)
        recv_displ = tl.load(recv_displs_ptr + i)

        num_tiles = tl.cdiv(send_count, BLOCK_SIZE)

        # Each SM handles a subset of tiles for this peer
        for tile_id in range(pid, num_tiles, COMM_SMS):
            offset = tile_id * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
            mask = offset < send_count

            # Load from local input at send_displ + offset
            data = tl.load(input_ptr + send_displ + offset, mask=mask)

            if i == group_rank:
                # Local copy: write to output at recv_displ
                tl.store(output_ptr + recv_displ + offset, data, mask=mask, cache_modifier=".wt")
            else:
                # Remote write: store to target rank's output at recv_displ for group_rank
                # Target rank's recv_displ for data from us is at recv_displs[group_rank] on target
                # We pre-compute this on the host and pass the right displacement
                iris.store(
                    output_ptr + recv_displ + offset,
                    data,
                    iris_rank,
                    target_rank,
                    heap_bases,
                    mask=mask,
                )


def all_to_all_v(
    output_tensor,
    input_tensor,
    send_counts,
    send_displs,
    recv_counts,
    recv_displs,
    ctx,
    group=None,
    async_op=False,
    config=None,
):
    """
    Variable-size all-to-all collective operation.

    Each rank sends send_counts[i] elements starting at send_displs[i] to rank i,
    and receives recv_counts[i] elements from rank i at recv_displs[i].

    Args:
        output_tensor: Output tensor (1D, flat) on symmetric heap.
        input_tensor: Input tensor (1D, flat) on symmetric heap.
        send_counts: list[int] of length world_size — elements to send to each rank.
        send_displs: list[int] of length world_size — element offsets in input for each rank.
        recv_counts: list[int] of length world_size — elements to receive from each rank.
        recv_displs: list[int] of length world_size — element offsets in output for each rank.
        ctx: Iris context.
        group: ProcessGroup or None.
        async_op: If False, barrier at end.
        config: Config instance.

    Note:
        Caller must ensure send_counts[i] on rank A == recv_counts[A] on rank i.
        iris does not validate this.
        Input and output tensors MUST be on the symmetric heap with identical
        allocation sizes across all ranks (symmetric heap invariant).
    """
    import torch

    if config is None:
        config = Config(block_size_m=32, block_size_n=128)

    rank_in_group, rank_global, world_size, rank_start, rank_stride = extract_group_info(group, ctx)

    device = input_tensor.device

    # Convert lists to device tensors for kernel access
    send_counts_t = torch.tensor(send_counts, dtype=torch.int64, device=device)
    send_displs_t = torch.tensor(send_displs, dtype=torch.int64, device=device)

    # For remote stores, we need the receiver's displacement for data from us.
    # recv_displs[i] on THIS rank is where data from rank i goes in OUR output.
    # When we write to rank j, we need rank j's recv_displs[group_rank].
    # We gather this via all_to_all on the displacements themselves.
    #
    # However, to keep this simple and avoid circular dependency, we require
    # the caller to pass recv_displs that are consistent across ranks:
    # recv_displs[i] on rank j == the offset in rank j's output for data from rank i.
    #
    # The kernel writes to the remote rank's output at the displacement that
    # the remote rank expects for data from us. We collect this info via
    # an all-to-all exchange of displacements.

    # For remote stores, we need each remote rank's recv_displ for data from us.
    # We gather all recv_displs via all_gather, then index into them.
    import torch.distributed as dist

    # All-gather recv_displs from all ranks into a [world_size, world_size] matrix.
    # all_recv_displs[j][i] = rank j's recv_displs[i] = where rank j stores data from rank i.
    local_recv_displs_t = torch.tensor(recv_displs, dtype=torch.int64, device=device)
    all_recv_displs_list = [torch.zeros(world_size, dtype=torch.int64, device=device) for _ in range(world_size)]
    dist.all_gather(all_recv_displs_list, local_recv_displs_t, group=group)

    # kernel_recv_displs[i] = rank i's recv_displs[group_rank] = where rank i stores data from us.
    kernel_recv_displs = torch.zeros(world_size, dtype=torch.int64, device=device)
    for i in range(world_size):
        kernel_recv_displs[i] = all_recv_displs_list[i][rank_in_group].item()
    kernel_recv_displs_t = kernel_recv_displs.to(device)

    block_size = config.block_size_n  # Use block_size_n as the 1D tile size

    persistent_all_to_all_v[(config.comm_sms,)](
        input_tensor,
        output_tensor,
        send_counts_t,
        send_displs_t,
        kernel_recv_displs_t,
        kernel_recv_displs_t,  # recv_displs not used separately in kernel
        ctx.get_heap_bases(),
        rank_in_group,
        rank_global,
        world_size,
        rank_start,
        rank_stride,
        block_size,
        config.comm_sms,
        config.num_xcds,
        config.chunk_size,
        num_stages=config.num_stages,
        num_warps=config.num_warps,
        waves_per_eu=config.waves_per_eu,
    )

    if not async_op:
        ctx.barrier()
