# SPDX-License-Identifier: MIT
# Copyright (c) 2025 Advanced Micro Devices, Inc. All rights reserved.

"""
Triton kernel for all-to-all collective communication.

Direct P2P writes via iris symmetric heap.
Direct P2P writes via iris symmetric heap.
where each rank sends a different chunk to every other rank using independent P2P
operations. For contention reduction on XGMI links, remote ranks are visited in
ring order (offset by group_rank) for contention avoidance
.

Additionally provides AllToAllv for variable-size per-rank chunks, matching
torch.distributed.all_to_all_single with split_sizes.
"""

import triton
import triton.language as tl
import iris
from iris.host.tracing.kernel_artifacts import iris_launch
from ..utils import chiplet_transform_chunked


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
    Persistent all-to-all kernel using direct P2P writes.

    Each rank reads its local input buffer
    and writes (iris.store) to remote ranks' output buffers. Remote ranks are
    visited in ring order (offset by group_rank) to reduce XGMI link contention,
    for contention avoidance.

    Data layout:
      input[M, N*world_size]:  input[:, i*N:(i+1)*N] -> data destined for rank i
      output[M, N*world_size]: output[:, i*N:(i+1)*N] <- data received from rank i

    Algorithm:
      For each rank r in [0, world_size):
        Send input[:, r*N:(r+1)*N] to rank r's output[:, group_rank*N:(group_rank+1)*N]

    Ring ordering for contention avoidance:
      Instead of iterating ranks 0,1,...,world_size-1, we iterate as:
        (group_rank+1)%ws, (group_rank+2)%ws, ..., group_rank (local last)
      This staggers traffic across XGMI links.
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
            # Ring ordering: visit remote ranks in staggered order
            # to spread traffic across XGMI links (for contention avoidance).
            # Local rank is handled first for cache locality.
            #
            # Process local rank first (direct copy, no iris RMA needed)
            input_offset_local = input_base_m + (input_base_n + group_rank * N * stride_in_n)
            output_offset_local = output_base_m + (output_base_n + group_rank * N * stride_out_n)
            input_ptr_local = input_ptr + input_offset_local
            output_ptr_local = output_ptr + output_offset_local
            input_ptr_local = tl.multiple_of(input_ptr_local, (BLOCK_SIZE_M, BLOCK_SIZE_N))
            output_ptr_local = tl.multiple_of(output_ptr_local, (BLOCK_SIZE_M, BLOCK_SIZE_N))

            data = tl.load(input_ptr_local)
            tl.store(output_ptr_local, data)

            # Process all remote ranks in ring order: (group_rank+1)%ws, (group_rank+2)%ws, ...
            # This staggered ordering reduces contention on XGMI links by ensuring
            # different ranks target different remote ranks at the same time.
            for hop in range(1, world_size):
                i = (group_rank + hop) % world_size
                target_rank = rank_start + i * rank_stride

                # Read chunk destined for rank i from local input
                input_offset_remote = input_base_m + (input_base_n + i * N * stride_in_n)
                # Write to rank i's output at position group_rank
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
            tl.store(output_ptr_local, data, mask=mask)

            # Process all remote ranks in ring order
            for hop in range(1, world_size):
                i = (group_rank + hop) % world_size
                target_rank = rank_start + i * rank_stride

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


@triton.jit()
def persistent_all_to_all_v(
    input_ptr,
    output_ptr,
    M,
    input_split_offsets,
    remote_output_offsets,
    input_split_sizes,
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
    Persistent all-to-all-v kernel for variable-size per-rank chunks.

    Extension of AllToAll where each rank can send/receive different amounts
    of data to/from each other rank. This matches torch.distributed.all_to_all
    with variable-size tensor lists.

    Data layout:
      input[M, total_input_cols]:  variable-width column slices, one per destination
      output[M, total_output_cols]: variable-width column slices, one per source

    input_split_offsets[world_size]: column offset for each destination in LOCAL input
    remote_output_offsets[world_size]: column offset at which rank i expects data from us
    input_split_sizes[world_size]: number of columns to send to each rank

    Uses ring ordering for XGMI contention avoidance, same as AllToAll.
    """
    pid = tl.program_id(0)

    if NUM_XCDS != 1:
        pid = chiplet_transform_chunked(pid, COMM_SMS, NUM_XCDS, CHUNK_SIZE)

    # Process each rank's chunk separately due to variable sizes.
    # For AllToAllv, we iterate over destination ranks and tile over
    # each rank's variable-size chunk independently.
    for hop in range(world_size):
        # Ring ordering for contention avoidance
        i = (group_rank + hop) % world_size
        target_rank = rank_start + i * rank_stride

        # Load the split sizes and offsets for this rank pair
        in_offset = tl.load(input_split_offsets + i)
        # remote_output_offsets[i] = offset in rank i's output buffer for data from us
        out_offset = tl.load(remote_output_offsets + i)
        N_send = tl.load(input_split_sizes + i)

        if N_send > 0:
            num_pid_m = tl.cdiv(M, BLOCK_SIZE_M)
            num_pid_n = tl.cdiv(N_send, BLOCK_SIZE_N)
            total_tiles = num_pid_m * num_pid_n

            for tile_id in range(pid, total_tiles, COMM_SMS):
                num_pid_in_group = GROUP_SIZE_M * num_pid_n
                group_id = tile_id // num_pid_in_group
                first_pid_m = group_id * GROUP_SIZE_M
                group_size_m = min(num_pid_m - first_pid_m, GROUP_SIZE_M)
                pid_m = first_pid_m + ((tile_id % num_pid_in_group) % group_size_m)
                pid_n = (tile_id % num_pid_in_group) // group_size_m

                rm_base = pid_m * BLOCK_SIZE_M
                rn_base = pid_n * BLOCK_SIZE_N

                rm = rm_base + tl.arange(0, BLOCK_SIZE_M)
                rn = rn_base + tl.arange(0, BLOCK_SIZE_N)

                mask = (rm[:, None] < M) & (rn[None, :] < N_send)

                # Input: read from local input at chunk for rank i
                in_ptrs = input_ptr + rm[:, None] * stride_in_m + (rn[None, :] + in_offset) * stride_in_n
                data = tl.load(in_ptrs, mask=mask)

                # Output: write to target rank's output at correct offset
                out_ptrs = output_ptr + rm[:, None] * stride_out_m + (rn[None, :] + out_offset) * stride_out_n

                if hop == 0:
                    # Local copy
                    tl.store(out_ptrs, data, mask=mask)
                else:
                    iris.store(
                        out_ptrs,
                        data,
                        iris_rank,
                        target_rank,
                        heap_bases,
                        mask=mask,
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
):
    """Launch the Triton all-to-all kernel."""
    M, total_N = input_tensor.shape[:2]
    N = total_N // world_size

    stride_in_m, stride_in_n = input_tensor.stride(0), input_tensor.stride(1)
    stride_out_m, stride_out_n = output_tensor.stride(0), output_tensor.stride(1)

    iris_launch(
        persistent_all_to_all,
        (config.comm_sms,),
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
        algorithm="all_to_all",
        rank=rank_global,
        dtype=input_tensor.dtype,
    )


def launch_v(
    input_tensor,
    output_tensor,
    input_split_sizes_tensor,
    input_split_offsets_tensor,
    remote_output_offsets_tensor,
    ctx,
    rank_in_group,
    rank_global,
    world_size,
    rank_start,
    rank_stride,
    config,
):
    """Launch the Triton all-to-all-v kernel.

    Args:
        remote_output_offsets_tensor: For each rank i, the column offset in rank i's
            output buffer where data from this rank should be written.
    """
    M = input_tensor.shape[0]

    stride_in_m, stride_in_n = input_tensor.stride(0), input_tensor.stride(1)
    stride_out_m, stride_out_n = output_tensor.stride(0), output_tensor.stride(1)

    iris_launch(
        persistent_all_to_all_v,
        (config.comm_sms,),
        input_tensor,
        output_tensor,
        M,
        input_split_offsets_tensor,
        remote_output_offsets_tensor,
        input_split_sizes_tensor,
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
        algorithm="all_to_all_v",
        rank=rank_global,
        dtype=input_tensor.dtype,
    )
