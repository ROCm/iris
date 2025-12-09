# SPDX-License-Identifier: MIT
# Copyright (c) 2025 Advanced Micro Devices, Inc. All rights reserved.

"""
Tile-level all-to-all primitive for Iris.

Each rank sends input data to all ranks and receives data from all ranks.
Similar to all-scatter but bidirectional.
"""

import triton
import triton.language as tl
import iris
from .common import compute_tile_indices, compute_tile_offsets


@triton.jit()
def all_to_all(
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
    All-to-all for a single tile.

    Each rank sends input data to all ranks and receives data from all ranks.
    Input/output tensors should have shape (M, N * world_size) where each chunk
    of N columns corresponds to one rank.

    Args:
        input_ptr: Pointer to input tensor (local rank's data to send) of shape (M, N * world_size)
        output_ptr: Pointer to output tensor (will receive from all ranks) of shape (M, N * world_size)
        pid_m: Tile coordinate in M dimension
        pid_n: Tile coordinate in N dimension (for the per-rank chunk)
        M: Number of rows
        N: Number of columns per rank (output will be N * world_size)
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

    # Pre-compute base offsets for better memory access patterns and vectorization
    # Base offset for input rows (M dimension)
    input_base_m = rm[:, None] * stride_in_m
    # Base offset for output rows (M dimension)
    output_base_m = rm[:, None] * stride_out_m
    # Base offset for input columns (N dimension) - will be adjusted per rank
    input_base_n = rn[None, :] * stride_in_n
    # Base offset for output columns (N dimension) - will be adjusted per rank
    output_base_n = rn[None, :] * stride_out_n

    # Process local rank first for better cache locality
    # Local path: copy input[cur_rank] chunk to output[cur_rank] chunk
    input_offset_local = input_base_m + (input_base_n + cur_rank * N * stride_in_n)
    output_offset_local = output_base_m + (output_base_n + cur_rank * N * stride_out_n)
    input_ptr_local = input_ptr + input_offset_local
    output_ptr_local = output_ptr + output_offset_local
    # Vectorization hints for 2D access pattern
    input_ptr_local = tl.multiple_of(input_ptr_local, (BLOCK_SIZE_M, BLOCK_SIZE_N))
    output_ptr_local = tl.multiple_of(output_ptr_local, (BLOCK_SIZE_M, BLOCK_SIZE_N))

    data = tl.load(input_ptr_local, mask=mask)
    tl.store(output_ptr_local, data, mask=mask, cache_modifier=".wt")

    # Process all remote ranks: load each chunk and scatter to corresponding target
    # Each target_rank may have different input data, so we must load separately
    for target_rank in range(world_size):
        if target_rank != cur_rank:
            # Compute input pointer for this target_rank's chunk
            input_offset = input_base_m + (input_base_n + target_rank * N * stride_in_n)
            input_ptr_send = input_ptr + input_offset
            input_ptr_send = tl.multiple_of(input_ptr_send, (BLOCK_SIZE_M, BLOCK_SIZE_N))

            # Compute output pointer (write into target's output at columns [cur_rank*N : (cur_rank+1)*N])
            output_offset = output_base_m + (output_base_n + cur_rank * N * stride_out_n)
            output_ptr_remote = output_ptr + output_offset
            output_ptr_remote = tl.multiple_of(output_ptr_remote, (BLOCK_SIZE_M, BLOCK_SIZE_N))

            # Load data chunk for this target rank
            data = tl.load(input_ptr_send, mask=mask)

            # Scatter to target rank's output
            iris.store(
                output_ptr_remote,
                data,
                cur_rank,
                target_rank,
                heap_bases,
                mask=mask,
            )

