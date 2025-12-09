# SPDX-License-Identifier: MIT
# Copyright (c) 2025 Advanced Micro Devices, Inc. All rights reserved.

"""
Tile-level all-gather primitive for Iris.

Gathers tiles from all ranks and concatenates them along the output dimension.
"""

import triton
import triton.language as tl
import iris
from .common import compute_tile_indices, compute_tile_offsets


@triton.jit()
def all_gather(
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
    All-gather for a single tile.

    Each rank sends its input tile to all ranks, and all ranks receive
    and place it in the output at the appropriate location.
    For all-gather, output has shape (world_size * M, N) and each rank's
    input goes to output[cur_rank * M : (cur_rank + 1) * M, :].

    Args:
        input_ptr: Pointer to input tensor (local rank's data to send) of shape (M, N)
        output_ptr: Pointer to output tensor (will receive from all ranks) of shape (world_size * M, N)
        pid_m: Tile coordinate in M dimension (for input)
        pid_n: Tile coordinate in N dimension
        M: Number of rows per rank in input (output will be world_size * M rows)
        N: Number of columns
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

    # Compute tile indices and mask for input
    rm_input, rn, input_mask = compute_tile_indices(pid_m, pid_n, M, N, BLOCK_SIZE_M, BLOCK_SIZE_N)
    
    # Compute input offset
    input_base_m = rm_input[:, None] * stride_in_m
    input_base_n = rn[None, :] * stride_in_n
    input_offset = input_base_m + input_base_n
    input_ptr_source = input_ptr + input_offset
    input_ptr_source = tl.multiple_of(input_ptr_source, (BLOCK_SIZE_M, BLOCK_SIZE_N))
    
    # Load local input data once for this tile
    data = tl.load(input_ptr_source, mask=input_mask, other=0.0)

    # Send local shard data to all destination ranks
    # Each rank's input goes to output[cur_rank * M : (cur_rank + 1) * M, :] on all ranks
    for rank in range(world_size):
        # Compute global output row indices: offset by cur_rank * M
        # This rank's data should be placed at output[cur_rank * M : (cur_rank + 1) * M, :]
        rm_output = rm_input + cur_rank * M
        
        # Output mask: check bounds for output tensor (world_size * M rows, N cols)
        output_mask = (rm_output[:, None] < (world_size * M)) & (rn[None, :] < N)
        
        # Combine masks: must be valid in both input and output
        combined_mask = input_mask & output_mask

        # Compute output offset: write to output at rows [cur_rank * M : (cur_rank + 1) * M]
        # This is the same location on all destination ranks
        output_base_m = rm_output[:, None] * stride_out_m
        output_base_n = rn[None, :] * stride_out_n
        output_offset = output_base_m + output_base_n
        output_ptr_target = output_ptr + output_offset
        output_ptr_target = tl.multiple_of(output_ptr_target, (BLOCK_SIZE_M, BLOCK_SIZE_N))

        if rank == cur_rank:
            # Local destination: use direct store
            tl.store(output_ptr_target, data, mask=combined_mask, cache_modifier=".wt")
        else:
            # Remote destination: use iris.put to send from local source to remote destination
            iris.put(
                input_ptr_source,
                output_ptr_target,
                cur_rank,
                rank,
                heap_bases,
                mask=combined_mask,
            )

