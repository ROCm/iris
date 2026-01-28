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
    gather_dim: tl.constexpr = 0,
):
    """
    All-gather for a single tile with configurable gather dimension.

    Each rank sends its input tile to all ranks, and all ranks receive
    and place it in the output at the appropriate location.

    Args:
        input_ptr: Pointer to input tensor (local rank's data to send)
        output_ptr: Pointer to output tensor (will receive from all ranks)
        pid_m: Tile coordinate in M dimension (for input)
        pid_n: Tile coordinate in N dimension (for input)
        M: Number of rows per rank in input
        N: Number of columns per rank in input
        stride_in_m, stride_in_n: Strides for input tensor
        stride_out_m, stride_out_n: Strides for output tensor
        heap_bases: Heap base pointers for all ranks
        cur_rank: Current rank
        world_size: Total number of ranks
        BLOCK_SIZE_M: Block size for M dimension
        BLOCK_SIZE_N: Block size for N dimension
        gather_dim: Dimension to gather along (0 for rows/M, 1 for columns/N)
    
    Gather dimension behavior:
        - gather_dim=0: Input (M, N) -> Output (world_size * M, N)
          Each rank's data goes to output[cur_rank * M : (cur_rank+1) * M, :]
        - gather_dim=1: Input (M, N) -> Output (M, world_size * N)
          Each rank's data goes to output[:, cur_rank * N : (cur_rank+1) * N]
    """
    tl.assume(pid_m >= 0)
    tl.assume(pid_n >= 0)

    # Compute tile indices and mask for input
    rm_input, rn_input, input_mask = compute_tile_indices(pid_m, pid_n, M, N, BLOCK_SIZE_M, BLOCK_SIZE_N)
    
    # Compute input offset
    input_base_m = rm_input[:, None] * stride_in_m
    input_base_n = rn_input[None, :] * stride_in_n
    input_offset = input_base_m + input_base_n
    input_ptr_source = input_ptr + input_offset
    input_ptr_source = tl.multiple_of(input_ptr_source, (BLOCK_SIZE_M, BLOCK_SIZE_N))
    
    # Load local input data once for this tile
    data = tl.load(input_ptr_source, mask=input_mask, other=0.0)

    # Gather dimension: 0 = rows (M), 1 = columns (N)
    # Compile-time branch for zero-overhead dimension selection
    if gather_dim == 0:
        # Gather along rows: output has shape (world_size * M, N)
        # Each rank's data goes to output[cur_rank * M : (cur_rank+1) * M, :]
        
        for rank in range(world_size):
            # Compute global output row indices: offset by cur_rank * M
            rm_output = rm_input + cur_rank * M
            rn_output = rn_input
            
            # Output mask: check bounds for output tensor (world_size * M rows, N cols)
            output_mask = (rm_output[:, None] < (world_size * M)) & (rn_output[None, :] < N)
            
            # Combine masks: must be valid in both input and output
            combined_mask = input_mask & output_mask

            # Compute output offset
            output_base_m = rm_output[:, None] * stride_out_m
            output_base_n = rn_output[None, :] * stride_out_n
            output_offset = output_base_m + output_base_n
            output_ptr_target = output_ptr + output_offset
            output_ptr_target = tl.multiple_of(output_ptr_target, (BLOCK_SIZE_M, BLOCK_SIZE_N))

            if rank == cur_rank:
                # Local destination: use direct store
                tl.store(output_ptr_target, data, mask=combined_mask, cache_modifier=".wt")
            else:
                # Remote destination: use iris.put
                iris.put(
                    input_ptr_source,
                    output_ptr_target,
                    cur_rank,
                    rank,
                    heap_bases,
                    mask=combined_mask,
                )
    else:
        # Gather along columns: output has shape (M, world_size * N)
        # Each rank's data goes to output[:, cur_rank * N : (cur_rank+1) * N]
        
        for rank in range(world_size):
            # Compute global output column indices: offset by cur_rank * N
            rm_output = rm_input
            rn_output = rn_input + cur_rank * N
            
            # Output mask: check bounds for output tensor (M rows, world_size * N cols)
            output_mask = (rm_output[:, None] < M) & (rn_output[None, :] < (world_size * N))
            
            # Combine masks: must be valid in both input and output
            combined_mask = input_mask & output_mask

            # Compute output offset
            output_base_m = rm_output[:, None] * stride_out_m
            output_base_n = rn_output[None, :] * stride_out_n
            output_offset = output_base_m + output_base_n
            output_ptr_target = output_ptr + output_offset
            output_ptr_target = tl.multiple_of(output_ptr_target, (BLOCK_SIZE_M, BLOCK_SIZE_N))

            if rank == cur_rank:
                # Local destination: use direct store
                tl.store(output_ptr_target, data, mask=combined_mask, cache_modifier=".wt")
            else:
                # Remote destination: use iris.put
                iris.put(
                    input_ptr_source,
                    output_ptr_target,
                    cur_rank,
                    rank,
                    heap_bases,
                    mask=combined_mask,
                )

