# SPDX-License-Identifier: MIT
# Copyright (c) 2025 Advanced Micro Devices, Inc. All rights reserved.

"""
All-to-all collective communication primitive for Iris.
"""

import triton
import triton.language as tl
import iris
from .config import Config


@triton.jit()
def chiplet_transform_chunked(
    pid, 
    num_workgroups: tl.constexpr, 
    num_xcds: tl.constexpr, 
    chunk_size: tl.constexpr
):
    if pid > (num_workgroups // (num_xcds * chunk_size)) * (num_xcds * chunk_size):
        # Outside of the contiguous chunked region, leave unchanged.
        return pid
    
    local_pid = pid // num_xcds 
    # Calculate chunk index and position within chunk
    chunk_idx = local_pid // chunk_size 
    pos_in_chunk = local_pid % chunk_size 

    # Calculate new PID
    xcd = pid % num_xcds 
    new_pid = chunk_idx * num_xcds * chunk_size + xcd * chunk_size + pos_in_chunk
    return new_pid

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
    """
    pid = tl.program_id(0)

    if NUM_XCDS != 1:
        pid = chiplet_transform_chunked(pid, NUM_XCDS, COMM_SMS, CHUNK_SIZE)
    
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

        # Compute row and column indices
        rm = (pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)) % M
        rn = (pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)) % N
        rm = tl.max_contiguous(tl.multiple_of(rm, BLOCK_SIZE_M), BLOCK_SIZE_M)
        rn = tl.max_contiguous(tl.multiple_of(rn, BLOCK_SIZE_N), BLOCK_SIZE_N)
        mask = (rm[:, None] < M) & (rn[None, :] < N)

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

        # Pre-compute constant parts that don't depend on target_rank
        # Base offset for input (without rank-specific column offset)
        input_base_offset = input_base_m + input_base_n
        # Remote store offset: write into target's output at columns [cur_rank*N : (cur_rank+1)*N]
        # This is constant for all target_rank iterations since it only depends on cur_rank
        output_offset_remote = output_base_m + (output_base_n + cur_rank * N * stride_out_n)
        output_ptr_remote = output_ptr + output_offset_remote
        output_ptr_remote = tl.multiple_of(output_ptr_remote, (BLOCK_SIZE_M, BLOCK_SIZE_N))
        
        # Pre-compute rank stride for input (N * stride_in_n)
        rank_stride_in = N * stride_in_n
        
        # Traffic shaping: Break each tile into 64x64 sub-blocks and process them
        # This creates better memory access patterns and allows hardware to distribute
        # traffic across XGMI links based on access patterns
        SUB_BLOCK_M: tl.constexpr = 64
        SUB_BLOCK_N: tl.constexpr = 64
        
        # Calculate number of 64x64 sub-blocks needed to cover the tile
        num_sub_blocks_m = tl.cdiv(BLOCK_SIZE_M, SUB_BLOCK_M)
        num_sub_blocks_n = tl.cdiv(BLOCK_SIZE_N, SUB_BLOCK_N)
        total_sub_blocks = num_sub_blocks_m * num_sub_blocks_n
        
        # Base row/column indices for the tile
        tile_base_m = pid_m * BLOCK_SIZE_M
        tile_base_n = pid_n * BLOCK_SIZE_N
        
        # Process all remote ranks: load each chunk and scatter to corresponding target
        # Each target_rank may have different input data, so we must load separately
        for target_rank in range(world_size):
            # Skip local rank as it's already processed above
            if target_rank != cur_rank:
                # Traffic shaping: Process tile in 64x64 sub-blocks
                # Loop over all sub-blocks to ensure complete coverage
                for sub_block_id in range(total_sub_blocks):
                    # Calculate sub-block position within the tile
                    sub_block_m = (sub_block_id // num_sub_blocks_n) * SUB_BLOCK_M
                    sub_block_n = (sub_block_id % num_sub_blocks_n) * SUB_BLOCK_N
                    
                    # Compute row and column indices for this 64x64 sub-block
                    # Start from tile base and add sub-block offset, then create arrays
                    sub_rm_base = tile_base_m + sub_block_m
                    sub_rn_base = tile_base_n + sub_block_n
                    sub_rm = sub_rm_base + tl.arange(0, SUB_BLOCK_M)
                    sub_rn = sub_rn_base + tl.arange(0, SUB_BLOCK_N)
                    
                    # Create mask for this sub-block
                    sub_mask = (sub_rm[:, None] < M) & (sub_rn[None, :] < N) & \
                               (sub_rm[:, None] < (tile_base_m + BLOCK_SIZE_M)) & \
                               (sub_rn[None, :] < (tile_base_n + BLOCK_SIZE_N))
                    
                    # Compute offsets for this sub-block
                    sub_input_base_m = sub_rm[:, None] * stride_in_m
                    sub_input_base_n = sub_rn[None, :] * stride_in_n
                    sub_output_base_m = sub_rm[:, None] * stride_out_m
                    sub_output_base_n = sub_rn[None, :] * stride_out_n
                    
                    # Compute input pointer for this target_rank's chunk (sub-block)
                    sub_input_offset = sub_input_base_m + (sub_input_base_n + target_rank * N * stride_in_n)
                    sub_input_ptr_send = input_ptr + sub_input_offset
                    sub_input_ptr_send = tl.multiple_of(sub_input_ptr_send, (SUB_BLOCK_M, SUB_BLOCK_N))
                    
                    # Compute output pointer (sub-block)
                    sub_output_offset = sub_output_base_m + (sub_output_base_n + cur_rank * N * stride_out_n)
                    sub_output_ptr_remote = output_ptr + sub_output_offset
                    sub_output_ptr_remote = tl.multiple_of(sub_output_ptr_remote, (SUB_BLOCK_M, SUB_BLOCK_N))
                    
                    # Load data chunk for this target rank (64x64 sub-block)
                    sub_data = tl.load(sub_input_ptr_send, mask=sub_mask)
                    
                    # Scatter to target rank's output
                    # Processing in 64x64 sub-blocks creates better memory access patterns
                    # that allow hardware to distribute traffic across XGMI links
                    iris.store(
                        sub_output_ptr_remote,
                        sub_data,
                        cur_rank,
                        target_rank,
                        heap_bases,
                        mask=sub_mask,
                    )


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
        shmem: Iris shmem context
        config: Config instance with kernel parameters (default: None).
                If None, uses default Config values.
        async_op: If False, performs a barrier at the end. If True, returns immediately.
                  Default: False.
    """
    # Use provided config or create default one
    if config is None:
        config = Config()
    
    rank = shmem.get_rank()
    world_size = shmem.get_num_ranks()
    
    M, total_N = input_tensor.shape[:2]
    N = total_N // world_size
    
    stride_in_m, stride_in_n = input_tensor.stride(0), input_tensor.stride(1)
    stride_out_m, stride_out_n = output_tensor.stride(0), output_tensor.stride(1)
    
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
