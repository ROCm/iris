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
        pid = (pid % NUM_XCDS) * (COMM_SMS // NUM_XCDS) + (pid // NUM_XCDS)
    
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

        # For each target rank, BSP-style exchange using only remote PUTs
        # Rank cur_rank sends input[cur_rank][target_rank] to rank target_rank
        # Remote rank writes into output[target_rank][cur_rank]
        for target_rank in range(world_size):
            # Send our chunk destined to target_rank from input at columns [target_rank*N : (target_rank+1)*N]
            input_offset_send = rm[:, None] * stride_in_m + (rn[None, :] + target_rank * N) * stride_in_n

            if target_rank == cur_rank:
                # Local path: copy input[cur_rank] chunk to output[cur_rank] chunk
                data = tl.load(input_ptr + input_offset_send, mask=mask)
                output_offset_local = rm[:, None] * stride_out_m + (rn[None, :] + cur_rank * N) * stride_out_n
                tl.store(output_ptr + output_offset_local, data, mask=mask, cache_modifier=".wt")
            else:
                # Remote PUT: write into target's output at columns [cur_rank*N : (cur_rank+1)*N]
                output_offset_remote = rm[:, None] * stride_out_m + (rn[None, :] + cur_rank * N) * stride_out_n
                iris.put(
                    input_ptr + input_offset_send,
                    output_ptr + output_offset_remote,
                    cur_rank,
                    target_rank,
                    heap_bases,
                    mask=mask,
                )


def all_to_all(output_tensor, input_tensor, shmem, config=None):
    """
    All-to-all collective operation.

    Each rank sends a tensor chunk to each other rank and receives
    a tensor chunk from each other rank. Input/output tensors should have
    shape (M, N * world_size) where each chunk of N columns corresponds to one rank.

    Args:
        output_tensor: Output tensor of shape (M, N * world_size)
        input_tensor: Input tensor of shape (M, N * world_size)
        shmem: Iris shmem context
        config: Config instance with kernel parameters (default: None).
                If None, uses default Config values.

    Example:
        >>> # Basic usage with default config
        >>> all_to_all(output_tensor, input_tensor, shmem)
        
        >>> # Custom configuration
        >>> from iris.ccl import Config
        >>> config = Config(
        ...     block_size_m=128,
        ...     block_size_n=32,
        ...     swizzle_size=8,
        ...     comm_sms=64
        ... )
        >>> all_to_all(output_tensor, input_tensor, shmem, config=config)
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
    )
