# SPDX-License-Identifier: MIT
# Copyright (c) 2025 Advanced Micro Devices, Inc. All rights reserved.

"""
Triton kernel for scatter collective communication.
"""

import triton
import triton.language as tl
import iris
from iris.host.tracing.kernel_artifacts import iris_launch


@triton.jit()
def persistent_scatter(
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
    src: tl.constexpr,
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
    GROUP_SIZE_M: tl.constexpr,
    COMM_SMS: tl.constexpr,
    NUM_XCDS: tl.constexpr,
    CHUNK_SIZE: tl.constexpr,
):
    """
    Persistent scatter kernel.

    Only the root rank (src) does work. For each destination rank i, it loads
    chunk input[i*M:(i+1)*M, :] and writes to rank i's output tensor.
    Non-root ranks are idle (the kernel is still launched but does no work).

    Args:
        input_ptr: Pointer to input tensor on root (world_size * M, N), unused on non-root
        output_ptr: Pointer to output tensor (M, N)
        M: Number of rows per rank in output
        N: Number of columns
        stride_in_m, stride_in_n: Strides for input tensor
        stride_out_m, stride_out_n: Strides for output tensor
        heap_bases: Heap base pointers for all ranks
        group_rank: Rank within the ProcessGroup (0 to group_size-1)
        iris_rank: Rank in the iris context, used for iris RMA operations
        world_size: Total number of ranks in the group
        rank_start: First iris rank in the group
        rank_stride: Stride between consecutive iris ranks in the group
        src: Source (root) rank within the group
        BLOCK_SIZE_M, BLOCK_SIZE_N: Block sizes for tiling
        GROUP_SIZE_M: Group size for M dimension tiling
        COMM_SMS: Number of SMs for communication
        NUM_XCDS: Number of XCDs
        CHUNK_SIZE: Chunk size for chiplet transform
    """
    # Only root rank does work
    if group_rank != src:
        return

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

        # Compute output row and column indices (within M x N output)
        rm_base = pid_m * BLOCK_SIZE_M
        rn_base = pid_n * BLOCK_SIZE_N
        rm = rm_base + tl.arange(0, BLOCK_SIZE_M)
        rn = rn_base + tl.arange(0, BLOCK_SIZE_N)
        rm = tl.max_contiguous(tl.multiple_of(rm, BLOCK_SIZE_M), BLOCK_SIZE_M)
        rn = tl.max_contiguous(tl.multiple_of(rn, BLOCK_SIZE_N), BLOCK_SIZE_N)

        # Output mask (within M x N)
        output_mask = (rm[:, None] < M) & (rn[None, :] < N)

        # Output offset
        output_base_m = rm[:, None] * stride_out_m
        output_base_n = rn[None, :] * stride_out_n
        output_offset = output_base_m + output_base_n
        output_ptr_target = output_ptr + output_offset
        output_ptr_target = tl.multiple_of(output_ptr_target, (BLOCK_SIZE_M, BLOCK_SIZE_N))

        # Send chunk[i] to each destination rank i
        for i in tl.static_range(world_size):
            target_rank = rank_start + i * rank_stride

            # Input rows for rank i: input[i*M:(i+1)*M, :]
            rm_input = rm + i * M
            input_mask = (rm_input[:, None] < (i + 1) * M) & (rn[None, :] < N)
            combined_mask = output_mask & input_mask

            # Load chunk from input
            input_base_m = rm_input[:, None] * stride_in_m
            input_base_n = rn[None, :] * stride_in_n
            input_offset = input_base_m + input_base_n
            input_ptr_source = input_ptr + input_offset
            input_ptr_source = tl.multiple_of(input_ptr_source, (BLOCK_SIZE_M, BLOCK_SIZE_N))

            data = tl.load(input_ptr_source, mask=combined_mask, other=0.0)

            if i == src:
                # Local destination (root writes to its own output)
                tl.store(output_ptr_target, data, mask=combined_mask, cache_modifier=".wt")
            else:
                # Remote destination: use iris.store
                iris.store(
                    output_ptr_target,
                    data,
                    iris_rank,
                    target_rank,
                    heap_bases,
                    mask=combined_mask,
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
    src,
    config,
):
    """Launch the Triton scatter kernel."""
    M, N = output_tensor.shape[:2]
    stride_out_m, stride_out_n = output_tensor.stride(0), output_tensor.stride(1)

    # Input strides: only meaningful on root, but Triton specializes on stride values
    # so we must pass non-zero strides to avoid zero-stride specialization issues.
    # Non-root ranks pass output strides as a safe placeholder (kernel early-exits anyway).
    if rank_in_group == src:
        stride_in_m, stride_in_n = input_tensor.stride(0), input_tensor.stride(1)
    else:
        stride_in_m, stride_in_n = stride_out_m, stride_out_n

    heap_bases = ctx.get_heap_bases()

    iris_launch(
        persistent_scatter,
        (config.comm_sms,),
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
        src,
        config.block_size_m,
        config.block_size_n,
        config.swizzle_size,
        config.comm_sms,
        config.num_xcds,
        config.chunk_size,
        num_stages=config.num_stages,
        num_warps=config.num_warps,
        waves_per_eu=config.waves_per_eu,
        algorithm="scatter",
        rank=rank_global,
        dtype=output_tensor.dtype,
    )
