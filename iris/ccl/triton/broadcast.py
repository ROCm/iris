# SPDX-License-Identifier: MIT
# Copyright (c) 2025 Advanced Micro Devices, Inc. All rights reserved.

"""
Triton kernels for broadcast collective communication.

Pull-based broadcast using iris symmetric heap: all ranks read directly
from root's input buffer via XGMI, then store to their local output.
Root does a simple local copy.
"""

import triton
import triton.language as tl
import iris
from iris.host.tracing.kernel_artifacts import iris_launch


@triton.jit()
def broadcast_kernel(
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
    src_group_rank: tl.constexpr,
    src_iris_rank: tl.constexpr,
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
    GROUP_SIZE_M: tl.constexpr,
    COMM_SMS: tl.constexpr,
    NUM_XCDS: tl.constexpr,
    CHUNK_SIZE: tl.constexpr,
):
    """
    Pull-based broadcast kernel: all ranks read from root's input buffer.

    Pull-based broadcast: all ranks read from root's input buffer via XGMI.

    Root does a local copy. All other ranks use iris.load from root's
    input buffer, achieving parallel bandwidth across all XGMI links.

    Args:
        input_ptr: Pointer to input tensor (M, N) — only root's matters
        output_ptr: Pointer to output tensor (M, N) — receives root's data
        M, N: Tensor dimensions
        stride_in_m, stride_in_n: Input strides
        stride_out_m, stride_out_n: Output strides
        heap_bases: Heap base pointers for all ranks
        group_rank: This rank's position within the group
        iris_rank: This rank's global iris rank (for RMA)
        world_size: Number of ranks in the group
        rank_start, rank_stride: Group rank mapping
        src_group_rank: Broadcast root within the group
        src_iris_rank: Root's global iris rank
        BLOCK_SIZE_M, BLOCK_SIZE_N: Tile dimensions
        GROUP_SIZE_M: Swizzle group size
        COMM_SMS: Number of persistent kernel blocks
        NUM_XCDS, CHUNK_SIZE: Chiplet distribution params
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

        rm = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
        rn = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
        rm = tl.max_contiguous(tl.multiple_of(rm, BLOCK_SIZE_M), BLOCK_SIZE_M)
        rn = tl.max_contiguous(tl.multiple_of(rn, BLOCK_SIZE_N), BLOCK_SIZE_N)

        mask = (rm[:, None] < M) & (rn[None, :] < N)

        # Load from root's input buffer
        input_offset = rm[:, None] * stride_in_m + rn[None, :] * stride_in_n
        input_ptrs = input_ptr + input_offset
        input_ptrs = tl.multiple_of(input_ptrs, (BLOCK_SIZE_M, BLOCK_SIZE_N))

        if group_rank == src_group_rank:
            # Root: local load
            data = tl.load(input_ptrs, mask=mask, other=0.0)
        else:
            # Non-root: remote load from root via XGMI
            data = iris.load(
                input_ptrs,
                iris_rank,
                src_iris_rank,
                heap_bases,
                mask=mask,
                hint=(1, BLOCK_SIZE_N),
            )

        # Store to local output
        output_offset = rm[:, None] * stride_out_m + rn[None, :] * stride_out_n
        output_ptrs = output_ptr + output_offset
        output_ptrs = tl.multiple_of(output_ptrs, (BLOCK_SIZE_M, BLOCK_SIZE_N))
        tl.store(output_ptrs, data, mask=mask, cache_modifier=".wt")


def launch(
    input_tensor,
    output_tensor,
    ctx,
    rank_in_group,
    rank_global,
    world_size,
    rank_start,
    rank_stride,
    src_rank,
    config,
):
    """
    Launch the Triton broadcast kernel.

    Args:
        input_tensor: Input tensor (M, N). Only src rank's data matters.
        output_tensor: Output tensor (M, N). Will contain src's data on all ranks.
        ctx: Iris context.
        rank_in_group: This rank's position within the group.
        rank_global: This rank's global iris rank.
        world_size: Number of ranks in the group.
        rank_start: Starting global rank of the group.
        rank_stride: Stride between consecutive ranks.
        src_rank: Source rank within the group (broadcast root).
        config: Config with kernel parameters.
    """
    M, N = input_tensor.shape[:2]
    stride_in_m, stride_in_n = input_tensor.stride(0), input_tensor.stride(1)
    stride_out_m, stride_out_n = output_tensor.stride(0), output_tensor.stride(1)

    heap_bases = ctx.get_heap_bases()
    src_iris_rank = rank_start + src_rank * rank_stride

    iris_launch(
        broadcast_kernel,
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
        src_rank,
        src_iris_rank,
        config.block_size_m,
        config.block_size_n,
        config.swizzle_size,
        config.comm_sms,
        config.num_xcds,
        config.chunk_size,
        num_stages=config.num_stages,
        num_warps=config.num_warps,
        waves_per_eu=config.waves_per_eu,
        algorithm="broadcast",
        rank=rank_global,
        dtype=input_tensor.dtype,
    )
