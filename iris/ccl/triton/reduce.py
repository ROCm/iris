# SPDX-License-Identifier: MIT
# Copyright (c) 2025 Advanced Micro Devices, Inc. All rights reserved.

"""
Triton kernel for reduce collective communication.
Uses the one-shot approach: root rank gathers all partials via iris.load
and accumulates locally. Non-root ranks are no-ops.
"""

import triton
import triton.language as tl
import iris
from iris.host.tracing.kernel_artifacts import iris_launch
from ..utils import chiplet_transform_chunked


@triton.jit()
def persistent_reduce_one_shot(
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
    dst_rank: tl.constexpr,
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
    GROUP_SIZE_M: tl.constexpr,
    COMM_SMS: tl.constexpr,
    NUM_XCDS: tl.constexpr,
    CHUNK_SIZE: tl.constexpr,
):
    """
    One-shot reduce kernel.

    Only the root rank (dst_rank) does work: it gathers all partials from
    every rank via iris.load, accumulates them in float32, and stores the
    result to its local output tensor. Non-root ranks exit immediately.
    """
    # Non-root ranks have nothing to do
    if group_rank != dst_rank:
        return

    pid = tl.program_id(0)

    if NUM_XCDS != 1:
        pid = chiplet_transform_chunked(pid, COMM_SMS, NUM_XCDS, CHUNK_SIZE)

    num_pid_m = tl.cdiv(M, BLOCK_SIZE_M)
    num_pid_n = tl.cdiv(N, BLOCK_SIZE_N)
    total_tiles = num_pid_m * num_pid_n

    acc_dtype = tl.float32 if output_ptr.type.element_ty != tl.int8 else tl.int32

    for tile_id in range(pid, total_tiles, COMM_SMS):
        num_pid_in_group = GROUP_SIZE_M * num_pid_n
        group_id = tile_id // num_pid_in_group
        first_pid_m = group_id * GROUP_SIZE_M
        group_size_m = min(num_pid_m - first_pid_m, GROUP_SIZE_M)
        pid_m = first_pid_m + ((tile_id % num_pid_in_group) % group_size_m)
        pid_n = (tile_id % num_pid_in_group) // group_size_m

        tl.assume(pid_m >= 0)
        tl.assume(pid_n >= 0)

        rm_base = pid_m * BLOCK_SIZE_M
        rn_base = pid_n * BLOCK_SIZE_N
        rm = rm_base + tl.arange(0, BLOCK_SIZE_M)
        rn = rn_base + tl.arange(0, BLOCK_SIZE_N)
        rm = tl.max_contiguous(tl.multiple_of(rm, BLOCK_SIZE_M), BLOCK_SIZE_M)
        rn = tl.max_contiguous(tl.multiple_of(rn, BLOCK_SIZE_N), BLOCK_SIZE_N)
        mask = (rm[:, None] < M) & (rn[None, :] < N)

        input_offset = rm[:, None] * stride_in_m + rn[None, :] * stride_in_n
        output_offset = rm[:, None] * stride_out_m + rn[None, :] * stride_out_n

        acc = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=acc_dtype)

        for i in range(world_size):
            remote_rank = rank_start + i * rank_stride
            partial = iris.load(
                input_ptr + input_offset,
                iris_rank,
                remote_rank,
                heap_bases,
                mask=mask,
            )
            acc += partial.to(acc_dtype)

        tl.store(
            output_ptr + output_offset,
            acc.to(output_ptr.type.element_ty),
            mask=mask,
        )


def launch(
    output_tensor,
    input_tensor,
    ctx,
    rank_in_group,
    rank_global,
    world_size,
    rank_start,
    rank_stride,
    dst,
    config,
):
    """Launch the Triton reduce kernel."""
    M, N = input_tensor.shape[:2]
    stride_in_m, stride_in_n = input_tensor.stride(0), input_tensor.stride(1)
    stride_out_m, stride_out_n = output_tensor.stride(0), output_tensor.stride(1)

    heap_bases = ctx.get_heap_bases()

    iris_launch(
        persistent_reduce_one_shot,
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
        dst,
        config.block_size_m,
        config.block_size_n,
        config.swizzle_size,
        config.comm_sms,
        config.num_xcds,
        config.chunk_size,
        algorithm="reduce",
        rank=rank_global,
        dtype=input_tensor.dtype,
    )
