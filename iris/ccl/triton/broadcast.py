# SPDX-License-Identifier: MIT
# Copyright (c) 2025 Advanced Micro Devices, Inc. All rights reserved.

"""
Broadcast — Triton kernel.

Pull-based broadcast: all non-root ranks read from root's heap via
iris.load and store locally.  Parallelizes BW across all ranks instead
of serializing writes from root.  Pre-kernel barrier ensures root's
data is visible before reads begin.
"""

import triton
import triton.language as tl
import iris
from iris.host.tracing.kernel_artifacts import iris_launch


@triton.jit()
def persistent_broadcast_direct(
    tensor_ptr,
    M,
    N,
    stride_m,
    stride_n,
    heap_bases: tl.tensor,
    group_rank: tl.constexpr,
    iris_rank: tl.constexpr,
    src: tl.constexpr,
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
    """Pull-based broadcast — all non-root ranks read from root's heap."""
    pid = tl.program_id(0)

    if group_rank == src:
        return

    src_iris_rank = rank_start + src * rank_stride

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

        rm_base = pid_m * BLOCK_SIZE_M
        rn_base = pid_n * BLOCK_SIZE_N
        rm = rm_base + tl.arange(0, BLOCK_SIZE_M)
        rn = rn_base + tl.arange(0, BLOCK_SIZE_N)
        rm = tl.max_contiguous(tl.multiple_of(rm, BLOCK_SIZE_M), BLOCK_SIZE_M)
        rn = tl.max_contiguous(tl.multiple_of(rn, BLOCK_SIZE_N), BLOCK_SIZE_N)

        mask = (rm[:, None] < M) & (rn[None, :] < N)

        offset = rm[:, None] * stride_m + rn[None, :] * stride_n
        ptrs = tensor_ptr + offset

        data = iris.load(ptrs, iris_rank, src_iris_rank, heap_bases, mask=mask)
        tl.store(ptrs, data, mask=mask)


def launch(
    tensor,
    ctx,
    rank_in_group,
    rank_global,
    world_size,
    rank_start,
    rank_stride,
    src,
    config,
):
    """Launch the pull-based broadcast kernel."""
    M, N = tensor.shape[:2]
    stride_m, stride_n = tensor.stride(0), tensor.stride(1)

    heap_bases = ctx.get_heap_bases()

    iris_launch(
        persistent_broadcast_direct,
        (config.comm_sms,),
        tensor,
        M,
        N,
        stride_m,
        stride_n,
        heap_bases,
        rank_in_group,
        rank_global,
        src,
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
        algorithm="broadcast",
        rank=rank_global,
        dtype=tensor.dtype,
    )
