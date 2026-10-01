# SPDX-License-Identifier: MIT
# Copyright (c) 2025 Advanced Micro Devices, Inc. All rights reserved.

"""
Gluon kernel for all-to-all collective communication with traffic shaping.

This module is lazily imported only when config.use_gluon=True.
If gluon is not installed, the import itself raises ValueError.
"""

try:
    from triton.experimental import gluon
    from triton.experimental.gluon import language as gl
except ImportError as e:
    raise ValueError("Gluon is not available. Install Triton with Gluon support or set use_gluon=False.") from e

from iris.mem.gluon.context import Context as IrisDeviceCtx
from iris.host.tracing.kernel_artifacts import iris_launch


@gluon.jit
def chiplet_transform_chunked_gluon(
    pid, num_xcds: gl.constexpr, num_workgroups: gl.constexpr, chunk_size: gl.constexpr
):
    if pid > (num_workgroups // (num_xcds * chunk_size)) * (num_xcds * chunk_size):
        return pid

    local_pid = pid // num_xcds
    chunk_idx = local_pid // chunk_size
    pos_in_chunk = local_pid % chunk_size

    xcd = pid % num_xcds
    new_pid = chunk_idx * num_xcds * chunk_size + xcd * chunk_size + pos_in_chunk
    return new_pid


@gluon.jit
def persistent_all_to_all_gluon(
    IrisDeviceCtx: gl.constexpr,
    context_tensor,
    input_ptr,
    output_ptr,
    M,
    N,
    stride_in_m,
    stride_in_n,
    stride_out_m,
    stride_out_n,
    group_rank: gl.constexpr,
    iris_rank: gl.constexpr,
    world_size: gl.constexpr,
    rank_start: gl.constexpr,
    rank_stride: gl.constexpr,
    BLOCK_SIZE_M: gl.constexpr,
    BLOCK_SIZE_N: gl.constexpr,
    GROUP_SIZE_M: gl.constexpr,
    COMM_SMS: gl.constexpr,
    NUM_XCDS: gl.constexpr,
    CHUNK_SIZE: gl.constexpr,
):
    """
    Persistent all-to-all kernel using Gluon.

    Each rank sends input data to all ranks and receives data from all ranks.
    Simplified version that mirrors the Triton implementation.
    """
    ctx = IrisDeviceCtx.initialize(context_tensor)

    pid = gl.program_id(0)

    if NUM_XCDS != 1:
        pid = chiplet_transform_chunked_gluon(pid, NUM_XCDS, COMM_SMS, CHUNK_SIZE)

    num_pid_m = gl.cdiv(M, BLOCK_SIZE_M)
    num_pid_n = gl.cdiv(N, BLOCK_SIZE_N)
    total_tiles = num_pid_m * num_pid_n

    for tile_id in range(pid, total_tiles, COMM_SMS):
        num_pid_in_group = GROUP_SIZE_M * num_pid_n
        group_id = tile_id // num_pid_in_group
        first_pid_m = group_id * GROUP_SIZE_M
        group_size_m = min(num_pid_m - first_pid_m, GROUP_SIZE_M)
        pid_m = first_pid_m + ((tile_id % num_pid_in_group) % group_size_m)
        pid_n = (tile_id % num_pid_in_group) // group_size_m

        # Optimized layout for maximum VGPR usage and dwordx4 vectorization
        # Use layout that maximizes register utilization and enables wider loads
        # For AMD: 64 threads/warp, 4 warps = 256 threads total
        # BlockedLayout: [size_per_thread], [threads_per_warp], [warps_per_cta], [order]
        layout_col: gl.constexpr = gl.BlockedLayout([1], [64], [4], [0])  # Column access
        layout_row: gl.constexpr = gl.BlockedLayout([1], [64], [4], [0])  # Row indices

        rm = (pid_m * BLOCK_SIZE_M + gl.arange(0, BLOCK_SIZE_M, layout=layout_row)) % M
        rn = (pid_n * BLOCK_SIZE_N + gl.arange(0, BLOCK_SIZE_N, layout=layout_col)) % N
        # Strong hints for coalesced access and dwordx4
        rm = gl.max_contiguous(gl.multiple_of(rm, BLOCK_SIZE_M), BLOCK_SIZE_M)
        rn = gl.max_contiguous(gl.multiple_of(rn, BLOCK_SIZE_N), BLOCK_SIZE_N)

        # Pre-compute base offsets - maximize VGPR usage by keeping all offsets in registers
        row_offsets_m = rm * stride_in_m
        row_offsets_out_m = rm * stride_out_m
        col_offsets_n = rn * stride_in_n
        col_offsets_out_n = rn * stride_out_n

        # Process local rank - optimized access pattern for dwordx4
        # Process rows to maximize VGPR usage (BLOCK_SIZE_N elements per row)
        for i in range(BLOCK_SIZE_M):
            row_idx = (pid_m * BLOCK_SIZE_M + i) % M

            if row_idx < M:
                row_offset_m = row_idx * stride_in_m
                row_offset_out_m = row_idx * stride_out_m
                col_mask = rn < N

                # Compute offsets - compiler should see contiguous access pattern
                input_offset_local = row_offset_m + (col_offsets_n + group_rank * N * stride_in_n)
                output_offset_local = row_offset_out_m + (col_offsets_out_n + group_rank * N * stride_out_n)
                input_ptr_local = input_ptr + input_offset_local
                output_ptr_local = output_ptr + output_offset_local
                # Critical: multiple_of(4) enables dwordx4 for aligned fp16 access
                # This tells compiler that addresses are aligned to 4-element boundaries
                input_ptr_local = gl.multiple_of(input_ptr_local, 4)
                output_ptr_local = gl.multiple_of(output_ptr_local, 4)

                # Load/store - should generate dwordx4 for 4 consecutive fp16 elements
                data = gl.load(input_ptr_local, mask=col_mask)
                gl.store(output_ptr_local, data, mask=col_mask, cache_modifier=".wt")

        # Process remote ranks - same optimized pattern
        for rank_idx in range(world_size):
            target_rank = rank_start + rank_idx * rank_stride
            if rank_idx != group_rank:
                for i in range(BLOCK_SIZE_M):
                    row_idx = (pid_m * BLOCK_SIZE_M + i) % M

                    if row_idx < M:
                        row_offset_m = row_idx * stride_in_m
                        row_offset_out_m = row_idx * stride_out_m
                        col_mask = rn < N

                        # Use rank_idx for input chunk offset (based on position in group)
                        input_offset_remote = row_offset_m + (col_offsets_n + rank_idx * N * stride_in_n)
                        output_offset_remote = row_offset_out_m + (col_offsets_out_n + group_rank * N * stride_out_n)
                        input_ptr_remote = input_ptr + input_offset_remote
                        output_ptr_remote = output_ptr + output_offset_remote
                        # Strong hints for dwordx4
                        input_ptr_remote = gl.multiple_of(input_ptr_remote, 4)
                        output_ptr_remote = gl.multiple_of(output_ptr_remote, 4)

                        remote_data = gl.load(input_ptr_remote, mask=col_mask)
                        ctx.store(output_ptr_remote, remote_data, target_rank, mask=col_mask)


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
    """Launch the Gluon all-to-all kernel."""
    M, total_N = input_tensor.shape[:2]
    N = total_N // world_size

    stride_in_m, stride_in_n = input_tensor.stride(0), input_tensor.stride(1)
    stride_out_m, stride_out_n = output_tensor.stride(0), output_tensor.stride(1)

    context_tensor = ctx.get_device_context()

    iris_launch(
        persistent_all_to_all_gluon,
        (config.comm_sms,),
        IrisDeviceCtx,
        context_tensor,
        input_tensor,
        output_tensor,
        M,
        N,
        stride_in_m,
        stride_in_n,
        stride_out_m,
        stride_out_n,
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
