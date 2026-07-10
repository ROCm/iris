# SPDX-License-Identifier: MIT
# Copyright (c) 2025 Advanced Micro Devices, Inc. All rights reserved.

"""
Triton kernel for all-to-all collective communication.

Direct P2P writes via iris symmetric heap.
Direct P2P writes via iris symmetric heap.
where each rank sends a different chunk to every other rank using independent P2P
operations. For contention reduction on XGMI links, remote ranks are visited in
ring order (offset by group_rank) for contention avoidance
.

Additionally provides AllToAllv for variable-size per-rank chunks, matching
torch.distributed.all_to_all_single with split_sizes.
"""

import triton
import triton.language as tl
import iris
from iris.host.tracing.kernel_artifacts import iris_launch
from ..utils import chiplet_transform_chunked


@triton.jit()
def persistent_all_to_all(
    input_ptr,
    output_ptr,
    N_PER_RANK,
    N_TOTAL,
    heap_bases: tl.tensor,
    group_rank: tl.constexpr,
    iris_rank: tl.constexpr,
    world_size: tl.constexpr,
    rank_start: tl.constexpr,
    rank_stride: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
    COMM_SMS: tl.constexpr,
):
    """All-to-all with flat 1D tiling. Minimal constexprs for clean codegen."""
    pid = tl.program_id(0)
    total_tiles = tl.cdiv(N_PER_RANK, BLOCK_SIZE)

    for tile_id in range(pid, total_tiles, COMM_SMS):
        base = tile_id * BLOCK_SIZE
        offsets = base + tl.arange(0, BLOCK_SIZE)
        mask = offsets < N_PER_RANK

        # Local copy
        local_in = input_ptr + group_rank * N_PER_RANK + offsets
        local_out = output_ptr + group_rank * N_PER_RANK + offsets
        data = tl.load(local_in, mask=mask)
        tl.store(local_out, data, mask=mask)

        # Remote ranks in ring order
        for hop in tl.static_range(1, world_size):
            i = (group_rank + hop) % world_size
            target_rank = rank_start + i * rank_stride

            remote_in = input_ptr + i * N_PER_RANK + offsets
            remote_out = output_ptr + group_rank * N_PER_RANK + offsets

            remote_data = tl.load(remote_in, mask=mask)
            iris.store(remote_out, remote_data, iris_rank, target_rank, heap_bases, mask=mask)


@triton.jit()
def persistent_all_to_all_v(
    input_ptr,
    output_ptr,
    M,
    input_split_offsets,
    remote_output_offsets,
    input_split_sizes,
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
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
    GROUP_SIZE_M: tl.constexpr,
    COMM_SMS: tl.constexpr,
    NUM_XCDS: tl.constexpr,
    CHUNK_SIZE: tl.constexpr,
):
    """
    Persistent all-to-all-v kernel for variable-size per-rank chunks.

    Extension of AllToAll where each rank can send/receive different amounts
    of data to/from each other rank. This matches torch.distributed.all_to_all
    with variable-size tensor lists.

    Data layout:
      input[M, total_input_cols]:  variable-width column slices, one per destination
      output[M, total_output_cols]: variable-width column slices, one per source

    input_split_offsets[world_size]: column offset for each destination in LOCAL input
    remote_output_offsets[world_size]: column offset at which rank i expects data from us
    input_split_sizes[world_size]: number of columns to send to each rank

    Uses ring ordering for XGMI contention avoidance, same as AllToAll.
    """
    pid = tl.program_id(0)

    if NUM_XCDS != 1:
        pid = chiplet_transform_chunked(pid, COMM_SMS, NUM_XCDS, CHUNK_SIZE)

    # Process each rank's chunk separately due to variable sizes.
    # For AllToAllv, we iterate over destination ranks and tile over
    # each rank's variable-size chunk independently.
    for hop in range(world_size):
        # Ring ordering for contention avoidance
        i = (group_rank + hop) % world_size
        target_rank = rank_start + i * rank_stride

        # Load the split sizes and offsets for this rank pair
        in_offset = tl.load(input_split_offsets + i)
        # remote_output_offsets[i] = offset in rank i's output buffer for data from us
        out_offset = tl.load(remote_output_offsets + i)
        N_send = tl.load(input_split_sizes + i)

        if N_send > 0:
            num_pid_m = tl.cdiv(M, BLOCK_SIZE_M)
            num_pid_n = tl.cdiv(N_send, BLOCK_SIZE_N)
            total_tiles = num_pid_m * num_pid_n

            for tile_id in range(pid, total_tiles, COMM_SMS):
                num_pid_in_group = GROUP_SIZE_M * num_pid_n
                group_id = tile_id // num_pid_in_group
                first_pid_m = group_id * GROUP_SIZE_M
                group_size_m = min(num_pid_m - first_pid_m, GROUP_SIZE_M)
                pid_m = first_pid_m + ((tile_id % num_pid_in_group) % group_size_m)
                pid_n = (tile_id % num_pid_in_group) // group_size_m

                rm_base = pid_m * BLOCK_SIZE_M
                rn_base = pid_n * BLOCK_SIZE_N

                rm = rm_base + tl.arange(0, BLOCK_SIZE_M)
                rn = rn_base + tl.arange(0, BLOCK_SIZE_N)

                mask = (rm[:, None] < M) & (rn[None, :] < N_send)

                # Input: read from local input at chunk for rank i
                in_ptrs = input_ptr + rm[:, None] * stride_in_m + (rn[None, :] + in_offset) * stride_in_n
                data = tl.load(in_ptrs, mask=mask)

                # Output: write to target rank's output at correct offset
                out_ptrs = output_ptr + rm[:, None] * stride_out_m + (rn[None, :] + out_offset) * stride_out_n

                if hop == 0:
                    # Local copy
                    tl.store(out_ptrs, data, mask=mask)
                else:
                    iris.store(
                        out_ptrs,
                        data,
                        iris_rank,
                        target_rank,
                        heap_bases,
                        mask=mask,
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
    config,
):
    """Launch the Triton all-to-all kernel."""
    numel = input_tensor.numel()
    n_per_rank = numel // world_size
    n_total = numel

    flat_input = input_tensor.contiguous().view(-1)
    flat_output = output_tensor.contiguous().view(-1)

    BLOCK_SIZE = 2048
    total_tiles = (n_per_rank + BLOCK_SIZE - 1) // BLOCK_SIZE
    num_sms = max(1, min(total_tiles, 16))

    iris_launch(
        persistent_all_to_all,
        (num_sms,),
        flat_input,
        flat_output,
        n_per_rank,
        n_total,
        ctx.get_heap_bases(),
        rank_in_group,
        rank_global,
        world_size,
        rank_start,
        rank_stride,
        BLOCK_SIZE,
        num_sms,
        num_warps=8,
        num_stages=1,
        waves_per_eu=1,
        algorithm="all_to_all",
        rank=rank_global,
        dtype=input_tensor.dtype,
    )


def launch_v(
    input_tensor,
    output_tensor,
    input_split_sizes_tensor,
    input_split_offsets_tensor,
    remote_output_offsets_tensor,
    ctx,
    rank_in_group,
    rank_global,
    world_size,
    rank_start,
    rank_stride,
    config,
):
    """Launch the Triton all-to-all-v kernel.

    Args:
        remote_output_offsets_tensor: For each rank i, the column offset in rank i's
            output buffer where data from this rank should be written.
    """
    M = input_tensor.shape[0]

    stride_in_m, stride_in_n = input_tensor.stride(0), input_tensor.stride(1)
    stride_out_m, stride_out_n = output_tensor.stride(0), output_tensor.stride(1)

    iris_launch(
        persistent_all_to_all_v,
        (config.comm_sms,),
        input_tensor,
        output_tensor,
        M,
        input_split_offsets_tensor,
        remote_output_offsets_tensor,
        input_split_sizes_tensor,
        stride_in_m,
        stride_in_n,
        stride_out_m,
        stride_out_n,
        ctx.get_heap_bases(),
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
        algorithm="all_to_all_v",
        rank=rank_global,
        dtype=input_tensor.dtype,
    )
