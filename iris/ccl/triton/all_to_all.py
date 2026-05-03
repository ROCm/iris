# SPDX-License-Identifier: MIT
# Copyright (c) 2025 Advanced Micro Devices, Inc. All rights reserved.

"""
Triton kernel for all-to-all collective communication.

Rank-parallel design: work items are (tile, target_rank) pairs distributed
across SMs, so different SMs send to different remote ranks simultaneously.
This avoids the original bottleneck of W-1 sequential iris.store per SM.
"""

import triton
import triton.language as tl
import iris
from iris.host.tracing.kernel_artifacts import iris_launch
from ..utils import chiplet_transform_chunked
from iris.ccl.utils import inline_device_barrier


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
    group_rank: tl.constexpr,
    iris_rank: tl.constexpr,
    world_size: tl.constexpr,
    rank_start: tl.constexpr,
    rank_stride: tl.constexpr,
    barrier_flags_ptr,
    wg_done_ptr,
    barrier_sense_ptr,
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
    GROUP_SIZE_M: tl.constexpr,
    COMM_SMS: tl.constexpr,
    NUM_XCDS: tl.constexpr,
    CHUNK_SIZE: tl.constexpr,
    INLINE_BARRIER: tl.constexpr = False,
):
    pid_raw = tl.program_id(0)
    pid = pid_raw
    if NUM_XCDS != 1:
        pid = chiplet_transform_chunked(pid_raw, COMM_SMS, NUM_XCDS, CHUNK_SIZE)

    num_pid_m = tl.cdiv(M, BLOCK_SIZE_M)
    num_pid_n = tl.cdiv(N, BLOCK_SIZE_N)
    total_tiles = num_pid_m * num_pid_n

    # Total work items: tiles * world_size (including local copy)
    total_work = total_tiles * world_size

    for work_id in range(pid, total_work, COMM_SMS):
        tile_id = work_id // world_size
        target_group_rank = work_id % world_size

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

        is_full = (rm_base + BLOCK_SIZE_M <= M) & (rn_base + BLOCK_SIZE_N <= N)

        input_base_m = rm[:, None] * stride_in_m
        output_base_m = rm[:, None] * stride_out_m
        input_base_n = rn[None, :] * stride_in_n
        output_base_n = rn[None, :] * stride_out_n

        input_offset = input_base_m + (input_base_n + target_group_rank * N * stride_in_n)
        output_offset = output_base_m + (output_base_n + group_rank * N * stride_out_n)

        in_ptr = input_ptr + input_offset
        out_ptr = output_ptr + output_offset
        in_ptr = tl.multiple_of(in_ptr, (BLOCK_SIZE_M, BLOCK_SIZE_N))
        out_ptr = tl.multiple_of(out_ptr, (BLOCK_SIZE_M, BLOCK_SIZE_N))

        if target_group_rank == group_rank:
            if is_full:
                data = tl.load(in_ptr)
                tl.store(out_ptr, data, cache_modifier=".wt")
            else:
                mask = (rm[:, None] < M) & (rn[None, :] < N)
                data = tl.load(in_ptr, mask=mask)
                tl.store(out_ptr, data, mask=mask, cache_modifier=".wt")
        else:
            target_rank = rank_start + target_group_rank * rank_stride
            if is_full:
                data = tl.load(in_ptr)
                iris.store(
                    out_ptr,
                    data,
                    iris_rank,
                    target_rank,
                    heap_bases,
                    hint=(1, BLOCK_SIZE_N),
                )
            else:
                mask = (rm[:, None] < M) & (rn[None, :] < N)
                data = tl.load(in_ptr, mask=mask)
                iris.store(
                    out_ptr,
                    data,
                    iris_rank,
                    target_rank,
                    heap_bases,
                    mask=mask,
                    hint=(1, BLOCK_SIZE_N),
                )

    if INLINE_BARRIER:
        inline_device_barrier(
            pid,
            barrier_flags_ptr,
            wg_done_ptr,
            barrier_sense_ptr,
            heap_bases,
            iris_rank,
            world_size,
            rank_start,
            rank_stride,
            COMM_SMS,
        )


_dummy_barrier_cache: dict = {}


def _get_dummy_barrier(device):
    """Return cached dummy barrier tensors for the no-inline-barrier path."""
    if device not in _dummy_barrier_cache:
        import torch

        _dummy_barrier_cache[device] = tuple(torch.zeros(1, dtype=torch.int32, device=device) for _ in range(3))
    return _dummy_barrier_cache[device]


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
    inline_barrier=False,
    barrier_state=None,
):
    """Launch the Triton all-to-all kernel."""
    M, total_N = input_tensor.shape[:2]
    N = total_N // world_size

    stride_in_m, stride_in_n = input_tensor.stride(0), input_tensor.stride(1)
    stride_out_m, stride_out_n = output_tensor.stride(0), output_tensor.stride(1)

    if inline_barrier and barrier_state is not None:
        barrier_flags, wg_done, barrier_sense = barrier_state
    else:
        barrier_flags, wg_done, barrier_sense = _get_dummy_barrier(input_tensor.device)

    iris_launch(
        persistent_all_to_all,
        (config.comm_sms,),
        input_tensor,
        output_tensor,
        M,
        N,
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
        barrier_flags,
        wg_done,
        barrier_sense,
        config.block_size_m,
        config.block_size_n,
        config.swizzle_size,
        config.comm_sms,
        config.num_xcds,
        config.chunk_size,
        inline_barrier,
        num_stages=config.num_stages,
        num_warps=config.num_warps,
        waves_per_eu=config.waves_per_eu,
        algorithm="all_to_all",
        rank=rank_global,
        dtype=input_tensor.dtype,
    )
