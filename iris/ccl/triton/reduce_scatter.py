# SPDX-License-Identifier: MIT
# Copyright (c) 2025 Advanced Micro Devices, Inc. All rights reserved.

"""
Triton kernel for reduce-scatter collective communication.
Two-shot variant: reduce assigned tiles and store only to own rank.
One-shot-fused variant: per-block fused barriers, no host coordination.
"""

import torch
import triton
import triton.language as tl
import iris
from iris.host.tracing.kernel_artifacts import iris_launch
from ..utils import chiplet_transform_chunked
from iris.ccl.utils import inline_device_barrier
from iris.ccl.triton.all_reduce import _per_block_barrier


@triton.jit()
def persistent_reduce_scatter_two_shot(
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
    DISTRIBUTION: tl.constexpr,
    INLINE_BARRIER: tl.constexpr = False,
):
    """
    Reduce-scatter using two-shot approach.

    Each rank reduces its assigned tiles from all ranks and stores the result
    only to its own output (no broadcast to other ranks).
    """
    pid = tl.program_id(0)

    if NUM_XCDS != 1:
        pid = chiplet_transform_chunked(pid, COMM_SMS, NUM_XCDS, CHUNK_SIZE)

    num_pid_m = tl.cdiv(M, BLOCK_SIZE_M)
    num_pid_n = tl.cdiv(N, BLOCK_SIZE_N)
    total_tiles = num_pid_m * num_pid_n

    acc_dtype = tl.float32 if output_ptr.type.element_ty != tl.int8 else tl.int32

    tiles_per_rank = tl.cdiv(total_tiles, world_size)
    if DISTRIBUTION == 0:
        start_tile = group_rank
        stride = world_size
        remaining = total_tiles - start_tile
        remaining = tl.maximum(remaining, 0)
        max_tile_offset = tl.cdiv(remaining, stride)
    else:
        start_tile = group_rank * tiles_per_rank
        stride = 1
        remaining = total_tiles - start_tile
        remaining = tl.maximum(remaining, 0)
        max_tile_offset = tl.minimum(tiles_per_rank, remaining)

    for tile_offset in range(pid, max_tile_offset, COMM_SMS):
        tile_id = start_tile + tile_offset * stride

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

        is_full = (rm_base + BLOCK_SIZE_M <= M) & (rn_base + BLOCK_SIZE_N <= N)

        rm = rm_base + tl.arange(0, BLOCK_SIZE_M)
        rn = rn_base + tl.arange(0, BLOCK_SIZE_N)

        rm = tl.max_contiguous(tl.multiple_of(rm, BLOCK_SIZE_M), BLOCK_SIZE_M)
        rn = tl.max_contiguous(tl.multiple_of(rn, BLOCK_SIZE_N), BLOCK_SIZE_N)

        input_offset = rm[:, None] * stride_in_m + rn[None, :] * stride_in_n
        base_ptr = input_ptr + input_offset

        if DISTRIBUTION == 0:
            out_rm_base = tile_offset * BLOCK_SIZE_M
        else:
            out_rm_base = (tile_id - start_tile) * BLOCK_SIZE_M
        out_rm = out_rm_base + tl.arange(0, BLOCK_SIZE_M)
        out_rm = tl.max_contiguous(tl.multiple_of(out_rm, BLOCK_SIZE_M), BLOCK_SIZE_M)
        output_offset = out_rm[:, None] * stride_out_m + rn[None, :] * stride_out_n
        out_ptr = output_ptr + output_offset

        if is_full:
            start_rank_idx = pid % world_size
            start_rank_global = rank_start + start_rank_idx * rank_stride
            acc = iris.load(base_ptr, iris_rank, start_rank_global, heap_bases).to(acc_dtype)
            for i in tl.static_range(1, world_size):
                remote_rank_idx = (start_rank_idx + i) % world_size
                remote_rank = rank_start + remote_rank_idx * rank_stride
                acc += iris.load(base_ptr, iris_rank, remote_rank, heap_bases).to(acc_dtype)

            reduced = acc.to(output_ptr.type.element_ty)

            tl.store(out_ptr, reduced, cache_modifier=".wt")

        else:
            mask = (rm[:, None] < M) & (rn[None, :] < N)

            start_rank_idx = pid % world_size
            start_rank_global = rank_start + start_rank_idx * rank_stride
            acc = iris.load(base_ptr, iris_rank, start_rank_global, heap_bases, mask=mask).to(acc_dtype)
            for i in tl.static_range(1, world_size):
                remote_rank_idx = (start_rank_idx + i) % world_size
                remote_rank = rank_start + remote_rank_idx * rank_stride
                acc += iris.load(base_ptr, iris_rank, remote_rank, heap_bases, mask=mask).to(acc_dtype)

            reduced = acc.to(output_ptr.type.element_ty)

            tl.store(out_ptr, reduced, mask=mask, cache_modifier=".wt")

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


@triton.jit()
def persistent_reduce_scatter_fused(
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
    start_flags_ptr,
    end_flags_ptr,
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
    GROUP_SIZE_M: tl.constexpr,
    COMM_SMS: tl.constexpr,
    NUM_XCDS: tl.constexpr,
    CHUNK_SIZE: tl.constexpr,
    DISTRIBUTION: tl.constexpr,
):
    """
    Reduce-scatter with fused per-block barriers.

    START BARRIER → each rank reduces only its assigned tiles → END BARRIER.
    Per-block monotonic flags — graph-capturable, no host coordination.
    """
    pid = tl.program_id(0)

    if NUM_XCDS != 1:
        pid = chiplet_transform_chunked(pid, COMM_SMS, NUM_XCDS, CHUNK_SIZE)

    # --- START BARRIER ---
    _per_block_barrier(
        pid,
        start_flags_ptr,
        heap_bases,
        group_rank,
        iris_rank,
        world_size,
        rank_start,
        rank_stride,
    )

    num_pid_m = tl.cdiv(M, BLOCK_SIZE_M)
    num_pid_n = tl.cdiv(N, BLOCK_SIZE_N)
    total_tiles = num_pid_m * num_pid_n

    acc_dtype = tl.float32 if output_ptr.type.element_ty != tl.int8 else tl.int32

    tiles_per_rank = tl.cdiv(total_tiles, world_size)
    if DISTRIBUTION == 0:
        start_tile = group_rank
        stride = world_size
        remaining = total_tiles - start_tile
        remaining = tl.maximum(remaining, 0)
        max_tile_offset = tl.cdiv(remaining, stride)
    else:
        start_tile = group_rank * tiles_per_rank
        stride = 1
        remaining = total_tiles - start_tile
        remaining = tl.maximum(remaining, 0)
        max_tile_offset = tl.minimum(tiles_per_rank, remaining)

    out_M = tl.cdiv(M, world_size) * BLOCK_SIZE_M
    num_out_pid_m = tl.cdiv(out_M, BLOCK_SIZE_M)

    for tile_offset in range(pid, max_tile_offset, COMM_SMS):
        tile_id = start_tile + tile_offset * stride

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

        input_offset = rm[:, None] * stride_in_m + rn[None, :] * stride_in_n
        base_ptr = input_ptr + input_offset

        # Output uses local offset within this rank's chunk
        if DISTRIBUTION == 0:
            out_rm_base = tile_offset * BLOCK_SIZE_M
        else:
            out_rm_base = (tile_id - start_tile) * BLOCK_SIZE_M
        out_rm = out_rm_base + tl.arange(0, BLOCK_SIZE_M)
        out_rm = tl.max_contiguous(tl.multiple_of(out_rm, BLOCK_SIZE_M), BLOCK_SIZE_M)
        output_offset = out_rm[:, None] * stride_out_m + rn[None, :] * stride_out_n
        out_ptr = output_ptr + output_offset

        is_full = (rm_base + BLOCK_SIZE_M <= M) & (rn_base + BLOCK_SIZE_N <= N)

        if is_full:
            start_rank_idx = pid % world_size
            start_rank_global = rank_start + start_rank_idx * rank_stride
            acc = iris.load(base_ptr, iris_rank, start_rank_global, heap_bases).to(acc_dtype)
            for i in tl.static_range(1, world_size):
                remote_rank_idx = (start_rank_idx + i) % world_size
                remote_rank = rank_start + remote_rank_idx * rank_stride
                acc += iris.load(base_ptr, iris_rank, remote_rank, heap_bases).to(acc_dtype)
            tl.store(out_ptr, acc.to(output_ptr.type.element_ty), cache_modifier=".wt")
        else:
            mask = (rm[:, None] < M) & (rn[None, :] < N)
            start_rank_idx = pid % world_size
            start_rank_global = rank_start + start_rank_idx * rank_stride
            acc = iris.load(base_ptr, iris_rank, start_rank_global, heap_bases, mask=mask).to(acc_dtype)
            for i in tl.static_range(1, world_size):
                remote_rank_idx = (start_rank_idx + i) % world_size
                remote_rank = rank_start + remote_rank_idx * rank_stride
                acc += iris.load(base_ptr, iris_rank, remote_rank, heap_bases, mask=mask).to(acc_dtype)
            tl.store(out_ptr, acc.to(output_ptr.type.element_ty), mask=mask, cache_modifier=".wt")

    # --- END BARRIER ---
    _per_block_barrier(
        pid,
        end_flags_ptr,
        heap_bases,
        group_rank,
        iris_rank,
        world_size,
        rank_start,
        rank_stride,
    )


_fused_flags_cache: dict = {}


def _get_fused_flags(ctx, comm_sms, group=None):
    """Get or create per-block start/end flags for fused reduce-scatter."""
    key = ("rs_fused", comm_sms, group)
    if key not in _fused_flags_cache:
        num_ranks = ctx.get_num_ranks()
        needed = comm_sms * num_ranks
        start_flags = ctx.zeros((needed,), dtype=torch.int32)
        end_flags = ctx.zeros((needed,), dtype=torch.int32)
        _fused_flags_cache[key] = (start_flags, end_flags)
    return _fused_flags_cache[key]


_dummy_barrier_cache: dict = {}


def _get_dummy_barrier(device):
    """Return cached dummy barrier tensors for the no-inline-barrier path."""
    if device not in _dummy_barrier_cache:
        _dummy_barrier_cache[device] = tuple(torch.zeros(1, dtype=torch.int32, device=device) for _ in range(3))
    return _dummy_barrier_cache[device]


def launch(
    output_tensor,
    input_tensor,
    ctx,
    rank_in_group,
    rank_global,
    world_size,
    rank_start,
    rank_stride,
    config,
    inline_barrier=False,
    barrier_state=None,
    variant="two_shot",
):
    """Launch the Triton reduce-scatter kernel."""
    M, N = input_tensor.shape[:2]
    stride_in_m, stride_in_n = input_tensor.stride(0), input_tensor.stride(1)
    stride_out_m, stride_out_n = output_tensor.stride(0), output_tensor.stride(1)

    heap_bases = ctx.get_heap_bases()
    distribution = config.all_reduce_distribution

    if variant == "fused":
        start_flags, end_flags = _get_fused_flags(ctx, config.comm_sms)
        num_pid_m = (M + config.block_size_m - 1) // config.block_size_m
        num_pid_n = (N + config.block_size_n - 1) // config.block_size_n
        total_tiles = num_pid_m * num_pid_n
        tiles_per_rank = (total_tiles + world_size - 1) // world_size
        fused_sms = min(tiles_per_rank, config.comm_sms)
        fused_sms = max(fused_sms, 1)
        iris_launch(
            persistent_reduce_scatter_fused,
            (fused_sms,),
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
            start_flags,
            end_flags,
            config.block_size_m,
            config.block_size_n,
            config.swizzle_size,
            fused_sms,
            config.num_xcds,
            config.chunk_size,
            distribution,
            algorithm="reduce_scatter",
            rank=rank_global,
            dtype=input_tensor.dtype,
        )
    else:
        if inline_barrier and barrier_state is not None:
            barrier_flags, wg_done, barrier_sense = barrier_state
        else:
            barrier_flags, wg_done, barrier_sense = _get_dummy_barrier(input_tensor.device)

        iris_launch(
            persistent_reduce_scatter_two_shot,
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
            barrier_flags,
            wg_done,
            barrier_sense,
            config.block_size_m,
            config.block_size_n,
            config.swizzle_size,
            config.comm_sms,
            config.num_xcds,
            config.chunk_size,
            distribution,
            inline_barrier,
            num_stages=config.num_stages,
            num_warps=config.num_warps,
            waves_per_eu=config.waves_per_eu,
            algorithm="reduce_scatter",
            rank=rank_global,
            dtype=input_tensor.dtype,
        )
