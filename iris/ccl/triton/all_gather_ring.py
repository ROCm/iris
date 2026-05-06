# SPDX-License-Identifier: MIT
# Copyright (c) 2025 Advanced Micro Devices, Inc. All rights reserved.

"""
Ring all-gather kernel with point-to-point synchronization.

Each rank starts with chunk_rows rows of data at output[group_rank * chunk_rows].
Output is (world_size * chunk_rows, N). After W-1 ring steps, every rank has all data.

Uses p2p step counters — same pattern as broadcast_ring.py.
"""

import triton
import triton.language as tl
import iris
from iris.host.tracing.kernel_artifacts import iris_launch
from iris.host.distributed.helpers import _translate_ptr
from ..utils import chiplet_transform_chunked


@triton.jit()
def _p2p_signal(
    step_flags_ptr,
    iris_rank: tl.constexpr,
    heap_bases: tl.tensor,
    COMM_SMS: tl.constexpr,
):
    pid = tl.program_id(0)
    tl.debug_barrier()
    if pid == 0:
        own_ptr = step_flags_ptr + iris_rank
        own_translated = _translate_ptr(own_ptr, iris_rank, iris_rank, heap_bases)
        tl.atomic_add(own_translated, 1, sem="release", scope="sys")


@triton.jit()
def _p2p_wait(
    step_flags_ptr,
    target,
    remote_iris_rank,
    iris_rank: tl.constexpr,
    heap_bases: tl.tensor,
):
    pid = tl.program_id(0)
    if pid == 0:
        remote_ptr = step_flags_ptr + remote_iris_rank
        remote_translated = _translate_ptr(remote_ptr, iris_rank, remote_iris_rank, heap_bases)
        while tl.atomic_cas(remote_translated, target, target, sem="acquire", scope="sys") < target:
            pass


@triton.jit()
def persistent_all_gather_ring(
    input_ptr,
    output_ptr,
    chunk_rows,
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
    step_flags_ptr,
    step_base,
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
    COMM_SMS: tl.constexpr,
    NUM_XCDS: tl.constexpr,
    CHUNK_SIZE: tl.constexpr,
):
    """
    Ring all-gather. Data is pre-reshaped so chunk_rows >> 1 for efficient tiling.

    Step 0: copy local input to output[group_rank * chunk_rows].
    Steps 1..W-1: pull chunk from predecessor's output, store locally.
    """
    pid = tl.program_id(0)

    if NUM_XCDS != 1:
        pid = chiplet_transform_chunked(pid, COMM_SMS, NUM_XCDS, CHUNK_SIZE)

    num_pid_m = tl.cdiv(chunk_rows, BLOCK_SIZE_M)
    num_pid_n = tl.cdiv(N, BLOCK_SIZE_N)
    total_tiles = num_pid_m * num_pid_n

    pred_group = (group_rank - 1 + world_size) % world_size
    pred_iris = rank_start + pred_group * rank_stride

    # Step 0: copy local input to output[group_rank * chunk_rows : (group_rank+1) * chunk_rows]
    out_row_base = group_rank * chunk_rows

    for tile_id in range(pid, total_tiles, COMM_SMS):
        pid_m = tile_id // num_pid_n
        pid_n = tile_id % num_pid_n

        rm_base = pid_m * BLOCK_SIZE_M
        rn_base = pid_n * BLOCK_SIZE_N

        rm = rm_base + tl.arange(0, BLOCK_SIZE_M)
        rn = rn_base + tl.arange(0, BLOCK_SIZE_N)
        rm = tl.max_contiguous(tl.multiple_of(rm, BLOCK_SIZE_M), BLOCK_SIZE_M)
        rn = tl.max_contiguous(tl.multiple_of(rn, BLOCK_SIZE_N), BLOCK_SIZE_N)

        in_offset = rm[:, None] * stride_in_m + rn[None, :] * stride_in_n
        rm_out = rm + out_row_base
        out_offset = rm_out[:, None] * stride_out_m + rn[None, :] * stride_out_n

        is_full = (rm_base + BLOCK_SIZE_M <= chunk_rows) & (rn_base + BLOCK_SIZE_N <= N)

        if is_full:
            data = tl.load(input_ptr + in_offset)
            tl.store(output_ptr + out_offset, data, cache_modifier=".wt")
        else:
            mask = (rm[:, None] < chunk_rows) & (rn[None, :] < N)
            data = tl.load(input_ptr + in_offset, mask=mask, other=0.0)
            tl.store(output_ptr + out_offset, data, mask=mask, cache_modifier=".wt")

    # Signal: local copy done
    _p2p_signal(step_flags_ptr, iris_rank, heap_bases, COMM_SMS)

    # Ring steps
    for step in tl.static_range(world_size - 1):
        chunk_owner = (group_rank - step - 1 + world_size) % world_size

        # Wait for predecessor to have this chunk
        target = step_base + step + 1
        _p2p_wait(step_flags_ptr, target, pred_iris, iris_rank, heap_bases)
        tl.debug_barrier()

        # Read chunk from predecessor's output, write to own output
        chunk_row_start = chunk_owner * chunk_rows
        for tile_id in range(pid, total_tiles, COMM_SMS):
            pid_m = tile_id // num_pid_n
            pid_n = tile_id % num_pid_n

            rm_base = pid_m * BLOCK_SIZE_M
            rn_base = pid_n * BLOCK_SIZE_N

            rm = rm_base + tl.arange(0, BLOCK_SIZE_M)
            rn = rn_base + tl.arange(0, BLOCK_SIZE_N)
            rm = tl.max_contiguous(tl.multiple_of(rm, BLOCK_SIZE_M), BLOCK_SIZE_M)
            rn = tl.max_contiguous(tl.multiple_of(rn, BLOCK_SIZE_N), BLOCK_SIZE_N)

            rm_out = rm + chunk_row_start
            out_offset = rm_out[:, None] * stride_out_m + rn[None, :] * stride_out_n
            out_ptrs = output_ptr + out_offset

            is_full = (rm_base + BLOCK_SIZE_M <= chunk_rows) & (rn_base + BLOCK_SIZE_N <= N)

            if is_full:
                data = iris.load(out_ptrs, iris_rank, pred_iris, heap_bases)
                tl.store(out_ptrs, data, cache_modifier=".wt")
            else:
                mask = (rm[:, None] < chunk_rows) & (rn[None, :] < N)
                data = iris.load(out_ptrs, iris_rank, pred_iris, heap_bases, mask=mask)
                tl.store(out_ptrs, data, mask=mask, cache_modifier=".wt")

        # Signal: step done
        _p2p_signal(step_flags_ptr, iris_rank, heap_bases, COMM_SMS)


_step_flags_cache: dict = {}


def _get_step_flags(ctx, group=None):
    key = ("all_gather_ring", group)
    if key not in _step_flags_cache:
        import torch
        _step_flags_cache[key] = ctx.zeros((ctx.num_ranks,), dtype=torch.int32)
        ctx.device_barrier(group)
    return _step_flags_cache[key]


_step_base_cache: dict = {}


def _advance_step_base(world_size, group=None):
    key = ("all_gather_ring", group)
    if key not in _step_base_cache:
        _step_base_cache[key] = 0
    old = _step_base_cache[key]
    _step_base_cache[key] = old + world_size
    return old


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
    group=None,
):
    """Launch ring all-gather kernel with proper 2D reshape."""
    numel_in = input_tensor.numel()
    numel_out = output_tensor.numel()

    block_n = config.block_size_n

    # Reshape input: flatten to (-1, block_n) for efficient tiling
    input_flat = input_tensor.contiguous().view(-1)
    if numel_in >= block_n:
        input_2d = input_flat.view(-1, block_n)
    else:
        input_2d = input_flat.view(1, -1)

    # Reshape output: flatten to (-1, block_n) matching
    output_flat = output_tensor.contiguous().view(-1)
    if numel_out >= block_n:
        output_2d = output_flat.view(-1, block_n)
    else:
        output_2d = output_flat.view(1, -1)

    M_in, N = input_2d.shape
    M_out = output_2d.shape[0]
    chunk_rows = M_in

    stride_in_m, stride_in_n = input_2d.stride(0), input_2d.stride(1)
    stride_out_m, stride_out_n = output_2d.stride(0), output_2d.stride(1)

    heap_bases = ctx.get_heap_bases()

    step_flags = _get_step_flags(ctx, group)
    step_base = _advance_step_base(world_size, group)

    iris_launch(
        persistent_all_gather_ring,
        (config.comm_sms,),
        input_2d,
        output_2d,
        chunk_rows,
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
        step_flags,
        step_base,
        config.block_size_m,
        config.block_size_n,
        config.comm_sms,
        config.num_xcds,
        config.chunk_size,
        num_stages=config.num_stages,
        num_warps=config.num_warps,
        waves_per_eu=config.waves_per_eu,
        algorithm="all_gather_ring",
        rank=rank_global,
        dtype=input_tensor.dtype,
    )
