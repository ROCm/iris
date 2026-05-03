# SPDX-License-Identifier: MIT
# Copyright (c) 2025 Advanced Micro Devices, Inc. All rights reserved.

"""
Two-phase broadcast kernel for large messages.

Phase 1 (scatter): Each non-root rank pulls its assigned chunk from root via iris.load.
    All ranks read in parallel — distributes XGMI traffic across all links.
Phase 2 (all-gather): Each rank pushes its chunk to all other ranks via iris.store.

Single kernel launch with inline mid-barrier between phases.
"""

import triton
import triton.language as tl
import iris
from iris.host.tracing.kernel_artifacts import iris_launch
from iris.ccl.utils import inline_device_barrier
from ..utils import chiplet_transform_chunked


@triton.jit()
def persistent_broadcast_twophase(
    tensor_ptr,
    M,
    N,
    stride_m,
    stride_n,
    chunk_rows,
    heap_bases: tl.tensor,
    group_rank: tl.constexpr,
    iris_rank: tl.constexpr,
    src: tl.constexpr,
    world_size: tl.constexpr,
    rank_start: tl.constexpr,
    rank_stride: tl.constexpr,
    barrier_flags_ptr,
    wg_done_ptr,
    barrier_sense_ptr,
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
    COMM_SMS: tl.constexpr,
    NUM_XCDS: tl.constexpr,
    CHUNK_SIZE: tl.constexpr,
    INLINE_BARRIER: tl.constexpr = True,
):
    """Two-phase broadcast: parallel pull from root + all-gather."""
    pid = tl.program_id(0)

    if NUM_XCDS != 1:
        pid = chiplet_transform_chunked(pid, COMM_SMS, NUM_XCDS, CHUNK_SIZE)

    num_pid_n = tl.cdiv(N, BLOCK_SIZE_N)
    src_global = rank_start + src * rank_stride

    # ---- Phase 1: Each non-root rank pulls its chunk from root ----
    # Root already has its data — no work needed.
    # Non-root ranks read their assigned chunk from root in PARALLEL.
    if group_rank != src:
        my_row_start = group_rank * chunk_rows
        my_actual_rows = tl.minimum(chunk_rows, M - my_row_start)
        num_pid_m_chunk = tl.cdiv(my_actual_rows, BLOCK_SIZE_M)
        total_tiles = num_pid_m_chunk * num_pid_n

        for tile_id in range(pid, total_tiles, COMM_SMS):
            pid_m = tile_id // num_pid_n
            pid_n = tile_id % num_pid_n

            rm_base = my_row_start + pid_m * BLOCK_SIZE_M
            rn_base = pid_n * BLOCK_SIZE_N

            rm = rm_base + tl.arange(0, BLOCK_SIZE_M)
            rn = rn_base + tl.arange(0, BLOCK_SIZE_N)
            rm = tl.max_contiguous(tl.multiple_of(rm, BLOCK_SIZE_M), BLOCK_SIZE_M)
            rn = tl.max_contiguous(tl.multiple_of(rn, BLOCK_SIZE_N), BLOCK_SIZE_N)

            offset = rm[:, None] * stride_m + rn[None, :] * stride_n
            ptrs = tensor_ptr + offset

            is_full = (rm_base + BLOCK_SIZE_M <= M) & (rn_base + BLOCK_SIZE_N <= N)

            if is_full:
                data = iris.load(ptrs, iris_rank, src_global, heap_bases)
                tl.store(ptrs, data, cache_modifier=".wt")
            else:
                mask = (rm[:, None] < M) & (rn[None, :] < N)
                data = iris.load(ptrs, iris_rank, src_global, heap_bases, mask=mask)
                tl.store(ptrs, data, mask=mask, cache_modifier=".wt")

    # ---- Mid-barrier: all ranks have their chunk from root ----
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

    # ---- Phase 2: All-gather — each rank pushes its chunk to all others ----
    my_row_start = group_rank * chunk_rows
    my_actual_rows = tl.minimum(chunk_rows, M - my_row_start)
    num_pid_m_chunk = tl.cdiv(my_actual_rows, BLOCK_SIZE_M)
    total_tiles = num_pid_m_chunk * num_pid_n

    for tile_id in range(pid, total_tiles, COMM_SMS):
        pid_m = tile_id // num_pid_n
        pid_n = tile_id % num_pid_n

        rm_base = my_row_start + pid_m * BLOCK_SIZE_M
        rn_base = pid_n * BLOCK_SIZE_N

        rm = rm_base + tl.arange(0, BLOCK_SIZE_M)
        rn = rn_base + tl.arange(0, BLOCK_SIZE_N)
        rm = tl.max_contiguous(tl.multiple_of(rm, BLOCK_SIZE_M), BLOCK_SIZE_M)
        rn = tl.max_contiguous(tl.multiple_of(rn, BLOCK_SIZE_N), BLOCK_SIZE_N)

        offset = rm[:, None] * stride_m + rn[None, :] * stride_n
        ptrs = tensor_ptr + offset

        is_full = (rm_base + BLOCK_SIZE_M <= M) & (rn_base + BLOCK_SIZE_N <= N)

        if is_full:
            data = tl.load(ptrs)
            for dest_idx in tl.static_range(world_size):
                if dest_idx != group_rank:
                    dest_rank = rank_start + dest_idx * rank_stride
                    iris.store(ptrs, data, iris_rank, dest_rank, heap_bases, hint=(1, BLOCK_SIZE_N))
        else:
            mask = (rm[:, None] < M) & (rn[None, :] < N)
            data = tl.load(ptrs, mask=mask, other=0.0)
            for dest_idx in tl.static_range(world_size):
                if dest_idx != group_rank:
                    dest_rank = rank_start + dest_idx * rank_stride
                    iris.store(ptrs, data, iris_rank, dest_rank, heap_bases, mask=mask, hint=(1, BLOCK_SIZE_N))

    # ---- Post-barrier: all ranks have complete data ----
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
    tensor,
    ctx,
    rank_in_group,
    rank_global,
    world_size,
    rank_start,
    rank_stride,
    src,
    config,
    inline_barrier=True,
    barrier_state=None,
):
    """Launch two-phase broadcast kernel."""
    M, N = tensor.shape[:2]
    stride_m, stride_n = tensor.stride(0), tensor.stride(1)

    # Divide rows into W equal chunks
    chunk_rows = (M + world_size - 1) // world_size
    # Align to block_size_m
    chunk_rows = ((chunk_rows + config.block_size_m - 1) // config.block_size_m) * config.block_size_m

    heap_bases = ctx.get_heap_bases()

    if inline_barrier and barrier_state is not None:
        barrier_flags, wg_done, barrier_sense = barrier_state
    else:
        barrier_flags, wg_done, barrier_sense = _get_dummy_barrier(tensor.device)

    iris_launch(
        persistent_broadcast_twophase,
        (config.comm_sms,),
        tensor,
        M,
        N,
        stride_m,
        stride_n,
        chunk_rows,
        heap_bases,
        rank_in_group,
        rank_global,
        src,
        world_size,
        rank_start,
        rank_stride,
        barrier_flags,
        wg_done,
        barrier_sense,
        config.block_size_m,
        config.block_size_n,
        config.comm_sms,
        config.num_xcds,
        config.chunk_size,
        inline_barrier,
        num_stages=config.num_stages,
        num_warps=config.num_warps,
        waves_per_eu=config.waves_per_eu,
        algorithm="broadcast_twophase",
        rank=rank_global,
        dtype=tensor.dtype,
    )
