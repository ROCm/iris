# SPDX-License-Identifier: MIT
# Copyright (c) 2025 Advanced Micro Devices, Inc. All rights reserved.

"""
Broadcast — Triton kernel.

Ring broadcast: data pipelines from root through the ring.  Root sends to
next, each rank receives from prev, stores locally, and forwards to next.
Per-tile flag signaling provides synchronization — no host barriers needed
inside the algorithm.

For small messages (< RING_THRESHOLD elements), falls back to the direct
push kernel where root sends to all ranks in one shot.
"""

import torch
import triton
import triton.language as tl
import iris
from iris.host.tracing.kernel_artifacts import iris_launch
from ..utils import chiplet_transform_chunked

RING_THRESHOLD = 1 << 30  # Disabled — ring broadcast has a deadlock bug, using direct push only


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
    """Direct push broadcast — root sends to all ranks. Fast for small messages."""
    pid = tl.program_id(0)

    if group_rank != src:
        return

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
        tl.assume(stride_m >= 0)
        tl.assume(stride_n >= 0)

        rm_base = pid_m * BLOCK_SIZE_M
        rn_base = pid_n * BLOCK_SIZE_N
        rm = rm_base + tl.arange(0, BLOCK_SIZE_M)
        rn = rn_base + tl.arange(0, BLOCK_SIZE_N)
        rm = tl.max_contiguous(tl.multiple_of(rm, BLOCK_SIZE_M), BLOCK_SIZE_M)
        rn = tl.max_contiguous(tl.multiple_of(rn, BLOCK_SIZE_N), BLOCK_SIZE_N)

        mask = (rm[:, None] < M) & (rn[None, :] < N)

        offset = rm[:, None] * stride_m + rn[None, :] * stride_n
        src_ptrs = tensor_ptr + offset
        src_ptrs = tl.multiple_of(src_ptrs, (BLOCK_SIZE_M, BLOCK_SIZE_N))
        data = tl.load(src_ptrs, mask=mask, other=0.0)

        for i in tl.static_range(world_size):
            if i != src:
                target_rank = rank_start + i * rank_stride
                dst_ptrs = tensor_ptr + offset
                dst_ptrs = tl.multiple_of(dst_ptrs, (BLOCK_SIZE_M, BLOCK_SIZE_N))
                iris.store(
                    dst_ptrs,
                    data,
                    iris_rank,
                    target_rank,
                    heap_bases,
                    mask=mask,
                    hint=(1, BLOCK_SIZE_N),
                )


@triton.jit()
def persistent_broadcast_ring(
    tensor_ptr,
    ring_buffer,
    flags,
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
    next_rank: tl.constexpr,
    ring_pos: tl.constexpr,
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
    GROUP_SIZE_M: tl.constexpr,
    COMM_SMS: tl.constexpr,
    NUM_XCDS: tl.constexpr,
    CHUNK_SIZE: tl.constexpr,
):
    """
    Ring broadcast kernel — data pipelines from root through the ring.

    Ring order: root -> root+1 -> root+2 -> ... -> root-1
    ring_pos: 0 = root, 1 = first after root, ..., world_size-1 = last

    Root (ring_pos == 0):
        Load tile, write to next_rank's ring_buffer, signal next.

    Middle ranks (0 < ring_pos < world_size - 1):
        Wait for signal from predecessor, load from own ring_buffer,
        store locally, forward to next_rank's ring_buffer, signal next.

    Last rank (ring_pos == world_size - 1):
        Wait for signal from predecessor, load from own ring_buffer,
        store locally. No forwarding.

    The per-tile flag signaling IS the synchronization — no host-side
    barriers needed inside the algorithm.
    """
    pid = tl.program_id(0)

    if NUM_XCDS != 1:
        pid = chiplet_transform_chunked(pid, COMM_SMS, NUM_XCDS, CHUNK_SIZE)

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
        tile_offset = rm[:, None] * stride_m + rn[None, :] * stride_n

        flag_ptr = flags + tile_id
        remote_flag_ptr = flags + tile_id

        if ring_pos == 0:
            # Root: load local data, push to next rank's ring buffer, signal
            data = tl.load(tensor_ptr + tile_offset, mask=mask, other=0.0)

            # Wait until next rank is ready (flag == 0)
            while (
                iris.atomic_cas(
                    remote_flag_ptr,
                    0,
                    0,
                    iris_rank,
                    next_rank,
                    heap_bases,
                    sem="acquire",
                    scope="sys",
                )
                != 0
            ):
                pass

            iris.store(
                ring_buffer + tile_offset,
                data,
                iris_rank,
                next_rank,
                heap_bases,
                mask=mask,
                hint=(1, BLOCK_SIZE_N),
            )
            tl.debug_barrier()
            iris.atomic_xchg(
                remote_flag_ptr,
                1,
                iris_rank,
                next_rank,
                heap_bases,
                sem="release",
                scope="sys",
            )

        elif ring_pos == world_size - 1:
            # Last rank: wait for data, store locally. No forwarding.
            while tl.atomic_cas(flag_ptr, 0, 0, sem="acquire", scope="sys") != 1:
                pass

            data = tl.load(ring_buffer + tile_offset, mask=mask, other=0.0)
            tl.store(tensor_ptr + tile_offset, data, mask=mask)

            tl.debug_barrier()
            tl.atomic_xchg(flag_ptr, 0, sem="release", scope="sys")

        else:
            # Middle rank: wait, load from ring buffer, store locally, forward
            while tl.atomic_cas(flag_ptr, 0, 0, sem="acquire", scope="sys") != 1:
                pass

            data = tl.load(ring_buffer + tile_offset, mask=mask, other=0.0)
            tl.store(tensor_ptr + tile_offset, data, mask=mask)

            # Reset own flag
            tl.debug_barrier()
            tl.atomic_xchg(flag_ptr, 0, sem="release", scope="sys")

            # Forward to next rank
            while (
                iris.atomic_cas(
                    remote_flag_ptr,
                    0,
                    0,
                    iris_rank,
                    next_rank,
                    heap_bases,
                    sem="acquire",
                    scope="sys",
                )
                != 0
            ):
                pass

            iris.store(
                ring_buffer + tile_offset,
                data,
                iris_rank,
                next_rank,
                heap_bases,
                mask=mask,
                hint=(1, BLOCK_SIZE_N),
            )
            tl.debug_barrier()
            iris.atomic_xchg(
                remote_flag_ptr,
                1,
                iris_rank,
                next_rank,
                heap_bases,
                sem="release",
                scope="sys",
            )


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
    """Launch the broadcast kernel — ring for large messages, direct for small."""
    M, N = tensor.shape[:2]
    stride_m, stride_n = tensor.stride(0), tensor.stride(1)

    heap_bases = ctx.get_heap_bases()

    if M * N < RING_THRESHOLD or world_size <= 2:
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
    else:
        # Ring broadcast: allocate ring buffer + flags on symmetric heap
        ring_buffer = ctx.zeros((M, N), dtype=tensor.dtype)
        num_pid_m = (M + config.block_size_m - 1) // config.block_size_m
        num_pid_n = (N + config.block_size_n - 1) // config.block_size_n
        total_tiles = num_pid_m * num_pid_n
        flags = ctx.zeros((total_tiles,), dtype=torch.int32)

        # Ring position: root is 0, then root+1, root+2, ..., root-1
        ring_pos = (rank_in_group - src) % world_size

        # Next rank in ring (global iris rank)
        next_group_rank = (rank_in_group + 1) % world_size
        next_rank = rank_start + next_group_rank * rank_stride

        # Pre-kernel barrier to ensure ring_buffer and flags are visible
        ctx.device_barrier()

        iris_launch(
            persistent_broadcast_ring,
            (config.comm_sms,),
            tensor,
            ring_buffer,
            flags,
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
            next_rank,
            ring_pos,
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
