# SPDX-License-Identifier: MIT
# Copyright (c) 2025 Advanced Micro Devices, Inc. All rights reserved.

"""
Ring broadcast kernel — per-tile independent processing, no separate ring buffer.

Each CTA independently processes its assigned tiles through the ring.
No inter-CTA barriers, no pid-0 coordination.

Ring order: root -> root+1 -> ... -> root-1
Root writes its data directly to rank 1's tensor. Rank 1 reads from its own tensor
(root already wrote there) and forwards to rank 2's tensor. Etc.

Uses per-tile flags on the symmetric heap for producer-consumer handshake.
"""

import torch
import triton
import triton.language as tl
import iris
from iris.host.tracing.kernel_artifacts import iris_launch
from iris.ccl.utils import inline_device_barrier
from ..utils import chiplet_transform_chunked


@triton.jit()
def persistent_ring_broadcast(
    data_ptr,
    flags,
    M,
    N,
    stride_m,
    stride_n,
    heap_bases: tl.tensor,
    group_rank: tl.constexpr,
    iris_rank: tl.constexpr,
    world_size: tl.constexpr,
    rank_start: tl.constexpr,
    rank_stride: tl.constexpr,
    root: tl.constexpr,
    next_rank: tl.constexpr,
    barrier_flags_ptr,
    wg_done_ptr,
    barrier_sense_ptr,
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
    GROUP_SIZE_M: tl.constexpr,
    COMM_SMS: tl.constexpr,
    NUM_XCDS: tl.constexpr,
    CHUNK_SIZE: tl.constexpr,
    FLAGS_PER_TILE: tl.constexpr,
    INLINE_BARRIER: tl.constexpr = False,
):
    pid_raw = tl.program_id(0)
    pid = pid_raw
    if NUM_XCDS != 1:
        pid = chiplet_transform_chunked(pid_raw, COMM_SMS, NUM_XCDS, CHUNK_SIZE)

    tl.static_assert(FLAGS_PER_TILE >= 1, "FLAGS_PER_TILE must be >= 1")

    ring_pos = (group_rank - root + world_size) % world_size
    is_root = ring_pos == 0
    is_last = ring_pos == world_size - 1

    num_pid_m = tl.cdiv(M, BLOCK_SIZE_M)
    num_pid_n = tl.cdiv(N, BLOCK_SIZE_N)
    total_tiles = num_pid_m * num_pid_n

    if total_tiles > 0:
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

            flag_offset = tile_id * FLAGS_PER_TILE
            remote_flag_ptr = flags + flag_offset
            local_flag_ptr = flags + flag_offset

            if is_root:
                tile_data = tl.load(data_ptr + tile_offset, mask=mask, other=0)

                if not is_last:
                    iris.store(
                        data_ptr + tile_offset, tile_data,
                        iris_rank, next_rank, heap_bases,
                        mask=mask, hint=(1, BLOCK_SIZE_N),
                    )
                    tl.debug_barrier()
                    iris.atomic_xchg(
                        remote_flag_ptr, 1, iris_rank, next_rank, heap_bases,
                        sem="release", scope="sys",
                    )
            else:
                while tl.atomic_cas(local_flag_ptr, 0, 0, sem="acquire", scope="sys") != 1:
                    pass

                tl.atomic_xchg(local_flag_ptr, 0, sem="release", scope="sys")

                if not is_last:
                    tile_data = tl.load(data_ptr + tile_offset, mask=mask, other=0)

                    iris.store(
                        data_ptr + tile_offset, tile_data,
                        iris_rank, next_rank, heap_bases,
                        mask=mask, hint=(1, BLOCK_SIZE_N),
                    )
                    tl.debug_barrier()
                    iris.atomic_xchg(
                        remote_flag_ptr, 1, iris_rank, next_rank, heap_bases,
                        sem="release", scope="sys",
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
    if device not in _dummy_barrier_cache:
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
    root,
    config,
    inline_barrier=False,
    barrier_state=None,
):
    """Launch per-tile ring broadcast kernel."""
    M, N = tensor.shape[:2]
    stride_m, stride_n = tensor.stride(0), tensor.stride(1)

    block_m = config.block_size_m
    block_n = config.block_size_n
    num_pid_m = (M + block_m - 1) // block_m
    num_pid_n = (N + block_n - 1) // block_n
    total_tiles = num_pid_m * num_pid_n
    flags_per_tile = 1

    total_flags = total_tiles * flags_per_tile
    flags = ctx.zeros((total_flags,), dtype=torch.int32)

    ring_pos = (rank_in_group - root + world_size) % world_size
    next_in_ring = (ring_pos + 1) % world_size
    next_group_rank = (root + next_in_ring) % world_size
    next_rank = rank_start + next_group_rank * rank_stride

    heap_bases = ctx.get_heap_bases()

    if inline_barrier and barrier_state is not None:
        barrier_flags, wg_done, barrier_sense = barrier_state
    else:
        barrier_flags, wg_done, barrier_sense = _get_dummy_barrier(tensor.device)

    iris_launch(
        persistent_ring_broadcast,
        (config.comm_sms,),
        tensor,
        flags,
        M,
        N,
        stride_m,
        stride_n,
        heap_bases,
        rank_in_group,
        rank_global,
        world_size,
        rank_start,
        rank_stride,
        root,
        next_rank,
        barrier_flags,
        wg_done,
        barrier_sense,
        config.block_size_m,
        config.block_size_n,
        config.swizzle_size,
        config.comm_sms,
        config.num_xcds,
        config.chunk_size,
        flags_per_tile,
        inline_barrier,
        num_stages=config.num_stages,
        num_warps=config.num_warps,
        waves_per_eu=config.waves_per_eu,
        algorithm="ring_broadcast",
        rank=rank_global,
        dtype=tensor.dtype,
    )
