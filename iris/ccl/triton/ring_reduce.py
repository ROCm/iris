# SPDX-License-Identifier: MIT
# Copyright (c) 2025 Advanced Micro Devices, Inc. All rights reserved.

"""
Ring reduce kernel — per-tile independent processing.

Each CTA independently processes its assigned tiles through the ring.
No inter-CTA barriers, no pid-0 coordination.

Ring order (data flows towards root):
  Farthest rank from root -> ... -> root's predecessor -> root

First sender stores local input to output. Each subsequent rank reads
predecessor's output (via iris.load), reduces with local input, stores result
to own output, then writes result to next rank's output (via iris.store).
Root receives the accumulated partial sum and does the final reduce.

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
def persistent_ring_reduce(
    output_ptr,
    input_ptr,
    flags,
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
    root: tl.constexpr,
    next_rank: tl.constexpr,
    prev_rank: tl.constexpr,
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

    # ring_pos: 0 = first sender (farthest from root), W-1 = root
    ring_pos = (group_rank - root - 1 + world_size) % world_size
    is_first = ring_pos == 0
    is_root = group_rank == root

    num_pid_m = tl.cdiv(M, BLOCK_SIZE_M)
    num_pid_n = tl.cdiv(N, BLOCK_SIZE_N)
    total_tiles = num_pid_m * num_pid_n

    acc_dtype = tl.float32 if output_ptr.type.element_ty != tl.int8 else tl.int32

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
            in_offset = rm[:, None] * stride_in_m + rn[None, :] * stride_in_n
            out_offset = rm[:, None] * stride_out_m + rn[None, :] * stride_out_n

            flag_offset = tile_id * FLAGS_PER_TILE
            remote_flag_ptr = flags + flag_offset
            local_flag_ptr = flags + flag_offset

            local_data = tl.load(input_ptr + in_offset, mask=mask, other=0)

            if is_first:
                tl.store(output_ptr + out_offset, local_data, mask=mask)

                iris.store(
                    output_ptr + out_offset, local_data,
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

                remote_data = tl.load(output_ptr + out_offset, mask=mask, other=0)

                acc = remote_data.to(acc_dtype) + local_data.to(acc_dtype)
                result = acc.to(output_ptr.type.element_ty)

                tl.store(output_ptr + out_offset, result, mask=mask)

                if not is_root:
                    iris.store(
                        output_ptr + out_offset, result,
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
    output_tensor,
    input_tensor,
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
    """Launch per-tile ring reduce kernel."""
    M, N = input_tensor.shape[:2]
    stride_in_m, stride_in_n = input_tensor.stride(0), input_tensor.stride(1)
    stride_out_m, stride_out_n = output_tensor.stride(0), output_tensor.stride(1)

    block_m = config.block_size_m
    block_n = config.block_size_n
    num_pid_m = (M + block_m - 1) // block_m
    num_pid_n = (N + block_n - 1) // block_n
    total_tiles = num_pid_m * num_pid_n
    flags_per_tile = 1

    total_flags = total_tiles * flags_per_tile
    flags = ctx.zeros((total_flags,), dtype=torch.int32)

    # Ring topology for reduce: data flows towards root
    ring_pos = (rank_in_group - root - 1 + world_size) % world_size

    # Next in ring = one step closer to root
    next_in_ring = (ring_pos + 1) % world_size
    next_group_rank = (root + 1 + next_in_ring) % world_size
    next_rank_global = rank_start + next_group_rank * rank_stride

    # Previous in ring = one step farther from root
    prev_in_ring = (ring_pos - 1 + world_size) % world_size
    prev_group_rank = (root + 1 + prev_in_ring) % world_size
    prev_rank_global = rank_start + prev_group_rank * rank_stride

    heap_bases = ctx.get_heap_bases()

    if inline_barrier and barrier_state is not None:
        barrier_flags, wg_done, barrier_sense = barrier_state
    else:
        barrier_flags, wg_done, barrier_sense = _get_dummy_barrier(input_tensor.device)

    iris_launch(
        persistent_ring_reduce,
        (config.comm_sms,),
        output_tensor,
        input_tensor,
        flags,
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
        root,
        next_rank_global,
        prev_rank_global,
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
        algorithm="ring_reduce",
        rank=rank_global,
        dtype=input_tensor.dtype,
    )
