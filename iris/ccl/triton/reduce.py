# SPDX-License-Identifier: MIT
# Copyright (c) 2025 Advanced Micro Devices, Inc. All rights reserved.

"""
Reduce — Triton kernel.

Ring reduce: partial sums pipeline around the ring toward root.  Each rank
receives the running accumulator from its predecessor, adds its local data,
and forwards to the next rank.  Per-tile flag signaling provides
synchronization — barriers are part of the algorithm.

For small messages (< RING_THRESHOLD elements), falls back to the lock-based
kernel where non-root ranks do atomic read-modify-write on root's heap.
"""

import torch
import triton
import triton.language as tl
import iris
from iris.host.tracing.kernel_artifacts import iris_launch
from ..utils import chiplet_transform_chunked

RING_THRESHOLD = 1 << 30  # Disabled — ring reduce has bugs, using lock-based only


@triton.jit()
def persistent_reduce_lock(
    input_ptr,
    output_ptr,
    locks_ptr,
    M,
    N,
    stride_in_m,
    stride_in_n,
    stride_out_m,
    stride_out_n,
    heap_bases: tl.tensor,
    group_rank: tl.constexpr,
    iris_rank: tl.constexpr,
    dst: tl.constexpr,
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
    """Lock-based reduce — fallback for small messages."""
    pid = tl.program_id(0)

    if NUM_XCDS != 1:
        pid = chiplet_transform_chunked(pid, COMM_SMS, NUM_XCDS, CHUNK_SIZE)

    num_pid_m = tl.cdiv(M, BLOCK_SIZE_M)
    num_pid_n = tl.cdiv(N, BLOCK_SIZE_N)
    total_tiles = num_pid_m * num_pid_n

    dst_iris_rank = rank_start + dst * rank_stride

    acc_dtype = tl.float32 if output_ptr.type.element_ty != tl.int8 else tl.int32

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

        if group_rank != dst:
            input_offset = rm[:, None] * stride_in_m + rn[None, :] * stride_in_n
            output_offset = rm[:, None] * stride_out_m + rn[None, :] * stride_out_n

            local_data = tl.load(input_ptr + input_offset, mask=mask, other=0.0)

            while (
                iris.atomic_cas(
                    locks_ptr + tile_id,
                    0,
                    1,
                    iris_rank,
                    dst_iris_rank,
                    heap_bases,
                    sem="acquire",
                    scope="sys",
                )
                != 0
            ):
                pass

            current_value = iris.load(
                output_ptr + output_offset,
                iris_rank,
                dst_iris_rank,
                heap_bases,
                mask=mask,
            )

            acc = current_value.to(acc_dtype) + local_data.to(acc_dtype)
            result = acc.to(output_ptr.type.element_ty)

            iris.store(
                output_ptr + output_offset,
                result,
                iris_rank,
                dst_iris_rank,
                heap_bases,
                mask=mask,
                hint=(1, BLOCK_SIZE_N),
            )

            iris.atomic_xchg(
                locks_ptr + tile_id,
                0,
                iris_rank,
                dst_iris_rank,
                heap_bases,
                sem="release",
                scope="sys",
            )


@triton.jit()
def persistent_reduce_ring(
    input_ptr,
    output_ptr,
    ring_buffer,
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
    dst: tl.constexpr,
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
    Ring reduce kernel — partial sums pipeline toward root.

    Ring order for reduce: data flows dst+1 -> dst+2 -> ... -> dst
    ring_pos: 0 = first sender (dst+1), ..., world_size-2 = last before root,
              world_size-1 = root (dst)

    First sender (ring_pos == 0):
        Load local data, push to next_rank's ring_buffer, signal.

    Middle ranks (0 < ring_pos < world_size - 1):
        Wait for signal, load accumulated value from ring_buffer,
        add local data, forward to next_rank's ring_buffer, signal.

    Root (ring_pos == world_size - 1):
        Wait for signal, load accumulated value from ring_buffer,
        add local data, store final result to output.
    """
    pid = tl.program_id(0)

    if NUM_XCDS != 1:
        pid = chiplet_transform_chunked(pid, COMM_SMS, NUM_XCDS, CHUNK_SIZE)

    num_pid_m = tl.cdiv(M, BLOCK_SIZE_M)
    num_pid_n = tl.cdiv(N, BLOCK_SIZE_N)
    total_tiles = num_pid_m * num_pid_n
    tl.assume(total_tiles > 0)

    acc_dtype = tl.float32 if output_ptr.type.element_ty != tl.int8 else tl.int32

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

        input_offset = rm[:, None] * stride_in_m + rn[None, :] * stride_in_n
        tile_offset = rm[:, None] * stride_out_m + rn[None, :] * stride_out_n

        flag_ptr = flags + tile_id
        remote_flag_ptr = flags + tile_id

        local_data = tl.load(input_ptr + input_offset, mask=mask, other=0.0)

        if ring_pos == 0:
            # First sender: push local data to next rank, signal
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
                local_data,
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
            # Root: wait for accumulated data, add local, store final result
            while tl.atomic_cas(flag_ptr, 0, 0, sem="acquire", scope="sys") != 1:
                pass

            incoming = tl.load(ring_buffer + tile_offset, mask=mask, other=0.0)
            acc = incoming.to(acc_dtype) + local_data.to(acc_dtype)
            tl.store(
                output_ptr + tile_offset,
                acc.to(output_ptr.type.element_ty),
                mask=mask,
            )

            tl.debug_barrier()
            tl.atomic_xchg(flag_ptr, 0, sem="release", scope="sys")

        else:
            # Middle: wait, load accumulated, add local, forward
            while tl.atomic_cas(flag_ptr, 0, 0, sem="acquire", scope="sys") != 1:
                pass

            incoming = tl.load(ring_buffer + tile_offset, mask=mask, other=0.0)
            acc = incoming.to(acc_dtype) + local_data.to(acc_dtype)
            send_data = acc.to(output_ptr.type.element_ty)

            tl.debug_barrier()
            tl.atomic_xchg(flag_ptr, 0, sem="release", scope="sys")

            # Forward reduced data to next rank
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
                send_data,
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
    output_tensor,
    input_tensor,
    ctx,
    rank_in_group,
    rank_global,
    dst,
    world_size,
    rank_start,
    rank_stride,
    config,
):
    """Launch reduce — ring for large messages, lock-based for small."""
    M, N = input_tensor.shape[:2]
    stride_in_m, stride_in_n = input_tensor.stride(0), input_tensor.stride(1)
    stride_out_m, stride_out_n = output_tensor.stride(0), output_tensor.stride(1)

    heap_bases = ctx.get_heap_bases()

    if M * N < RING_THRESHOLD or world_size <= 2:
        # Lock-based fallback for small messages
        num_pid_m = (M + config.block_size_m - 1) // config.block_size_m
        num_pid_n = (N + config.block_size_n - 1) // config.block_size_n
        total_tiles = num_pid_m * num_pid_n

        locks = ctx.zeros((total_tiles,), dtype=torch.int32)

        if rank_in_group == dst:
            output_tensor.copy_(input_tensor)
        else:
            output_tensor.zero_()
        torch.cuda.synchronize()
        ctx.device_barrier()

        iris_launch(
            persistent_reduce_lock,
            (config.comm_sms,),
            input_tensor,
            output_tensor,
            locks,
            M,
            N,
            stride_in_m,
            stride_in_n,
            stride_out_m,
            stride_out_n,
            heap_bases,
            rank_in_group,
            rank_global,
            dst,
            world_size,
            rank_start,
            rank_stride,
            config.block_size_m,
            config.block_size_n,
            config.swizzle_size,
            config.comm_sms,
            config.num_xcds,
            config.chunk_size,
            algorithm="reduce",
            rank=rank_global,
            dtype=input_tensor.dtype,
        )
    else:
        # Ring reduce
        ring_buffer = ctx.zeros((M, N), dtype=input_tensor.dtype)
        num_pid_m = (M + config.block_size_m - 1) // config.block_size_m
        num_pid_n = (N + config.block_size_n - 1) // config.block_size_n
        total_tiles = num_pid_m * num_pid_n
        flags = ctx.zeros((total_tiles,), dtype=torch.int32)

        # Ring position: first sender is dst+1, root (dst) is last
        ring_pos = (rank_in_group - dst - 1) % world_size

        # Next rank in ring (global iris rank)
        next_group_rank = (rank_in_group + 1) % world_size
        next_rank = rank_start + next_group_rank * rank_stride

        # Pre-kernel barrier to ensure ring_buffer and flags are visible
        ctx.device_barrier()

        iris_launch(
            persistent_reduce_ring,
            (config.comm_sms,),
            input_tensor,
            output_tensor,
            ring_buffer,
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
            dst,
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
            algorithm="reduce",
            rank=rank_global,
            dtype=input_tensor.dtype,
        )
