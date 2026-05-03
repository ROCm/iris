# SPDX-License-Identifier: MIT
# Copyright (c) 2025 Advanced Micro Devices, Inc. All rights reserved.

"""
Ring reduce kernel for iris CCL.

Pipeline reduce through a ring of ranks. Each rank reads partial sums
from its predecessor, reduces with local data, and signals the next rank.

Ring order (relative to root):
  root+1 -> root+2 -> ... -> root-1 -> root
  ring_pos = (group_rank - root - 1 + world_size) % world_size

Uses per-chunk flag signaling and inter-CTA arrive/wait barriers.
"""

import torch
import triton
import triton.language as tl
import iris
from iris.host.tracing.kernel_artifacts import iris_launch
from ..utils import chiplet_transform_chunked


@triton.jit()
def persistent_ring_reduce(
    output_ptr,
    input_ptr,
    flags_ptr,
    arrive_ptr,
    ready_ptr,
    M,
    N,
    stride_in_m,
    stride_in_n,
    stride_out_m,
    stride_out_n,
    num_chunks,
    chunk_rows,
    heap_bases: tl.tensor,
    group_rank: tl.constexpr,
    iris_rank: tl.constexpr,
    world_size: tl.constexpr,
    rank_start: tl.constexpr,
    rank_stride: tl.constexpr,
    root: tl.constexpr,
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
    COMM_SMS: tl.constexpr,
    NUM_XCDS: tl.constexpr,
    CHUNK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)

    if NUM_XCDS != 1:
        pid = chiplet_transform_chunked(pid, COMM_SMS, NUM_XCDS, CHUNK_SIZE)

    ring_pos = (group_rank - root - 1 + world_size) % world_size
    prev_group_rank = (group_rank - 1 + world_size) % world_size
    prev_global = rank_start + prev_group_rank * rank_stride

    from_base = tl.load(heap_bases + iris_rank)

    num_pid_n = tl.cdiv(N, BLOCK_SIZE_N)

    acc_dtype = tl.float32

    if ring_pos == 0:
        # First sender: copy local input to output, signal ready
        for c in range(num_chunks):
            row_start = c * chunk_rows
            actual_chunk_rows = tl.minimum(chunk_rows, M - row_start)
            num_pid_m_chunk = tl.cdiv(actual_chunk_rows, BLOCK_SIZE_M)
            total_tiles = num_pid_m_chunk * num_pid_n

            for tile_offset in range(pid, total_tiles, COMM_SMS):
                pid_m = tile_offset // num_pid_n
                pid_n = tile_offset % num_pid_n

                rm_base = row_start + pid_m * BLOCK_SIZE_M
                rn_base = pid_n * BLOCK_SIZE_N

                rm = rm_base + tl.arange(0, BLOCK_SIZE_M)
                rn = rn_base + tl.arange(0, BLOCK_SIZE_N)
                rm = tl.max_contiguous(tl.multiple_of(rm, BLOCK_SIZE_M), BLOCK_SIZE_M)
                rn = tl.max_contiguous(tl.multiple_of(rn, BLOCK_SIZE_N), BLOCK_SIZE_N)

                in_offset = rm[:, None] * stride_in_m + rn[None, :] * stride_in_n
                out_offset = rm[:, None] * stride_out_m + rn[None, :] * stride_out_n

                is_full = (rm_base + BLOCK_SIZE_M <= M) & (rn_base + BLOCK_SIZE_N <= N)

                if is_full:
                    data = tl.load(input_ptr + in_offset)
                    tl.store(output_ptr + out_offset, data, cache_modifier=".wt")
                else:
                    mask = (rm[:, None] < M) & (rn[None, :] < N)
                    data = tl.load(input_ptr + in_offset, mask=mask, other=0.0)
                    tl.store(output_ptr + out_offset, data, mask=mask, cache_modifier=".wt")

            # Inter-CTA arrive barrier
            expected_arrive = (c + 1) * COMM_SMS
            tl.atomic_add(arrive_ptr, 1, sem="release", scope="gpu")

            if pid == 0:
                while (
                    tl.atomic_cas(arrive_ptr, expected_arrive, expected_arrive, sem="acquire", scope="gpu")
                    < expected_arrive
                ):
                    pass

                own_flag_ptr = flags_ptr + c
                own_offset = tl.cast(own_flag_ptr, tl.uint64) - from_base
                own_translated = tl.cast(
                    tl.cast(from_base, tl.pointer_type(tl.int8)) + own_offset,
                    own_flag_ptr.dtype,
                )
                tl.atomic_xchg(own_translated, 1, sem="release", scope="sys")
    else:
        # Middle ranks and root: wait for prev, reduce, forward
        expected_step = ring_pos

        prev_base = tl.load(heap_bases + prev_global)

        for c in range(num_chunks):
            row_start = c * chunk_rows
            actual_chunk_rows = tl.minimum(chunk_rows, M - row_start)

            # pid 0 waits for predecessor's flag, then releases all CTAs
            if pid == 0:
                prev_flag_ptr = flags_ptr + c
                prev_offset = tl.cast(prev_flag_ptr, tl.uint64) - from_base
                prev_translated = tl.cast(
                    tl.cast(prev_base, tl.pointer_type(tl.int8)) + prev_offset,
                    prev_flag_ptr.dtype,
                )
                while (
                    tl.atomic_cas(prev_translated, expected_step, expected_step, sem="acquire", scope="sys")
                    < expected_step
                ):
                    pass
                tl.debug_barrier()
                tl.atomic_xchg(prev_translated, 0, sem="release", scope="sys")

                tl.atomic_xchg(ready_ptr, c + 1, sem="release", scope="gpu")

            if pid != 0:
                while tl.atomic_cas(ready_ptr, c + 1, c + 1, sem="acquire", scope="gpu") < (c + 1):
                    pass

            num_pid_m_chunk = tl.cdiv(actual_chunk_rows, BLOCK_SIZE_M)
            total_tiles = num_pid_m_chunk * num_pid_n

            for tile_offset in range(pid, total_tiles, COMM_SMS):
                pid_m = tile_offset // num_pid_n
                pid_n = tile_offset % num_pid_n

                rm_base = row_start + pid_m * BLOCK_SIZE_M
                rn_base = pid_n * BLOCK_SIZE_N

                rm = rm_base + tl.arange(0, BLOCK_SIZE_M)
                rn = rn_base + tl.arange(0, BLOCK_SIZE_N)
                rm = tl.max_contiguous(tl.multiple_of(rm, BLOCK_SIZE_M), BLOCK_SIZE_M)
                rn = tl.max_contiguous(tl.multiple_of(rn, BLOCK_SIZE_N), BLOCK_SIZE_N)

                in_offset = rm[:, None] * stride_in_m + rn[None, :] * stride_in_n
                out_offset = rm[:, None] * stride_out_m + rn[None, :] * stride_out_n
                src_ptr = output_ptr + out_offset

                is_full = (rm_base + BLOCK_SIZE_M <= M) & (rn_base + BLOCK_SIZE_N <= N)

                if is_full:
                    remote_data = iris.load(src_ptr, iris_rank, prev_global, heap_bases, hint=(1, BLOCK_SIZE_N))
                    local_data = tl.load(input_ptr + in_offset)
                    acc = remote_data.to(acc_dtype) + local_data.to(acc_dtype)
                    tl.store(output_ptr + out_offset, acc.to(output_ptr.type.element_ty), cache_modifier=".wt")
                else:
                    mask = (rm[:, None] < M) & (rn[None, :] < N)
                    remote_data = iris.load(
                        src_ptr, iris_rank, prev_global, heap_bases, mask=mask, hint=(1, BLOCK_SIZE_N)
                    )
                    local_data = tl.load(input_ptr + in_offset, mask=mask, other=0.0)
                    acc = remote_data.to(acc_dtype) + local_data.to(acc_dtype)
                    tl.store(
                        output_ptr + out_offset, acc.to(output_ptr.type.element_ty), mask=mask, cache_modifier=".wt"
                    )

            # Inter-CTA arrive barrier
            expected_arrive = (c + 1) * COMM_SMS
            tl.atomic_add(arrive_ptr, 1, sem="release", scope="gpu")

            if pid == 0:
                while (
                    tl.atomic_cas(arrive_ptr, expected_arrive, expected_arrive, sem="acquire", scope="gpu")
                    < expected_arrive
                ):
                    pass

                own_flag_ptr = flags_ptr + c
                own_offset = tl.cast(own_flag_ptr, tl.uint64) - from_base
                own_translated = tl.cast(
                    tl.cast(from_base, tl.pointer_type(tl.int8)) + own_offset,
                    own_flag_ptr.dtype,
                )
                tl.atomic_xchg(own_translated, expected_step + 1, sem="release", scope="sys")


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
    flags=None,
):
    """Launch ring reduce kernel."""
    M, N = input_tensor.shape[:2]
    stride_in_m, stride_in_n = input_tensor.stride(0), input_tensor.stride(1)
    stride_out_m, stride_out_n = output_tensor.stride(0), output_tensor.stride(1)

    chunk_rows = max(config.block_size_m, M // (world_size * 2))
    chunk_rows = ((chunk_rows + config.block_size_m - 1) // config.block_size_m) * config.block_size_m
    num_chunks = (M + chunk_rows - 1) // chunk_rows

    if flags is None:
        flags = ctx.zeros((num_chunks,), dtype=torch.int32)

    dev = ctx.get_device()
    ring_sms = config.comm_sms
    arrive_counter = torch.zeros((1,), dtype=torch.int32, device=dev)
    ready_counter = torch.zeros((1,), dtype=torch.int32, device=dev)

    heap_bases = ctx.get_heap_bases()

    iris_launch(
        persistent_ring_reduce,
        (ring_sms,),
        output_tensor,
        input_tensor,
        flags,
        arrive_counter,
        ready_counter,
        M,
        N,
        stride_in_m,
        stride_in_n,
        stride_out_m,
        stride_out_n,
        num_chunks,
        chunk_rows,
        heap_bases,
        rank_in_group,
        rank_global,
        world_size,
        rank_start,
        rank_stride,
        root,
        config.block_size_m,
        config.block_size_n,
        ring_sms,
        config.num_xcds,
        config.chunk_size,
        num_stages=config.num_stages,
        num_warps=config.num_warps,
        waves_per_eu=config.waves_per_eu,
        algorithm="ring_reduce",
        rank=rank_global,
        dtype=input_tensor.dtype,
    )

    return flags
