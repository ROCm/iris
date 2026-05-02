# SPDX-License-Identifier: MIT
# Copyright (c) 2025 Advanced Micro Devices, Inc. All rights reserved.

"""
Triton kernel for reduce collective communication.
Single-kernel lock-based accumulation: each non-root rank acquires a
per-tile spinlock on the root rank's heap, does a read-modify-write,
and releases the lock.  One kernel launch, no host barriers between steps.
"""

import torch
import triton
import triton.language as tl
import iris
from iris.host.tracing.kernel_artifacts import iris_launch
from ..utils import chiplet_transform_chunked


@triton.jit()
def persistent_reduce(
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
    """
    Lock-based reduce kernel.

    Root rank initializes output with its own input data (output was
    pre-zeroed and root's data is copied by the host before launch).
    Non-root ranks acquire a per-tile spinlock on the root rank's heap,
    load the current accumulated value, add their local contribution,
    store the result, and release the lock.
    """
    pid = tl.program_id(0)

    if NUM_XCDS != 1:
        pid = chiplet_transform_chunked(pid, COMM_SMS, NUM_XCDS, CHUNK_SIZE)

    num_pid_m = tl.cdiv(M, BLOCK_SIZE_M)
    num_pid_n = tl.cdiv(N, BLOCK_SIZE_N)
    total_tiles = num_pid_m * num_pid_n

    dst_iris_rank = rank_start + dst * rank_stride

    acc_dtype = tl.float32 if output_ptr.type.element_ty != tl.int8 else tl.int32

    for tile_id in range(pid, total_tiles, COMM_SMS):
        # Compute tile coordinates (swizzled)
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
        output_offset = rm[:, None] * stride_out_m + rn[None, :] * stride_out_n

        # Load local contribution
        local_data = tl.load(input_ptr + input_offset, mask=mask, other=0.0)

        if group_rank == dst:
            # Root: data was already copied to output on the host side
            # before the barrier.  Nothing to do here.
            pass
        else:
            # Non-root: acquire lock, read-modify-write on root's output
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

            # Load current accumulated value from root's output tile
            current_value = iris.load(
                output_ptr + output_offset,
                iris_rank,
                dst_iris_rank,
                heap_bases,
                mask=mask,
            )

            # Accumulate
            acc = current_value.to(acc_dtype) + local_data.to(acc_dtype)
            result = acc.to(output_ptr.type.element_ty)

            # Store back to root's output tile
            iris.store(
                output_ptr + output_offset,
                result,
                iris_rank,
                dst_iris_rank,
                heap_bases,
                mask=mask,
                hint=(1, BLOCK_SIZE_N),
            )

            # Release lock
            iris.atomic_xchg(
                locks_ptr + tile_id,
                0,
                iris_rank,
                dst_iris_rank,
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
    """Launch the lock-based reduce kernel."""
    M, N = input_tensor.shape[:2]
    stride_in_m, stride_in_n = input_tensor.stride(0), input_tensor.stride(1)
    stride_out_m, stride_out_n = output_tensor.stride(0), output_tensor.stride(1)

    # Allocate per-tile locks on the symmetric heap
    num_pid_m = (M + config.block_size_m - 1) // config.block_size_m
    num_pid_n = (N + config.block_size_n - 1) // config.block_size_n
    total_tiles = num_pid_m * num_pid_n

    locks = ctx.zeros((total_tiles,), dtype=torch.int32)

    # Root initializes output with its own data on the host side,
    # before the barrier.  This avoids a race between root's in-kernel
    # tl.store and non-root's iris.load (non-root could read stale zeros
    # or partial root data).  Non-root ranks zero their output (unused
    # but keeps heap symmetric).
    if rank_in_group == dst:
        output_tensor.copy_(input_tensor)
    else:
        output_tensor.zero_()
    ctx.barrier()

    heap_bases = ctx.get_heap_bases()

    iris_launch(
        persistent_reduce,
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
