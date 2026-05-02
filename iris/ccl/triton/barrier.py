# SPDX-License-Identifier: MIT
# Copyright (c) 2025 Advanced Micro Devices, Inc. All rights reserved.

"""
Triton kernel for barrier collective communication.

GPU-side barrier using atomic flag signaling on the symmetric heap.
Each rank signals all peers via iris.atomic_xchg, then spin-waits
until all peers have signaled back.
"""

import triton
import triton.language as tl

import iris
from iris.host.tracing.kernel_artifacts import iris_launch


@triton.jit
def barrier_kernel(
    flags_ptr,
    heap_bases: tl.tensor,
    iris_rank: tl.constexpr,
    world_size: tl.constexpr,
    rank_start: tl.constexpr,
    rank_stride: tl.constexpr,
    MAX_SPINS: tl.constexpr = 1_000_000_000,
):
    """
    Device-side barrier using atomic flag signaling.

    Single workgroup (pid==0) does all work:
    1. Atomically set this rank's flag to 1 on every other rank
       (via iris.atomic_xchg on the remote heap).
    2. Spin-wait on local flags array until every entry is 1
       (meaning every peer has signaled us).
    3. Clear all local flags back to 0 for the next barrier invocation.

    Args:
        flags_ptr: Pointer to int32[world_size] on symmetric heap.
                   flags_ptr[r] == 1 means rank r has arrived.
        heap_bases: Heap base pointers for all ranks.
        iris_rank: Global rank of this process.
        world_size: Number of ranks in the group.
        rank_start: Starting global rank of the group.
        rank_stride: Stride between consecutive ranks in the group.
    """
    # Only pid 0 participates
    if tl.program_id(0) != 0:
        return

    # Step 1: Signal all remote ranks by setting our flag on their heap.
    # On each remote rank's flags array, flags_ptr[iris_rank] = 1 tells
    # them that we have arrived at the barrier.
    for i in range(world_size):
        target_rank = rank_start + i * rank_stride
        if target_rank != iris_rank:
            # Set flags_ptr[iris_rank] on target_rank's heap to 1
            iris.atomic_xchg(
                flags_ptr + iris_rank,
                1,
                iris_rank,
                target_rank,
                heap_bases,
                sem="release",
                scope="sys",
            )

    # Step 2: Spin-wait until all peers have signaled us.
    # Each remote rank will set flags_ptr[remote_rank] on OUR heap to 1.
    for i in range(world_size):
        remote_rank = rank_start + i * rank_stride
        if remote_rank != iris_rank:
            spin_count = 0
            # Poll flags_ptr[remote_rank] in our own address space
            while (
                tl.atomic_cas(
                    flags_ptr + remote_rank,
                    1,
                    1,
                    sem="acquire",
                    scope="sys",
                )
                != 1
            ):
                spin_count += 1
                tl.device_assert(spin_count < MAX_SPINS, "barrier: timeout waiting for peer")

    # Step 3: Clear all flags for next use.
    # Only clear entries written by remote ranks (index != iris_rank),
    # plus our own entry which remote ranks will have read.
    for i in range(world_size):
        rank_i = rank_start + i * rank_stride
        tl.atomic_xchg(flags_ptr + rank_i, 0, sem="release", scope="sys")


def launch(flags, ctx, rank_global, world_size, rank_start, rank_stride):
    """
    Launch the barrier Triton kernel.

    Args:
        flags: int32 tensor on symmetric heap, shape (world_size,).
        ctx: Iris instance.
        rank_global: Global rank of this process.
        world_size: Number of ranks in the group.
        rank_start: Starting global rank of the group.
        rank_stride: Stride between consecutive ranks.
    """
    heap_bases = ctx.get_heap_bases()

    iris_launch(
        barrier_kernel,
        (1,),  # Single workgroup
        flags,
        heap_bases,
        rank_global,
        world_size,
        rank_start,
        rank_stride,
        algorithm="barrier",
        rank=rank_global,
    )
