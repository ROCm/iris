# SPDX-License-Identifier: MIT
# Copyright (c) 2025 Advanced Micro Devices, Inc. All rights reserved.

"""
Triton kernel for broadcast collective communication.

Only the root rank does work: it loads tiles from its local tensor and
stores them to every other rank via iris.store (and locally via tl.store).
Non-root ranks exit immediately.
"""

import triton
import triton.language as tl
import iris
from iris.host.tracing.kernel_artifacts import iris_launch
from ..utils import chiplet_transform_chunked


@triton.jit()
def persistent_broadcast(
    tensor_ptr,
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
    src_rank: tl.constexpr,
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
    GROUP_SIZE_M: tl.constexpr,
    COMM_SMS: tl.constexpr,
    NUM_XCDS: tl.constexpr,
    CHUNK_SIZE: tl.constexpr,
):
    """
    Persistent broadcast kernel.

    Only the root rank (src_rank) does work. It loads tiles from its local
    tensor and stores them to all ranks: iris.store for remote ranks,
    tl.store for the local copy. Non-root ranks exit immediately; they
    receive data via the symmetric heap writes from root.

    Args:
        tensor_ptr: Pointer to tensor of shape (M, N). In-place: root reads
                    from here and writes the same data to all ranks' tensors.
        M: Number of rows.
        N: Number of columns.
        stride_m, stride_n: Strides for the tensor.
        heap_bases: Heap base pointers for all ranks.
        group_rank: Rank within the ProcessGroup (0 to group_size-1).
        iris_rank: Rank in the iris context (for heap_bases indexing).
        world_size: Total number of ranks in the group.
        rank_start: Starting global rank of the group.
        rank_stride: Stride between consecutive ranks in the group.
        src_rank: Source rank (within group) that broadcasts.
        BLOCK_SIZE_M, BLOCK_SIZE_N: Block sizes for tiling.
        GROUP_SIZE_M: Group size for M dimension tiling (swizzle).
        COMM_SMS: Number of SMs for persistent scheduling.
        NUM_XCDS: Number of XCDs.
        CHUNK_SIZE: Chunk size for chiplet transform.
    """
    # Non-root ranks have nothing to do
    if group_rank != src_rank:
        return

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

        # Load tile from root's local tensor
        data = tl.load(src_ptrs, mask=mask, other=0.0)

        # Store to all ranks: stagger write order to reduce contention
        for rank_idx in tl.static_range(world_size):
            dest_idx = (group_rank + rank_idx) % world_size
            target_rank = rank_start + dest_idx * rank_stride

            if dest_idx == group_rank:
                # Local: write-through store (data is already here, but
                # ensures cache coherence for subsequent reads)
                tl.store(src_ptrs, data, mask=mask, cache_modifier=".wt")
            else:
                # Remote: iris.store sends tile to target rank's tensor
                iris.store(
                    src_ptrs,
                    data,
                    iris_rank,
                    target_rank,
                    heap_bases,
                    mask=mask,
                    hint=(1, BLOCK_SIZE_N),
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
    """Launch the Triton broadcast kernel."""
    M, N = tensor.shape[:2]
    stride_m, stride_n = tensor.stride(0), tensor.stride(1)

    heap_bases = ctx.get_heap_bases()

    iris_launch(
        persistent_broadcast,
        (config.comm_sms,),
        tensor,
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
        src,
        config.block_size_m,
        config.block_size_n,
        config.swizzle_size,
        config.comm_sms,
        config.num_xcds,
        config.chunk_size,
        algorithm="broadcast",
        rank=rank_global,
        dtype=tensor.dtype,
    )
