# SPDX-License-Identifier: MIT
# Copyright (c) 2025 Advanced Micro Devices, Inc. All rights reserved.

"""
Broadcast — Triton kernel.

Single-kernel all-ranks push: the root rank loads each tile once and
iris.store's it to every other rank.  Non-root ranks exit immediately.
One kernel launch, one trailing barrier — no tree, no host-side steps.
"""

import triton
import triton.language as tl
import iris
from iris.host.tracing.kernel_artifacts import iris_launch


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
    """
    Persistent broadcast kernel — single-kernel all-ranks push.

    Only the root rank (group_rank == src) does any work.  It loads each
    tile of the source tensor once and pushes it to every other rank via
    iris.store.  Non-root ranks return immediately.

    Args:
        tensor_ptr: Pointer to the tensor on the symmetric heap (in-place).
        M: Number of rows.
        N: Number of columns.
        stride_m, stride_n: Strides of the tensor.
        heap_bases: Heap base pointers for all ranks.
        group_rank: Rank within the ProcessGroup (0 to group_size-1).
        iris_rank: Rank in the iris context (for heap_bases indexing).
        src: Root rank within the group that owns the data.
        world_size: Total number of ranks in the group.
        rank_start: First iris rank in the group.
        rank_stride: Stride between iris ranks in the group.
        BLOCK_SIZE_M, BLOCK_SIZE_N: Block sizes for tiling.
        GROUP_SIZE_M: Group size for M dimension swizzle.
        COMM_SMS: Number of SMs for communication (grid size).
        NUM_XCDS: Number of XCDs.
        CHUNK_SIZE: Chunk size for chiplet transform.
    """
    pid = tl.program_id(0)

    # Only root rank does work
    if group_rank != src:
        return

    num_pid_m = tl.cdiv(M, BLOCK_SIZE_M)
    num_pid_n = tl.cdiv(N, BLOCK_SIZE_N)
    total_tiles = num_pid_m * num_pid_n
    tl.assume(total_tiles > 0)

    for tile_id in range(pid, total_tiles, COMM_SMS):
        # Swizzle tiling (same pattern as all_gather)
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

        # Compute row and column indices
        rm_base = pid_m * BLOCK_SIZE_M
        rn_base = pid_n * BLOCK_SIZE_N
        rm = rm_base + tl.arange(0, BLOCK_SIZE_M)
        rn = rn_base + tl.arange(0, BLOCK_SIZE_N)
        rm = tl.max_contiguous(tl.multiple_of(rm, BLOCK_SIZE_M), BLOCK_SIZE_M)
        rn = tl.max_contiguous(tl.multiple_of(rn, BLOCK_SIZE_N), BLOCK_SIZE_N)

        mask = (rm[:, None] < M) & (rn[None, :] < N)

        # Compute offset and load tile once
        offset = rm[:, None] * stride_m + rn[None, :] * stride_n
        src_ptrs = tensor_ptr + offset
        src_ptrs = tl.multiple_of(src_ptrs, (BLOCK_SIZE_M, BLOCK_SIZE_N))
        data = tl.load(src_ptrs, mask=mask, other=0.0)

        # Push to every other rank
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
    """Launch the Triton broadcast kernel.

    Args:
        tensor: Flattened 1-D tensor on the symmetric heap.
        ctx: Iris instance.
        rank_in_group: Rank within the ProcessGroup.
        rank_global: Global iris rank.
        world_size: Number of ranks in the group.
        rank_start: First iris rank in the group.
        rank_stride: Stride between iris ranks.
        src: Root rank within the group.
        config: Config with kernel parameters.
    """
    M = tensor.shape[0]
    N = 1
    stride_m = tensor.stride(0)
    stride_n = 1

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
