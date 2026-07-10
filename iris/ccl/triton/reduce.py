# SPDX-License-Identifier: MIT
# Copyright (c) 2025 Advanced Micro Devices, Inc. All rights reserved.

"""
Triton kernels for reduce collective communication.

Two variants:
  - "one_shot": Gathers all inputs to root via iris.load (good for small messages)
  - "two_shot": Each rank reduces its tile partition, then sends result to root
                 (good for large messages, leverages symmetric heap)
"""

from dataclasses import dataclass
from typing import Optional, Tuple

import triton
import triton.language as tl
import torch
import iris
from iris.host.tracing.kernel_artifacts import iris_launch
from ..utils import chiplet_transform_chunked

# Variant types
VARIANT_ONE_SHOT = "one_shot"
VARIANT_TWO_SHOT = "two_shot"
VARIANT_RS_AG = "rs_ag"
VARIANT_RING = "ring"


@dataclass
class ReduceWorkspace:
    """
    Holds reusable workspace allocations for reduce variants.

    Attributes:
        variant: Selected reduce variant.
        shape: Tuple of (M, N) for tensor shape.
        dtype: Torch dtype of buffers.
        ring_buffer: Temporary buffer used by ring-based algorithm.
        flags: Synchronization flags for ring-based algorithm.
        root: Root rank for the reduce operation.
        prepared: Indicates whether preamble has been executed since last use.
    """

    variant: str = ""
    shape: Tuple[int, int] = ()
    dtype: Optional[torch.dtype] = None
    ring_buffer: Optional[torch.Tensor] = None
    flags: Optional[torch.Tensor] = None
    root: int = 0
    flags_per_tile: int = 0
    prepared: bool = False


def reduce_preamble(
    output_tensor,
    input_tensor,
    ctx,
    root=0,
    config=None,
    workspace=None,
):
    """
    Allocate and reset temporary buffers for the chosen variant.

    Returns:
        ReduceWorkspace instance ready for the next call to reduce.
    """
    from ..config import Config

    if config is None:
        config = Config()

    variant = getattr(config, "reduce_variant", "two_shot")

    M, N = input_tensor.shape[:2]
    dtype = input_tensor.dtype

    if workspace is None:
        workspace = ReduceWorkspace()

    workspace.variant = variant
    workspace.shape = (M, N)
    workspace.dtype = dtype
    workspace.root = root
    workspace.prepared = False

    if variant == VARIANT_ONE_SHOT:
        output_tensor.zero_()
        ctx.barrier()

    elif variant == VARIANT_RING:
        num_pid_m = (M + config.block_size_m - 1) // config.block_size_m
        num_pid_n = (N + config.block_size_n - 1) // config.block_size_n
        total_tiles = num_pid_m * num_pid_n
        workspace.flags_per_tile = 1
        total_flags = total_tiles * workspace.flags_per_tile
        if (
            workspace.ring_buffer is None
            or workspace.ring_buffer.shape != (M, N)
            or workspace.ring_buffer.dtype != dtype
        ):
            workspace.ring_buffer = ctx.zeros((M, N), dtype=dtype)
        else:
            workspace.ring_buffer.zero_()

        if workspace.flags is None or workspace.flags.numel() != total_flags:
            workspace.flags = ctx.zeros((total_flags,), dtype=torch.int32)
        else:
            workspace.flags.zero_()

        output_tensor.zero_()
        ctx.barrier()

    elif variant == VARIANT_TWO_SHOT:
        # Two-shot needs no extra workspace beyond output zeroing
        pass

    elif variant == VARIANT_RS_AG:
        # RS+AG: needs a scatter buffer on each rank to hold intermediate reduce-scatter results
        if (
            workspace.ring_buffer is None
            or workspace.ring_buffer.shape != (M, N)
            or workspace.ring_buffer.dtype != dtype
        ):
            workspace.ring_buffer = ctx.zeros((M, N), dtype=dtype)
        else:
            workspace.ring_buffer.zero_()
        ctx.barrier()

    workspace.prepared = True
    return workspace


@triton.jit()
def persistent_reduce_one_shot(
    input_ptr,
    output_ptr,
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
    ROOT: tl.constexpr,
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
    GROUP_SIZE_M: tl.constexpr,
    COMM_SMS: tl.constexpr,
    NUM_XCDS: tl.constexpr,
    CHUNK_SIZE: tl.constexpr,
):
    """
    One-shot reduce for small/latency-bound buffers.

    Only root rank gathers all partials via iris.load and writes the reduced result.
    Non-root ranks do nothing — where only root receives the result.

    root gathers all data directly (similar to recvReduceCopy for each peer).
    """
    # Only root rank does work
    if group_rank != ROOT:
        return

    pid = tl.program_id(0)

    if NUM_XCDS != 1:
        pid = chiplet_transform_chunked(pid, COMM_SMS, NUM_XCDS, CHUNK_SIZE)

    num_pid_m = tl.cdiv(M, BLOCK_SIZE_M)
    num_pid_n = tl.cdiv(N, BLOCK_SIZE_N)
    total_tiles = num_pid_m * num_pid_n

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

        is_full = (rm_base + BLOCK_SIZE_M <= M) & (rn_base + BLOCK_SIZE_N <= N)

        input_offset = rm[:, None] * stride_in_m + rn[None, :] * stride_in_n
        output_offset = rm[:, None] * stride_out_m + rn[None, :] * stride_out_n

        base_ptr = input_ptr + input_offset
        out_ptr = output_ptr + output_offset

        if is_full:
            # Fast path: no masks needed for full tiles
            # Gather from first rank
            first_rank = rank_start
            acc = iris.load(base_ptr, iris_rank, first_rank, heap_bases).to(acc_dtype)
            # Accumulate from remaining ranks
            for i in tl.static_range(1, world_size):
                remote_rank = rank_start + i * rank_stride
                acc += iris.load(base_ptr, iris_rank, remote_rank, heap_bases).to(acc_dtype)

            tl.store(out_ptr, acc.to(output_ptr.type.element_ty), cache_modifier=".wt")
        else:
            # Slow path: masked for boundary tiles
            mask = (rm[:, None] < M) & (rn[None, :] < N)

            first_rank = rank_start
            acc = iris.load(base_ptr, iris_rank, first_rank, heap_bases, mask=mask).to(acc_dtype)
            for i in tl.static_range(1, world_size):
                remote_rank = rank_start + i * rank_stride
                acc += iris.load(base_ptr, iris_rank, remote_rank, heap_bases, mask=mask).to(acc_dtype)

            tl.store(out_ptr, acc.to(output_ptr.type.element_ty), mask=mask, cache_modifier=".wt")


@triton.jit()
def persistent_reduce_two_shot(
    input_ptr,
    output_ptr,
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
    ROOT: tl.constexpr,
    ROOT_GLOBAL: tl.constexpr,
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
    GROUP_SIZE_M: tl.constexpr,
    COMM_SMS: tl.constexpr,
    NUM_XCDS: tl.constexpr,
    CHUNK_SIZE: tl.constexpr,
    DISTRIBUTION: tl.constexpr,
):
    """
    Two-shot reduce: each rank reduces its assigned tiles from all ranks, then
    sends the result to root.

    Leverages
    iris's symmetric heap for direct remote reads instead of ring forwarding.

    Phase 1 (Reduce): Each rank reads its assigned tiles from all other ranks
    via iris.load, accumulates locally in float32.

    Phase 2 (Send to root): Non-root ranks send their reduced tiles to root
    via iris.store. Root writes its own tiles locally.

    with iris's symmetric heap (no sequential forwarding through ring).
    """
    pid = tl.program_id(0)

    num_pid_m = tl.cdiv(M, BLOCK_SIZE_M)
    num_pid_n = tl.cdiv(N, BLOCK_SIZE_N)
    total_tiles = num_pid_m * num_pid_n

    acc_dtype = tl.float32 if output_ptr.type.element_ty != tl.int8 else tl.int32

    tiles_per_rank = tl.cdiv(total_tiles, world_size)
    if DISTRIBUTION == 0:
        start_tile = group_rank
        stride = world_size
        remaining = total_tiles - start_tile
        remaining = tl.maximum(remaining, 0)
        max_tile_offset = tl.cdiv(remaining, stride)
    else:
        start_tile = group_rank * tiles_per_rank
        stride = 1
        remaining = total_tiles - start_tile
        remaining = tl.maximum(remaining, 0)
        max_tile_offset = tl.minimum(tiles_per_rank, remaining)

    # Persistent traversal
    for tile_offset in range(pid, max_tile_offset, COMM_SMS):
        tile_id = start_tile + tile_offset * stride

        num_pid_in_group = GROUP_SIZE_M * num_pid_n
        group_id = tile_id // num_pid_in_group
        first_pid_m = group_id * GROUP_SIZE_M
        group_size_m = min(num_pid_m - first_pid_m, GROUP_SIZE_M)
        pid_m = first_pid_m + ((tile_id % num_pid_in_group) % group_size_m)
        pid_n = (tile_id % num_pid_in_group) // group_size_m

        rm_base = pid_m * BLOCK_SIZE_M
        rn_base = pid_n * BLOCK_SIZE_N

        is_full = (rm_base + BLOCK_SIZE_M <= M) & (rn_base + BLOCK_SIZE_N <= N)

        # Build indices
        rm = rm_base + tl.arange(0, BLOCK_SIZE_M)
        rn = rn_base + tl.arange(0, BLOCK_SIZE_N)

        rm = tl.max_contiguous(tl.multiple_of(rm, BLOCK_SIZE_M), BLOCK_SIZE_M)
        rn = tl.max_contiguous(tl.multiple_of(rn, BLOCK_SIZE_N), BLOCK_SIZE_N)

        input_offset = rm[:, None] * stride_in_m + rn[None, :] * stride_in_n
        output_offset = rm[:, None] * stride_out_m + rn[None, :] * stride_out_n

        base_ptr = input_ptr + input_offset
        out_ptr = output_ptr + output_offset

        # Fast path: NO MASKS (full tiles)
        if is_full:
            # Phase 1: Reduce — gather from all ranks
            start_rank_idx = pid % world_size
            start_rank_global = rank_start + start_rank_idx * rank_stride
            acc = iris.load(base_ptr, iris_rank, start_rank_global, heap_bases).to(acc_dtype)
            for i in tl.static_range(1, world_size):
                remote_rank_idx = (start_rank_idx + i) % world_size
                remote_rank = rank_start + remote_rank_idx * rank_stride
                acc += iris.load(base_ptr, iris_rank, remote_rank, heap_bases).to(acc_dtype)

            reduced = acc.to(output_ptr.type.element_ty)

            # Phase 2: Send to root only
            if group_rank == ROOT:
                # Root: write locally
                tl.store(out_ptr, reduced, cache_modifier=".wt")
            else:
                # Non-root: send to root via RMA
                iris.store(out_ptr, reduced, iris_rank, ROOT_GLOBAL, heap_bases, hint=(1, BLOCK_SIZE_N))

        # Slow path: MASKED (only boundary tiles land here)
        else:
            mask = (rm[:, None] < M) & (rn[None, :] < N)

            start_rank_idx = pid % world_size
            start_rank_global = rank_start + start_rank_idx * rank_stride
            acc = iris.load(base_ptr, iris_rank, start_rank_global, heap_bases, mask=mask).to(acc_dtype)
            for i in tl.static_range(1, world_size):
                remote_rank_idx = (start_rank_idx + i) % world_size
                remote_rank = rank_start + remote_rank_idx * rank_stride
                acc += iris.load(base_ptr, iris_rank, remote_rank, heap_bases, mask=mask).to(acc_dtype)

            reduced = acc.to(output_ptr.type.element_ty)

            if group_rank == ROOT:
                tl.store(out_ptr, reduced, mask=mask, cache_modifier=".wt")
            else:
                iris.store(
                    out_ptr,
                    reduced,
                    iris_rank,
                    ROOT_GLOBAL,
                    heap_bases,
                    mask=mask,
                    hint=(1, BLOCK_SIZE_N),
                )


@triton.jit()
def persistent_reduce_scatter_then_gather(
    input_ptr,
    output_ptr,
    scatter_buf,
    M,
    N,
    stride_in_m,
    stride_in_n,
    stride_out_m,
    stride_out_n,
    stride_sb_m,
    stride_sb_n,
    heap_bases: tl.tensor,
    group_rank: tl.constexpr,
    iris_rank: tl.constexpr,
    world_size: tl.constexpr,
    rank_start: tl.constexpr,
    rank_stride: tl.constexpr,
    ROOT: tl.constexpr,
    ROOT_GLOBAL: tl.constexpr,
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
    GROUP_SIZE_M: tl.constexpr,
    COMM_SMS: tl.constexpr,
    NUM_XCDS: tl.constexpr,
    CHUNK_SIZE: tl.constexpr,
    DISTRIBUTION: tl.constexpr,
):
    """
    Reduce via reduce-scatter + gather-to-root.

    Phase 1 (reduce-scatter): Each rank reduces its assigned 1/W partition of tiles
    from all ranks, storing the partial result into scatter_buf locally.
    Total XGMI reads: W × (data/W) per rank = data per rank.

    Phase 2 (gather to root): Root rank reads all partitions from other ranks'
    scatter_buf via iris.load. Non-root ranks do nothing in phase 2.
    Total XGMI reads (root only): (W-1)/W × data.

    Total XGMI traffic: data + (W-1)/W × data = (2W-1)/W × data ≈ 2× data
    vs two_shot: data + (W-1)/W × data (writes) = same total but writes cause
    contention at root, while this approach uses reads which are pull-based.

    Ring reduce approach (reduce-scatter + gather) but uses
    iris's symmetric heap for direct RMA instead of ring forwarding.
    """
    pid = tl.program_id(0)

    num_pid_m = tl.cdiv(M, BLOCK_SIZE_M)
    num_pid_n = tl.cdiv(N, BLOCK_SIZE_N)
    total_tiles = num_pid_m * num_pid_n

    acc_dtype = tl.float32 if output_ptr.type.element_ty != tl.int8 else tl.int32

    tiles_per_rank = tl.cdiv(total_tiles, world_size)
    if DISTRIBUTION == 0:
        start_tile = group_rank
        stride = world_size
        remaining = total_tiles - start_tile
        remaining = tl.maximum(remaining, 0)
        max_tile_offset = tl.cdiv(remaining, stride)
    else:
        start_tile = group_rank * tiles_per_rank
        stride = 1
        remaining = total_tiles - start_tile
        remaining = tl.maximum(remaining, 0)
        max_tile_offset = tl.minimum(tiles_per_rank, remaining)

    # ========== Phase 1: Reduce-scatter ==========
    # Each rank reduces its assigned tiles and stores locally into scatter_buf
    for tile_offset in range(pid, max_tile_offset, COMM_SMS):
        tile_id = start_tile + tile_offset * stride

        num_pid_in_group = GROUP_SIZE_M * num_pid_n
        group_id = tile_id // num_pid_in_group
        first_pid_m = group_id * GROUP_SIZE_M
        group_size_m = min(num_pid_m - first_pid_m, GROUP_SIZE_M)
        pid_m = first_pid_m + ((tile_id % num_pid_in_group) % group_size_m)
        pid_n = (tile_id % num_pid_in_group) // group_size_m

        rm_base = pid_m * BLOCK_SIZE_M
        rn_base = pid_n * BLOCK_SIZE_N

        is_full = (rm_base + BLOCK_SIZE_M <= M) & (rn_base + BLOCK_SIZE_N <= N)

        rm = rm_base + tl.arange(0, BLOCK_SIZE_M)
        rn = rn_base + tl.arange(0, BLOCK_SIZE_N)
        rm = tl.max_contiguous(tl.multiple_of(rm, BLOCK_SIZE_M), BLOCK_SIZE_M)
        rn = tl.max_contiguous(tl.multiple_of(rn, BLOCK_SIZE_N), BLOCK_SIZE_N)

        input_offset = rm[:, None] * stride_in_m + rn[None, :] * stride_in_n
        sb_offset = rm[:, None] * stride_sb_m + rn[None, :] * stride_sb_n

        base_ptr = input_ptr + input_offset
        sb_ptr = scatter_buf + sb_offset

        if is_full:
            start_rank_idx = pid % world_size
            start_rank_global = rank_start + start_rank_idx * rank_stride
            acc = iris.load(base_ptr, iris_rank, start_rank_global, heap_bases).to(acc_dtype)
            for i in tl.static_range(1, world_size):
                remote_rank_idx = (start_rank_idx + i) % world_size
                remote_rank = rank_start + remote_rank_idx * rank_stride
                acc += iris.load(base_ptr, iris_rank, remote_rank, heap_bases).to(acc_dtype)
            tl.store(sb_ptr, acc.to(output_ptr.type.element_ty), cache_modifier=".wt")
        else:
            mask = (rm[:, None] < M) & (rn[None, :] < N)
            start_rank_idx = pid % world_size
            start_rank_global = rank_start + start_rank_idx * rank_stride
            acc = iris.load(base_ptr, iris_rank, start_rank_global, heap_bases, mask=mask).to(acc_dtype)
            for i in tl.static_range(1, world_size):
                remote_rank_idx = (start_rank_idx + i) % world_size
                remote_rank = rank_start + remote_rank_idx * rank_stride
                acc += iris.load(base_ptr, iris_rank, remote_rank, heap_bases, mask=mask).to(acc_dtype)
            tl.store(sb_ptr, acc.to(output_ptr.type.element_ty), mask=mask, cache_modifier=".wt")

    # ========== Phase 2: Gather to root ==========
    # Only root rank gathers all partitions
    if group_rank == ROOT:
        # Root needs to gather ALL tiles — its own partition AND from every other rank
        for tile_id in range(pid, total_tiles, COMM_SMS):
            num_pid_in_group = GROUP_SIZE_M * num_pid_n
            group_id = tile_id // num_pid_in_group
            first_pid_m = group_id * GROUP_SIZE_M
            group_size_m = min(num_pid_m - first_pid_m, GROUP_SIZE_M)
            pid_m = first_pid_m + ((tile_id % num_pid_in_group) % group_size_m)
            pid_n = (tile_id % num_pid_in_group) // group_size_m

            rm_base = pid_m * BLOCK_SIZE_M
            rn_base = pid_n * BLOCK_SIZE_N

            rm = rm_base + tl.arange(0, BLOCK_SIZE_M)
            rn = rn_base + tl.arange(0, BLOCK_SIZE_N)
            rm = tl.max_contiguous(tl.multiple_of(rm, BLOCK_SIZE_M), BLOCK_SIZE_M)
            rn = tl.max_contiguous(tl.multiple_of(rn, BLOCK_SIZE_N), BLOCK_SIZE_N)

            sb_offset = rm[:, None] * stride_sb_m + rn[None, :] * stride_sb_n
            out_offset = rm[:, None] * stride_out_m + rn[None, :] * stride_out_n

            is_full = (rm_base + BLOCK_SIZE_M <= M) & (rn_base + BLOCK_SIZE_N <= N)

            # Determine which rank owns this tile
            if DISTRIBUTION == 0:
                owner_rank = tile_id % world_size
            else:
                owner_rank = tile_id // tiles_per_rank
                owner_rank = tl.minimum(owner_rank, world_size - 1)

            owner_global = rank_start + owner_rank * rank_stride

            if is_full:
                if owner_rank == ROOT:
                    # Root's own tiles — read from local scatter_buf
                    tile_data = tl.load(scatter_buf + sb_offset)
                else:
                    # Read from remote rank's scatter_buf
                    tile_data = iris.load(scatter_buf + sb_offset, iris_rank, owner_global, heap_bases)
                tl.store(output_ptr + out_offset, tile_data, cache_modifier=".wt")
            else:
                mask = (rm[:, None] < M) & (rn[None, :] < N)
                if owner_rank == ROOT:
                    tile_data = tl.load(scatter_buf + sb_offset, mask=mask, other=0)
                else:
                    tile_data = iris.load(scatter_buf + sb_offset, iris_rank, owner_global, heap_bases, mask=mask)
                tl.store(output_ptr + out_offset, tile_data, mask=mask, cache_modifier=".wt")


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
    world_size: tl.constexpr,
    rank_start: tl.constexpr,
    rank_stride: tl.constexpr,
    next_rank: tl.constexpr,
    prev_rank: tl.constexpr,
    ROOT: tl.constexpr,
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
    GROUP_SIZE_M: tl.constexpr,
    COMM_SMS: tl.constexpr,
    NUM_XCDS: tl.constexpr,
    CHUNK_SIZE: tl.constexpr,
    FLAGS_PER_TILE: tl.constexpr,
):
    """
    Ring-based reduce kernel that streams tiles around the ring toward root.

    Ring reduce kernel:
    - prevRank == root: just sends its data to next rank
    - rank == root: receives and reduces (recvReduceCopy)
    - intermediate: receives, reduces with local data, and forwards (recvReduceSend)

    The ring is ordered so data flows through all ranks and converges on root.
    Each hop reduces one more rank's contribution.

    Uses flag-based producer/consumer handshake for synchronization,
    with step-based flow control.
    """
    pid_raw = tl.program_id(0)

    pid = pid_raw
    if NUM_XCDS != 1:
        pid = chiplet_transform_chunked(pid_raw, COMM_SMS, NUM_XCDS, CHUNK_SIZE)

    tl.static_assert(FLAGS_PER_TILE >= 1, "FLAGS_PER_TILE must be at least 1")

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
            rm = tl.max_contiguous(tl.multiple_of(rm, BLOCK_SIZE_M), BLOCK_SIZE_M)

            rn = rn_base + tl.arange(0, BLOCK_SIZE_N)
            mask = (rm[:, None] < M) & (rn[None, :] < N)
            tile_offset = rm[:, None] * stride_in_m + rn[None, :] * stride_in_n
            out_tile_offset = rm[:, None] * stride_out_m + rn[None, :] * stride_out_n

            # Load local data
            local_tile = tl.load(input_ptr + tile_offset, mask=mask, other=0)

            flag_offset = tile_id * FLAGS_PER_TILE
            remote_flag_ptr = flags + flag_offset
            local_flag_ptr = flags + flag_offset

            # 3-phase logic:
            # Phase 1: prevRank == root → just send
            # Phase 2: rank == root → recvReduceCopy
            # Phase 3: intermediate → recvReduceSend

            # Determine rank's role in the ring
            # prevRank is the rank that sends TO this rank
            # The ring flows: ... → prev_rank → this_rank → next_rank → ...
            # Data converges on root

            if prev_rank == ROOT:
                # Phase 1: We are the rank right after root in the ring.
                # Root's predecessor just sends its own data.
                # But here, WE are prev_rank==root's successor, meaning
                # we need to send our data and it eventually reaches root.
                # "if prevRank == root" means the previous
                # rank in the ring is root, so THIS rank is the first sender.
                # It just sends its data without receiving anything.
                # Send local data to next rank
                # Wait for next rank's buffer to be ready
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
                    local_tile,
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

            elif group_rank == ROOT:
                # Phase 2: Root rank — receive and reduce.
                # Root receives from its predecessor, reduces with local data,
                # and stores the final result.
                # Wait for incoming data
                while tl.atomic_cas(local_flag_ptr, 0, 0, sem="acquire", scope="sys") != 1:
                    pass

                recv_tile = tl.load(ring_buffer + tile_offset, mask=mask, other=0)
                acc = local_tile.to(acc_dtype) + recv_tile.to(acc_dtype)
                tl.debug_barrier()
                tl.atomic_xchg(local_flag_ptr, 0, sem="release", scope="sys")

                # Store final reduced result to output
                tl.store(
                    output_ptr + out_tile_offset,
                    acc.to(output_ptr.type.element_ty),
                    mask=mask,
                )

            else:
                # Phase 3: Intermediate rank — receive, reduce, forward.
                # Wait for incoming data from predecessor
                while tl.atomic_cas(local_flag_ptr, 0, 0, sem="acquire", scope="sys") != 1:
                    pass

                recv_tile = tl.load(ring_buffer + tile_offset, mask=mask, other=0)
                acc = local_tile.to(acc_dtype) + recv_tile.to(acc_dtype)
                send_data = acc.to(input_ptr.type.element_ty)
                tl.debug_barrier()
                tl.atomic_xchg(local_flag_ptr, 0, sem="release", scope="sys")

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
    world_size,
    rank_start,
    rank_stride,
    root,
    config,
    workspace,
    group=None,
):
    """Launch the appropriate Triton reduce kernel variant."""
    M, N = input_tensor.shape[:2]
    stride_in_m, stride_in_n = input_tensor.stride(0), input_tensor.stride(1)
    stride_out_m, stride_out_n = output_tensor.stride(0), output_tensor.stride(1)

    variant = getattr(config, "reduce_variant", "two_shot")

    needs_prepare = (
        workspace is None
        or not getattr(workspace, "prepared", False)
        or workspace.variant != variant
        or workspace.shape != (M, N)
        or workspace.dtype != input_tensor.dtype
        or workspace.root != root
    )

    if needs_prepare:
        workspace = reduce_preamble(
            output_tensor,
            input_tensor,
            ctx,
            root=root,
            config=config,
            workspace=workspace,
        )

    heap_bases = ctx.get_heap_bases()

    # Calculate root's global rank for iris RMA operations
    root_global = rank_start + root * rank_stride

    if variant == VARIANT_ONE_SHOT:
        iris_launch(
            persistent_reduce_one_shot,
            (config.comm_sms,),
            input_tensor,
            output_tensor,
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

    elif variant == VARIANT_TWO_SHOT:
        iris_launch(
            persistent_reduce_two_shot,
            (config.comm_sms,),
            input_tensor,
            output_tensor,
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
            root_global,
            config.block_size_m,
            config.block_size_n,
            config.swizzle_size,
            config.comm_sms,
            config.num_xcds,
            config.chunk_size,
            config.all_reduce_distribution,
            num_warps=8,
            num_stages=1,
            waves_per_eu=1,
            algorithm="reduce",
            rank=rank_global,
            dtype=input_tensor.dtype,
        )

    elif variant == VARIANT_RS_AG:
        if workspace is None or workspace.ring_buffer is None:
            raise RuntimeError("rs_ag variant requires workspace preparation. Call reduce_preamble before reduce.")
        scatter_buf = workspace.ring_buffer
        stride_sb_m, stride_sb_n = scatter_buf.stride(0), scatter_buf.stride(1)

        iris_launch(
            persistent_reduce_scatter_then_gather,
            (config.comm_sms,),
            input_tensor,
            output_tensor,
            scatter_buf,
            M,
            N,
            stride_in_m,
            stride_in_n,
            stride_out_m,
            stride_out_n,
            stride_sb_m,
            stride_sb_n,
            heap_bases,
            rank_in_group,
            rank_global,
            world_size,
            rank_start,
            rank_stride,
            root,
            root_global,
            config.block_size_m,
            config.block_size_n,
            config.swizzle_size,
            config.comm_sms,
            config.num_xcds,
            config.chunk_size,
            config.all_reduce_distribution,
            num_warps=8,
            num_stages=1,
            waves_per_eu=1,
            algorithm="reduce",
            rank=rank_global,
            dtype=input_tensor.dtype,
        )

    elif variant == VARIANT_RING:
        if workspace is None or workspace.ring_buffer is None or workspace.flags is None:
            raise RuntimeError("Ring variant requires workspace preparation. Call reduce_preamble before reduce.")

        # Build ring topology for reduce toward root
        # Ring order: root+1 → root+2 → ... → root-1 → root
        # Data flows through the ring and converges on root
        if group is None:
            next_rank_in_group = (rank_in_group + 1) % world_size
            prev_rank_in_group = (rank_in_group - 1) % world_size
            next_rank = next_rank_in_group
            prev_rank = prev_rank_in_group
        else:
            import torch.distributed as dist

            group_ranks = dist.get_process_group_ranks(group)
            next_rank_in_group = (rank_in_group + 1) % world_size
            prev_rank_in_group = (rank_in_group - 1) % world_size
            next_rank = group_ranks[next_rank_in_group]
            prev_rank = prev_rank_in_group  # prev_rank is group-local for role determination

        iris_launch(
            persistent_reduce_ring,
            (config.comm_sms,),
            input_tensor,
            output_tensor,
            workspace.ring_buffer,
            workspace.flags,
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
            next_rank,
            prev_rank,
            root,
            config.block_size_m,
            config.block_size_n,
            config.swizzle_size,
            config.comm_sms,
            config.num_xcds,
            config.chunk_size,
            workspace.flags_per_tile,
            algorithm="reduce",
            rank=rank_global,
            dtype=input_tensor.dtype,
        )

    return workspace
