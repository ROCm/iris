# SPDX-License-Identifier: MIT
# Copyright (c) 2025 Advanced Micro Devices, Inc. All rights reserved.

"""
All-reduce collective communication primitive for Iris.
Supports multiple variants: atomic, spinlock, ring, two-shot, and one-shot.
"""

from dataclasses import dataclass
from typing import Optional, Tuple

import triton
import triton.language as tl
import torch
import iris
from iris.tracing.kernel_artifacts import iris_launch
from .config import Config
from .utils import chiplet_transform_chunked, ReduceOp, extract_group_info

# Conditional import for Gluon
try:
    from triton.experimental import gluon
    from triton.experimental.gluon import language as gl
    from iris.experimental.iris_gluon import IrisDeviceCtx

    GLUON_AVAILABLE = True
except ImportError:
    GLUON_AVAILABLE = False

# Variant types
VARIANT_ATOMIC = "atomic"
VARIANT_RING = "ring"
VARIANT_TWO_SHOT = "two_shot"
VARIANT_ONE_SHOT = "one_shot"
VARIANT_SPINLOCK = "spinlock"
VARIANT_LL = "ll"
VARIANT_LL128 = "ll128"
VARIANT_RCCL_LL = "rccl_ll"


@dataclass
class AllReduceWorkspace:
    """
    Holds reusable workspace allocations for all-reduce variants.

    Attributes:
        variant: Selected all-reduce variant.
        shape: Tuple of (M, N) for tensor shape.
        dtype: Torch dtype of buffers.
        ring_buffer: Temporary buffer used by ring-based algorithm.
        flags: Synchronization flags for ring-based algorithm.
        num_rings: Number of concurrent rings prepared for ring-based variant.
        prepared: Indicates whether preamble has been executed since last use.
    """

    variant: str = ""
    shape: Tuple[int, int] = ()
    dtype: Optional[torch.dtype] = None
    ring_buffer: Optional[torch.Tensor] = None
    flags: Optional[torch.Tensor] = None
    locks: Optional[torch.Tensor] = None
    num_rings: int = 1
    flags_per_tile: int = 0
    prepared: bool = False


def all_reduce_preamble(
    output_tensor,
    input_tensor,
    ctx,
    config: Optional[Config] = None,
    workspace: Optional[AllReduceWorkspace] = None,
):
    """
    Allocate and reset temporary buffers for the chosen variant.

    Returns:
        AllReduceWorkspace instance ready for the next call to all_reduce.
    """
    if config is None:
        config = Config()

    variant = config.all_reduce_variant.lower()
    if variant not in [
        VARIANT_ATOMIC,
        VARIANT_RING,
        VARIANT_TWO_SHOT,
        VARIANT_ONE_SHOT,
        VARIANT_SPINLOCK,
        VARIANT_LL,
        VARIANT_LL128,
        VARIANT_RCCL_LL,
    ]:
        raise ValueError(
            f"Invalid all_reduce_variant: {variant}. Must be one of: {VARIANT_ATOMIC}, {VARIANT_RING}, {VARIANT_TWO_SHOT}, {VARIANT_ONE_SHOT}, {VARIANT_SPINLOCK}, {VARIANT_LL}, {VARIANT_LL128}, {VARIANT_RCCL_LL}"
        )

    M, N = input_tensor.shape[:2]
    dtype = input_tensor.dtype

    if workspace is None:
        workspace = AllReduceWorkspace()

    workspace.variant = variant
    workspace.shape = (M, N)
    workspace.dtype = dtype
    workspace.num_rings = getattr(config, "all_reduce_num_rings", 1)
    workspace.prepared = False

    if variant in (VARIANT_ATOMIC, VARIANT_SPINLOCK, VARIANT_ONE_SHOT):
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

    elif variant == VARIANT_LL:
        # LL uses per-rank flags for in-kernel synchronization (no barrier).
        num_ranks = ctx.get_num_ranks()
        if workspace.flags is None or workspace.flags.numel() != num_ranks:
            workspace.flags = ctx.zeros((num_ranks,), dtype=torch.int32)
        if not hasattr(workspace, "ll_epoch"):
            workspace.ll_epoch = 0

    elif variant == VARIANT_LL128:
        # LL128: f32 staging buffer with data+flag per cache line.
        # 31 f32 data + 1 f32 flag = 32 f32 = 128 bytes per cache line.
        payload = 31
        total_elems = M * N
        num_lines = (total_elems + payload - 1) // payload
        staging_size = num_lines * 32  # 32 f32 per line
        if workspace.ring_buffer is None or workspace.ring_buffer.numel() != staging_size:
            workspace.ring_buffer = ctx.zeros((staging_size,), dtype=torch.float32)
        if not hasattr(workspace, "ll_epoch"):
            workspace.ll_epoch = 0

    elif variant == VARIANT_RCCL_LL:
        # RCCL-style LL: interleaved [data_f32, flag_i32] per element.
        # Buffer size: 2 * M * N float32 values (8 bytes per element).
        # Flag = step counter. Epoch tracks generation to avoid stale flags.
        buf_size = 2 * M * N
        if workspace.ring_buffer is None or workspace.ring_buffer.numel() != buf_size:
            workspace.ring_buffer = ctx.zeros((buf_size,), dtype=torch.float32)
        if not hasattr(workspace, "ll_epoch"):
            workspace.ll_epoch = 0

    elif variant == VARIANT_TWO_SHOT:
        pass

    if variant == VARIANT_SPINLOCK:
        num_pid_m = (M + config.block_size_m - 1) // config.block_size_m
        num_pid_n = (N + config.block_size_n - 1) // config.block_size_n
        total_tiles = num_pid_m * num_pid_n
        if workspace.locks is None or workspace.locks.numel() != total_tiles:
            workspace.locks = ctx.zeros((total_tiles,), dtype=torch.int32)
        else:
            workspace.locks.zero_()

    workspace.prepared = True
    return workspace


@triton.jit()
def persistent_all_reduce_atomic(
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
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
    GROUP_SIZE_M: tl.constexpr,
    COMM_SMS: tl.constexpr,
    NUM_XCDS: tl.constexpr,
    CHUNK_SIZE: tl.constexpr,
):
    """
    Atomic-based all-reduce kernel.

    Each rank atomically adds its local partial result to the global output buffer.
    All ranks write to all locations using atomic operations.

    Args:
        input_ptr: Pointer to input tensor (local rank's partial data)
        output_ptr: Pointer to output tensor (will contain sum of all ranks)
        M: Number of rows
        N: Number of columns
        heap_bases: Heap base pointers for all ranks
        group_rank: Rank within the ProcessGroup (0 to group_size-1), used for tile assignment and comparisons
        iris_rank: Rank in the iris context, used for iris RMA operations (heap_bases indexing)
        world_size: Total number of ranks in the group
    """
    pid = tl.program_id(0)

    # Use chiplet transform to distribute program IDs across XCDs
    if NUM_XCDS != 1:
        pid = chiplet_transform_chunked(pid, COMM_SMS, NUM_XCDS, CHUNK_SIZE)

    num_pid_m = tl.cdiv(M, BLOCK_SIZE_M)
    num_pid_n = tl.cdiv(N, BLOCK_SIZE_N)
    total_tiles = num_pid_m * num_pid_n

    for tile_id in range(pid, total_tiles, COMM_SMS):
        num_pid_in_group = GROUP_SIZE_M * num_pid_n
        group_id = tile_id // num_pid_in_group
        first_pid_m = group_id * GROUP_SIZE_M
        group_size_m = min(num_pid_m - first_pid_m, GROUP_SIZE_M)
        pid_m = first_pid_m + ((tile_id % num_pid_in_group) % group_size_m)
        pid_n = (tile_id % num_pid_in_group) // group_size_m

        tl.assume(pid_m >= 0)
        tl.assume(pid_n >= 0)

        # Compute row and column indices
        # Calculate base indices without modulo to avoid double-counting when blocks are larger than dimensions
        rm_base = pid_m * BLOCK_SIZE_M
        rn_base = pid_n * BLOCK_SIZE_N
        rm = rm_base + tl.arange(0, BLOCK_SIZE_M)
        rn = rn_base + tl.arange(0, BLOCK_SIZE_N)
        rm = tl.max_contiguous(tl.multiple_of(rm, BLOCK_SIZE_M), BLOCK_SIZE_M)
        rn = tl.max_contiguous(tl.multiple_of(rn, BLOCK_SIZE_N), BLOCK_SIZE_N)
        # Create mask to prevent out-of-bounds access
        mask = (rm[:, None] < M) & (rn[None, :] < N)

        # Use the original rm/rn for offsets (mask will prevent out-of-bounds access)
        # This avoids double-counting that occurs with modulo when block_size > dimension
        input_offset = rm[:, None] * stride_in_m + rn[None, :] * stride_in_n
        output_offset = rm[:, None] * stride_out_m + rn[None, :] * stride_out_n

        input_ptr_local = input_ptr + input_offset
        input_ptr_local = tl.multiple_of(input_ptr_local, (BLOCK_SIZE_M, BLOCK_SIZE_N))

        # Load local partial result
        data = tl.load(input_ptr_local, mask=mask)

        # Atomically add to output buffer on all ranks
        # Each rank's output tensor is in its own heap, accessible via RMA
        for i in range(world_size):
            target_rank = rank_start + i * rank_stride
            if i == group_rank:
                # For the current rank (i == group_rank), use local atomic add
                # output_ptr is already in current rank's address space
                tl.atomic_add(output_ptr + output_offset, data, mask=mask)
            else:
                # For remote ranks, use iris.atomic_add to translate pointer
                # This accesses the remote rank's heap via RMA
                # Use iris_rank for iris operations (heap_bases indexing)
                iris.atomic_add(
                    output_ptr + output_offset,
                    data,
                    iris_rank,
                    target_rank,
                    heap_bases,
                    mask=mask,
                )
        # Ensure all atomic operations complete before moving to next tile
        tl.debug_barrier()


@triton.jit()
def persistent_all_reduce_spinlock(
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
    Spinlock-based all-reduce kernel that mimics an “atomic add” by using a lock per tile.

    Each tile acquires its lock across the entire system before accumulating remote
    partials locally, then writes the reduced result once and releases the lock.
    Atomics are used only for CAS/XCHG (lock/unlock); the accumulation itself is done
    with ordinary loads/stores.
    """
    pid = tl.program_id(0)

    if NUM_XCDS != 1:
        pid = chiplet_transform_chunked(pid, COMM_SMS, NUM_XCDS, CHUNK_SIZE)

    num_pid_m = tl.cdiv(M, BLOCK_SIZE_M)
    num_pid_n = tl.cdiv(N, BLOCK_SIZE_N)
    total_tiles = num_pid_m * num_pid_n

    acc_dtype = tl.float32 if output_ptr.type.element_ty != tl.int8 else tl.int32

    for tile_id in range(pid, total_tiles, COMM_SMS):
        # Compute tile coordinates
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

        # For each destination rank, do spinlock-protected read-modify-write
        for i in range(world_size):
            dest_rank = rank_start + i * rank_stride

            # Acquire lock for this tile at dest_rank using iris RMA
            while (
                iris.atomic_cas(locks_ptr + tile_id, 0, 1, iris_rank, dest_rank, heap_bases, sem="acquire", scope="sys")
                != 0
            ):
                pass

            # Load current value from dest_rank's output tile
            current_value = iris.load(
                output_ptr + output_offset,
                iris_rank,
                dest_rank,
                heap_bases,
                mask=mask,
                hint=(1, BLOCK_SIZE_N),
            )

            # Add our local contribution
            acc = current_value.to(acc_dtype) + local_data.to(acc_dtype)

            # Store accumulated result back to dest_rank
            result = acc.to(output_ptr.type.element_ty)
            iris.store(
                output_ptr + output_offset,
                result,
                iris_rank,
                dest_rank,
                heap_bases,
                mask=mask,
                hint=(1, BLOCK_SIZE_N),
            )

            # Release lock for this tile at dest_rank
            iris.atomic_xchg(locks_ptr + tile_id, 0, iris_rank, dest_rank, heap_bases, sem="release", scope="sys")


@triton.jit()
def persistent_all_reduce_one_shot(
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
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
    GROUP_SIZE_M: tl.constexpr,
    COMM_SMS: tl.constexpr,
    NUM_XCDS: tl.constexpr,
    CHUNK_SIZE: tl.constexpr,
):
    """
    One-shot all-reduce for small/latency-bound buffers.

    Each CTA gathers all partials directly using iris.load and writes the final result once.
    """
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
        mask = (rm[:, None] < M) & (rn[None, :] < N)

        input_offset = rm[:, None] * stride_in_m + rn[None, :] * stride_in_n
        output_offset = rm[:, None] * stride_out_m + rn[None, :] * stride_out_n

        acc = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=acc_dtype)

        for i in range(world_size):
            remote_rank = rank_start + i * rank_stride
            partial = iris.load(
                input_ptr + input_offset,
                iris_rank,
                remote_rank,
                heap_bases,
                mask=mask,
                hint=(1, BLOCK_SIZE_N),
            )
            acc += partial.to(acc_dtype)

        tl.store(
            output_ptr + output_offset,
            acc.to(output_ptr.type.element_ty),
            mask=mask,
        )


@triton.jit()
def persistent_all_reduce_ring(
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
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
    GROUP_SIZE_M: tl.constexpr,
    COMM_SMS: tl.constexpr,
    NUM_XCDS: tl.constexpr,
    CHUNK_SIZE: tl.constexpr,
    NUM_RINGS: tl.constexpr,
    SLICE_SIZE_N: tl.constexpr,
    FLAGS_PER_TILE: tl.constexpr,
):
    """
    Ring-based all-reduce kernel that streams whole tiles around the ring using a
    single-buffer, producer/consumer handshake.

    Each rank keeps a running accumulator for its local tile, forwards the tile it
    just received to its successor, and consumes the predecessor's contribution in
    lock-step.  After (world_size - 1) hops every rank has seen all partial tiles,
    so the accumulator holds the fully reduced result which is written back locally.
    """
    pid_raw = tl.program_id(0)

    # Use chiplet transform to distribute program IDs across XCDs
    pid = pid_raw
    if NUM_XCDS != 1:
        pid = chiplet_transform_chunked(pid_raw, COMM_SMS, NUM_XCDS, CHUNK_SIZE)

    tl.static_assert(NUM_RINGS > 0, "NUM_RINGS must be >= 1")
    tl.static_assert(FLAGS_PER_TILE >= 1, "FLAGS_PER_TILE must be at least 1")

    num_pid_m = tl.cdiv(M, BLOCK_SIZE_M)
    num_pid_n = tl.cdiv(N, BLOCK_SIZE_N)
    total_tiles = num_pid_m * num_pid_n

    # Ring topology: next_rank is passed in from Python side
    # for group support

    acc_dtype = tl.float32 if output_ptr.type.element_ty != tl.int8 else tl.int32
    elem_ty = input_ptr.type.element_ty

    # Partition CTAs across rings to form NUM_RINGS concurrent rings.
    ctas_per_ring = (COMM_SMS + NUM_RINGS - 1) // NUM_RINGS
    ring_id = pid % NUM_RINGS
    cta_in_ring = pid // NUM_RINGS

    if (cta_in_ring < ctas_per_ring) and (total_tiles > 0) and (total_tiles > ring_id):
        tiles_per_ring = (total_tiles - ring_id + NUM_RINGS - 1) // NUM_RINGS
        for tile_index_in_ring in range(cta_in_ring, tiles_per_ring, ctas_per_ring):
            tile_id = ring_id + tile_index_in_ring * NUM_RINGS
            if tile_id < total_tiles:
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

                local_tile = tl.load(input_ptr + tile_offset, mask=mask, other=0)
                acc = local_tile.to(acc_dtype)
                send_data = local_tile

                flag_offset = tile_id * FLAGS_PER_TILE
                remote_flag_ptr = flags + flag_offset
                local_flag_ptr = flags + flag_offset

                for _step in range(0, world_size - 1):
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

                    while tl.atomic_cas(local_flag_ptr, 0, 0, sem="acquire", scope="sys") != 1:
                        pass

                    recv_tile = tl.load(ring_buffer + tile_offset, mask=mask, other=0)
                    acc += recv_tile.to(acc_dtype)
                    send_data = recv_tile
                    tl.debug_barrier()
                    tl.atomic_xchg(local_flag_ptr, 0, sem="release", scope="sys")

                tl.store(
                    output_ptr + tile_offset,
                    acc.to(output_ptr.type.element_ty),
                    mask=mask,
                )


@triton.jit()
def persistent_all_reduce_ll(
    input_ptr,
    output_ptr,
    flags_ptr,
    epoch,
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
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
    GROUP_SIZE_M: tl.constexpr,
    COMM_SMS: tl.constexpr,
    NUM_XCDS: tl.constexpr,
    CHUNK_SIZE: tl.constexpr,
):
    """
    Low-latency all-reduce with in-kernel flag synchronization.

    Every rank reads all tiles from all peers, reduces, writes locally.
    Synchronization via per-rank flags in shared memory -- no external
    barrier needed.

    Protocol:
    1. CTA 0 on each rank writes epoch to its own flag slot on every peer
       (signals "my input is ready").
    2. All CTAs poll local flag slots for all ranks until they all == epoch.
    3. All CTAs read from all ranks, reduce, write locally.
    """
    pid = tl.program_id(0)

    # --- Phase 1: Signal readiness to all peers ---
    if pid == 0:
        # Set our own flag locally (iris may not handle self-writes)
        tl.atomic_xchg(flags_ptr + group_rank, epoch, sem="release", scope="sys")
        # Signal to all remote peers
        for i in tl.static_range(world_size):
            remote_rank = rank_start + i * rank_stride
            iris.atomic_xchg(
                flags_ptr + group_rank,
                epoch,
                iris_rank,
                remote_rank,
                heap_bases,
                sem="release",
                scope="sys",
            )

    # --- Phase 2: Wait for all peers to be ready ---
    # Use >= comparison: a higher epoch means the peer already finished
    # this epoch's data, so it's safe to read.  atomic_add(0) is an
    # atomic load.
    for i in tl.static_range(world_size):
        while tl.atomic_add(flags_ptr + i, 0, sem="acquire", scope="sys") < epoch:
            pass

    # --- Phase 3: Read-reduce-write ---
    num_pid_m = tl.cdiv(M, BLOCK_SIZE_M)
    num_pid_n = tl.cdiv(N, BLOCK_SIZE_N)
    total_tiles = num_pid_m * num_pid_n

    acc_dtype = tl.float32 if output_ptr.type.element_ty != tl.int8 else tl.int32

    for tile_offset in range(pid, total_tiles, COMM_SMS):
        tile_id = tile_offset

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
        output_offset = rm[:, None] * stride_out_m + rn[None, :] * stride_out_n

        base_ptr = input_ptr + input_offset
        out_ptr = output_ptr + output_offset

        if is_full:
            start_rank_idx = pid % world_size
            start_rank_global = rank_start + start_rank_idx * rank_stride
            acc = iris.load(base_ptr, iris_rank, start_rank_global, heap_bases, hint=(1, BLOCK_SIZE_N)).to(acc_dtype)
            for i in tl.static_range(1, world_size):
                remote_rank_idx = (start_rank_idx + i) % world_size
                remote_rank = rank_start + remote_rank_idx * rank_stride
                acc += iris.load(base_ptr, iris_rank, remote_rank, heap_bases, hint=(1, BLOCK_SIZE_N)).to(acc_dtype)

            tl.store(out_ptr, acc.to(output_ptr.type.element_ty), cache_modifier=".wt")
        else:
            mask = (rm[:, None] < M) & (rn[None, :] < N)

            start_rank_idx = pid % world_size
            start_rank_global = rank_start + start_rank_idx * rank_stride
            acc = iris.load(base_ptr, iris_rank, start_rank_global, heap_bases, mask=mask, hint=(1, BLOCK_SIZE_N)).to(
                acc_dtype
            )
            for i in tl.static_range(1, world_size):
                remote_rank_idx = (start_rank_idx + i) % world_size
                remote_rank = rank_start + remote_rank_idx * rank_stride
                acc += iris.load(base_ptr, iris_rank, remote_rank, heap_bases, mask=mask, hint=(1, BLOCK_SIZE_N)).to(
                    acc_dtype
                )

            tl.store(out_ptr, acc.to(output_ptr.type.element_ty), mask=mask, cache_modifier=".wt")


@triton.jit()
def persistent_all_reduce_ll128(
    input_ptr,
    output_ptr,
    staging_ptr,
    epoch,
    total_elems,
    num_lines,
    heap_bases: tl.tensor,
    group_rank: tl.constexpr,
    iris_rank: tl.constexpr,
    world_size: tl.constexpr,
    rank_start: tl.constexpr,
    rank_stride: tl.constexpr,
    COMM_SMS: tl.constexpr,
    PAYLOAD: tl.constexpr,
):
    """
    LL128-style all-reduce: data+flag in same 128-byte cache line.

    Each rank stages its input (upcast to f32) into a buffer where every
    128-byte cache line contains 31 f32 data words + 1 f32 flag.  A
    coalesced 32×f32 write lands atomically on a cache line.  Readers
    poll the flag word; when it matches, the 31 data words are valid.
    No explicit atomics needed — relies on cache-line write atomicity.
    """
    pid = tl.program_id(0)

    LINE: tl.constexpr = PAYLOAD + 1  # 32 f32 = 128 bytes = one cache line
    line_offsets = tl.arange(0, LINE)
    # Vectorization hint: offsets are contiguous and aligned to LINE
    line_offsets = tl.max_contiguous(tl.multiple_of(line_offsets, LINE), LINE)
    is_data = line_offsets < PAYLOAD

    # --- Phase 1: Stage local data with embedded flag per cache line ---
    for line_idx in range(pid, num_lines, COMM_SMS):
        data_start = line_idx * PAYLOAD
        line_base = line_idx * LINE
        # Load input elems, upcast to f32; use LINE-sized vector with masking
        full_offsets = data_start + line_offsets
        full_mask = is_data & (full_offsets < total_elems)
        safe_offsets = tl.where(is_data, full_offsets, 0)
        line = tl.load(input_ptr + safe_offsets, mask=full_mask, other=0.0).to(tl.float32)
        # Set element PAYLOAD (idx 31) to epoch flag; keep data in 0..30
        line = tl.where(is_data, line, epoch)
        # Single coalesced 128-byte store — atomic on cache line boundary
        staging_line_ptr = staging_ptr + line_base + line_offsets
        staging_line_ptr = tl.max_contiguous(tl.multiple_of(staging_line_ptr, LINE), LINE)
        tl.store(staging_line_ptr, line, cache_modifier=".wt")

    # --- Phase 2: Read from all peers, poll flag (via full cache line read), reduce ---
    for line_idx in range(pid, num_lines, COMM_SMS):
        data_start = line_idx * PAYLOAD
        line_base = line_idx * LINE

        acc = tl.zeros((LINE,), dtype=tl.float32)

        # Pointer to this cache line in the staging buffer (contiguous, aligned)
        staging_line_ptr = staging_ptr + line_base + line_offsets
        staging_line_ptr = tl.max_contiguous(tl.multiple_of(staging_line_ptr, LINE), LINE)

        for r in tl.static_range(world_size):
            remote_rank = rank_start + r * rank_stride

            # Read the full 128-byte cache line (32 f32) and check flag (element 31)
            # Use cache_modifier=".cv" to bypass all GPU caches for cross-GPU coherence
            # hint=LINE tells iris.load to apply vectorization hints to the translated ptr
            remote_line = iris.load(
                staging_line_ptr,
                iris_rank,
                remote_rank,
                heap_bases,
                cache_modifier=".cv",
                hint=LINE,
            )
            # Extract flag value (element PAYLOAD = element 31)
            flag_vals = tl.where(line_offsets == PAYLOAD, remote_line, 0.0)
            flag_val = tl.sum(flag_vals)
            while flag_val < epoch:
                remote_line = iris.load(
                    staging_line_ptr,
                    iris_rank,
                    remote_rank,
                    heap_bases,
                    cache_modifier=".cv",
                    hint=LINE,
                )
                flag_vals = tl.where(line_offsets == PAYLOAD, remote_line, 0.0)
                flag_val = tl.sum(flag_vals)

            # Flag matched — data words 0..30 are valid (same cache line load)
            # Mask out flag position so it doesn't pollute the sum
            acc += tl.where(is_data, remote_line, 0.0)

        # Write reduced data to output (downcast f32 → bf16)
        out_offsets = data_start + line_offsets
        out_mask = is_data & (out_offsets < total_elems)
        safe_out = tl.where(is_data, out_offsets, 0)
        tl.store(output_ptr + safe_out, acc.to(output_ptr.type.element_ty), mask=out_mask)


@triton.jit
def persistent_all_reduce_two_shot(
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
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
    GROUP_SIZE_M: tl.constexpr,
    COMM_SMS: tl.constexpr,
    NUM_XCDS: tl.constexpr,  # unused here but kept for signature compatibility
    CHUNK_SIZE: tl.constexpr,  # unused here but kept for signature compatibility
    DISTRIBUTION: tl.constexpr,
):
    """Reduce assigned tiles for a rank and broadcast the result to all peers.
    Single kernel: unmasked fast path for full tiles, masked slow path for tails.
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

        # Build indices (used by both paths)
        rm = rm_base + tl.arange(0, BLOCK_SIZE_M)
        rn = rn_base + tl.arange(0, BLOCK_SIZE_N)

        rm = tl.max_contiguous(tl.multiple_of(rm, BLOCK_SIZE_M), BLOCK_SIZE_M)
        rn = tl.max_contiguous(tl.multiple_of(rn, BLOCK_SIZE_N), BLOCK_SIZE_N)

        input_offset = rm[:, None] * stride_in_m + rn[None, :] * stride_in_n
        output_offset = rm[:, None] * stride_out_m + rn[None, :] * stride_out_n

        base_ptr = input_ptr + input_offset
        out_ptr = output_ptr + output_offset

        # Fast path: NO MASKS (full tiles)
        # The masking is problem size dependent, and the compiler does not recognize it can have two paths
        # (one with masks and one without). Separate unmasked paths allow the compiler to generate
        # more efficient vectorized instructions.
        if is_full:
            start_rank_idx = pid % world_size
            start_rank_global = rank_start + start_rank_idx * rank_stride
            acc = iris.load(base_ptr, iris_rank, start_rank_global, heap_bases, hint=(1, BLOCK_SIZE_N)).to(acc_dtype)
            for i in tl.static_range(1, world_size):
                remote_rank_idx = (start_rank_idx + i) % world_size
                remote_rank = rank_start + remote_rank_idx * rank_stride
                acc += iris.load(base_ptr, iris_rank, remote_rank, heap_bases, hint=(1, BLOCK_SIZE_N)).to(acc_dtype)

            reduced = acc.to(output_ptr.type.element_ty)

            tl.store(out_ptr, reduced, cache_modifier=".wt")

            for i in tl.static_range(0, world_size):
                remote_rank_idx = (start_rank_idx + i) % world_size
                remote_rank = rank_start + remote_rank_idx * rank_stride
                if remote_rank_idx != group_rank:
                    iris.store(out_ptr, reduced, iris_rank, remote_rank, heap_bases, hint=(1, BLOCK_SIZE_N))

        # Slow path: MASKED (only boundary tiles land here)
        # This path handles tiles at tensor boundaries where not all elements are valid.
        else:
            mask = (rm[:, None] < M) & (rn[None, :] < N)

            start_rank_idx = pid % world_size
            start_rank_global = rank_start + start_rank_idx * rank_stride
            acc = iris.load(base_ptr, iris_rank, start_rank_global, heap_bases, mask=mask, hint=(1, BLOCK_SIZE_N)).to(
                acc_dtype
            )
            for i in tl.static_range(1, world_size):
                remote_rank_idx = (start_rank_idx + i) % world_size
                remote_rank = rank_start + remote_rank_idx * rank_stride
                acc += iris.load(base_ptr, iris_rank, remote_rank, heap_bases, mask=mask, hint=(1, BLOCK_SIZE_N)).to(
                    acc_dtype
                )

            reduced = acc.to(output_ptr.type.element_ty)

            tl.store(out_ptr, reduced, mask=mask, cache_modifier=".wt")

            for i in tl.static_range(0, world_size):
                remote_rank_idx = (start_rank_idx + i) % world_size
                remote_rank = rank_start + remote_rank_idx * rank_stride
                if remote_rank_idx != group_rank:
                    iris.store(
                        out_ptr,
                        reduced,
                        iris_rank,
                        remote_rank,
                        heap_bases,
                        mask=mask,
                        hint=(1, BLOCK_SIZE_N),
                    )


if GLUON_AVAILABLE:

    @gluon.jit
    def chiplet_transform_chunked_gluon(
        pid, num_xcds: gl.constexpr, num_workgroups: gl.constexpr, chunk_size: gl.constexpr
    ):
        if pid > (num_workgroups // (num_xcds * chunk_size)) * (num_xcds * chunk_size):
            return pid
        local_pid = pid // num_xcds
        chunk_idx = local_pid // chunk_size
        pos_in_chunk = local_pid % chunk_size
        xcd = pid % num_xcds
        new_pid = chunk_idx * num_xcds * chunk_size + xcd * chunk_size + pos_in_chunk
        return new_pid

    @gluon.jit
    def persistent_all_reduce_ll_gluon(
        IrisDeviceCtx: gl.constexpr,
        context_tensor,
        input_ptr,
        output_ptr,
        flags_ptr,
        epoch,
        M,
        N,
        stride_in_m,
        stride_in_n,
        stride_out_m,
        stride_out_n,
        group_rank: gl.constexpr,
        iris_rank: gl.constexpr,
        world_size: gl.constexpr,
        rank_start: gl.constexpr,
        rank_stride: gl.constexpr,
        BLOCK_SIZE_M: gl.constexpr,
        BLOCK_SIZE_N: gl.constexpr,
        GROUP_SIZE_M: gl.constexpr,
        COMM_SMS: gl.constexpr,
        NUM_XCDS: gl.constexpr,
        CHUNK_SIZE: gl.constexpr,
        THREADS_PER_WARP: gl.constexpr,
        WARPS_PER_CTA: gl.constexpr,
    ):
        """
        Low-latency all-reduce using Gluon with flat-2D tiling.

        Same algorithm as the Triton LL variant: per-rank epoch flags for
        in-kernel synchronization, every rank reads all peers and reduces locally.

        Gluon advantages: hoisted heap_bases, explicit BlockedLayout for
        guaranteed dwordx4 vectorization, manual ptr_delta to avoid redundant
        heap_bases loads.
        """
        ctx = IrisDeviceCtx.initialize(context_tensor, tracing=False)
        pid = gl.program_id(0)

        if NUM_XCDS != 1:
            pid = chiplet_transform_chunked_gluon(pid, NUM_XCDS, COMM_SMS, CHUNK_SIZE)

        # --- Phase 1: Signal readiness to all peers ---
        if pid == 0:
            gl.atomic_xchg(flags_ptr + group_rank, epoch, sem="release", scope="sys")
            for i in gl.static_range(world_size):
                remote_rank = rank_start + i * rank_stride
                ctx.atomic_xchg(flags_ptr + group_rank, epoch, remote_rank, sem="release", scope="sys")

        # --- Phase 2: Wait for all peers to be ready ---
        for i in gl.static_range(world_size):
            while gl.atomic_add(flags_ptr + i, 0, sem="acquire", scope="sys") < epoch:
                pass

        # --- Phase 3: Read-reduce-write (flat-2D tiling) ---
        num_pid_m = gl.cdiv(M, BLOCK_SIZE_M)
        num_pid_n = gl.cdiv(N, BLOCK_SIZE_N)
        total_tiles = num_pid_m * num_pid_n

        TOTAL_ELEMS: gl.constexpr = BLOCK_SIZE_M * BLOCK_SIZE_N
        ELEMS_PER_THREAD: gl.constexpr = TOTAL_ELEMS // (THREADS_PER_WARP * WARPS_PER_CTA)
        flat_layout: gl.constexpr = gl.BlockedLayout([ELEMS_PER_THREAD], [THREADS_PER_WARP], [WARPS_PER_CTA], [0])

        # Hoist local heap base outside the tile loop
        local_base = gl.load(ctx.heap_bases + iris_rank)

        for tile_id in range(pid, total_tiles, COMM_SMS):
            # Swizzled tile index computation
            num_pid_in_group = GROUP_SIZE_M * num_pid_n
            group_id = tile_id // num_pid_in_group
            first_pid_m = group_id * GROUP_SIZE_M
            group_size_m = min(num_pid_m - first_pid_m, GROUP_SIZE_M)
            pid_m = first_pid_m + ((tile_id % num_pid_in_group) % group_size_m)
            pid_n = (tile_id % num_pid_in_group) // group_size_m

            # Flat index -> 2D row/col within tile
            flat_idx = gl.arange(0, TOTAL_ELEMS, layout=flat_layout)
            row_local = flat_idx // BLOCK_SIZE_N
            col_local = flat_idx % BLOCK_SIZE_N

            row = pid_m * BLOCK_SIZE_M + row_local
            col = pid_n * BLOCK_SIZE_N + col_local
            mask = (row < M) & (col < N)

            input_offsets = row * stride_in_m + col * stride_in_n
            output_offsets = row * stride_out_m + col * stride_out_n

            # Stagger starting rank per CTA to spread load
            start_rank_idx = pid % world_size
            start_rank_global = rank_start + start_rank_idx * rank_stride

            # First load: use ctx.load for initial rank
            base_ptr = input_ptr + input_offsets
            acc = ctx.load(base_ptr, start_rank_global, mask=mask, other=0.0).to(gl.float32)

            # Remaining ranks: hoisted ptr_delta for efficiency
            for i in gl.static_range(1, world_size):
                remote_rank_idx = (start_rank_idx + i) % world_size
                remote_rank = rank_start + remote_rank_idx * rank_stride
                target_base = gl.load(ctx.heap_bases + remote_rank)
                ptr_delta = target_base - local_base
                remote_ptrs_int = tl.cast(base_ptr, gl.uint64) + ptr_delta
                remote_ptrs = tl.cast(remote_ptrs_int, base_ptr.dtype)
                acc += gl.load(remote_ptrs, mask=mask, other=0.0).to(gl.float32)

            # Write reduced result locally
            out_ptr = output_ptr + output_offsets
            gl.store(out_ptr, acc.to(output_ptr.type.element_ty), mask=mask, cache_modifier=".wt")

    @gluon.jit
    def persistent_all_reduce_rccl_ll_gluon(
        IrisDeviceCtx: gl.constexpr,
        context_tensor,
        input_ptr,
        output_ptr,
        ll_buffer_ptr,
        epoch,
        total_elems,
        group_rank: gl.constexpr,
        iris_rank: gl.constexpr,
        world_size: gl.constexpr,
        rank_start: gl.constexpr,
        rank_stride: gl.constexpr,
        BLOCK_SIZE: gl.constexpr,
        COMM_SMS: gl.constexpr,
        THREADS_PER_WARP: gl.constexpr,
        WARPS_PER_CTA: gl.constexpr,
    ):
        """
        RCCL-style LL all-reduce using Gluon.

        Ring reduce-scatter + ring all-gather with interleaved data+flag pairs.
        Buffer layout: ll_buffer[2*i] = data_f32, ll_buffer[2*i+1] = flag_f32.

        Key: prev_rank writes data+flag into MY buffer. I poll MY local buffer
        for the flag, then read the data from MY local buffer. I then write
        my result into next_rank's buffer.

        Flag values (epoch-based to avoid stale flags):
          reduce-scatter step k: epoch_base + k + 1  (k = 0..W-2)
          all-gather step k:     epoch_base + W + k   (k = 0..W-2)
        where epoch_base = epoch * (2 * world_size).
        """
        ctx = IrisDeviceCtx.initialize(context_tensor, tracing=False)
        pid = gl.program_id(0)

        ELEMS_PER_CTA: gl.constexpr = BLOCK_SIZE
        THREADS_PER_CTA: gl.constexpr = THREADS_PER_WARP * WARPS_PER_CTA
        ELEMS_PER_THREAD: gl.constexpr = ELEMS_PER_CTA // THREADS_PER_CTA
        flat_layout: gl.constexpr = gl.BlockedLayout([ELEMS_PER_THREAD], [THREADS_PER_WARP], [WARPS_PER_CTA], [0])

        # Ring neighbor: next_rank in ring
        next_iris: gl.constexpr = rank_start + ((group_rank + 1) % world_size) * rank_stride

        # Chunk size: total_elems / world_size
        chunk_elems = total_elems // world_size

        # Hoist heap bases
        local_base = gl.load(ctx.heap_bases + iris_rank)
        next_base = gl.load(ctx.heap_bases + next_iris)
        next_delta = next_base - local_base

        # Base epoch offset for flag values
        epoch_base = epoch * (2 * world_size)

        # --- Phase 1: Ring reduce-scatter (world_size - 1 steps) ---
        for step in gl.static_range(world_size - 1):
            chunk_idx = (group_rank - step) % world_size
            chunk_start = chunk_idx * chunk_elems
            flag_val_f32 = (epoch_base + step + 1).to(gl.float32)

            for elem_offset in range(pid * ELEMS_PER_CTA, chunk_elems, COMM_SMS * ELEMS_PER_CTA):
                idx = gl.arange(0, ELEMS_PER_CTA, layout=flat_layout) + elem_offset
                mask = idx < chunk_elems
                global_idx = chunk_start + idx

                # Load local input for this chunk
                local_data = gl.load(input_ptr + global_idx, mask=mask, other=0.0).to(gl.float32)

                if step == 0:
                    # First step: just send my data
                    acc = local_data
                else:
                    # Poll MY local buffer for data written by prev_rank
                    expected_f32 = (epoch_base + step).to(gl.float32)
                    my_flag_ptr = ll_buffer_ptr + global_idx * 2 + 1
                    flag = gl.load(my_flag_ptr, mask=mask, other=expected_f32, cache_modifier=".cv")
                    while gl.min(tl.where(mask, (flag == expected_f32).to(gl.int32), 1), axis=0) == 0:
                        flag = gl.load(my_flag_ptr, mask=mask, other=expected_f32, cache_modifier=".cv")

                    # Read accumulated data from my buffer
                    my_data_ptr = ll_buffer_ptr + global_idx * 2
                    prev_data = gl.load(my_data_ptr, mask=mask, other=0.0, cache_modifier=".cv")
                    acc = local_data + prev_data

                # Write data + flag to next_rank's buffer
                next_buf_data = ll_buffer_ptr + global_idx * 2
                next_buf_flag = ll_buffer_ptr + global_idx * 2 + 1
                next_data_int = tl.cast(next_buf_data, gl.uint64) + next_delta
                next_flag_int = tl.cast(next_buf_flag, gl.uint64) + next_delta
                next_data_ptr = tl.cast(next_data_int, next_buf_data.dtype)
                next_flag_ptr = tl.cast(next_flag_int, next_buf_flag.dtype)
                gl.store(next_data_ptr, acc, mask=mask, cache_modifier=".wt")
                gl.store(
                    next_flag_ptr,
                    tl.full([ELEMS_PER_CTA], flag_val_f32, gl.float32, layout=flat_layout),
                    mask=mask,
                    cache_modifier=".wt",
                )

        # --- Phase 2: Ring all-gather (world_size - 1 steps) ---
        # After reduce-scatter, rank r has the fully reduced chunk
        # (group_rank + 1) % world_size in its buffer (written by prev_rank).
        # We need to write that chunk to output AND forward all chunks around the ring.

        for step in gl.static_range(world_size - 1):
            # Which chunk arrives at this step
            chunk_idx = (group_rank - step + 1) % world_size
            chunk_start = chunk_idx * chunk_elems

            # For step 0: expect the last reduce-scatter flag = epoch_base + (W-1)
            # For step k: expect all-gather flag = epoch_base + W + (step - 1)
            # Unified: epoch_base + W - 1 + step
            expected_f32 = (epoch_base + world_size - 1 + step).to(gl.float32)
            ag_flag_f32 = (epoch_base + world_size + step).to(gl.float32)

            for elem_offset in range(pid * ELEMS_PER_CTA, chunk_elems, COMM_SMS * ELEMS_PER_CTA):
                idx = gl.arange(0, ELEMS_PER_CTA, layout=flat_layout) + elem_offset
                mask = idx < chunk_elems
                global_idx = chunk_start + idx

                # Poll MY local buffer for data written by prev_rank
                my_flag_ptr = ll_buffer_ptr + global_idx * 2 + 1
                flag = gl.load(my_flag_ptr, mask=mask, other=expected_f32, cache_modifier=".cv")
                while gl.min(tl.where(mask, (flag == expected_f32).to(gl.int32), 1), axis=0) == 0:
                    flag = gl.load(my_flag_ptr, mask=mask, other=expected_f32, cache_modifier=".cv")

                # Read data from my buffer
                my_data_ptr = ll_buffer_ptr + global_idx * 2
                data = gl.load(my_data_ptr, mask=mask, other=0.0, cache_modifier=".cv")

                # Write to output
                gl.store(output_ptr + global_idx, data.to(output_ptr.type.element_ty), mask=mask)

                # Forward to next rank
                next_buf_data = ll_buffer_ptr + global_idx * 2
                next_buf_flag = ll_buffer_ptr + global_idx * 2 + 1
                next_data_int = tl.cast(next_buf_data, gl.uint64) + next_delta
                next_flag_int = tl.cast(next_buf_flag, gl.uint64) + next_delta
                next_data_ptr = tl.cast(next_data_int, next_buf_data.dtype)
                next_flag_ptr = tl.cast(next_flag_int, next_buf_flag.dtype)
                gl.store(next_data_ptr, data, mask=mask, cache_modifier=".wt")
                gl.store(
                    next_flag_ptr,
                    tl.full([ELEMS_PER_CTA], ag_flag_f32, gl.float32, layout=flat_layout),
                    mask=mask,
                    cache_modifier=".wt",
                )

        # Write own fully-reduced chunk to output
        # This is the chunk that ended at me after reduce-scatter, i.e., chunk
        # (group_rank + 1) % world_size. It was written by prev_rank with
        # flag = epoch_base + (world_size - 1).
        own_chunk_idx = (group_rank + 1) % world_size
        own_chunk_start = own_chunk_idx * chunk_elems
        own_flag_expected = (epoch_base + world_size - 1).to(gl.float32)

        for elem_offset in range(pid * ELEMS_PER_CTA, chunk_elems, COMM_SMS * ELEMS_PER_CTA):
            idx = gl.arange(0, ELEMS_PER_CTA, layout=flat_layout) + elem_offset
            mask = idx < chunk_elems
            global_idx = own_chunk_start + idx

            # Poll for reduce-scatter completion
            my_flag_ptr = ll_buffer_ptr + global_idx * 2 + 1
            flag = gl.load(my_flag_ptr, mask=mask, other=own_flag_expected, cache_modifier=".cv")
            while gl.min(tl.where(mask, (flag == own_flag_expected).to(gl.int32), 1), axis=0) == 0:
                flag = gl.load(my_flag_ptr, mask=mask, other=own_flag_expected, cache_modifier=".cv")

            my_data_ptr = ll_buffer_ptr + global_idx * 2
            data = gl.load(my_data_ptr, mask=mask, other=0.0, cache_modifier=".cv")
            gl.store(output_ptr + global_idx, data.to(output_ptr.type.element_ty), mask=mask)


def all_reduce(
    output_tensor,
    input_tensor,
    ctx,
    op=ReduceOp.SUM,
    group=None,
    async_op=False,
    config=None,
    workspace: Optional[AllReduceWorkspace] = None,
):
    """
    Internal all-reduce collective operation implementation.

    This function is called internally by ctx.ccl.all_reduce().
    Users should use the Iris instance method instead:
        >>> ctx.ccl.all_reduce(output_tensor, input_tensor)

    Each rank has a local input tensor, and all ranks compute the sum of all
    input tensors. The result is written to output_tensor on all ranks.

    Args:
        output_tensor: Output tensor of shape (M, N) - will contain sum of all inputs
        input_tensor: Input tensor of shape (M, N) - local rank's partial data
        ctx: Iris context
        op: Reduction operation to apply. Currently only ReduceOp.SUM is supported.
            Default: ReduceOp.SUM.
        group: ProcessGroup or None. If None, uses all ranks in iris context.
               Default: None.
        async_op: If False, performs a barrier at the end. If True, returns immediately.
                  Default: False.
        config: Config instance with kernel parameters (default: None).
                If None, uses default Config values.
                Set config.all_reduce_variant to choose variant: "atomic", "spinlock", "ring", "two_shot", or "one_shot"
        workspace: Optional AllReduceWorkspace instance prepared via all_reduce_preamble.
    """
    # Validate op parameter
    if op != ReduceOp.SUM:
        raise ValueError(
            f"Only ReduceOp.SUM is currently supported, got {op}. "
            "Support for other operations (PRODUCT, MAX, MIN, etc.) will be added in a future release."
        )
    if config is None:
        config = Config(block_size_m=32, block_size_n=64, all_reduce_distribution=1)

    # Extract group information
    # rank_in_group: position within the ProcessGroup (0, 1, 2, ...) - passed as group_rank to kernel
    # rank_global: global rank in iris context - passed as iris_rank to kernel for RMA operations
    rank_in_group, rank_global, world_size, rank_start, rank_stride = extract_group_info(group, ctx)
    M, N = input_tensor.shape[:2]

    stride_in_m, stride_in_n = input_tensor.stride(0), input_tensor.stride(1)
    stride_out_m, stride_out_n = output_tensor.stride(0), output_tensor.stride(1)

    variant = config.all_reduce_variant.lower()
    if variant not in [
        VARIANT_ATOMIC,
        VARIANT_SPINLOCK,
        VARIANT_RING,
        VARIANT_TWO_SHOT,
        VARIANT_ONE_SHOT,
        VARIANT_LL,
        VARIANT_LL128,
        VARIANT_RCCL_LL,
    ]:
        raise ValueError(
            f"Invalid all_reduce_variant: {variant}. Must be one of: {VARIANT_ATOMIC}, {VARIANT_SPINLOCK}, {VARIANT_RING}, {VARIANT_TWO_SHOT}, {VARIANT_ONE_SHOT}, {VARIANT_LL}, {VARIANT_LL128}, {VARIANT_RCCL_LL}"
        )

    slice_n = config.all_reduce_ring_slice_n
    if variant == VARIANT_RING:
        if config.block_size_n % world_size != 0:
            raise ValueError(
                f"block_size_n ({config.block_size_n}) must be divisible by world_size ({world_size}) for ring variant"
            )
        expected_slice = config.block_size_n // world_size
        if slice_n is None or slice_n * world_size != config.block_size_n:
            slice_n = expected_slice
        config.all_reduce_ring_slice_n = slice_n

    needs_prepare = (
        workspace is None
        or not getattr(workspace, "prepared", False)
        or workspace.variant != variant
        or workspace.shape != (M, N)
        or workspace.dtype != input_tensor.dtype
        or (variant == VARIANT_RING and workspace.num_rings != config.all_reduce_num_rings)
        or (variant == VARIANT_RING and workspace.flags_per_tile != 1)
        or (variant == VARIANT_SPINLOCK and (workspace.locks is None))
        or (variant == VARIANT_LL and (workspace.flags is None or not hasattr(workspace, "ll_epoch")))
        or (variant == VARIANT_LL128 and (workspace.ring_buffer is None or not hasattr(workspace, "ll_epoch")))
    )

    if needs_prepare:
        workspace = all_reduce_preamble(
            output_tensor,
            input_tensor,
            ctx,
            config=config,
            workspace=workspace,
        )

    heap_bases = ctx.get_heap_bases()

    if variant == VARIANT_ATOMIC:
        iris_launch(
            persistent_all_reduce_atomic,
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
            config.block_size_m,
            config.block_size_n,
            config.swizzle_size,
            config.comm_sms,
            config.num_xcds,
            config.chunk_size,
            algorithm="all_reduce",
            rank=rank_global,
            dtype=input_tensor.dtype,
        )

    elif variant == VARIANT_SPINLOCK:
        if workspace is None or workspace.locks is None:
            raise RuntimeError(
                "Spinlock variant requires workspace preparation. Call all_reduce_preamble before all_reduce."
            )

        num_pid_m = (M + config.block_size_m - 1) // config.block_size_m
        num_pid_n = (N + config.block_size_n - 1) // config.block_size_n
        total_tiles = num_pid_m * num_pid_n
        if workspace.locks.numel() < total_tiles:
            raise ValueError(
                f"Lock array too small: have {workspace.locks.numel()} but need {total_tiles}. "
                f"Pre-allocate workspace with the smallest block sizes you intend to use."
            )

        iris_launch(
            persistent_all_reduce_spinlock,
            (config.comm_sms,),
            input_tensor,
            output_tensor,
            workspace.locks,
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
            config.block_size_m,
            config.block_size_n,
            config.swizzle_size,
            config.comm_sms,
            config.num_xcds,
            config.chunk_size,
            algorithm="all_reduce",
            rank=rank_global,
            dtype=input_tensor.dtype,
        )

    elif variant == VARIANT_RING:
        if workspace is None or workspace.ring_buffer is None or workspace.flags is None:
            raise RuntimeError(
                "Ring variant requires workspace preparation. Call all_reduce_preamble before all_reduce."
            )

        num_pid_m = (M + config.block_size_m - 1) // config.block_size_m
        num_pid_n = (N + config.block_size_n - 1) // config.block_size_n
        total_tiles = num_pid_m * num_pid_n
        total_flags = total_tiles * workspace.flags_per_tile
        if workspace.flags.numel() < total_flags:
            raise ValueError(
                f"Flags array too small: have {workspace.flags.numel()} but need {total_flags}. "
                f"Pre-allocate workspace with the smallest block sizes you intend to use."
            )

        # Calculate next rank in the ring for group support
        # next_rank must be a global rank for iris RMA operations
        if group is None:
            # Simple case: next rank is just (rank_in_group + 1) % world_size (which equals global rank)
            next_rank = (rank_in_group + 1) % world_size
        else:
            # Group case: get the group ranks and find next in ring
            import torch.distributed as dist

            group_ranks = dist.get_process_group_ranks(group)
            next_rank_in_group = (rank_in_group + 1) % world_size
            next_rank = group_ranks[next_rank_in_group]

        iris_launch(
            persistent_all_reduce_ring,
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
            config.block_size_m,
            config.block_size_n,
            config.swizzle_size,
            config.comm_sms,
            config.num_xcds,
            config.chunk_size,
            config.all_reduce_num_rings,
            slice_n,
            workspace.flags_per_tile,
            algorithm="all_reduce",
            rank=rank_global,
            dtype=input_tensor.dtype,
        )

    elif variant == VARIANT_TWO_SHOT:
        # Use num_warps=8 for latency hiding, BLOCK_SIZE_N>=128 so
        # 512 threads still get 8 bf16/thread = 16 bytes = dwordx4.
        ar_block_n = max(config.block_size_n, 128)
        iris_launch(
            persistent_all_reduce_two_shot,
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
            config.block_size_m,
            ar_block_n,
            config.swizzle_size,
            config.comm_sms,
            config.num_xcds,
            config.chunk_size,
            config.all_reduce_distribution,
            num_warps=8,
            num_stages=1,
            algorithm="all_reduce",
            rank=rank_global,
            dtype=input_tensor.dtype,
        )
    elif variant == VARIANT_ONE_SHOT:
        iris_launch(
            persistent_all_reduce_one_shot,
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
            config.block_size_m,
            config.block_size_n,
            config.swizzle_size,
            config.comm_sms,
            config.num_xcds,
            config.chunk_size,
            algorithm="all_reduce",
            rank=rank_global,
            dtype=input_tensor.dtype,
        )

    elif variant == VARIANT_LL:
        workspace.ll_epoch += 1
        ar_block_n = max(config.block_size_n, 128)

        if config.use_gluon and GLUON_AVAILABLE:
            # Gluon path: flat-2D tiling with explicit BlockedLayout
            block_m = config.block_size_m
            block_n = ar_block_n
            total_elems = block_m * block_n
            threads_per_cta = config.threads_per_warp * config.num_warps
            if total_elems < threads_per_cta or total_elems % threads_per_cta != 0:
                raise ValueError(
                    f"Gluon LL requires block_size_m * block_size_n to be a "
                    f"multiple of threads_per_warp * num_warps ({threads_per_cta}), "
                    f"got {block_m} * {block_n} = {total_elems}."
                )
            context_tensor = ctx.get_device_context()
            iris_launch(
                persistent_all_reduce_ll_gluon,
                (config.comm_sms,),
                IrisDeviceCtx,
                context_tensor,
                input_tensor,
                output_tensor,
                workspace.flags,
                workspace.ll_epoch,
                M,
                N,
                stride_in_m,
                stride_in_n,
                stride_out_m,
                stride_out_n,
                rank_in_group,
                rank_global,
                world_size,
                rank_start,
                rank_stride,
                block_m,
                block_n,
                config.swizzle_size,
                config.comm_sms,
                config.num_xcds,
                config.chunk_size,
                config.threads_per_warp,
                config.num_warps,
                num_stages=config.num_stages,
                num_warps=config.num_warps,
                waves_per_eu=config.waves_per_eu,
                algorithm="all_reduce",
                rank=rank_global,
                dtype=input_tensor.dtype,
            )
        else:
            iris_launch(
                persistent_all_reduce_ll,
                (config.comm_sms,),
                input_tensor,
                output_tensor,
                workspace.flags,
                workspace.ll_epoch,
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
                config.block_size_m,
                ar_block_n,
                config.swizzle_size,
                config.comm_sms,
                config.num_xcds,
                config.chunk_size,
                num_warps=8,
                num_stages=1,
                algorithm="all_reduce",
                rank=rank_global,
                dtype=input_tensor.dtype,
            )

    elif variant == VARIANT_LL128:
        workspace.ll_epoch += 1
        total_elems = M * N
        payload = 31
        num_lines = (total_elems + payload - 1) // payload
        iris_launch(
            persistent_all_reduce_ll128,
            (config.comm_sms,),
            input_tensor,
            output_tensor,
            workspace.ring_buffer,
            float(workspace.ll_epoch),
            total_elems,
            num_lines,
            heap_bases,
            rank_in_group,
            rank_global,
            world_size,
            rank_start,
            rank_stride,
            config.comm_sms,
            payload,
            algorithm="all_reduce",
            rank=rank_global,
            dtype=input_tensor.dtype,
        )

    elif variant == VARIANT_RCCL_LL:
        if not GLUON_AVAILABLE:
            raise ValueError("rccl_ll variant requires Gluon. Install Triton with Gluon support.")
        workspace.ll_epoch += 1
        total_elems = M * N

        # Require total_elems divisible by world_size for ring chunking
        if total_elems % world_size != 0:
            raise ValueError(f"rccl_ll requires total elements ({total_elems}) divisible by world_size ({world_size})")

        block_size = 256  # elements per CTA tile
        threads_per_cta = config.threads_per_warp * config.num_warps
        if block_size % threads_per_cta != 0:
            raise ValueError(
                f"rccl_ll BLOCK_SIZE ({block_size}) must be divisible by "
                f"threads_per_warp * num_warps ({threads_per_cta})"
            )

        context_tensor = ctx.get_device_context()
        iris_launch(
            persistent_all_reduce_rccl_ll_gluon,
            (config.comm_sms,),
            IrisDeviceCtx,
            context_tensor,
            input_tensor,
            output_tensor,
            workspace.ring_buffer,
            workspace.ll_epoch,
            total_elems,
            rank_in_group,
            rank_global,
            world_size,
            rank_start,
            rank_stride,
            block_size,
            config.comm_sms,
            config.threads_per_warp,
            config.num_warps,
            num_stages=config.num_stages,
            num_warps=config.num_warps,
            waves_per_eu=config.waves_per_eu,
            algorithm="all_reduce",
            rank=rank_global,
            dtype=input_tensor.dtype,
        )

    if workspace is not None:
        workspace.prepared = False

    if not async_op and variant not in (VARIANT_LL, VARIANT_LL128, VARIANT_RCCL_LL):
        ctx.barrier()

    return workspace
