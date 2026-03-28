# SPDX-License-Identifier: MIT
# Copyright (c) 2025 Advanced Micro Devices, Inc. All rights reserved.

"""
Reduce-scatter collective communication primitive for Iris.

Supports two variants:
- two_shot: Each rank reads all other ranks' data for its assigned tiles (all-pairs read).
- ring_chunked: Ring-based Rabenseifner-style reduce-scatter with flag-based synchronization.
"""

import functools

import torch
import triton
import triton.language as tl
import iris
from .config import Config
from .utils import chiplet_transform_chunked, ReduceOp, extract_group_info


@functools.lru_cache(maxsize=8)
def _default_config(block_size_m, block_size_n, comm_sms=64, distribution=1):
    """Cache default Config objects to avoid repeated HIP queries (subprocess overhead)."""
    return Config(
        block_size_m=block_size_m, block_size_n=block_size_n, comm_sms=comm_sms, all_reduce_distribution=distribution
    )


@triton.jit()
def _apply_delta(ptr, delta, hint: tl.constexpr = None):
    """Translate pointer using pre-computed delta (remote_base - local_base)."""
    ptr_byte = tl.cast(ptr, tl.pointer_type(tl.int8))
    translated_byte = ptr_byte + delta
    translated = tl.cast(translated_byte, ptr.dtype)
    if hint is not None:
        translated = tl.max_contiguous(tl.multiple_of(translated, hint), hint)
    return translated


@triton.jit()
def persistent_reduce_scatter_two_shot(
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
    DISTRIBUTION: tl.constexpr,
):
    """
    Reduce-scatter using two-shot approach with delta-based pointer translation.

    Each rank reduces its assigned tiles from all ranks and stores the result
    only to its own output (no broadcast to other ranks).

    Optimizations:
    - Pre-computes pointer deltas (remote_base - local_base) per rank once.
    - Each remote load is just ptr + delta (single addition vs subtract+add).
    - local_base loaded once, remote bases loaded once each outside tile loop.
    """
    pid = tl.program_id(0)

    if NUM_XCDS != 1:
        pid = chiplet_transform_chunked(pid, COMM_SMS, NUM_XCDS, CHUNK_SIZE)

    num_pid_m = tl.cdiv(M, BLOCK_SIZE_M)
    num_pid_n = tl.cdiv(N, BLOCK_SIZE_N)
    total_tiles = num_pid_m * num_pid_n

    acc_dtype = tl.float32 if output_ptr.type.element_ty != tl.int8 else tl.int32

    # Pre-compute pointer deltas: delta[r] = heap_bases[r] - heap_bases[iris_rank]
    # This makes translation just ptr + delta (one add vs subtract+add).
    local_base = tl.load(heap_bases + iris_rank)

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

    for tile_offset in range(pid, max_tile_offset, COMM_SMS):
        tile_id = start_tile + tile_offset * stride

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
        if is_full:
            start_rank_idx = pid % world_size
            # Load from first rank using delta-based translation
            first_rank = rank_start + start_rank_idx * rank_stride
            delta_0 = tl.load(heap_bases + first_rank) - local_base
            acc = tl.load(_apply_delta(base_ptr, delta_0, hint=(1, BLOCK_SIZE_N))).to(acc_dtype)
            for i in tl.static_range(1, world_size):
                remote_rank_idx = (start_rank_idx + i) % world_size
                remote_rank = rank_start + remote_rank_idx * rank_stride
                delta_i = tl.load(heap_bases + remote_rank) - local_base
                acc += tl.load(_apply_delta(base_ptr, delta_i, hint=(1, BLOCK_SIZE_N))).to(acc_dtype)

            reduced = acc.to(output_ptr.type.element_ty)
            tl.store(out_ptr, reduced, cache_modifier=".wt")

        # Slow path: MASKED (only boundary tiles)
        else:
            mask = (rm[:, None] < M) & (rn[None, :] < N)

            start_rank_idx = pid % world_size
            first_rank = rank_start + start_rank_idx * rank_stride
            delta_0 = tl.load(heap_bases + first_rank) - local_base
            acc = tl.load(_apply_delta(base_ptr, delta_0, hint=(1, BLOCK_SIZE_N)), mask=mask, other=0.0).to(acc_dtype)
            for i in tl.static_range(1, world_size):
                remote_rank_idx = (start_rank_idx + i) % world_size
                remote_rank = rank_start + remote_rank_idx * rank_stride
                delta_i = tl.load(heap_bases + remote_rank) - local_base
                acc += tl.load(_apply_delta(base_ptr, delta_i, hint=(1, BLOCK_SIZE_N)), mask=mask, other=0.0).to(
                    acc_dtype
                )

            reduced = acc.to(output_ptr.type.element_ty)
            tl.store(out_ptr, reduced, mask=mask, cache_modifier=".wt")


@triton.jit()
def persistent_reduce_scatter_ring(
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
    DISTRIBUTION: tl.constexpr,
):
    """
    Ring-based reduce-scatter (Rabenseifner-style).

    All ranks participate in reducing ALL tiles through the ring. Each rank:
      1. Loads its local data for the tile
      2. Stores to ring_buffer on the next rank and signals
      3. Waits for data from the previous rank
      4. Accumulates the received data with its local partial sum
      5. Forwards the received data to the next rank
      6. After world_size-1 steps, the tile is fully reduced
      7. Only the owning rank writes the result to output

    This achieves O(1) remote reads/writes per step instead of O(N) in two_shot,
    while each rank only stores results for its assigned tiles.
    """
    pid_raw = tl.program_id(0)

    pid = pid_raw
    if NUM_XCDS != 1:
        pid = chiplet_transform_chunked(pid_raw, COMM_SMS, NUM_XCDS, CHUNK_SIZE)

    num_pid_m = tl.cdiv(M, BLOCK_SIZE_M)
    num_pid_n = tl.cdiv(N, BLOCK_SIZE_N)
    total_tiles = num_pid_m * num_pid_n

    acc_dtype = tl.float32 if output_ptr.type.element_ty != tl.int8 else tl.int32

    # Tile ownership: determine which tiles this rank will store results for
    tiles_per_rank = tl.cdiv(total_tiles, world_size)

    # ALL ranks process ALL tiles through the ring
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
        tile_offset_in = rm[:, None] * stride_in_m + rn[None, :] * stride_in_n
        tile_offset_out = rm[:, None] * stride_out_m + rn[None, :] * stride_out_n

        # Load local data for this tile
        local_tile = tl.load(input_ptr + tile_offset_in, mask=mask, other=0)
        acc = local_tile.to(acc_dtype)
        send_data = local_tile

        flag_offset = tile_id
        remote_flag_ptr = flags + flag_offset
        local_flag_ptr = flags + flag_offset

        # Ring reduce: world_size - 1 steps
        for _step in range(0, world_size - 1):
            # 1. Wait for next rank's ring_buffer to be free (flag == 0)
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

            # 2. Store data to next rank's ring_buffer
            iris.store(
                ring_buffer + tile_offset_in,
                send_data,
                iris_rank,
                next_rank,
                heap_bases,
                mask=mask,
                hint=(1, BLOCK_SIZE_N),
            )
            tl.debug_barrier()

            # 3. Signal next rank that data is ready (set flag to 1)
            iris.atomic_xchg(
                remote_flag_ptr,
                1,
                iris_rank,
                next_rank,
                heap_bases,
                sem="release",
                scope="sys",
            )

            # 4. Wait for previous rank's data (our flag becomes 1)
            while tl.atomic_cas(local_flag_ptr, 0, 0, sem="acquire", scope="sys") != 1:
                pass

            # 5. Read received data from our ring_buffer
            recv_tile = tl.load(ring_buffer + tile_offset_in, mask=mask, other=0)

            # 6. Accumulate
            acc += recv_tile.to(acc_dtype)

            # Forward the received data (not the accumulated result)
            send_data = recv_tile

            # 7. Reset our flag for next step
            tl.debug_barrier()
            tl.atomic_xchg(local_flag_ptr, 0, sem="release", scope="sys")

        # Only the owning rank stores the result (reduce-scatter vs all-reduce)
        if DISTRIBUTION == 0:
            # Striding: rank owns tiles rank, rank+world_size, rank+2*world_size, ...
            is_owner = (tile_id % world_size) == group_rank
        else:
            # Block: rank owns tiles [rank*tiles_per_rank, (rank+1)*tiles_per_rank)
            owner_start = group_rank * tiles_per_rank
            is_owner = (tile_id >= owner_start) & (tile_id < owner_start + tiles_per_rank)

        if is_owner:
            tl.store(
                output_ptr + tile_offset_out,
                acc.to(output_ptr.type.element_ty),
                mask=mask,
            )


class ReduceScatterWorkspace:
    """Workspace for ring-based reduce-scatter."""

    __slots__ = ("ring_buffer", "flags", "prepared", "variant")

    def __init__(self):
        self.ring_buffer = None
        self.flags = None
        self.prepared = False
        self.variant = None


def _prepare_ring_workspace(shmem, M, N, dtype, total_tiles, workspace):
    """Allocate ring buffer and flags for ring_chunked variant."""
    if workspace.ring_buffer is None or workspace.ring_buffer.shape != (M, N) or workspace.ring_buffer.dtype != dtype:
        workspace.ring_buffer = shmem.zeros((M, N), dtype=dtype)
    else:
        workspace.ring_buffer.zero_()

    if workspace.flags is None or workspace.flags.numel() != total_tiles:
        workspace.flags = shmem.zeros((total_tiles,), dtype=torch.int32)
    else:
        workspace.flags.zero_()

    workspace.prepared = True
    workspace.variant = "ring_chunked"


def reduce_scatter(
    output_tensor,
    input_tensor,
    shmem,
    op=ReduceOp.SUM,
    group=None,
    async_op=False,
    config=None,
    workspace=None,
):
    """
    Internal reduce-scatter collective operation implementation.

    This function is called internally by ctx.ccl.reduce_scatter().
    Users should use the Iris instance method instead:
        >>> ctx.ccl.reduce_scatter(output_tensor, input_tensor)

    Each rank reduces its assigned tiles from all ranks' inputs and stores
    the result only to its own output tensor. This is similar to all-reduce
    but without broadcasting the result to all ranks.

    Args:
        output_tensor: Output tensor of shape (M, N) - will contain reduced tiles for this rank
        input_tensor: Input tensor of shape (M, N) - local rank's partial data
        shmem: Iris context
        op: Reduction operation to apply. Currently only ReduceOp.SUM is supported.
            Default: ReduceOp.SUM.
        group: ProcessGroup or None. If None, uses all ranks in context.
               Default: None.
        async_op: If False, performs a barrier at the end. If True, returns immediately.
                  Default: False.
        config: Config instance with kernel parameters (default: None).
                If None, uses default Config values.
                Supports reduce_scatter_variant="two_shot" or "ring_chunked".
        workspace: ReduceScatterWorkspace for reusing ring buffers across calls.
                   Only used by ring_chunked variant. If None, allocated internally.

    Example:
        >>> ctx = iris.iris()
        >>> ctx.ccl.reduce_scatter(output_tensor, input_tensor)

        >>> # Ring-chunked variant
        >>> from iris.ccl import Config
        >>> config = Config(reduce_scatter_variant="ring_chunked")
        >>> ctx.ccl.reduce_scatter(output_tensor, input_tensor, config=config)
    """
    # Validate op parameter
    if op != ReduceOp.SUM:
        raise ValueError(
            f"Only ReduceOp.SUM is currently supported, got {op}. "
            "Support for other operations (PRODUCT, MAX, MIN, etc.) will be added in a future release."
        )
    if config is None:
        # Adaptive defaults tuned on MI308X×4 vs RCCL.
        # Key insight: for very large tensors, fewer SMs (comm_sms=48) reduces
        # XGMI link contention and improves bandwidth. Smaller block_size_m=16
        # creates more tiles for better SM utilization.
        # Uses _default_config() cache to avoid repeated HIP subprocess queries.
        M_in, N_in = input_tensor.shape[:2]
        total_elems = M_in * N_in
        if total_elems >= 64 * 1024 * 1024:  # >= 64M elements
            config = _default_config(16, 64, comm_sms=48)
        elif total_elems >= 16 * 1024 * 1024:  # >= 16M elements
            config = _default_config(16, 64)
        else:
            config = _default_config(32, 64)

    # Check for unsupported options
    if config.use_gluon:
        raise ValueError(
            "reduce_scatter does not support use_gluon=True. "
            "Gluon implementation is not available for reduce_scatter. "
            "Use default config (use_gluon=False)."
        )

    variant = getattr(config, "reduce_scatter_variant", "two_shot")

    # Extract group information
    rank_in_group, rank_global, world_size, rank_start, rank_stride = extract_group_info(group, shmem)
    M, N = input_tensor.shape[:2]

    # Validate output shape matches input shape
    if output_tensor.shape[:2] != (M, N):
        raise ValueError(
            f"Output tensor shape {output_tensor.shape[:2]} does not match input shape {(M, N)}. "
            f"For reduce-scatter, output should have the same shape as input."
        )

    stride_in_m, stride_in_n = input_tensor.stride(0), input_tensor.stride(1)
    stride_out_m, stride_out_n = output_tensor.stride(0), output_tensor.stride(1)

    heap_bases = shmem.get_heap_bases()
    distribution = config.all_reduce_distribution

    if variant == "two_shot":
        persistent_reduce_scatter_two_shot[(config.comm_sms,)](
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
            distribution,
            num_stages=config.num_stages,
            num_warps=config.num_warps,
            waves_per_eu=config.waves_per_eu,
        )

    elif variant == "ring_chunked":
        # Compute total tiles for workspace allocation
        num_pid_m = (M + config.block_size_m - 1) // config.block_size_m
        num_pid_n = (N + config.block_size_n - 1) // config.block_size_n
        total_tiles = num_pid_m * num_pid_n

        # Prepare workspace
        if workspace is None:
            workspace = ReduceScatterWorkspace()
        if not workspace.prepared or workspace.variant != "ring_chunked":
            _prepare_ring_workspace(shmem, M, N, input_tensor.dtype, total_tiles, workspace)

        # Validate workspace sizes
        if workspace.flags.numel() < total_tiles:
            raise ValueError(
                f"Flags array too small: have {workspace.flags.numel()} but need {total_tiles}. "
                f"Pre-allocate workspace with the smallest block sizes you intend to use."
            )

        # Calculate next rank in the ring
        if group is None:
            next_rank_global = rank_start + ((rank_in_group + 1) % world_size) * rank_stride
        else:
            group_ranks = list(range(rank_start, rank_start + world_size * rank_stride, rank_stride))
            next_idx = (rank_in_group + 1) % world_size
            next_rank_global = group_ranks[next_idx]

        persistent_reduce_scatter_ring[(config.comm_sms,)](
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
            next_rank_global,
            config.block_size_m,
            config.block_size_n,
            config.swizzle_size,
            config.comm_sms,
            config.num_xcds,
            config.chunk_size,
            distribution,
            num_stages=config.num_stages,
            num_warps=config.num_warps,
            waves_per_eu=config.waves_per_eu,
        )
    else:
        raise ValueError(f"reduce_scatter_variant must be 'two_shot' or 'ring_chunked', got '{variant}'.")

    if not async_op:
        shmem.barrier()
