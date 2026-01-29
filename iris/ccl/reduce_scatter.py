# SPDX-License-Identifier: MIT
# Copyright (c) 2025 Advanced Micro Devices, Inc. All rights reserved.

"""
Reduce-scatter collective communication primitive for Iris.
Uses the two-shot approach: reduce assigned tiles and store only to own rank.
"""

import triton
import triton.language as tl
import iris
from .config import Config
from .utils import chiplet_transform_chunked, ReduceOp, extract_group_info


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
    Reduce-scatter using two-shot approach.

    Each rank reduces its assigned tiles from all ranks and stores the result
    only to its own output (no broadcast to other ranks).
    """
    pid = tl.program_id(0)

    if NUM_XCDS != 1:
        pid = chiplet_transform_chunked(pid, COMM_SMS, NUM_XCDS, CHUNK_SIZE)

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
        # The masking is problem size dependent, and the compiler does not recognize it can have two paths
        # (one with masks and one without). Separate unmasked paths allow the compiler to generate
        # more efficient vectorized instructions.
        if is_full:
            start_rank_idx = pid % world_size
            start_rank_global = rank_start + start_rank_idx * rank_stride
            acc = iris.load(base_ptr, iris_rank, start_rank_global, heap_bases).to(acc_dtype)
            for i in tl.static_range(1, world_size):
                remote_rank_idx = (start_rank_idx + i) % world_size
                remote_rank = rank_start + remote_rank_idx * rank_stride
                acc += iris.load(base_ptr, iris_rank, remote_rank, heap_bases).to(acc_dtype)

            reduced = acc.to(output_ptr.type.element_ty)

            # Store only to own rank (no broadcast)
            tl.store(out_ptr, reduced, cache_modifier=".wt")

        # Slow path: MASKED (only boundary tiles land here)
        # This path handles tiles at tensor boundaries where not all elements are valid.
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

            # Store only to own rank (no broadcast)
            tl.store(out_ptr, reduced, mask=mask, cache_modifier=".wt")


def reduce_scatter(
    output_tensor,
    input_tensor,
    shmem,
    op=ReduceOp.SUM,
    group=None,
    async_op=False,
    config=None,
):
    """
    Internal reduce-scatter collective operation implementation.

    This function is called internally by shmem.ccl.reduce_scatter().
    Users should use the Iris instance method instead:
        >>> shmem.ccl.reduce_scatter(output_tensor, input_tensor)

    Each rank reduces its assigned tiles from all ranks' inputs and stores
    the result only to its own output tensor. This is similar to all-reduce
    but without broadcasting the result to all ranks.

    Args:
        output_tensor: Output tensor of shape (M, N) - will contain reduced tiles for this rank
        input_tensor: Input tensor of shape (M, N) - local rank's partial data
        shmem: Iris shmem context
        op: Reduction operation to apply. Currently only ReduceOp.SUM is supported.
            Default: ReduceOp.SUM.
        group: ProcessGroup or None. If None, uses all ranks in shmem context.
               Default: None.
        async_op: If False, performs a barrier at the end. If True, returns immediately.
                  Default: False.
        config: Config instance with kernel parameters (default: None).
                If None, uses default Config values.
                Only supports reduce_scatter_variant="two_shot".

    Example:
        >>> shmem = iris.iris()
        >>> shmem.ccl.reduce_scatter(output_tensor, input_tensor)

        >>> # Custom configuration
        >>> from iris.ccl import Config
        >>> config = Config(reduce_scatter_variant="two_shot", all_reduce_distribution=1)
        >>> shmem.ccl.reduce_scatter(output_tensor, input_tensor, config=config)
    """
    # Validate op parameter
    if op != ReduceOp.SUM:
        raise ValueError(
            f"Only ReduceOp.SUM is currently supported, got {op}. "
            "Support for other operations (PRODUCT, MAX, MIN, etc.) will be added in a future release."
        )
    if config is None:
        config = Config(block_size_m=32, block_size_n=64, all_reduce_distribution=1)

    # Check for unsupported options
    if config.use_gluon:
        raise ValueError(
            "reduce_scatter does not support use_gluon=True. "
            "Gluon implementation is not available for reduce_scatter. "
            "Use default config (use_gluon=False)."
        )

    # Validate that only two_shot variant is used
    variant = getattr(config, "reduce_scatter_variant", "two_shot")
    if variant != "two_shot":
        raise ValueError(
            f"reduce_scatter only supports variant='two_shot', got '{variant}'. "
            f"Set config.reduce_scatter_variant='two_shot' or use default config."
        )

    # Extract group information
    # rank_in_group: position within the group (0, 1, 2, ...) - used for tile assignment
    # rank_global: global rank in iris context - passed as iris_rank to kernel for RMA operations
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

    # Use all_reduce_distribution for tile distribution
    distribution = config.all_reduce_distribution

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
    )

    if not async_op:
        shmem.barrier()
