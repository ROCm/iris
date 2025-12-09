# SPDX-License-Identifier: MIT
# Copyright (c) 2025 Advanced Micro Devices, Inc. All rights reserved.

"""
Tile-level reduce-scatter primitive for Iris.

Reduces tiles from all ranks and stores the result only to the assigned rank.
"""

import triton
import triton.language as tl
import iris
from .common import compute_tile_indices, compute_tile_offsets


@triton.jit()
def reduce_scatter(
    input_ptr,
    output_ptr,
    pid_m,
    pid_n,
    M,
    N,
    stride_in_m,
    stride_in_n,
    stride_out_m,
    stride_out_n,
    heap_bases: tl.tensor,
    cur_rank: tl.constexpr,
    world_size: tl.constexpr,
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
):
    """
    Reduce-scatter for a single tile.

    Each rank reduces its assigned tiles from all ranks' inputs and stores
    the result only to its own output tensor. This is similar to all-reduce
    but without broadcasting the result to all ranks.

    Note: This function assumes the tile belongs to the current rank.
    Users should only call this for tiles assigned to their rank.

    Args:
        input_ptr: Pointer to input tensor (local rank's partial data) of shape (M, N)
        output_ptr: Pointer to output tensor (will contain reduced tiles for this rank) of shape (M, N)
        pid_m: Tile coordinate in M dimension
        pid_n: Tile coordinate in N dimension
        M: Number of rows in full tensor
        N: Number of columns in full tensor
        stride_in_m, stride_in_n: Strides for input tensor
        stride_out_m, stride_out_n: Strides for output tensor
        heap_bases: Heap base pointers for all ranks
        cur_rank: Current rank
        world_size: Total number of ranks
        BLOCK_SIZE_M: Block size for M dimension
        BLOCK_SIZE_N: Block size for N dimension
    """
    tl.assume(pid_m >= 0)
    tl.assume(pid_n >= 0)

    # Compute tile indices and mask
    rm, rn, mask = compute_tile_indices(pid_m, pid_n, M, N, BLOCK_SIZE_M, BLOCK_SIZE_N)
    input_offset, output_offset = compute_tile_offsets(
        rm, rn, stride_in_m, stride_in_n, stride_out_m, stride_out_n
    )

    acc_dtype = tl.float32 if output_ptr.type.element_ty != tl.int8 else tl.int32
    acc = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=acc_dtype)

    # Reduce: sum contributions from all ranks
    for remote_rank in range(world_size):
        partial = iris.load(
            input_ptr + input_offset,
            cur_rank,
            remote_rank,
            heap_bases,
            mask=mask,
        )
        acc += partial.to(acc_dtype)

    reduced = acc.to(output_ptr.type.element_ty)

    # Store only to own rank (no broadcast)
    tl.store(output_ptr + output_offset, reduced, mask=mask, cache_modifier=".wt")

