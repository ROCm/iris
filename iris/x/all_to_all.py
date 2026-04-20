# SPDX-License-Identifier: MIT
# Copyright (c) 2025-2026 Advanced Micro Devices, Inc. All rights reserved.

"""
Tile-level all-to-all primitive for Iris.

Performs all-to-all communication where each rank sends and receives data to/from all other ranks.
Provides both Triton (@triton.jit) and Gluon (@gluon.jit) implementations.
"""

import triton
import triton.language as tl
import iris
from iris.iris import DeviceContext
from .core import Tile, TensorView

# Conditional import for Gluon
try:
    from triton.experimental import gluon
    from triton.experimental.gluon import language as gl
    from iris.experimental.iris_gluon import IrisDeviceCtx as _IrisDeviceCtx  # noqa: F401

    GLUON_AVAILABLE = True
except ImportError:
    GLUON_AVAILABLE = False


@triton.jit()
def all_to_all(
    tile: Tile,
    src_view: TensorView,
    dst_view: TensorView,
    N_per_rank: tl.constexpr,
    ctx: DeviceContext,
):
    """
    Tile-level all-to-all for iris.x.

    Each rank sends portions of its data to every other rank and receives data
    from every other rank. The data is organized by columns (N dimension).

    Args:
        tile: Tile object with position and dimensions.
        src_view: TensorView for input tensor.
        dst_view: TensorView for output tensor.
        N_per_rank: Number of columns each rank sends/receives per rank.
        ctx: DeviceContext with rank, world_size, and heap_bases.

    Semantics:
        Input: Each rank has (M, world_size * N_per_rank)
        Output: Each rank has (M, world_size * N_per_rank)

        PyTorch semantics: rank dst receives chunk dst from each rank src
        - Rank dst's output columns [src*N:(src+1)*N] contain rank src's input columns [dst*N:(dst+1)*N]

        For a tile processing output columns [pid_n*BLOCK_N:(pid_n+1)*BLOCK_N]:
        - The tile might span multiple source ranks' chunks
        - We need to iterate over all source ranks whose data appears in this tile
    """
    output_col_start = tile.pid_n * tile.block_n
    output_col_end = output_col_start + tile.block_n

    # Determine which source ranks contribute to this output tile
    first_src_rank = output_col_start // N_per_rank
    last_src_rank = tl.minimum((output_col_end - 1) // N_per_rank, ctx.world_size - 1)

    # Process each source rank that contributes to this tile
    for src_rank in range(first_src_rank, last_src_rank + 1):
        # Determine the column range within this tile that comes from src_rank
        src_chunk_out_start = src_rank * N_per_rank  # Where src_rank's chunk starts in output
        src_chunk_out_end = (src_rank + 1) * N_per_rank  # Where it ends

        # Intersect with this tile's output range
        tile_src_start = tl.maximum(output_col_start, src_chunk_out_start)
        tile_src_end = tl.minimum(output_col_end, src_chunk_out_end)

        # Offset within the tile
        offset_in_tile = tile_src_start - output_col_start
        num_cols = tile_src_end - tile_src_start

        # Offset within src_rank's chunk
        offset_in_src_chunk = tile_src_start - src_chunk_out_start

        # Compute source column offset: where to read from src_rank's input
        src_col_offset = ctx.rank * N_per_rank + offset_in_src_chunk

        # Compute indices
        src_indices_m = tile.pid_m * tile.block_m + tl.arange(0, tile.block_m)
        # For source columns, we need to read starting from src_col_offset
        # But we only read num_cols columns, starting at offset_in_tile within the tile
        src_col_base = src_col_offset - offset_in_tile  # Adjust so that offset_in_tile aligns correctly
        src_indices_n = src_col_base + tl.arange(0, tile.block_n)

        # Source mask: only read columns in the valid range for this src_rank
        mask_m = src_indices_m < src_view.M
        # Check if column index is in the range [src_col_offset, src_col_offset + num_cols)
        col_in_range = (src_indices_n >= src_col_offset) & (src_indices_n < src_col_offset + num_cols)
        mask_n = (src_indices_n < src_view.N) & (src_indices_n >= 0) & col_in_range
        mask = mask_m[:, None] & mask_n[None, :]

        # Compute offsets
        src_offsets = src_indices_m[:, None] * src_view.stride_m + src_indices_n[None, :] * src_view.stride_n

        # Destination indices
        dst_indices_m = tile.pid_m * tile.block_m + tl.arange(0, tile.block_m)
        dst_indices_n = output_col_start + tl.arange(0, tile.block_n)
        dst_offsets = dst_indices_m[:, None] * dst_view.stride_m + dst_indices_n[None, :] * dst_view.stride_n

        # Destination mask
        dst_mask_m = dst_indices_m < dst_view.M
        dst_mask_n = (
            (dst_indices_n >= output_col_start + offset_in_tile)
            & (dst_indices_n < output_col_start + offset_in_tile + num_cols)
            & (dst_indices_n < dst_view.N)
        )
        dst_mask = dst_mask_m[:, None] & dst_mask_n[None, :]

        # Combined mask
        combined_mask = mask & dst_mask

        if src_rank == ctx.rank:
            # Local: read from our own input and write to our own output
            data = tl.load(src_view.ptr + src_offsets, mask=combined_mask, other=0.0)
            tl.store(dst_view.ptr + dst_offsets, data, mask=combined_mask)
        else:
            # Remote: read from src_rank's input and write to our output
            data = iris.load(
                src_view.ptr + src_offsets,
                ctx.rank,  # to_rank (current rank)
                src_rank,  # from_rank (where to read from)
                ctx.heap_bases,
                mask=combined_mask,
            )
            tl.store(dst_view.ptr + dst_offsets, data, mask=combined_mask)


# Gluon implementation
if GLUON_AVAILABLE:

    @gluon.jit
    def all_to_all_gluon(
        IrisDeviceCtx: gl.constexpr,
        context_tensor,
        src_ptr,
        dst_ptr,
        M,
        N,
        stride_src_m,
        stride_src_n,
        stride_dst_m,
        stride_dst_n,
        pid_m,
        pid_n,
        N_per_rank: gl.constexpr,
        cur_rank: gl.constexpr,
        world_size: gl.constexpr,
        BLOCK_SIZE_M: gl.constexpr,
        BLOCK_SIZE_N: gl.constexpr,
    ):
        """
        Gluon tile-level all-to-all for iris.x.

        Gluon port of all_to_all using IrisDeviceCtx. Can be called from
        within a @gluon.jit kernel. Iterates over all source ranks with a
        compile-time-unrolled loop (world_size is constexpr) and applies
        masking to handle tiles that span rank-chunk boundaries.

        Args:
            IrisDeviceCtx: IrisDeviceCtx class (constexpr, passed as first arg).
            context_tensor: Encoded context tensor from shmem.get_device_context().
            src_ptr: Pointer to source tensor (local rank's input).
            dst_ptr: Pointer to destination tensor (local rank's output).
            M: Number of rows.
            N: Total number of columns (world_size * N_per_rank).
            stride_src_m, stride_src_n: Strides for source tensor.
            stride_dst_m, stride_dst_n: Strides for destination tensor.
            pid_m: Tile row index.
            pid_n: Tile column index.
            N_per_rank: Number of columns per rank (constexpr).
            cur_rank: Current rank (constexpr).
            world_size: Total number of ranks (constexpr).
            BLOCK_SIZE_M: Block size for M dimension (constexpr).
            BLOCK_SIZE_N: Block size for N dimension (constexpr).

        Semantics:
            Input:  Each rank has (M, world_size * N_per_rank)
            Output: Each rank has (M, world_size * N_per_rank)

            rank dst's output columns [src*N:(src+1)*N] receive rank src's
            input columns [dst*N:(dst+1)*N].

        Example:
            @gluon.jit
            def my_kernel(IrisDeviceCtx: gl.constexpr, context_tensor, ...):
                pid_m = ...
                pid_n = ...
                iris.x.all_to_all_gluon(
                    IrisDeviceCtx, context_tensor,
                    src_ptr, dst_ptr, M, N,
                    stride_src_m, stride_src_n,
                    stride_dst_m, stride_dst_n,
                    pid_m, pid_n, N_per_rank, rank, world_size,
                    BLOCK_SIZE_M, BLOCK_SIZE_N,
                )
        """
        ctx = IrisDeviceCtx.initialize(context_tensor)

        # 1-D layout covering BLOCK_SIZE_N elements across 4 warps of 64 threads.
        # Mirrors the layout used in persistent_all_to_all_gluon.
        col_layout: gl.constexpr = gl.BlockedLayout([1], [64], [4], [0])

        output_col_start = pid_n * BLOCK_SIZE_N
        output_col_end = output_col_start + BLOCK_SIZE_N

        # Destination column indices are the same regardless of source rank.
        rn_dst = output_col_start + gl.arange(0, BLOCK_SIZE_N, layout=col_layout)

        # Iterate over all source ranks (loop is unrolled because world_size is constexpr).
        for src_rank in range(world_size):
            src_chunk_out_start = src_rank * N_per_rank
            src_chunk_out_end = (src_rank + 1) * N_per_rank

            # Intersection of this tile's output range with src_rank's chunk.
            tile_src_start = tl.maximum(output_col_start, src_chunk_out_start)
            tile_src_end = tl.minimum(output_col_end, src_chunk_out_end)
            num_cols = tile_src_end - tile_src_start

            # Where the intersection starts within this tile and within src's chunk.
            offset_in_tile = tile_src_start - output_col_start
            offset_in_src_chunk = tile_src_start - src_chunk_out_start

            # Source column in src_rank's input that maps to this output region.
            src_col_offset = cur_rank * N_per_rank + offset_in_src_chunk

            # Source column indices adjusted so that col offset_in_tile aligns correctly.
            src_col_base = src_col_offset - offset_in_tile
            rn_src = src_col_base + gl.arange(0, BLOCK_SIZE_N, layout=col_layout)

            # Column validity masks.
            src_col_valid = (
                (rn_src >= src_col_offset) & (rn_src < src_col_offset + num_cols) & (rn_src >= 0) & (rn_src < N)
            )
            dst_col_valid = (
                (rn_dst >= output_col_start + offset_in_tile)
                & (rn_dst < output_col_start + offset_in_tile + num_cols)
                & (rn_dst < N)
            )
            # Also skip this rank entirely when there is no overlap.
            col_mask = src_col_valid & dst_col_valid & (num_cols > 0)

            # Process each row in the tile (unrolled since BLOCK_SIZE_M is constexpr).
            for i in range(BLOCK_SIZE_M):
                row_m = pid_m * BLOCK_SIZE_M + i

                src_offsets = row_m * stride_src_m + rn_src * stride_src_n
                dst_offsets = row_m * stride_dst_m + rn_dst * stride_dst_n

                # Combine column mask with row bounds check.
                row_col_mask = col_mask & (row_m < M)

                if src_rank == cur_rank:
                    # Local copy: read from our own input.
                    data = gl.load(src_ptr + src_offsets, mask=row_col_mask)
                    gl.store(dst_ptr + dst_offsets, data, mask=row_col_mask)
                else:
                    # Remote read: translate pointer to src_rank's address space.
                    data = ctx.load(src_ptr + src_offsets, src_rank, mask=row_col_mask)
                    gl.store(dst_ptr + dst_offsets, data, mask=row_col_mask)
