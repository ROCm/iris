# SPDX-License-Identifier: MIT
# Copyright (c) 2025 Advanced Micro Devices, Inc. All rights reserved.

"""
Tile-level all-to-all primitive for Iris.

Performs all-to-all communication where each rank sends and receives data to/from all other ranks.
"""

import triton
import triton.language as tl
import iris
from .core import Tile, TensorView, DeviceContext


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
