# SPDX-License-Identifier: MIT
# Copyright (c) 2025 Advanced Micro Devices, Inc. All rights reserved.

"""
Tile-level all-gather primitive for Iris.

Gathers tiles from all ranks and concatenates them along the output dimension.
"""

import triton
import triton.language as tl
import iris
from .core import Tile, TensorView, DeviceContext


@triton.jit()
def all_gather(
    tile: Tile,
    src_view: TensorView,
    dst_view: TensorView,
    dim: tl.constexpr,
    ctx: DeviceContext,
):
    """
    Tile-level all-gather for iris.x.

    Gathers data from all ranks and concatenates along the specified dimension.
    Each rank contributes one tile worth of data.

    Args:
        tile: Tile object with position and dimensions.
        src_view: TensorView for input tensor.
        dst_view: TensorView for output tensor.
        dim: Dimension to gather along (0 for M, 1 for N).
        ctx: DeviceContext with rank, world_size, and heap_bases.
    
    Gather dimension behavior:
        - dim=0: Input (M, N) -> Output (world_size * M, N)
          Each rank's data goes to output[rank * M : (rank+1) * M, :]
        - dim=1: Input (M, N) -> Output (M, world_size * N)
          Each rank's data goes to output[:, rank * N : (rank+1) * N]
    """
    # Get tile pointer and mask
    src_tile_ptr, mask = src_view.tile_ptr(tile)

    # Load local tile
    local_tile = tl.load(src_tile_ptr, mask=mask, other=0.0)

    # Determine output layout based on gather dimension
    if dim == 0:
        # Gather along M dimension: output is (world_size * M) x N
        # Each rank writes to a different row block
        for r in range(ctx.world_size):
            if r == ctx.rank:
                # Write local tile to output
                out_offset_m = ctx.rank * src_view.M + tile.pid_m * tile.block_m
                out_offset_n = tile.pid_n * tile.block_n
                out_indices_m = out_offset_m + tl.arange(0, tile.block_m)
                out_indices_n = out_offset_n + tl.arange(0, tile.block_n)
                out_offsets = out_indices_m[:, None] * dst_view.stride_m + out_indices_n[None, :] * dst_view.stride_n
                tl.store(dst_view.ptr + out_offsets, local_tile, mask=mask)
            else:
                # Read from remote rank
                remote_tile = iris.load(
                    src_tile_ptr,
                    ctx.rank,  # to_rank (current rank)
                    r,  # from_rank (remote rank)
                    ctx.heap_bases,
                    mask=mask,
                )
                # Write to output at rank r's section
                out_offset_m = r * src_view.M + tile.pid_m * tile.block_m
                out_offset_n = tile.pid_n * tile.block_n
                out_indices_m = out_offset_m + tl.arange(0, tile.block_m)
                out_indices_n = out_offset_n + tl.arange(0, tile.block_n)
                out_offsets = out_indices_m[:, None] * dst_view.stride_m + out_indices_n[None, :] * dst_view.stride_n
                tl.store(dst_view.ptr + out_offsets, remote_tile, mask=mask)
    else:
        # Gather along N dimension: output is M x (world_size * N)
        # Each rank writes to a different column block
        for r in range(ctx.world_size):
            if r == ctx.rank:
                # Write local tile to output
                out_offset_m = tile.pid_m * tile.block_m
                out_offset_n = ctx.rank * src_view.N + tile.pid_n * tile.block_n
                out_indices_m = out_offset_m + tl.arange(0, tile.block_m)
                out_indices_n = out_offset_n + tl.arange(0, tile.block_n)
                out_offsets = out_indices_m[:, None] * dst_view.stride_m + out_indices_n[None, :] * dst_view.stride_n
                tl.store(dst_view.ptr + out_offsets, local_tile, mask=mask)
            else:
                # Read from remote rank
                remote_tile = iris.load(
                    src_tile_ptr,
                    ctx.rank,  # to_rank (current rank)
                    r,  # from_rank (remote rank)
                    ctx.heap_bases,
                    mask=mask,
                )
                # Write to output at rank r's section
                out_offset_m = tile.pid_m * tile.block_m
                out_offset_n = r * src_view.N + tile.pid_n * tile.block_n
                out_indices_m = out_offset_m + tl.arange(0, tile.block_m)
                out_indices_n = out_offset_n + tl.arange(0, tile.block_n)
                out_offsets = out_indices_m[:, None] * dst_view.stride_m + out_indices_n[None, :] * dst_view.stride_n
                tl.store(dst_view.ptr + out_offsets, remote_tile, mask=mask)
