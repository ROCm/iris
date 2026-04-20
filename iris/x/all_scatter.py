# SPDX-License-Identifier: MIT
# Copyright (c) 2025-2026 Advanced Micro Devices, Inc. All rights reserved.

"""
Tile-level all-scatter primitive for Iris.

Each rank pushes its pre-computed tile to all other ranks at the rank's column
offset in the global output tensor.  After the operation every rank holds the
full result.
"""

import triton
import iris
from iris.iris import DeviceContext
from .core import Tile, TensorView


@triton.jit()
def all_scatter(
    tile: Tile,
    dst_view: TensorView,
    ctx: DeviceContext,
):
    """
    Tile-level all-scatter operation.

    Each rank scatters its pre-computed tile to all ranks (including itself) at
    its column-stripe offset in the global output.  Automatically derives
    N_local from ``dst_view`` and ``ctx.world_size``.

    Args:
        tile:     Tile object containing the pre-computed data (e.g. a GEMM
                  result in registers).
        dst_view: TensorView for the full output tensor of shape (M, N) where
                  ``N = N_local * world_size``.
        ctx:      DeviceContext carrying rank, world_size, and heap_bases.

    Layout:
        Current rank's column stripe occupies
        ``output[:, ctx.rank * N_local : (ctx.rank + 1) * N_local]``
        where ``N_local = dst_view.N // world_size``.

    Example::

        tile_obj = iris.x.Tile(pid_m, pid_n, BLOCK_SIZE_M, BLOCK_SIZE_N, c)
        dst_view = iris.x.make_tensor_view(C, M, N, stride_cm, stride_cn)
        iris.x.all_scatter(tile_obj, dst_view, ctx)
    """
    N_local = dst_view.N // ctx.world_size

    # Scatter this rank's tile to all destination ranks
    for dest_rank in range(ctx.world_size):
        # Compute pointer at this rank's column-stripe offset
        dst_ptr, combined_mask = dst_view.offset_tile_ptr(tile, offset_n=ctx.rank * N_local, src_mask=None)

        iris.store(
            dst_ptr,
            tile.data,
            ctx.rank,
            dest_rank,
            ctx.heap_bases,
            mask=combined_mask,
        )
