# SPDX-License-Identifier: MIT
# Copyright (c) 2025 Advanced Micro Devices, Inc. All rights reserved.

"""
iris-x: Device-side tile-level primitives for fine-grained compute and collective operations.

This module provides composable tile-level functions that users can call from their own kernels.
Unlike iris.ccl which handles full tensors with internal tiling, iris.x provides functions
that operate on individual tiles, allowing users to manage tile iteration themselves.

Example (API with default algorithms):
    >>> import iris
    >>> import iris.x
    >>> import triton
    >>> import triton.language as tl
    >>>
    >>> @triton.jit
    >>> def my_kernel(input_ptr, output_ptr, ...):
    >>>     tile = iris.x.TileView(pid_m, pid_n, BLOCK_M, BLOCK_N)
    >>>     src_view = iris.x.TensorView(input_ptr, M, N, stride_m, stride_n)
    >>>     dst_view = iris.x.TensorView(output_ptr, M, N, stride_m, stride_n)
    >>>     ctx = iris.x.DeviceContext(rank, world_size, heap_bases)
    >>>
    >>>     # Call collectives on ctx directly (default algorithms)
    >>>     ctx.all_reduce(tile, src_view, dst_view)
    >>>     ctx.all_gather(tile, src_view, dst_view, dim=0)
    >>>     ctx.all_to_all(tile, src_view, dst_view, N_per_rank)
    >>>     ctx.reduce_scatter(tile, src_view, dst_view)

Example (API with AllReduceConfig for algorithm selection):
    >>> @triton.jit
    >>> def my_kernel(input_ptr, output_ptr, locks_ptr, ...):
    >>>     tile = iris.x.TileView(pid_m, pid_n, BLOCK_M, BLOCK_N)
    >>>     src_view = iris.x.TensorView(input_ptr, M, N, stride_m, stride_n)
    >>>     dst_view = iris.x.TensorView(output_ptr, M, N, stride_m, stride_n)
    >>>     ctx = iris.x.DeviceContext(rank, world_size, heap_bases)
    >>>
    >>>     # Use ring algorithm
    >>>     config = iris.x.AllReduceConfig("ring")
    >>>     ctx.all_reduce(tile, src_view, dst_view, config=config)
    >>>
    >>>     # Use spinlock with locks
    >>>     config = iris.x.AllReduceConfig("spinlock", locks_ptr)
    >>>     tile_id = pid_m * num_tiles_n + pid_n
    >>>     ctx.all_reduce(tile, src_view, dst_view, config=config, tile_id=tile_id)

Example (Standalone API):
    >>> @triton.jit
    >>> def my_kernel(input_ptr, output_ptr, ...):
    >>>     iris.x.all_reduce_atomic(tile, src_view, dst_view, ctx)
    >>>     iris.x.all_gather(tile, src_view, dst_view, dim, ctx)
"""

from .core import Tile, TileView, TensorView, DeviceContext, AllReduceConfig, tile_layout, tile_ptr, offset_ptr
from .all_reduce import (
    all_reduce_atomic,
    all_reduce_ring,
    all_reduce_two_shot,
    all_reduce_one_shot,
    all_reduce_spinlock,
)
from .gather import gather
from .all_gather import all_gather
from .all_to_all import all_to_all
from .reduce_scatter import reduce_scatter

__all__ = [
    # Core abstractions
    "Tile",
    "TileView",
    "TensorView",
    "DeviceContext",
    "AllReduceConfig",
    "tile_layout",
    "tile_ptr",
    "offset_ptr",
    # Device-side collectives
    "all_reduce_atomic",
    "all_reduce_ring",
    "all_reduce_two_shot",
    "all_reduce_one_shot",
    "all_reduce_spinlock",
    "gather",
    "all_gather",
    "all_to_all",
    "reduce_scatter",
]
