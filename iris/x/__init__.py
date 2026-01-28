# SPDX-License-Identifier: MIT
# Copyright (c) 2025 Advanced Micro Devices, Inc. All rights reserved.

"""
iris-x: Device-side tile-level primitives for fine-grained compute and collective operations.

This module provides composable tile-level functions that users can call from their own kernels.
Unlike iris.ccl which handles full tensors with internal tiling, iris.x provides functions
that operate on individual tiles, allowing users to manage tile iteration themselves.

Example:
    >>> import iris
    >>> import iris.x
    >>> import triton
    >>> import triton.language as tl
    >>>
    >>> @triton.jit
    >>> def my_kernel(input_ptr, output_ptr, pid_m, pid_n, ...):
    >>>     # Process a single tile
    >>>     iris.x.all_reduce_atomic(
    >>>         input_ptr, output_ptr, pid_m, pid_n,
    >>>         M, N, stride_in_m, stride_in_n, stride_out_m, stride_out_n,
    >>>         heap_bases, cur_rank, world_size, BLOCK_SIZE_M, BLOCK_SIZE_N
    >>>     )
"""

from .core import Tile, TensorView, DeviceContext, tile_layout, tile_ptr, offset_ptr
from .all_reduce import (
    all_reduce_atomic,
    all_reduce_ring,
    all_reduce_two_shot,
    all_reduce_one_shot,
    all_reduce_spinlock,
)
from .all_gather import all_gather
from .all_to_all import all_to_all
from .reduce_scatter import reduce_scatter

# Try to import GEMM+Comm primitives (requires tritonBLAS)
try:
    from .gemm_all_gather import gemm_all_gather
    from .gemm_all_reduce import gemm_all_reduce
    from .all_gather_gemm import all_gather_gemm
    from .gemm_reduce_scatter import gemm_reduce_scatter

    __all__ = [
        # Core abstractions
        "Tile",
        "TensorView",
        "DeviceContext",
        "tile_layout",
        "tile_ptr",
        "offset_ptr",
        # Collectives
        "all_reduce_atomic",
        "all_reduce_ring",
        "all_reduce_two_shot",
        "all_reduce_one_shot",
        "all_reduce_spinlock",
        "all_gather",
        "all_to_all",
        "reduce_scatter",
        # GEMM+Comm
        "gemm_all_gather",
        "gemm_all_reduce",
        "all_gather_gemm",
        "gemm_reduce_scatter",
    ]
except ImportError:
    __all__ = [
        "all_reduce_atomic",
        "all_reduce_ring",
        "all_reduce_two_shot",
        "all_reduce_one_shot",
        "all_reduce_spinlock",
        "all_gather",
        "all_to_all",
        "reduce_scatter",
    ]
