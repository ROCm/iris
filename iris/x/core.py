# SPDX-License-Identifier: MIT
# Copyright (c) 2025-2026 Advanced Micro Devices, Inc. All rights reserved.

"""
Core abstractions for iris.x tile-level primitives.

This module provides device functions for tile layout and tensor view operations.
These are used by the collective primitives to compute memory pointers and masks.

The module provides both:
1. Device functions (tile_layout, tile_ptr, offset_ptr) - Always work, recommended
2. OOP classes (Tile) - Clean API using @constexpr_function pattern
"""

import triton
import triton.language as tl
from triton.language.core import _aggregate as aggregate


@triton.jit
def tile_layout(pid_m, pid_n, M, N, BLOCK_SIZE_M: tl.constexpr, BLOCK_SIZE_N: tl.constexpr):
    """
    Compute the memory layout for a tile.

    Args:
        pid_m: Tile coordinate in M dimension
        pid_n: Tile coordinate in N dimension
        M: Total number of rows in the tensor
        N: Total number of columns in the tensor
        BLOCK_SIZE_M: Block size for M dimension (constexpr)
        BLOCK_SIZE_N: Block size for N dimension (constexpr)

    Returns:
        rm: Row indices for this tile (1D tensor of size BLOCK_SIZE_M)
        rn: Column indices for this tile (1D tensor of size BLOCK_SIZE_N)
        mask: Bounds mask for valid elements (2D tensor of shape [BLOCK_SIZE_M, BLOCK_SIZE_N])

    The returned indices are optimized with max_contiguous and multiple_of hints
    to enable better vectorization.
    """
    # Calculate base indices
    rm_base = pid_m * BLOCK_SIZE_M
    rn_base = pid_n * BLOCK_SIZE_N
    rm = rm_base + tl.arange(0, BLOCK_SIZE_M)
    rn = rn_base + tl.arange(0, BLOCK_SIZE_N)

    # Add vectorization hints
    rm = tl.max_contiguous(tl.multiple_of(rm, BLOCK_SIZE_M), BLOCK_SIZE_M)
    rn = tl.max_contiguous(tl.multiple_of(rn, BLOCK_SIZE_N), BLOCK_SIZE_N)

    # Create bounds mask
    mask = (rm[:, None] < M) & (rn[None, :] < N)

    return rm, rn, mask


@triton.jit
def tile_ptr(ptr, M, N, stride_m, stride_n, pid_m, pid_n, BLOCK_SIZE_M: tl.constexpr, BLOCK_SIZE_N: tl.constexpr):
    """
    Compute pointer tensor and mask for a tile.

    Args:
        ptr: Base pointer to tensor data
        M: Number of rows
        N: Number of columns
        stride_m: Stride in M dimension
        stride_n: Stride in N dimension
        pid_m: Tile coordinate in M dimension
        pid_n: Tile coordinate in N dimension
        BLOCK_SIZE_M: Block size for M dimension (constexpr)
        BLOCK_SIZE_N: Block size for N dimension (constexpr)

    Returns:
        tile_ptr: Pointer tensor for the tile elements (2D of shape [BLOCK_SIZE_M, BLOCK_SIZE_N])
        mask: Bounds mask for valid elements

    The returned pointer can be used directly with tl.load/tl.store or
    iris.load/iris.store for remote access.
    """
    rm, rn, mask = tile_layout(pid_m, pid_n, M, N, BLOCK_SIZE_M, BLOCK_SIZE_N)
    offset = rm[:, None] * stride_m + rn[None, :] * stride_n
    tile_ptr = ptr + offset
    tile_ptr = tl.multiple_of(tile_ptr, (BLOCK_SIZE_M, BLOCK_SIZE_N))
    return tile_ptr, mask


@triton.jit
def offset_ptr(ptr, stride_m, stride_n, offset_m, offset_n):
    """
    Compute offset pointer.

    Args:
        ptr: Base pointer to tensor data
        stride_m: Stride in M dimension
        stride_n: Stride in N dimension
        offset_m: Offset in M dimension (rows)
        offset_n: Offset in N dimension (columns)

    Returns:
        New pointer with offset applied
    """
    return ptr + offset_m * stride_m + offset_n * stride_n


@aggregate
class TileView:
    """
    TileView storing BOTH runtime coordinates AND compile-time block sizes.

    This class uses the @constexpr_function pattern discovered from Triton's gluon examples:
    - Stores runtime coordinates (pid_m, pid_n) as tl.tensor (computed from tl.program_id)
    - Stores compile-time block sizes (block_m, block_n) as tl.constexpr
    - Constructor is decorated with @constexpr_function to execute at compile-time

    Example usage:
        pid = tl.program_id(0)
        pid_m = pid // num_tiles_n  # Tensor from arithmetic
        pid_n = pid % num_tiles_n   # Tensor from arithmetic
        tile = TileView(pid_m, pid_n, BLOCK_M, BLOCK_N)
        rm, rn, mask = tile.layout(M, N)  # Coords stored, dims passed
    """

    pid_m: tl.tensor
    pid_n: tl.tensor
    block_m: tl.constexpr
    block_n: tl.constexpr

    @triton.constexpr_function
    def __init__(self, pid_m, pid_n, block_m, block_n):
        """
        Create a tile view with runtime coordinates and compile-time sizes.

        Args:
            pid_m: Tile coordinate in M dimension (tensor from tl.program_id arithmetic)
            pid_n: Tile coordinate in N dimension (tensor from tl.program_id arithmetic)
            block_m: Block size in M dimension (constexpr)
            block_n: Block size in N dimension (constexpr)
        """
        self.pid_m = pid_m  # Already a tensor
        self.pid_n = pid_n  # Already a tensor
        self.block_m = tl.constexpr(block_m)
        self.block_n = tl.constexpr(block_n)

    @triton.jit
    def layout(self, M, N):
        """
        Compute memory layout using stored coordinates.

        Args:
            M: Total rows in tensor (can be runtime or constexpr)
            N: Total columns in tensor (can be runtime or constexpr)

        Returns:
            rm, rn, mask: Row indices, column indices, and bounds mask
        """
        return tile_layout(self.pid_m, self.pid_n, M, N, self.block_m, self.block_n)

    @triton.jit
    def indices(self):
        """
        Compute base row and column indices for this tile.

        Returns:
            rm, rn: Row indices, column indices (without bounds checking)

        Example:
            rm, rn = tile.indices()
            rm_offset = rm + offset_rows
            rn_offset = rn + offset_cols
        """
        rm = self.pid_m * self.block_m + tl.arange(0, self.block_m)
        rn = self.pid_n * self.block_n + tl.arange(0, self.block_n)
        return rm, rn


@aggregate
class Tile:
    """
    Tile with embedded data for computed results.

    Extends TileView to include a data field for storing computed tile data
    (e.g., GEMM results in registers).

    Example usage:
        pid = tl.program_id(0)
        pid_m = pid // num_tiles_n
        pid_n = pid % num_tiles_n
        c = tl.dot(a, b)  # Compute tile data
        tile = Tile(pid_m, pid_n, BLOCK_M, BLOCK_N, c)
        # Now tile.data contains the computed result
    """

    pid_m: tl.tensor
    pid_n: tl.tensor
    block_m: tl.constexpr
    block_n: tl.constexpr
    data: tl.tensor

    @triton.constexpr_function
    def __init__(self, pid_m, pid_n, block_m, block_n, data):
        """
        Create a tile with runtime coordinates, compile-time sizes, and data.

        Args:
            pid_m: Tile coordinate in M dimension (tensor from tl.program_id arithmetic)
            pid_n: Tile coordinate in N dimension (tensor from tl.program_id arithmetic)
            block_m: Block size in M dimension (constexpr)
            block_n: Block size in N dimension (constexpr)
            data: Computed tile data (e.g., GEMM result in registers)
        """
        self.pid_m = pid_m  # Already a tensor
        self.pid_n = pid_n  # Already a tensor
        self.block_m = tl.constexpr(block_m)
        self.block_n = tl.constexpr(block_n)
        self.data = data

    @triton.jit
    def layout(self, M, N):
        """
        Compute memory layout using stored coordinates.

        Args:
            M: Total rows in tensor (can be runtime or constexpr)
            N: Total columns in tensor (can be runtime or constexpr)

        Returns:
            rm, rn, mask: Row indices, column indices, and bounds mask
        """
        return tile_layout(self.pid_m, self.pid_n, M, N, self.block_m, self.block_n)

    @triton.jit
    def indices(self):
        """
        Compute base row and column indices for this tile.

        Returns:
            rm, rn: Row indices, column indices (without bounds checking)

        Example:
            rm, rn = tile.indices()
            rm_offset = rm + offset_rows
            rn_offset = rn + offset_cols
        """
        rm = self.pid_m * self.block_m + tl.arange(0, self.block_m)
        rn = self.pid_n * self.block_n + tl.arange(0, self.block_n)
        return rm, rn


@aggregate
class TensorView:
    """
    TensorView storing pointer and tensor metadata.

    This works when dimensions and strides are marked as tl.constexpr in the kernel signature!

    Example usage (with constexpr dimensions):
        @triton.jit
        def kernel(ptr, M: tl.constexpr, N: tl.constexpr,
                   stride_m: tl.constexpr, stride_n: tl.constexpr, ...):
            view = TensorView(ptr, M, N, stride_m, stride_n)
            tile = Tile(pid_m, pid_n, BLOCK_M, BLOCK_N)
            ptr, mask = view.tile_ptr(tile)

    Note: If M, N, strides are NOT constexpr (runtime kernel args), you cannot store them.
    In that case, use the device functions directly or pass them as method arguments.
    """

    ptr: tl.tensor
    M: tl.constexpr
    N: tl.constexpr
    stride_m: tl.constexpr
    stride_n: tl.constexpr

    @triton.constexpr_function
    def __init__(self, ptr, M, N, stride_m, stride_n):
        """
        Create a tensor view with pointer and constexpr dimensions/strides.

        Args:
            ptr: Pointer to tensor data (runtime tensor)
            M: Number of rows (must be constexpr in kernel signature)
            N: Number of columns (must be constexpr in kernel signature)
            stride_m: Stride in M dimension (must be constexpr in kernel signature)
            stride_n: Stride in N dimension (must be constexpr in kernel signature)
        """
        self.ptr = ptr
        self.M = tl.constexpr(M)
        self.N = tl.constexpr(N)
        self.stride_m = tl.constexpr(stride_m)
        self.stride_n = tl.constexpr(stride_n)

    @triton.jit
    def tile_ptr(self, tile: Tile):
        """
        Compute tile pointer and mask using stored dimensions/strides.

        Args:
            tile: Tile object with stored coordinates

        Returns:
            tile_ptr, mask: Pointer tensor and bounds mask
        """
        return tile_ptr(
            self.ptr, self.M, self.N, self.stride_m, self.stride_n, tile.pid_m, tile.pid_n, tile.block_m, tile.block_n
        )

    @triton.jit
    def tile_ptr_from_indices(self, rm, rn, block_m: tl.constexpr, block_n: tl.constexpr):
        """
        Compute tile pointer and mask from custom row/column indices.

        This is useful when you need to access a tile at a custom location
        (e.g., with an offset or transformation applied to the indices).

        Args:
            rm: Row indices (1D tensor of size block_m)
            rn: Column indices (1D tensor of size block_n)
            block_m: Block size in M dimension (constexpr)
            block_n: Block size in N dimension (constexpr)

        Returns:
            tile_ptr, mask: Pointer tensor and bounds mask

        Example:
            # Access tile with offset
            rm_offset = tile.pid_m * tile.block_m + offset + tl.arange(0, tile.block_m)
            rn = tile.pid_n * tile.block_n + tl.arange(0, tile.block_n)
            ptr, mask = view.tile_ptr_from_indices(rm_offset, rn, tile.block_m, tile.block_n)
        """
        # Add vectorization hints
        rm = tl.max_contiguous(tl.multiple_of(rm, block_m), block_m)
        rn = tl.max_contiguous(tl.multiple_of(rn, block_n), block_n)

        # Create bounds mask
        mask = (rm[:, None] < self.M) & (rn[None, :] < self.N)

        # Compute pointer offsets
        offset = rm[:, None] * self.stride_m + rn[None, :] * self.stride_n
        tile_ptr = self.ptr + offset
        tile_ptr = tl.multiple_of(tile_ptr, (block_m, block_n))

        return tile_ptr, mask

    @triton.jit
    def offset_tile_ptr(self, tile: Tile, offset_m=0, offset_n=0, src_mask=None):
        """
        Compute tile pointer with row/column offsets applied.

        This is a higher-level convenience method that combines tile.indices(),
        offset application, and tile_ptr_from_indices() into a single call.

        Args:
            tile: Tile object with position and block sizes
            offset_m: Offset to add to row indices (default: 0)
            offset_n: Offset to add to column indices (default: 0)
            src_mask: Optional source mask to combine with computed mask (default: None)

        Returns:
            tile_ptr, mask: Pointer tensor and combined mask

        Example:
            # Access tile with rank-specific offset
            ptr, mask = dst_view.offset_tile_ptr(
                tile, offset_m=rank * M, src_mask=input_mask
            )
        """
        # Get base indices from tile
        rm, rn = tile.indices()

        # Apply offsets
        rm_offset = rm + offset_m
        rn_offset = rn + offset_n

        # Compute pointer and mask
        tile_ptr, dst_mask = self.tile_ptr_from_indices(rm_offset, rn_offset, tile.block_m, tile.block_n)

        # Combine masks if source mask provided
        if src_mask is not None:
            mask = src_mask & dst_mask
        else:
            mask = dst_mask

        return tile_ptr, mask

    @triton.jit
    def offset(self, offset_m=0, offset_n=0):
        """
        Create a new view with pointer offset applied.

        Args:
            offset_m: Offset in M dimension (rows)
            offset_n: Offset in N dimension (columns)

        Returns:
            New TensorView with offset pointer
        """
        new_ptr = offset_ptr(self.ptr, self.stride_m, self.stride_n, offset_m, offset_n)
        return TensorView(new_ptr, self.M, self.N, self.stride_m, self.stride_n)

    @triton.jit
    def number_of_tiles(self, block_m, block_n):
        """
        Compute the total number of tiles needed to cover the tensor.

        Args:
            block_m: Tile size in M dimension (constexpr)
            block_n: Tile size in N dimension (constexpr)

        Returns:
            Total number of tiles (num_tiles_m * num_tiles_n)
        """
        num_tiles_m = tl.cdiv(self.M, block_m)
        num_tiles_n = tl.cdiv(self.N, block_n)
        return num_tiles_m * num_tiles_n


@aggregate
class AllReduceConfig:
    """
    Device-side configuration for all_reduce collective.

    This config is shared across all tiles and specifies which algorithm to use
    and any required temporary resources (like locks).

    Fields:
        variant_code: Integer code for algorithm variant (0=atomic, 1=ring, 2=one_shot, 3=two_shot, 4=spinlock)
        locks_ptr: Pointer to locks array (ALWAYS required - pass dummy tensor if variant doesn't need locks)

    Example:
        # For variants that don't need locks (atomic, ring, one_shot, two_shot):
        dummy_locks = tl.zeros((1,), dtype=tl.int32)
        config = AllReduceConfig(0, dummy_locks)  # 0 = atomic

        # For spinlock variant that needs locks:
        locks = shmem.zeros(num_tiles, dtype=torch.int32)
        config = AllReduceConfig(4, locks)  # 4 = spinlock

        # Then in kernel:
        ctx.all_reduce(tile, src_view, dst_view, config=config, tile_id=tile_id)

    Variant codes:
        0 = atomic
        1 = ring
        2 = one_shot
        3 = two_shot
        4 = spinlock
    """

    variant_code: tl.constexpr  # Integer code for variant
    locks_ptr: tl.tensor  # Pointer to locks (always required, may be dummy)

    @triton.constexpr_function
    def __init__(self, variant_code, locks_ptr):
        """
        Create an all_reduce configuration.

        Args:
            variant_code: Integer code for variant (0-4)
            locks_ptr: Pointer to locks array (REQUIRED - pass dummy if not used)
        """
        self.variant_code = tl.constexpr(variant_code)  # Wrap as constexpr
        self.locks_ptr = locks_ptr


__all__ = ["TileView", "Tile", "TensorView", "AllReduceConfig", "tile_layout", "tile_ptr", "offset_ptr"]
