# SPDX-License-Identifier: MIT
# Copyright (c) 2025-2026 Advanced Micro Devices, Inc. All rights reserved.

"""
Core types and utilities for tile-level primitives.

This module provides:
1. Device functions (tile_layout, tile_ptr, offset_ptr) for tile memory layout
2. Aggregate classes (Tile, TileView, TensorView, AllReduceConfig) for OOP-style tile operations
3. Common utilities (chiplet_transform_chunked, compute_tile_indices, compute_tile_offsets)
"""

import triton
import triton.language as tl
from triton.language.core import _aggregate as aggregate


# === Device functions ===


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


# === Aggregate classes ===


@aggregate
class TileView:
    """
    TileView storing BOTH runtime coordinates AND compile-time block sizes.

    This class uses the @constexpr_function pattern:
    - Stores runtime coordinates (pid_m, pid_n) as tl.tensor (computed from tl.program_id)
    - Stores compile-time block sizes (block_m, block_n) as tl.constexpr
    - Constructor is decorated with @constexpr_function to execute at compile-time

    Example usage:
        pid = tl.program_id(0)
        pid_m = pid // num_tiles_n
        pid_n = pid % num_tiles_n
        tile = TileView(pid_m, pid_n, BLOCK_M, BLOCK_N)
        rm, rn, mask = tile.layout(M, N)
    """

    pid_m: tl.tensor
    pid_n: tl.tensor
    block_m: tl.constexpr
    block_n: tl.constexpr

    @triton.constexpr_function
    def __init__(self, pid_m, pid_n, block_m, block_n):
        self.pid_m = pid_m
        self.pid_n = pid_n
        self.block_m = tl.constexpr(block_m)
        self.block_n = tl.constexpr(block_n)

    @triton.jit
    def layout(self, M, N):
        """Compute memory layout using stored coordinates."""
        return tile_layout(self.pid_m, self.pid_n, M, N, self.block_m, self.block_n)

    @triton.jit
    def indices(self):
        """Compute base row and column indices for this tile (without bounds checking)."""
        rm = self.pid_m * self.block_m + tl.arange(0, self.block_m)
        rn = self.pid_n * self.block_n + tl.arange(0, self.block_n)
        return rm, rn


@aggregate
class Tile:
    """
    Tile with embedded data for computed results.

    Extends TileView concept to include a data field for storing computed tile data
    (e.g., GEMM results in registers).

    Example usage:
        c = tl.dot(a, b)
        tile = Tile(pid_m, pid_n, BLOCK_M, BLOCK_N, c)
        ctx.all_reduce_atomic(tile, dst_view)
    """

    pid_m: tl.tensor
    pid_n: tl.tensor
    block_m: tl.constexpr
    block_n: tl.constexpr
    data: tl.tensor

    @triton.constexpr_function
    def __init__(self, pid_m, pid_n, block_m, block_n, data):
        self.pid_m = pid_m
        self.pid_n = pid_n
        self.block_m = tl.constexpr(block_m)
        self.block_n = tl.constexpr(block_n)
        self.data = data

    @triton.jit
    def layout(self, M, N):
        """Compute memory layout using stored coordinates."""
        return tile_layout(self.pid_m, self.pid_n, M, N, self.block_m, self.block_n)

    @triton.jit
    def indices(self):
        """Compute base row and column indices for this tile (without bounds checking)."""
        rm = self.pid_m * self.block_m + tl.arange(0, self.block_m)
        rn = self.pid_n * self.block_n + tl.arange(0, self.block_n)
        return rm, rn


@triton.jit
def make_tensor_view(ptr, M, N, stride_m, stride_n):
    """
    Factory function to create a TensorView inside a JIT context.

    This wrapper is needed because @triton.constexpr_function constructors
    require a JIT context for proper semantic handling. It also converts
    int/constexpr values to tensors using the +0 trick.

    Args:
        ptr: Pointer to tensor data
        M: Number of rows
        N: Number of columns
        stride_m: Stride in M dimension
        stride_n: Stride in N dimension

    Returns:
        TensorView instance
    """
    M_t = M + 0
    N_t = N + 0
    stride_m_t = stride_m + 0
    stride_n_t = stride_n + 0
    return TensorView(ptr, M_t, N_t, stride_m_t, stride_n_t)


@aggregate
class TensorView:
    """
    TensorView storing pointer and tensor metadata.

    Example usage:
        view = make_tensor_view(ptr, M, N, stride_m, stride_n)
        tile_ptr, mask = view.tile_ptr(tile)
    """

    ptr: tl.tensor
    M: tl.tensor
    N: tl.tensor
    stride_m: tl.tensor
    stride_n: tl.tensor

    @triton.constexpr_function
    def __init__(self, ptr, M, N, stride_m, stride_n):
        self.ptr = ptr
        self.M = M
        self.N = N
        self.stride_m = stride_m
        self.stride_n = stride_n

    @triton.jit
    def tile_ptr(self, tile: Tile):
        """Compute tile pointer and mask using stored dimensions/strides."""
        return tile_ptr(
            self.ptr, self.M, self.N, self.stride_m, self.stride_n, tile.pid_m, tile.pid_n, tile.block_m, tile.block_n
        )

    @triton.jit
    def tile_ptr_from_indices(self, rm, rn, block_m: tl.constexpr, block_n: tl.constexpr):
        """Compute tile pointer and mask from custom row/column indices."""
        rm = tl.max_contiguous(tl.multiple_of(rm, block_m), block_m)
        rn = tl.max_contiguous(tl.multiple_of(rn, block_n), block_n)
        mask = (rm[:, None] < self.M) & (rn[None, :] < self.N)
        offset = rm[:, None] * self.stride_m + rn[None, :] * self.stride_n
        tile_ptr = self.ptr + offset
        tile_ptr = tl.multiple_of(tile_ptr, (block_m, block_n))
        return tile_ptr, mask

    @triton.jit
    def offset_tile_ptr(self, tile: Tile, offset_m=0, offset_n=0, src_mask=None):
        """Compute tile pointer with row/column offsets applied."""
        rm, rn = tile.indices()
        rm_offset = rm + offset_m
        rn_offset = rn + offset_n
        tile_ptr, dst_mask = self.tile_ptr_from_indices(rm_offset, rn_offset, tile.block_m, tile.block_n)
        if src_mask is not None:
            mask = src_mask & dst_mask
        else:
            mask = dst_mask
        return tile_ptr, mask

    @triton.jit
    def offset(self, offset_m=0, offset_n=0):
        """Create a new view with pointer offset applied."""
        new_ptr = offset_ptr(self.ptr, self.stride_m, self.stride_n, offset_m, offset_n)
        return TensorView(new_ptr, self.M, self.N, self.stride_m, self.stride_n)

    @triton.jit
    def number_of_tiles(self, block_m, block_n):
        """Compute the total number of tiles needed to cover the tensor."""
        num_tiles_m = tl.cdiv(self.M, block_m)
        num_tiles_n = tl.cdiv(self.N, block_n)
        return num_tiles_m * num_tiles_n


@aggregate
class AllReduceConfig:
    """
    Device-side configuration for all_reduce collective.

    Fields:
        variant_code: Integer code for algorithm variant (0=atomic, 1=ring, 2=one_shot, 3=two_shot, 4=spinlock)
        locks_ptr: Pointer to locks array (ALWAYS required - pass dummy tensor if variant doesn't need locks)
    """

    variant_code: tl.constexpr
    locks_ptr: tl.tensor

    @triton.constexpr_function
    def __init__(self, variant_code, locks_ptr):
        self.variant_code = tl.constexpr(variant_code)
        self.locks_ptr = locks_ptr


# === Common utilities ===


@triton.jit()
def chiplet_transform_chunked(pid, num_workgroups: tl.constexpr, num_xcds: tl.constexpr, chunk_size: tl.constexpr):
    """Transform program ID to distribute work across XCDs in a chunked pattern."""
    if pid > (num_workgroups // (num_xcds * chunk_size)) * (num_xcds * chunk_size):
        return pid

    local_pid = pid // num_xcds
    chunk_idx = local_pid // chunk_size
    pos_in_chunk = local_pid % chunk_size

    xcd = pid % num_xcds
    new_pid = chunk_idx * num_xcds * chunk_size + xcd * chunk_size + pos_in_chunk
    return new_pid


@triton.jit()
def compute_tile_indices(
    pid_m,
    pid_n,
    M,
    N,
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
):
    """Compute row and column indices for a tile given pid_m and pid_n."""
    rm_base = pid_m * BLOCK_SIZE_M
    rn_base = pid_n * BLOCK_SIZE_N
    rm = rm_base + tl.arange(0, BLOCK_SIZE_M)
    rn = rn_base + tl.arange(0, BLOCK_SIZE_N)
    rm = tl.max_contiguous(tl.multiple_of(rm, BLOCK_SIZE_M), BLOCK_SIZE_M)
    rn = tl.max_contiguous(tl.multiple_of(rn, BLOCK_SIZE_N), BLOCK_SIZE_N)
    mask = (rm[:, None] < M) & (rn[None, :] < N)
    return rm, rn, mask


@triton.jit()
def compute_tile_offsets(
    rm,
    rn,
    stride_in_m,
    stride_in_n,
    stride_out_m,
    stride_out_n,
):
    """Compute input and output offsets for a tile given row/column indices and strides."""
    input_offset = rm[:, None] * stride_in_m + rn[None, :] * stride_in_n
    output_offset = rm[:, None] * stride_out_m + rn[None, :] * stride_out_n
    return input_offset, output_offset
