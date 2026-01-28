# SPDX-License-Identifier: MIT
# Copyright (c) 2025 Advanced Micro Devices, Inc. All rights reserved.

"""
GEMM + Reduce-Scatter primitive combining tritonBLAS GEMM stages with iris.x reduce-scatter.

This module provides a fused GEMM + Reduce-Scatter operation that computes matrix multiplication
and then reduces and scatters results to assigned ranks, useful for column-parallel workloads.
"""

import triton
import triton.language as tl
import torch

try:
    from tritonblas.kernels.stages.indexing import grid_setup, idx2coord
    from tritonblas.kernels.stages.algorithms import gemm_loop
    from tritonblas.kernels.stages.algorithms.binary import add_vector
    from tritonblas.kernels.stages.algorithms.unary import convert_dtype
    from tritonblas.kernels.stages.memory import store

    TRITONBLAS_AVAILABLE = True
except ImportError:
    TRITONBLAS_AVAILABLE = False

from .reduce_scatter import reduce_scatter


@triton.jit()
def gemm_reduce_scatter(
    A,
    B,
    C_full,
    C,
    bias_ptr,
    M,
    N,
    K,
    stride_am,
    stride_ak,
    stride_bn,
    stride_bk,
    stride_cm_full,
    stride_cn_full,
    stride_cm,
    stride_cn,
    stride_bias,
    heap_bases: tl.tensor,
    cur_rank: tl.constexpr,
    world_size: tl.constexpr,
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr,
    GROUP_SIZE_M: tl.constexpr,
    NUM_SMS: tl.constexpr,
    NUM_XCDS: tl.constexpr,
    CHUNK_SIZE: tl.constexpr,
    BIAS: tl.constexpr,
    EVEN_K: tl.constexpr,
    CACHE_MODIFIER_A: tl.constexpr,
    CACHE_MODIFIER_B: tl.constexpr,
    ALLOW_TF32: tl.constexpr,
):
    """
    Fused GEMM + Reduce-Scatter kernel.

    Computes C_full = A @ B (with optional bias) and then performs reduce-scatter on the result.
    This is useful for column-parallel workloads where each rank computes over full rows
    but only keeps a subset of columns after reduction.

    The kernel processes tiles persistently and for each tile:
    1. Computes GEMM: C_full_tile = A_tile @ B_tile (+ bias)
    2. Performs reduce-scatter: reduces C_full_tile from all ranks and stores only assigned portion to C

    Args:
        A: Pointer to input matrix A of shape (M, K) - replicated across ranks
        B: Pointer to input matrix B of shape (K, N) - replicated across ranks
        C_full: Pointer to full output buffer of shape (M, N) - temporary storage for full GEMM result
        C: Pointer to output matrix C of shape (M, N_local) - will contain reduced result for this rank
        bias_ptr: Optional pointer to bias vector of shape (M,)
        M: Number of rows in A and C
        N: Number of columns in B (full)
        K: Number of columns in A and rows in B
        stride_am, stride_ak: Strides for A tensor
        stride_bn, stride_bk: Strides for B tensor
        stride_cm_full, stride_cn_full: Strides for C_full tensor (full result buffer)
        stride_cm, stride_cn: Strides for C tensor (output after reduce-scatter)
        stride_bias: Stride for bias vector
        heap_bases: Heap base pointers for all ranks
        cur_rank: Current rank
        world_size: Total number of ranks
        BLOCK_SIZE_M: Block size for M dimension
        BLOCK_SIZE_N: Block size for N dimension
        BLOCK_SIZE_K: Block size for K dimension
        GROUP_SIZE_M: Group size for M dimension tiling
        NUM_SMS: Number of SMs to use
        NUM_XCDS: Number of XCDs
        CHUNK_SIZE: Chunk size for chiplet transform
        BIAS: Whether to add bias (1 for True, 0 for False)
        EVEN_K: Whether K is evenly divisible by BLOCK_SIZE_K
        CACHE_MODIFIER_A: Cache modifier for A (e.g., ".ca" for cached)
        CACHE_MODIFIER_B: Cache modifier for B
        ALLOW_TF32: Whether to allow TF32 precision
    """
    if not TRITONBLAS_AVAILABLE:
        tl.static_assert(False, "tritonBLAS is required for gemm_reduce_scatter. Install it from https://github.com/ROCm/tritonBLAS")

    # Stride guards
    tl.assume(stride_am > 0)
    tl.assume(stride_ak > 0)
    tl.assume(stride_bn > 0)
    tl.assume(stride_bk > 0)
    tl.assume(stride_cm_full > 0)
    tl.assume(stride_cn_full > 0)
    tl.assume(stride_cm > 0)
    tl.assume(stride_cn > 0)

    # Determine accumulator dtype based on output type
    acc_dtype = tl.int32 if C.type.element_ty != tl.int8 else tl.float32

    # Use chiplet-aware PID mapping if NUM_XCDS > 1
    USE_CHIPLET_PID = NUM_XCDS != 1

    # N_local is the local output size (N = world_size * N_local)
    N_local = N // world_size

    # Compute Global Grid information once (for full N dimension)
    pid, num_pid_m, num_pid_n, total_tiles = grid_setup(
        M, N, K,  # Problem Dimensions (using full N)
        BLOCK_SIZE_M, BLOCK_SIZE_N,  # Tile Dimensions
        NUM_SMS, NUM_XCDS, CHUNK_SIZE,  # Hardware Info
        USE_CHIPLET_PID,  # Enable chiplet swizzle
    )

    # Persistent loop: process multiple tiles per workgroup
    for tile_id in range(pid, total_tiles, NUM_SMS):
        # Compute tile coordinates for full output
        output_coord_m, output_coord_n, row_indices, col_indices, acc = idx2coord(
            tile_id,
            num_pid_m,
            num_pid_n,
            M,
            N,  # Full N dimension
            BLOCK_SIZE_M,
            BLOCK_SIZE_N,
            GROUP_SIZE_M,
            acc_dtype,
        )

        # Compute matrix multiplication over full K dimension
        acc = gemm_loop(
            A,
            B,
            row_indices,
            col_indices,  # Full N columns
            acc,
            K,
            stride_am,
            stride_ak,
            stride_bn,
            stride_bk,
            BLOCK_SIZE_K,
            CACHE_MODIFIER_A,
            CACHE_MODIFIER_B,
            False,  # QUANTIZED
            ALLOW_TF32,
            EVEN_K,
        )

        # Add bias and convert to output dtype
        if BIAS:
            bias_vector = tl.load(
                bias_ptr + row_indices * stride_bias, mask=row_indices < M, other=0.0
            )
            acc = add_vector(acc, bias_vector[:, None], QUANTIZED=False)

        # Convert to output dtype
        result = convert_dtype(acc, C_full.type.element_ty)

        # Store full result to C_full, then reduce-scatter
        # Store the computed result (full N columns) to C_full buffer

        store(
            C_full,
            result,
            row_indices,
            col_indices,  # Full N columns
            M,
            N,
            stride_cm_full,
            stride_cn_full,
        )

        # Perform reduce-scatter on the computed tile
        # reduce_scatter will read from C_full (full result) on all ranks
        # and write reduced result to C (local portion) only on the assigned rank
        # 
        # For reduce-scatter, tiles are assigned using striding:
        # rank 0 gets tiles 0, world_size, 2*world_size, ... in N dimension
        # rank 1 gets tiles 1, world_size+1, 2*world_size+1, ...
        # 
        # ALL ranks participate in the reduction, but only the assigned rank stores

        # Compute which rank owns this tile
        tile_rank = output_coord_n % world_size
        num_pid_n_local = (N_local + BLOCK_SIZE_N - 1) // BLOCK_SIZE_N
        local_pid_n = output_coord_n // world_size

        # Call reduce_scatter for all tiles
        # reduce_scatter will reduce from all ranks and store only if tile belongs to this rank
        # Note: reduce_scatter currently stores unconditionally, so we need to call it
        # only for tiles assigned to this rank. In a full implementation, reduce_scatter
        # would check tile ownership internally.
        
        if tile_rank == cur_rank and local_pid_n < num_pid_n_local:
            # This tile belongs to this rank, perform reduce-scatter
            reduce_scatter(
                C_full,  # input_ptr: full result (all ranks have this)
                C,  # output_ptr: local output (will contain reduced result)
                output_coord_m,
                local_pid_n,  # Local tile coordinate in N
                M,
                N,  # Full N for input
                stride_cm_full,
                stride_cn_full,  # input strides
                stride_cm,
                stride_cn,  # output strides
                heap_bases,
                cur_rank,
                world_size,
                BLOCK_SIZE_M,
                BLOCK_SIZE_N,
            )

