# SPDX-License-Identifier: MIT
# Copyright (c) 2025 Advanced Micro Devices, Inc. All rights reserved.

"""
GEMM + All-Gather primitive combining tritonBLAS GEMM stages with iris.x all-gather.

This module provides a fused GEMM + All-Gather operation that computes matrix multiplication
and then gathers results from all ranks, useful for distributed training scenarios.
"""

import triton
import triton.language as tl

try:
    from tritonblas.kernels.stages.indexing import grid_setup, idx2coord
    from tritonblas.kernels.stages.algorithms import gemm_loop
    from tritonblas.kernels.stages.algorithms.binary import add_vector
    from tritonblas.kernels.stages.algorithms.unary import convert_dtype
    from tritonblas.kernels.stages.memory import store

    TRITONBLAS_AVAILABLE = True
except ImportError:
    TRITONBLAS_AVAILABLE = False

from .core import Tile, TensorView, DeviceContext


@triton.jit()
def gemm_all_gather(
    A,
    B,
    C,
    bias_ptr,
    M: tl.constexpr,
    N: tl.constexpr,
    K: tl.constexpr,
    stride_am: tl.constexpr,
    stride_ak: tl.constexpr,
    stride_bn: tl.constexpr,
    stride_bk: tl.constexpr,
    stride_cm: tl.constexpr,
    stride_cn: tl.constexpr,
    stride_bias: tl.constexpr,
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
    Fused GEMM + All-Gather kernel.

    Computes C = A @ B (with optional bias) and then performs all-gather on the result.
    This is useful for distributed training where each rank computes a partial result
    and then gathers the full result from all ranks.

    The kernel processes tiles persistently and for each output tile:
    1. Computes GEMM: C_tile = A_tile @ B_tile (+ bias)
    2. Performs all-gather: gathers C_tile from all ranks

    Args:
        A: Pointer to input matrix A of shape (M, K)
        B: Pointer to input matrix B of shape (K, N)
        C: Pointer to output matrix C of shape (world_size * M, N) after all-gather
        bias_ptr: Optional pointer to bias vector of shape (M,)
        M: Number of rows per rank in A (output will be world_size * M rows after all-gather)
        N: Number of columns in B and C
        K: Number of columns in A and rows in B
        stride_am, stride_ak: Strides for A tensor
        stride_bn, stride_bk: Strides for B tensor
        stride_cm, stride_cn: Strides for C tensor (output after all-gather)
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
        tl.static_assert(
            False, "tritonBLAS is required for gemm_all_gather. Install it from https://github.com/ROCm/tritonBLAS"
        )

    # Stride guards
    tl.assume(stride_am > 0)
    tl.assume(stride_ak > 0)
    tl.assume(stride_bn > 0)
    tl.assume(stride_bk > 0)
    tl.assume(stride_cm > 0)
    tl.assume(stride_cn > 0)

    # Determine accumulator dtype based on output type
    acc_dtype = tl.int32 if C.type.element_ty == tl.int8 else tl.float32

    # Use chiplet-aware PID mapping if NUM_XCDS > 1
    USE_CHIPLET_PID = NUM_XCDS != 1

    # Compute Global Grid information once
    pid, num_pid_m, num_pid_n, total_tiles = grid_setup(
        M,
        N,
        K,  # Problem Dimensions
        BLOCK_SIZE_M,
        BLOCK_SIZE_N,  # Tile Dimensions
        NUM_SMS,
        NUM_XCDS,
        CHUNK_SIZE,  # Hardware Info
        USE_CHIPLET_PID,  # Enable chiplet swizzle
    )

    # Persistent loop: process multiple tiles per workgroup
    for tile_id in range(pid, total_tiles, NUM_SMS):
        # Compute tile coordinates and initialize accumulator
        output_coord_m, output_coord_n, row_indices, col_indices, acc = idx2coord(
            tile_id,
            num_pid_m,
            num_pid_n,
            M,
            N,
            BLOCK_SIZE_M,
            BLOCK_SIZE_N,
            GROUP_SIZE_M,
            acc_dtype,
        )

        # Compute matrix multiplication over full K dimension
        acc = gemm_loop(
            A,
            B,  # Pointers to A and B tensors
            row_indices,
            col_indices,  # The row and column indices to process
            acc,
            K,  # Accumulator and problem K dimension
            stride_am,
            stride_ak,  # A tensor strides
            stride_bn,
            stride_bk,  # B tensor strides
            BLOCK_SIZE_K,  # Block Size in K dimension
            CACHE_MODIFIER_A,
            CACHE_MODIFIER_B,  # Cache modifiers to control locality
            False,  # QUANTIZED: False for fp16/bf16/fp32
            ALLOW_TF32,
            EVEN_K,  # Extra compile time constants
        )

        # Add bias and convert to output dtype
        # Add bias if provided
        if BIAS:
            bias_vector = tl.load(
                bias_ptr + row_indices * stride_bias, mask=row_indices < M, other=0.0
            )  # Load Bias vector
            acc = add_vector(acc, bias_vector, QUANTIZED=False)  # Add bias vector to output accumulator

        # Convert to output dtype
        result = convert_dtype(acc, C.type.element_ty)  # Convert accumulator to output datatype

        # Store GEMM result to local portion of output buffer
        # Each rank stores its computed tile to C[cur_rank * M : (cur_rank + 1) * M, :]
        # Compute global output row indices for this rank's section
        # row_indices are local (0 to M-1), we offset by cur_rank * M
        output_row_indices = row_indices + cur_rank * M
        output_col_indices = col_indices

        # Store local GEMM result to this rank's section of output
        # C has shape (world_size * M, N), we write to rows [cur_rank * M : (cur_rank + 1) * M]
        store(
            C,
            result,  # Output tensor pointer and output accumulator
            output_row_indices,
            output_col_indices,  # Precomputed offsets
            world_size * M,
            N,  # M and N dimension for masking OOB writes
            stride_cm,
            stride_cn,  # Stride of output dimensions
        )

        # Perform all-gather on the computed tile
        # Now we need to gather this tile from all ranks
        # all_gather will read each rank's portion from C and write it to
        # the appropriate location in C on all ranks
        #
        # all_gather expects:
        # - input_ptr: source (local rank's data) at shape (M, N)
        # - output_ptr: destination (full result) at shape (world_size * M, N)
        # - pid_m, pid_n: tile coordinates in the local M dimension
        #
        # Since we've stored to C already, we use C as both input and output
        # The all_gather function handles the offset calculation internally

        # Compute pid_m and pid_n for the all_gather call
        # output_coord_m is the local tile coordinate in M dimension (0 to num_pid_m-1)
        # output_coord_n is the tile coordinate in N dimension (0 to num_pid_n-1)
        pid_m = output_coord_m
        pid_n = output_coord_n

        # Call all_gather to gather this tile from all ranks using ctx API
        # all_gather reads from C at [cur_rank * M : (cur_rank + 1) * M, :] (input)
        # and writes to C at [cur_rank * M : (cur_rank + 1) * M, :] on all ranks (output)
        tile = Tile(pid_m, pid_n, BLOCK_SIZE_M, BLOCK_SIZE_N)
        src_view = TensorView(C, M, N, stride_cm, stride_cn)
        dst_view = TensorView(C, M * world_size, N, stride_cm, stride_cn)
        ctx = DeviceContext(cur_rank, world_size, heap_bases)

        ctx.all_gather(tile, src_view, dst_view, 0)  # gather_dim=0 for rows
