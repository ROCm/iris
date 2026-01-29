# SPDX-License-Identifier: MIT
# Copyright (c) 2025 Advanced Micro Devices, Inc. All rights reserved.

"""
GEMM + All-Reduce primitive combining tritonBLAS GEMM stages with iris.x all-reduce.

This module provides a fused GEMM + All-Reduce operation that computes matrix multiplication
and then reduces results across all ranks, useful for data-parallel distributed training.
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

from .core import Tile, TensorView, DeviceContext, AllReduceConfig


@triton.jit()
def gemm_all_reduce(
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
    Fused GEMM + All-Reduce kernel.

    Computes C = A @ B (with optional bias) and then performs all-reduce on the result.
    This is useful for data-parallel distributed training where each rank computes
    a partial result over a subset of data, and then reduces across all ranks.

    The kernel processes tiles persistently and for each output tile:
    1. Computes GEMM: C_tile = A_tile @ B_tile (+ bias)
    2. Performs all-reduce: reduces C_tile across all ranks using atomic operations

    Args:
        A: Pointer to input matrix A of shape (M, K) - local rank's data
        B: Pointer to input matrix B of shape (K, N) - replicated across ranks
        C: Pointer to output matrix C of shape (M, N) - will contain reduced result
        bias_ptr: Optional pointer to bias vector of shape (M,)
        M: Number of rows in A and C
        N: Number of columns in B and C
        K: Number of columns in A and rows in B
        stride_am, stride_ak: Strides for A tensor
        stride_bn, stride_bk: Strides for B tensor
        stride_cm, stride_cn: Strides for C tensor
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
            False, "tritonBLAS is required for gemm_all_reduce. Install it from https://github.com/ROCm/tritonBLAS"
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

        # Perform all-reduce on the computed tile
        # Compute pid_m and pid_n for the all_reduce call
        pid_m = output_coord_m
        pid_n = output_coord_n

        # Store local GEMM result to C first
        # Each rank stores its computed result, then all-reduce gathers and sums them
        store(
            C,
            result,
            row_indices,
            col_indices,
            M,
            N,
            stride_cm,
            stride_cn,
        )

        # Perform all-reduce using one-shot approach with ctx API
        # all_reduce reads from all ranks' C (which now contains GEMM results)
        # and writes the summed result back to C
        tile = Tile(pid_m, pid_n, BLOCK_SIZE_M, BLOCK_SIZE_N)
        src_view = TensorView(C, M, N, stride_cm, stride_cn)
        dst_view = TensorView(C, M, N, stride_cm, stride_cn)
        ctx = DeviceContext(cur_rank, world_size, heap_bases)
        config = AllReduceConfig("one_shot")

        ctx.all_reduce(tile, src_view, dst_view, config=config)
