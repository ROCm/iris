# SPDX-License-Identifier: MIT
# Copyright (c) 2025 Advanced Micro Devices, Inc. All rights reserved.

"""
All-Gather + GEMM primitive combining iris.x all-gather with tritonBLAS GEMM stages.

This module provides a fused All-Gather + GEMM operation that first gathers sharded data
from all ranks and then computes matrix multiplication, useful for tensor-parallel workloads.
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

from .all_gather import all_gather
from .core import Tile, TensorView, DeviceContext


@triton.jit()
def all_gather_gemm(
    A_sharded,
    B,
    C,
    A_gathered,
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
    stride_ag_m: tl.constexpr,
    stride_ag_n: tl.constexpr,
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
    Fused All-Gather + GEMM kernel.

    First gathers sharded matrix A from all ranks, then computes C = A_gathered @ B.
    This is useful for tensor-parallel workloads where weights are sharded across ranks
    and need to be gathered before computation.

    The kernel processes tiles persistently and for each tile:
    1. Performs all-gather: gathers A_tile from all ranks to A_gathered
    2. Computes GEMM: C_tile = A_gathered_tile @ B_tile (+ bias)

    Args:
        A_sharded: Pointer to sharded input matrix A of shape (M_local, K) - local rank's shard
        B: Pointer to input matrix B of shape (K, N) - replicated across ranks
        C: Pointer to output matrix C of shape (M, N)
        A_gathered: Pointer to gathered matrix A of shape (M, K) - temporary buffer for gathered data
        bias_ptr: Optional pointer to bias vector of shape (M,)
        M: Number of rows in gathered A and output C (M = world_size * M_local)
        N: Number of columns in B and C
        K: Number of columns in A and rows in B
        stride_am, stride_ak: Strides for sharded A tensor
        stride_bn, stride_bk: Strides for B tensor
        stride_cm, stride_cn: Strides for C tensor
        stride_ag_m, stride_ag_n: Strides for gathered A tensor
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
        tl.static_assert(False, "tritonBLAS is required for all_gather_gemm. Install it from https://github.com/ROCm/tritonBLAS")

    # Stride guards
    tl.assume(stride_am > 0)
    tl.assume(stride_ak > 0)
    tl.assume(stride_bn > 0)
    tl.assume(stride_bk > 0)
    tl.assume(stride_cm > 0)
    tl.assume(stride_cn > 0)
    tl.assume(stride_ag_m > 0)
    tl.assume(stride_ag_n > 0)

    # Determine accumulator dtype based on output type
    acc_dtype = tl.int32 if C.type.element_ty != tl.int8 else tl.float32

    # Use chiplet-aware PID mapping if NUM_XCDS > 1
    USE_CHIPLET_PID = NUM_XCDS != 1

    # K_local is the local shard size (K = world_size * K_local)
    # A_sharded has shape (M, K_local), A_gathered has shape (M, K)
    K_local = K // world_size

    # Perform column-wise all-gather on A_sharded using the generalized all_gather primitive
    # A_sharded: (M, K_local) per rank -> A_gathered: (M, K) where K = world_size * K_local
    # Each rank's data goes to columns [cur_rank * K_local : (cur_rank+1) * K_local]
    num_tiles_m_gather = tl.cdiv(M, BLOCK_SIZE_M)
    num_tiles_k_gather = tl.cdiv(K_local, BLOCK_SIZE_K)  # Use BLOCK_SIZE_K for K dimension
    total_gather_tiles = num_tiles_m_gather * num_tiles_k_gather

    pid_base = tl.program_id(0)
    for gather_tile_id in range(pid_base, total_gather_tiles, NUM_SMS):
        gather_pid_m = gather_tile_id // num_tiles_k_gather
        gather_pid_k = gather_tile_id % num_tiles_k_gather

        # Call all_gather with gather_dim=1 for column-wise gathering using OOP API
        tile = Tile(gather_pid_m, gather_pid_k, BLOCK_SIZE_M, BLOCK_SIZE_K)
        src_view = TensorView(A_sharded, M, K_local, stride_am, stride_ak)
        dst_view = TensorView(A_gathered, M, K, stride_ag_m, stride_ag_n)
        ctx = DeviceContext(cur_rank, world_size, heap_bases)

        all_gather(tile, src_view, dst_view, 1, ctx)  # gather_dim=1 for columns

    # Synchronization barrier to ensure all-gather completes before GEMM
    tl.debug_barrier()

    # Compute Global Grid information once (for output C dimensions)
    pid, num_pid_m, num_pid_n, total_tiles = grid_setup(
        M, N, K,  # Problem Dimensions (using full K for gathered A)
        BLOCK_SIZE_M, BLOCK_SIZE_N,  # Tile Dimensions
        NUM_SMS, NUM_XCDS, CHUNK_SIZE,  # Hardware Info
        USE_CHIPLET_PID,  # Enable chiplet swizzle
    )

    # Persistent loop: process multiple tiles per workgroup
    for tile_id in range(pid, total_tiles, NUM_SMS):
        # Compute tile coordinates for output C
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

        # Compute GEMM using gathered A
        # Now compute GEMM: C = A_gathered @ B
        # Use gemm_loop with gathered A
        acc = gemm_loop(
            A_gathered,  # Use gathered A (shape M x K)
            B,
            row_indices,
            col_indices,
            acc,
            K,  # Full K dimension
            stride_ag_m,
            stride_ag_n,  # stride_ag_n should match stride_ak conceptually
            stride_bn,
            stride_bk,
            BLOCK_SIZE_K,
            CACHE_MODIFIER_A,
            CACHE_MODIFIER_B,
            False,  # QUANTIZED
            ALLOW_TF32,
            EVEN_K,
        )

        # Add bias if provided
        if BIAS:
            bias_vector = tl.load(
                bias_ptr + row_indices * stride_bias, mask=row_indices < M, other=0.0
            )
            acc = add_vector(acc, bias_vector[:, None], QUANTIZED=False)

        # Convert to output dtype
        result = convert_dtype(acc, C.type.element_ty)

        # Store result
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

