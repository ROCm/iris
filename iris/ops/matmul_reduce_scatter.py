# SPDX-License-Identifier: MIT
# Copyright (c) 2025 Advanced Micro Devices, Inc. All rights reserved.

"""
High-level API for fused matrix multiplication and reduce-scatter.

This module provides a torch-like interface for GEMM+Reduce-Scatter operations,
automatically inferring dimensions, strides, and hardware parameters.
"""

from typing import Optional
import torch
import triton
import triton.language as tl

from .config import FusedConfig
from .workspace import FusedWorkspace

from tritonblas.kernels.stages.indexing import grid_setup, idx2coord
from tritonblas.kernels.stages.algorithms import gemm_loop
from tritonblas.kernels.stages.algorithms.binary import add_vector
from tritonblas.kernels.stages.algorithms.unary import convert_dtype
from tritonblas.kernels.stages.memory import store
from iris.x.core import Tile, TensorView, DeviceContext


@triton.jit()
def _gemm_reduce_scatter_kernel(
    A,
    B,
    C_full,
    C,
    bias_ptr,
    M: tl.constexpr,
    N: tl.constexpr,
    K: tl.constexpr,
    stride_am: tl.constexpr,
    stride_ak: tl.constexpr,
    stride_bn: tl.constexpr,
    stride_bk: tl.constexpr,
    stride_cm_full: tl.constexpr,
    stride_cn_full: tl.constexpr,
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
    acc_dtype = tl.int32 if C.type.element_ty == tl.int8 else tl.float32

    # Use chiplet-aware PID mapping if NUM_XCDS > 1
    USE_CHIPLET_PID = NUM_XCDS != 1

    # N_local is the local output size (N = world_size * N_local)
    N_local = N // world_size

    # Compute Global Grid information once (for full N dimension)
    pid, num_pid_m, num_pid_n, total_tiles = grid_setup(
        M,
        N,
        K,  # Problem Dimensions (using full N)
        BLOCK_SIZE_M,
        BLOCK_SIZE_N,  # Tile Dimensions
        NUM_SMS,
        NUM_XCDS,
        CHUNK_SIZE,  # Hardware Info
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
            bias_vector = tl.load(bias_ptr + row_indices * stride_bias, mask=row_indices < M, other=0.0)
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
            # This tile belongs to this rank, perform reduce-scatter using ctx API
            tile = Tile(output_coord_m, local_pid_n, BLOCK_SIZE_M, BLOCK_SIZE_N)
            src_view = TensorView(C_full, M, N, stride_cm_full, stride_cn_full)
            dst_view = TensorView(C, M, N // world_size, stride_cm, stride_cn)
            ctx = DeviceContext(cur_rank, world_size, heap_bases)

            ctx.reduce_scatter(tile, src_view, dst_view)


def matmul_reduce_scatter_preamble(
    shmem,
    output_tensor: torch.Tensor,
    A: torch.Tensor,
    B: torch.Tensor,
    config: Optional[FusedConfig] = None,
    workspace: Optional[FusedWorkspace] = None,
) -> FusedWorkspace:
    """
    Allocate and reset temporary buffers for matmul_reduce_scatter.

    Args:
        shmem: Iris shmem context
        output_tensor: Output tensor (M, N_local) where N_local = N / world_size
        A: Input matrix A (M, K)
        B: Input matrix B (K, N)
        config: Optional FusedConfig
        workspace: Optional existing workspace

    Returns:
        FusedWorkspace instance ready for kernel launch.
    """
    if config is None:
        config = FusedConfig()

    M, K = A.shape[:2]
    N = B.shape[1]
    world_size = shmem.get_num_ranks()
    dtype = A.dtype

    if workspace is None:
        workspace = FusedWorkspace()

    workspace.operation = "matmul_reduce_scatter"
    workspace.shape = (M, N, K)
    workspace.dtype = dtype
    workspace.world_size = world_size
    workspace.variant = ""
    workspace.prepared = False

    # Allocate full buffer for intermediate GEMM result (M, N)
    if (
        workspace.full_buffer is None
        or workspace.full_buffer.shape != (M, N)
        or workspace.full_buffer.dtype != dtype
    ):
        workspace.full_buffer = shmem.zeros((M, N), dtype=dtype)
    else:
        workspace.full_buffer.zero_()

    shmem.barrier()
    workspace.prepared = True
    return workspace


def matmul_reduce_scatter(
    shmem,
    output_tensor: torch.Tensor,
    A: torch.Tensor,
    B: torch.Tensor,
    bias: Optional[torch.Tensor] = None,
    async_op: bool = False,
    config: Optional[FusedConfig] = None,
    workspace: Optional[FusedWorkspace] = None,
) -> FusedWorkspace:
    """
    Fused matrix multiplication and reduce-scatter.

    Computes: output = reduce_scatter(A @ B + bias) along N dimension

    Each rank computes the full GEMM result (M, N), then reduces across ranks
    and scatters along the N dimension so each rank keeps N/world_size columns.

    Args:
        shmem: Iris shmem context
        output_tensor: Output tensor (M, N_local) where N_local = N / world_size
        A: Input matrix A (M, K) - replicated across ranks
        B: Input matrix B (K, N) - replicated across ranks
        bias: Optional bias vector (M,) or (N,). Default: None.
        async_op: If False, performs barrier at end. Default: False.
        config: Optional FusedConfig for tuning
        workspace: Optional pre-allocated workspace

    Returns:
        workspace: Updated workspace object

    Raises:
        ValueError: If tensor shapes are incompatible.

    Example:
        >>> world_size = shmem.get_num_ranks()
        >>> N_local = 2048 // world_size
        >>> A = shmem.randn((1024, 512), dtype=torch.float16)
        >>> B = shmem.randn((512, 2048), dtype=torch.float16)
        >>> output = shmem.zeros((1024, N_local), dtype=torch.float16)
        >>> shmem.ops.matmul_reduce_scatter(output, A, B)
    """
    if config is None:
        config = FusedConfig()

    # Extract dimensions
    if A.ndim != 2 or B.ndim != 2:
        raise ValueError(f"A and B must be 2D tensors, got shapes {A.shape} and {B.shape}")

    M, K = A.shape
    K_B, N = B.shape
    world_size = shmem.get_num_ranks()

    if K != K_B:
        raise ValueError(
            f"Incompatible matrix dimensions: A is ({M}, {K}), B is ({K_B}, {N}). "
            f"Inner dimensions must match"
        )

    # Validate N is divisible by world_size
    if N % world_size != 0:
        raise ValueError(
            f"N dimension ({N}) must be divisible by world_size ({world_size}) for reduce-scatter. "
            f"Each rank will keep N/world_size = {N}/{world_size} columns"
        )

    N_local = N // world_size
    if output_tensor.shape != (M, N_local):
        raise ValueError(
            f"Output tensor shape {output_tensor.shape} doesn't match expected ({M}, {N_local}). "
            f"Output should be (M, N/world_size) = ({M}, {N}/{world_size})"
        )

    if A.dtype != B.dtype or A.dtype != output_tensor.dtype:
        raise ValueError(
            f"All tensors must have same dtype, got A:{A.dtype}, B:{B.dtype}, output:{output_tensor.dtype}"
        )

    # Validate bias
    has_bias = bias is not None
    if has_bias:
        if bias.ndim != 1:
            raise ValueError(f"Bias must be 1D tensor, got shape {bias.shape}")
        if bias.shape[0] not in (M, N):
            raise ValueError(f"Bias shape {bias.shape} incompatible with full output shape ({M}, {N})")
        if bias.dtype != A.dtype:
            raise ValueError(f"Bias dtype {bias.dtype} doesn't match input dtype {A.dtype}")

    # Get rank info
    rank = shmem.get_rank()

    # Auto-detect num_sms
    if config.num_sms is None:
        config.num_sms = torch.cuda.get_device_properties(rank).multi_processor_count

    # Prepare workspace
    needs_prepare = (
        workspace is None
        or not workspace.matches("matmul_reduce_scatter", (M, N, K), A.dtype, world_size, "")
    )

    if needs_prepare:
        workspace = matmul_reduce_scatter_preamble(shmem, output_tensor, A, B, config=config, workspace=workspace)

    C_full = workspace.full_buffer

    # Extract strides
    stride_am, stride_ak = A.stride()
    stride_bk, stride_bn = B.stride()
    stride_cm_full = C_full.stride(0)
    stride_cn_full = C_full.stride(1)
    stride_cm, stride_cn = output_tensor.stride()
    stride_bias = bias.stride(0) if has_bias else 0

    heap_bases = shmem.get_heap_bases()
    even_k = 1 if (K % config.block_size_k == 0) else 0

    # Launch kernel
    grid = (config.num_sms,)

    _gemm_reduce_scatter_kernel[grid](
        A,
        B,
        C_full,
        output_tensor,
        bias if has_bias else None,
        M,
        N,
        K,
        stride_am,
        stride_ak,
        stride_bk,
        stride_bn,
        stride_cm_full,
        stride_cn_full,
        stride_cm,
        stride_cn,
        stride_bias,
        heap_bases,
        rank,
        world_size,
        config.block_size_m,
        config.block_size_n,
        config.block_size_k,
        config.group_size_m,
        config.num_sms,
        config.num_xcds,
        config.chunk_size,
        1 if has_bias else 0,
        even_k,
        config.cache_modifier_a,
        config.cache_modifier_b,
        config.allow_tf32,
    )

    if workspace is not None:
        workspace.prepared = False

    if not async_op:
        shmem.barrier()

    return workspace
