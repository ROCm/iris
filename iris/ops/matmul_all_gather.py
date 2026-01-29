# SPDX-License-Identifier: MIT
# Copyright (c) 2025 Advanced Micro Devices, Inc. All rights reserved.

"""
High-level API for fused matrix multiplication and all-gather.

This module provides a torch-like interface for GEMM+All-Gather operations,
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
def _gemm_all_gather_kernel(
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


def matmul_all_gather(
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
    Fused matrix multiplication and all-gather.

    Computes: output = all_gather(A @ B + bias) along M dimension

    Each rank computes its local portion of the GEMM result (M, N), then
    all-gathers along the M dimension to produce final output (M*world_size, N).

    Args:
        shmem: Iris shmem context
        output_tensor: Output tensor (M*world_size, N) - gathered result on all ranks
        A: Input matrix A (M, K) - each rank has different rows
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
        >>> M_local = 1024 // world_size
        >>> A = shmem.randn((M_local, 512), dtype=torch.float16)
        >>> B = shmem.randn((512, 2048), dtype=torch.float16)
        >>> output = shmem.zeros((1024, 2048), dtype=torch.float16)  # M_local * world_size
        >>> shmem.ops.matmul_all_gather(output, A, B)
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
            f"Incompatible matrix dimensions: A is ({M}, {K}), B is ({K_B}, {N}). Inner dimensions must match"
        )

    expected_M = M * world_size
    if output_tensor.shape != (expected_M, N):
        raise ValueError(
            f"Output tensor shape {output_tensor.shape} doesn't match expected ({expected_M}, {N}). "
            f"Output should be (M * world_size, N) = ({M} * {world_size}, {N})"
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
            raise ValueError(f"Bias shape {bias.shape} incompatible with local output shape ({M}, {N})")
        if bias.dtype != A.dtype:
            raise ValueError(f"Bias dtype {bias.dtype} doesn't match input dtype {A.dtype}")

    # Extract strides
    stride_am, stride_ak = A.stride()
    stride_bk, stride_bn = B.stride()
    stride_cm, stride_cn = output_tensor.stride()
    stride_bias = bias.stride(0) if has_bias else 0

    # Get rank info
    rank = shmem.get_rank()

    # Auto-detect num_sms
    if config.num_sms is None:
        config.num_sms = torch.cuda.get_device_properties(rank).multi_processor_count

    # Prepare workspace (no temporary buffers needed - output tensor serves as gather destination)
    if workspace is None:
        workspace = FusedWorkspace()

    workspace.operation = "matmul_all_gather"
    workspace.shape = (M, N, K)
    workspace.dtype = A.dtype
    workspace.world_size = world_size
    workspace.variant = ""
    workspace.prepared = True

    heap_bases = shmem.get_heap_bases()
    even_k = 1 if (K % config.block_size_k == 0) else 0

    # Launch kernel
    grid = (config.num_sms,)

    _gemm_all_gather_kernel[grid](
        A,
        B,
        output_tensor,
        bias if has_bias else None,
        M,
        N,
        K,
        stride_am,
        stride_ak,
        stride_bk,
        stride_bn,
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
