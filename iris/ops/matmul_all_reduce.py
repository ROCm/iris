# SPDX-License-Identifier: MIT
# Copyright (c) 2025 Advanced Micro Devices, Inc. All rights reserved.

"""
High-level API for fused matrix multiplication and all-reduce.

This module provides a torch-like interface for GEMM+All-Reduce operations,
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
from iris.x.core import Tile, TensorView, DeviceContext, AllReduceConfig


@triton.jit()
def _gemm_all_reduce_kernel(
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


def matmul_all_reduce_preamble(
    shmem,
    output_tensor: torch.Tensor,
    A: torch.Tensor,
    B: torch.Tensor,
    config: Optional[FusedConfig] = None,
    workspace: Optional[FusedWorkspace] = None,
) -> FusedWorkspace:
    """
    Allocate and reset temporary buffers for matmul_all_reduce.

    This function prepares the workspace for the fused operation by allocating
    any necessary temporary buffers based on the selected algorithm variant.

    Args:
        shmem: Iris shmem context
        output_tensor: Output tensor (M, N)
        A: Input matrix A (M, K)
        B: Input matrix B (K, N)
        config: Optional FusedConfig. If None, uses defaults.
        workspace: Optional existing workspace to reuse. If None, creates new one.

    Returns:
        FusedWorkspace instance ready for kernel launch.
    """
    if config is None:
        config = FusedConfig()

    M, K = A.shape[:2]
    N = B.shape[1]
    dtype = A.dtype
    world_size = shmem.get_num_ranks()
    variant = config.all_reduce_variant

    # Validate config
    config.validate(world_size=world_size)

    if workspace is None:
        workspace = FusedWorkspace()

    workspace.operation = "matmul_all_reduce"
    workspace.shape = (M, N, K)
    workspace.dtype = dtype
    workspace.world_size = world_size
    workspace.variant = variant
    workspace.prepared = False

    # Allocate temporary buffers based on variant
    if variant in ("atomic", "one_shot"):
        # Zero output tensor for atomic accumulation
        output_tensor.zero_()
        shmem.barrier()

    elif variant == "ring":
        # Allocate ring buffer and flags
        num_pid_m = (M + config.block_size_m - 1) // config.block_size_m
        num_pid_n = (N + config.block_size_n - 1) // config.block_size_n
        total_tiles = num_pid_m * num_pid_n
        flags_per_tile = 1
        total_flags = total_tiles * flags_per_tile

        if (
            workspace.ring_buffer is None
            or workspace.ring_buffer.shape != (M, N)
            or workspace.ring_buffer.dtype != dtype
        ):
            workspace.ring_buffer = shmem.zeros((M, N), dtype=dtype)
        else:
            workspace.ring_buffer.zero_()

        if workspace.flags is None or workspace.flags.numel() != total_flags:
            workspace.flags = shmem.zeros((total_flags,), dtype=torch.int32)
        else:
            workspace.flags.zero_()

        output_tensor.zero_()
        shmem.barrier()

    elif variant == "spinlock":
        # Allocate locks
        num_pid_m = (M + config.block_size_m - 1) // config.block_size_m
        num_pid_n = (N + config.block_size_n - 1) // config.block_size_n
        total_tiles = num_pid_m * num_pid_n

        if workspace.locks is None or workspace.locks.numel() != total_tiles:
            workspace.locks = shmem.zeros((total_tiles,), dtype=torch.int32)
        else:
            workspace.locks.zero_()

    elif variant == "two_shot":
        # No temporary buffers needed
        pass

    workspace.prepared = True
    return workspace


def matmul_all_reduce(
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
    Fused matrix multiplication and all-reduce.

    Computes: output = all_reduce(A @ B + bias)

    This high-level API automatically infers dimensions, strides, and hardware
    parameters, providing a torch-like interface for the fused operation.

    Args:
        shmem: Iris shmem context
        output_tensor: Output tensor (M, N) - will contain reduced result on all ranks
        A: Input matrix A (M, K) - each rank has different data (data-parallel)
        B: Input matrix B (K, N) - replicated across ranks
        bias: Optional bias vector. Can be (M,) or (N,). Default: None.
        async_op: If False, performs barrier at end. Default: False.
        config: Optional FusedConfig for tuning. If None, uses defaults.
        workspace: Optional pre-allocated workspace. If None, creates new one.

    Returns:
        workspace: Updated workspace object (can be reused for subsequent calls)

    Raises:
        ValueError: If tensor shapes are incompatible or config is invalid.

    Example:
        >>> # Basic usage
        >>> A = shmem.randn((1024, 512), dtype=torch.float16)
        >>> B = shmem.randn((512, 2048), dtype=torch.float16)
        >>> output = shmem.zeros((1024, 2048), dtype=torch.float16)
        >>> shmem.ops.matmul_all_reduce(output, A, B)
        >>>
        >>> # With bias and custom config
        >>> bias = shmem.randn((1024,), dtype=torch.float16)
        >>> config = FusedConfig(block_size_m=128, all_reduce_variant="ring")
        >>> shmem.ops.matmul_all_reduce(output, A, B, bias=bias, config=config)
        >>>
        >>> # Reuse workspace for multiple calls
        >>> workspace = None
        >>> for _ in range(10):
        >>>     workspace = shmem.ops.matmul_all_reduce(
        >>>         output, A, B, workspace=workspace
        >>>     )
    """
    if config is None:
        config = FusedConfig()

    # Extract dimensions
    if A.ndim != 2 or B.ndim != 2:
        raise ValueError(f"A and B must be 2D tensors, got shapes {A.shape} and {B.shape}")

    M, K = A.shape
    K_B, N = B.shape

    if K != K_B:
        raise ValueError(
            f"Incompatible matrix dimensions: A is ({M}, {K}), B is ({K_B}, {N}). "
            f"Inner dimensions must match (K={K} != K_B={K_B})"
        )

    if output_tensor.shape != (M, N):
        raise ValueError(
            f"Output tensor shape {output_tensor.shape} doesn't match expected ({M}, {N})"
        )

    if A.dtype != B.dtype or A.dtype != output_tensor.dtype:
        raise ValueError(
            f"All tensors must have same dtype, got A:{A.dtype}, B:{B.dtype}, output:{output_tensor.dtype}"
        )

    # Validate bias if provided
    has_bias = bias is not None
    if has_bias:
        if bias.ndim != 1:
            raise ValueError(f"Bias must be 1D tensor, got shape {bias.shape}")
        if bias.shape[0] not in (M, N):
            raise ValueError(
                f"Bias shape {bias.shape} incompatible with output shape ({M}, {N}). "
                f"Bias must be size {M} or {N}"
            )
        if bias.dtype != A.dtype:
            raise ValueError(f"Bias dtype {bias.dtype} doesn't match input dtype {A.dtype}")

    # Extract strides
    stride_am, stride_ak = A.stride()
    stride_bk, stride_bn = B.stride()
    stride_cm, stride_cn = output_tensor.stride()
    stride_bias = bias.stride(0) if has_bias else 0

    # Get rank info
    rank = shmem.get_rank()
    world_size = shmem.get_num_ranks()

    # Auto-detect num_sms if not specified
    if config.num_sms is None:
        config.num_sms = torch.cuda.get_device_properties(rank).multi_processor_count

    # Prepare workspace if needed
    needs_prepare = (
        workspace is None
        or not workspace.matches("matmul_all_reduce", (M, N, K), A.dtype, world_size, config.all_reduce_variant)
    )

    if needs_prepare:
        workspace = matmul_all_reduce_preamble(shmem, output_tensor, A, B, config=config, workspace=workspace)

    # Get heap bases for RMA
    heap_bases = shmem.get_heap_bases()

    # Compute EVEN_K flag
    even_k = 1 if (K % config.block_size_k == 0) else 0

    # Launch kernel
    grid = (config.num_sms,)

    _gemm_all_reduce_kernel[grid](
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
        1 if has_bias else 0,  # BIAS
        even_k,
        config.cache_modifier_a,
        config.cache_modifier_b,
        config.allow_tf32,
    )

    # Mark workspace as used
    if workspace is not None:
        workspace.prepared = False

    # Barrier unless async
    if not async_op:
        shmem.barrier()

    return workspace
