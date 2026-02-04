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
import iris
import iris.x


@triton.jit()
def _fused_matmul_all_reduce_kernel(
    A,
    B,
    C,
    aux_buffer,
    locks,
    M: tl.constexpr,
    N: tl.constexpr,
    K: tl.constexpr,
    stride_am: tl.constexpr,
    stride_ak: tl.constexpr,
    stride_bk: tl.constexpr,
    stride_bn: tl.constexpr,
    stride_cm: tl.constexpr,
    stride_cn: tl.constexpr,
    context_tensor,
    cur_rank: tl.constexpr,
    world_size: tl.constexpr,
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr,
    VARIANT: tl.constexpr,
):
    """
    Fused GEMM + All-Reduce kernel with configurable all-reduce variant.

    Computes C = A @ B and then performs all-reduce on the result using the specified variant.
    This is useful for data-parallel distributed training where each rank computes
    a partial result over different data, and then reduces across all ranks.

    Supported variants:
    - 'atomic': Fast, lock-free atomic accumulation
    - 'spinlock': Mutex-based serialized read-modify-write
    - 'one_shot': Each rank reduces all tiles (duplicated work, no remote stores)
    - 'two_shot': Work distribution with reduce-scatter then all-gather pattern

    The kernel for each output tile:
    1. Computes GEMM: local_tile = A_tile @ B_tile
    2. Uses spinlock-protected read-modify-write to accumulate to all ranks

    Args:
        A: Pointer to input matrix A of shape (M, K) - local rank's data
        B: Pointer to input matrix B of shape (K, N) - replicated across ranks
        C: Pointer to output matrix C of shape (M, N) - will contain reduced result
        locks: Pointer to locks array (one lock per tile)
        M: Number of rows in A and C
        N: Number of columns in B and C
        K: Number of columns in A and rows in B
        stride_am, stride_ak: Strides for A tensor
        stride_bk, stride_bn: Strides for B tensor
        stride_cm, stride_cn: Strides for C tensor
        heap_bases: Heap base pointers for all ranks
        cur_rank: Current rank
        world_size: Total number of ranks
        BLOCK_SIZE_M: Block size for M dimension
        BLOCK_SIZE_N: Block size for N dimension
        BLOCK_SIZE_K: Block size for K dimension
    """
    # Get program ID and compute grid dimensions
    pid = tl.program_id(axis=0)
    num_pid_m = tl.cdiv(M, BLOCK_SIZE_M)
    num_pid_n = tl.cdiv(N, BLOCK_SIZE_N)

    # Compute which tile this program handles
    pid_m = pid // num_pid_n
    pid_n = pid % num_pid_n

    # Compute row and column indices for this tile
    rm = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    rn = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)

    # Initialize accumulator for GEMM
    acc = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)

    # GEMM loop over K dimension
    for k in range(0, tl.cdiv(K, BLOCK_SIZE_K)):
        rk = k * BLOCK_SIZE_K + tl.arange(0, BLOCK_SIZE_K)

        # Load A tile
        A_ptr = A + rm[:, None] * stride_am + rk[None, :] * stride_ak
        a = tl.load(A_ptr, mask=(rm[:, None] < M) & (rk[None, :] < K), other=0.0)

        # Load B tile
        B_ptr = B + rk[:, None] * stride_bk + rn[None, :] * stride_bn
        b = tl.load(B_ptr, mask=(rk[:, None] < K) & (rn[None, :] < N), other=0.0)

        # Accumulate
        acc += tl.dot(a, b)

    # Convert to output dtype
    c = acc.to(C.dtype.element_ty)

    # Create views and context
    ctx = iris.DeviceContext.initialize(context_tensor, cur_rank, world_size)
    dst_view = iris.x.TensorView(C, M, N, stride_cm, stride_cn)

    # For one_shot and two_shot: store tile to aux_buffer and signal ready with lock
    if VARIANT == "one_shot" or VARIANT == "two_shot":
        # Store GEMM result to aux_buffer (avoid race condition with final output)
        temp_ptr = aux_buffer + rm[:, None] * stride_cm + rn[None, :] * stride_cn
        tl.store(temp_ptr, c, mask=(rm[:, None] < M) & (rn[None, :] < N), cache_modifier=".wt")
        tl.debug_barrier()  # Ensures all stores are visible before the atomic_xchg

        # Signal tile is ready by unlocking (set lock to 1)
        # Use atomic_xchg with release semantics to ensure memory ordering
        num_tiles_n = tl.cdiv(N, BLOCK_SIZE_N)
        tile_id = pid_m * num_tiles_n + pid_n
        lock_ptr = locks + tile_id
        tl.atomic_xchg(lock_ptr, 1, sem="release", scope="gpu")  # Release ensures prior stores visible

        # Create src_view pointing to aux_buffer
        src_view = iris.x.TensorView(aux_buffer, M, N, stride_cm, stride_cn)

    # Dispatch to appropriate all-reduce variant
    if VARIANT == "atomic":
        # Atomic uses tile.data directly (no intermediate store needed)
        tile_obj = iris.x.Tile(pid_m, pid_n, BLOCK_SIZE_M, BLOCK_SIZE_N, c)
        iris.x.all_reduce_atomic(tile_obj, dst_view, ctx)
    elif VARIANT == "spinlock":
        # Spinlock uses tile.data directly and lock for mutual exclusion
        tile_obj = iris.x.Tile(pid_m, pid_n, BLOCK_SIZE_M, BLOCK_SIZE_N, c)
        iris.x.all_reduce_spinlock(tile_obj, dst_view, locks, ctx)
    elif VARIANT == "one_shot":
        # one_shot loads from all ranks (data already in memory, locks signal readiness)
        tile_obj = iris.x.Tile(pid_m, pid_n, BLOCK_SIZE_M, BLOCK_SIZE_N, c)
        iris.x.all_reduce_one_shot(tile_obj, src_view, dst_view, locks, ctx)
    elif VARIANT == "two_shot":
        # two_shot with work distribution (data in memory, locks signal readiness)
        tile_obj = iris.x.Tile(pid_m, pid_n, BLOCK_SIZE_M, BLOCK_SIZE_N, c)
        iris.x.all_reduce_two_shot(tile_obj, src_view, dst_view, locks, cur_rank, world_size, ctx)
    # elif VARIANT == "ring":
    #     # Store locally first and signal ready
    #     tile_obj = iris.x.Tile(pid_m, pid_n, BLOCK_SIZE_M, BLOCK_SIZE_N, c)
    #     iris.x.all_reduce_ring(tile_obj, src_view, dst_view, locks, ctx)


def matmul_all_reduce_preamble(
    shmem,
    C: torch.Tensor,
    A: torch.Tensor,
    B: torch.Tensor,
    config: Optional[FusedConfig] = None,
    workspace: Optional[FusedWorkspace] = None,
) -> FusedWorkspace:
    """
    Allocate and reset temporary buffers for matmul_all_reduce.

    Args:
        shmem: Iris shmem context
        C: Output tensor (M, N)
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

    # Validate config
    config.validate(world_size=world_size)

    if workspace is None:
        workspace = FusedWorkspace()

    workspace.operation = "matmul_all_reduce"
    workspace.shape = (M, N, K)
    workspace.dtype = dtype
    workspace.world_size = world_size
    workspace.variant = config.all_reduce_variant
    workspace.prepared = False

    # Allocate locks for spinlock-based all-reduce
    num_pid_m = (M + config.block_size_m - 1) // config.block_size_m
    num_pid_n = (N + config.block_size_n - 1) // config.block_size_n
    total_tiles = num_pid_m * num_pid_n

    # Allocate locks for spinlock, one_shot, and two_shot variants
    if config.all_reduce_variant in ["spinlock", "one_shot", "two_shot"]:
        if workspace.locks is None or workspace.locks.numel() != total_tiles:
            workspace.locks = shmem.zeros((total_tiles,), dtype=torch.int32)
        else:
            workspace.locks.zero_()
    else:
        workspace.locks = None

    # Allocate auxiliary buffer for one_shot and two_shot to avoid race conditions
    # (GEMM results stored here, then reduced to final output)
    if config.all_reduce_variant in ["one_shot", "two_shot"]:
        if workspace.aux_buffer is None or workspace.aux_buffer.shape != (M, N):
            workspace.aux_buffer = shmem.zeros((M, N), dtype=dtype)
        else:
            workspace.aux_buffer.zero_()
    else:
        workspace.aux_buffer = None

    # Zero output tensor
    C.zero_()
    shmem.barrier()

    workspace.prepared = True
    return workspace


def matmul_all_reduce(
    shmem,
    C: torch.Tensor,
    A: torch.Tensor,
    B: torch.Tensor,
    async_op: bool = False,
    config: Optional[FusedConfig] = None,
    workspace: Optional[FusedWorkspace] = None,
) -> FusedWorkspace:
    """
    Fused matrix multiplication and all-reduce using atomic operations.

    Computes: C = all_reduce(A @ B) across all ranks using atomic adds.

    Args:
        shmem: Iris shmem context
        C: Output tensor (M, N) - will contain reduced result on all ranks
        A: Input matrix A (M, K) - each rank has different data (data-parallel)
        B: Input matrix B (K, N) - replicated across ranks
        async_op: If False, performs barrier at end. Default: False.
        config: Optional FusedConfig for tuning. If None, uses defaults.
        workspace: Optional pre-allocated workspace. If None, creates new one.

    Returns:
        workspace: Updated workspace object (can be reused for subsequent calls)

    Example:
        >>> A = shmem.randn((1024, 512), dtype=torch.float16)
        >>> B = shmem.randn((512, 2048), dtype=torch.float16)
        >>> C = shmem.zeros((1024, 2048), dtype=torch.float16)
        >>> shmem.ops.matmul_all_reduce(C, A, B)
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

    if C.shape != (M, N):
        raise ValueError(f"Output tensor shape {C.shape} doesn't match expected ({M}, {N})")

    if A.dtype != B.dtype or A.dtype != C.dtype:
        raise ValueError(f"All tensors must have same dtype, got A:{A.dtype}, B:{B.dtype}, C:{C.dtype}")

    # Validate block sizes match problem dimensions
    assert M >= config.block_size_m, f"M={M} too small for block_size_m={config.block_size_m}"
    assert K >= config.block_size_k, f"K={K} too small for block_size_k={config.block_size_k}"
    assert N >= config.block_size_n, f"N={N} too small for block_size_n={config.block_size_n}"

    # Extract strides
    stride_am, stride_ak = A.stride()
    stride_bk, stride_bn = B.stride()
    stride_cm, stride_cn = C.stride()

    # Get rank info
    rank = shmem.get_rank()
    world_size = shmem.get_num_ranks()

    # Prepare workspace if needed
    needs_prepare = workspace is None or not workspace.matches(
        "matmul_all_reduce", (M, N, K), A.dtype, world_size, config.all_reduce_variant
    )

    if needs_prepare:
        workspace = matmul_all_reduce_preamble(shmem, C, A, B, config=config, workspace=workspace)

    # Get device context for RMA
    device_context = shmem.get_device_context()

    # Launch kernel
    num_pid_m = (M + config.block_size_m - 1) // config.block_size_m
    num_pid_n = (N + config.block_size_n - 1) // config.block_size_n
    grid = (num_pid_m * num_pid_n,)

    _fused_matmul_all_reduce_kernel[grid](
        A,
        B,
        C,
        workspace.aux_buffer,
        workspace.locks,
        M,
        N,
        K,
        stride_am,
        stride_ak,
        stride_bk,
        stride_bn,
        stride_cm,
        stride_cn,
        device_context,
        rank,
        world_size,
        config.block_size_m,
        config.block_size_n,
        config.block_size_k,
        config.all_reduce_variant,
    )

    # Mark workspace as used
    if workspace is not None:
        workspace.prepared = False

    # Barrier unless async
    if not async_op:
        shmem.barrier()

    return workspace
