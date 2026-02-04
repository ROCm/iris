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
import iris
import iris.x


@triton.jit()
def _fused_matmul_reduce_scatter_kernel(
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
):
    """
    Fused GEMM + Reduce-Scatter kernel.

    Computes C = A @ B and then performs reduce-scatter on the result.
    Each rank computes the full GEMM but only keeps its assigned tiles after reduction.

    Args:
        A: Pointer to input matrix A of shape (M, K) - replicated across ranks
        B: Pointer to input matrix B of shape (K, N) - replicated across ranks
        C: Pointer to output matrix C of shape (M, N) - will contain reduced result for assigned tiles
        aux_buffer: Auxiliary buffer for intermediate GEMM results
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
    pid = tl.program_id(axis=0)
    num_pid_m = tl.cdiv(M, BLOCK_SIZE_M)
    num_pid_n = tl.cdiv(N, BLOCK_SIZE_N)

    pid_m = pid // num_pid_n
    pid_n = pid % num_pid_n

    rm = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    rn = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)

    acc = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)

    for k in range(0, tl.cdiv(K, BLOCK_SIZE_K)):
        rk = k * BLOCK_SIZE_K + tl.arange(0, BLOCK_SIZE_K)

        A_ptr = A + rm[:, None] * stride_am + rk[None, :] * stride_ak
        a = tl.load(A_ptr, mask=(rm[:, None] < M) & (rk[None, :] < K), other=0.0)

        B_ptr = B + rk[:, None] * stride_bk + rn[None, :] * stride_bn
        b = tl.load(B_ptr, mask=(rk[:, None] < K) & (rn[None, :] < N), other=0.0)

        acc += tl.dot(a, b)

    c = acc.to(C.dtype.element_ty)

    temp_ptr = aux_buffer + rm[:, None] * stride_cm + rn[None, :] * stride_cn
    tl.store(temp_ptr, c, mask=(rm[:, None] < M) & (rn[None, :] < N), cache_modifier=".wt")
    tl.debug_barrier()

    tile_id = pid_m * num_pid_n + pid_n
    lock_ptr = locks + tile_id
    tl.atomic_xchg(lock_ptr, 1, sem="release", scope="gpu")

    tile_obj = iris.x.Tile(pid_m, pid_n, BLOCK_SIZE_M, BLOCK_SIZE_N, c)
    src_view = iris.x.TensorView(aux_buffer, M, N, stride_cm, stride_cn)
    dst_view = iris.x.TensorView(C, M, N, stride_cm, stride_cn)
    ctx = iris.DeviceContext.initialize(context_tensor, cur_rank, world_size)

    iris.x.reduce_scatter(tile_obj, src_view, dst_view, locks, ctx)


def matmul_reduce_scatter_preamble(
    shmem,
    C: torch.Tensor,
    A: torch.Tensor,
    B: torch.Tensor,
    config: Optional[FusedConfig] = None,
    workspace: Optional[FusedWorkspace] = None,
) -> FusedWorkspace:
    """
    Allocate and reset temporary buffers for matmul_reduce_scatter.

    Args:
        shmem: Iris shmem context
        C: Output tensor (M, N) - will contain reduced result for assigned tiles
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

    workspace.operation = "matmul_reduce_scatter"
    workspace.shape = (M, N, K)
    workspace.dtype = dtype
    workspace.world_size = world_size
    workspace.variant = "two_shot"
    workspace.prepared = False

    num_pid_m = (M + config.block_size_m - 1) // config.block_size_m
    num_pid_n = (N + config.block_size_n - 1) // config.block_size_n
    total_tiles = num_pid_m * num_pid_n

    if workspace.locks is None or workspace.locks.numel() != total_tiles:
        workspace.locks = shmem.zeros((total_tiles,), dtype=torch.int32)
    else:
        workspace.locks.zero_()

    if workspace.aux_buffer is None or workspace.aux_buffer.shape != (M, N):
        workspace.aux_buffer = shmem.zeros((M, N), dtype=dtype)
    else:
        workspace.aux_buffer.zero_()

    C.zero_()
    shmem.barrier()

    workspace.prepared = True
    return workspace


def matmul_reduce_scatter(
    shmem,
    C: torch.Tensor,
    A: torch.Tensor,
    B: torch.Tensor,
    async_op: bool = False,
    config: Optional[FusedConfig] = None,
    workspace: Optional[FusedWorkspace] = None,
) -> FusedWorkspace:
    """
    Fused matrix multiplication and reduce-scatter.

    Computes: C = reduce_scatter(A @ B) where each rank keeps only its assigned tiles.

    This is equivalent to:
    1. All ranks compute: result = A @ B
    2. All ranks reduce: reduced_result = sum(result across ranks)
    3. Each rank keeps only its contiguous block of tiles from reduced_result

    Args:
        shmem: Iris shmem context
        C: Output tensor (M, N) - will contain reduced tiles for this rank
        A: Input matrix A (M, K) - replicated across ranks
        B: Input matrix B (K, N) - replicated across ranks
        async_op: If True, returns immediately without synchronization
        config: Optional FusedConfig for tuning. If None, uses defaults.
        workspace: Optional workspace to reuse. If None, allocates new.

    Returns:
        FusedWorkspace with allocated temporary buffers.
    """
    if config is None:
        config = FusedConfig()

    workspace = matmul_reduce_scatter_preamble(shmem, C, A, B, config, workspace)

    M, K = A.shape[:2]
    N = B.shape[1]
    rank = shmem.get_rank()
    world_size = shmem.get_num_ranks()

    num_pid_m = (M + config.block_size_m - 1) // config.block_size_m
    num_pid_n = (N + config.block_size_n - 1) // config.block_size_n
    grid = (num_pid_m * num_pid_n,)

    _fused_matmul_reduce_scatter_kernel[grid](
        A,
        B,
        C,
        workspace.aux_buffer,
        workspace.locks,
        M,
        N,
        K,
        A.stride(0),
        A.stride(1),
        B.stride(0),
        B.stride(1),
        C.stride(0),
        C.stride(1),
        shmem.get_device_context(),
        rank,
        world_size,
        config.block_size_m,
        config.block_size_n,
        config.block_size_k,
    )

    if not async_op:
        torch.cuda.synchronize()
        shmem.barrier()

    return workspace
