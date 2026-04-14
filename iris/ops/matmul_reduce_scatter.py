# SPDX-License-Identifier: MIT
# Copyright (c) 2025 Advanced Micro Devices, Inc. All rights reserved.

"""
High-level API for fused matrix multiplication and reduce-scatter.

This module provides a torch-like interface for GEMM+Reduce-Scatter operations,
automatically inferring dimensions, strides, and hardware parameters.

Two reduce-scatter variants are supported:

  - "two_shot" (default): Each rank has replicated A (M, K) and B (K, N).
    Computes full GEMM to aux_buffer, signals via locks, then pulls and reduces
    tiles from all ranks. Needs aux_buffer + locks workspace.

  - "atomic": Each rank has K-sharded A (M, K_local) and full B (K_local, N).
    Computes partial GEMM and atomic_adds directly to the destination rank's
    output buffer via the symmetric heap. No workspace needed.
"""

from typing import Optional
import torch
import triton
import triton.language as tl

from tritonblas.kernels.stages import GemmContext, ScheduleContext, make_tensor_view

from .config import FusedConfig
from .workspace import FusedWorkspace
import iris
import iris.x
from iris.tracing.kernel_artifacts import iris_launch


@triton.jit()
def _fused_matmul_reduce_scatter_kernel(
    A,
    B,
    C,
    aux_buffer,
    locks,
    M,
    N,
    N_local,
    K,
    stride_am,
    stride_ak,
    stride_bk,
    stride_bn,
    stride_cm,
    stride_cn,
    context_tensor: tl.tensor,
    cur_rank: tl.constexpr,
    world_size: tl.constexpr,
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr,
    GROUP_SIZE_M: tl.constexpr,
    NUM_SMS: tl.constexpr,
    NUM_XCDS: tl.constexpr,
    EVEN_K: tl.constexpr,
    ALLOW_TF32: tl.constexpr,
    VARIANT: tl.constexpr,
):
    """
    Unified fused GEMM + Reduce-Scatter kernel.

    Supports two variants controlled by the VARIANT constexpr:

      VARIANT == "atomic":
        K-split approach. Each rank has A_shard (M, K_local) and B (K_local, N).
        Computes partial GEMM and atomic_adds directly to the destination rank's
        C buffer. No aux_buffer or locks needed.

      VARIANT == "two_shot":
        Replicated approach. Each rank has identical A (M, K) and B (K, N).
        Computes full GEMM to aux_buffer, signals via locks, then iris.x.reduce_scatter
        pulls and reduces tiles across ranks.
    """
    tensorA = make_tensor_view(A, M, K, stride_am, stride_ak)
    tensorB = make_tensor_view(B, K, N, stride_bk, stride_bn)
    gemm_ctx = GemmContext(
        BLOCK_SIZE_M,
        BLOCK_SIZE_N,
        BLOCK_SIZE_K,
        num_sms=NUM_SMS,
        num_xcds=NUM_XCDS,
        group_size_m=GROUP_SIZE_M,
        even_k=EVEN_K,
        allow_tf32=ALLOW_TF32,
    )
    sched = ScheduleContext(M, N, K, gemm_ctx)
    ctx = iris.DeviceContext.initialize(context_tensor, cur_rank, world_size)

    num_tiles_n = tl.cdiv(N, BLOCK_SIZE_N)
    num_tiles_n_local = tl.cdiv(N_local, BLOCK_SIZE_N)

    # Two-shot needs views for the pull-based RS
    if VARIANT == "two_shot":
        src_view = iris.x.make_tensor_view(aux_buffer, M, N, stride_cm, stride_cn)
        dst_view = iris.x.make_tensor_view(C, M, N, stride_cm, stride_cn)

    start, total, stride = sched.persistent_tile_range()
    for tile_idx in range(start, total, stride):
        out_tile = sched.get_tile_from_idx(tile_idx)

        # ── GEMM (shared across both variants) ──
        acc = gemm_ctx.reduce_axis(tensorA, tensorB, out_tile)
        c = acc.to(C.type.element_ty)

        # ── Reduction (variant-specific) ──
        if VARIANT == "atomic":
            # Push partial sum directly to dest rank's output buffer
            pid_n = out_tile.pid_n
            dest_rank = pid_n // num_tiles_n_local
            local_pid_n = pid_n % num_tiles_n_local

            rm = out_tile.pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
            rn = local_pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
            mask = (rm[:, None] < M) & (rn[None, :] < N_local)
            C_ptr = C + rm[:, None] * stride_cm + rn[None, :] * stride_cn

            iris.atomic_add(
                C_ptr,
                c,
                cur_rank,
                dest_rank,
                ctx.heap_bases,
                mask=mask,
                sem="relaxed",
            )
        else:
            # Store to aux_buffer, signal lock, pull-based RS
            rm, rn = out_tile.indices()
            temp_ptr = aux_buffer + rm[:, None] * stride_cm + rn[None, :] * stride_cn
            tl.store(temp_ptr, c, mask=(rm[:, None] < M) & (rn[None, :] < N), cache_modifier=".wt")
            tl.debug_barrier()

            tile_id = out_tile.pid_m * num_tiles_n + out_tile.pid_n
            lock_ptr = locks + tile_id
            tl.atomic_xchg(lock_ptr, 1, sem="release", scope="gpu")

            tile_obj = iris.x.Tile(out_tile.pid_m, out_tile.pid_n, BLOCK_SIZE_M, BLOCK_SIZE_N, c)
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
    Allocate and reset temporary buffers for matmul_reduce_scatter (two_shot variant).

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

    if workspace.locks is not None and workspace.locks.numel() < total_tiles:
        raise ValueError(
            f"Lock array too small: have {workspace.locks.numel()} but need {total_tiles}. "
            f"Pre-allocate workspace with the smallest block sizes you intend to use."
        )

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

    Two independent config axes:

      config.ksplit (input sharding):
        False (default): A (M, K) and B (K, N) replicated. C is (M, N).
        True: A is K-sharded (M, K_local), B is (K_local, N). C is (M, N_local).

      config.reduce_scatter_variant (reduction algorithm):
        "two_shot" (default): Store to aux_buffer, signal locks, pull-based RS.
        "atomic": atomic_add directly to dest rank's output buffer.

    Args:
        shmem: Iris shmem context
        C: Output tensor
        A: Input matrix A
        B: Input matrix B
        async_op: If True, returns immediately without synchronization
        config: Optional FusedConfig for tuning. If None, uses defaults.
        workspace: Optional workspace to reuse (non-ksplit only). If None, allocates new.

    Returns:
        FusedWorkspace with operation metadata.
    """
    if config is None:
        config = FusedConfig()

    M, K = A.shape[:2]
    N = B.shape[1]
    rank = shmem.get_rank()
    world_size = shmem.get_num_ranks()
    variant = config.reduce_scatter_variant

    device = A.device
    num_sms = config.num_sms
    if num_sms is None:
        props = torch.cuda.get_device_properties(device)
        num_sms = props.multi_processor_count

    even_k = K % config.block_size_k == 0

    # ── Shape setup: ksplit controls input/output sharding ──
    if config.ksplit:
        N_local = N // world_size
        assert N % world_size == 0, f"N ({N}) must be divisible by world_size ({world_size})"
        assert C.shape == (M, N_local), f"Output C must be ({M}, {N_local}) for ksplit, got {C.shape}"
    else:
        N_local = N

    # ── Workspace setup: two_shot needs aux_buffer + locks ──
    aux_buffer = None
    locks = None
    if variant == "two_shot":
        workspace = matmul_reduce_scatter_preamble(shmem, C, A, B, config, workspace)
        aux_buffer = workspace.aux_buffer
        locks = workspace.locks
    else:
        C.zero_()
        shmem.barrier()

    # ── Single kernel launch ──
    iris_launch(
        _fused_matmul_reduce_scatter_kernel,
        (num_sms,),
        A,
        B,
        C,
        aux_buffer,
        locks,
        M,
        N,
        N_local,
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
        config.group_size_m,
        num_sms,
        config.num_xcds,
        even_k,
        config.allow_tf32,
        variant,
        algorithm="matmul_reduce_scatter",
        rank=rank,
        dtype=A.dtype,
    )

    if not async_op:
        torch.cuda.synchronize()
        shmem.barrier()

    if workspace is None:
        workspace = FusedWorkspace()
    workspace.operation = f"matmul_reduce_scatter_{variant}"
    if config.ksplit:
        workspace.operation += "_ksplit"
    workspace.shape = (M, N, K)
    workspace.dtype = A.dtype
    workspace.world_size = world_size
    workspace.prepared = True
    return workspace
