# SPDX-License-Identifier: MIT
# Copyright (c) 2025 Advanced Micro Devices, Inc. All rights reserved.

"""
High-level API for fused matrix multiplication and all-reduce.

This module provides a torch-like interface for GEMM+All-Reduce operations,
automatically inferring dimensions, strides, and hardware parameters.
"""

import logging
from typing import Optional
import torch
import torch.distributed as dist
import triton
import triton.language as tl

from tritonblas.kernels.stages import GemmContext, make_tensor_view, Tile

from .config import FusedConfig
from .workspace import FusedWorkspace
import iris
from iris.host.tracing.kernel_artifacts import iris_launch


@triton.jit(do_not_specialize=["generation"])
def _fused_matmul_all_reduce_kernel(
    A,
    B,
    C,
    aux_buffer,
    locks,
    completion_locks,
    generation,
    M,
    N,
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
    EVEN_K: tl.constexpr,
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
    1. Computes GEMM using tritonblas GemmContext
    2. Uses the specified variant for all-reduce across ranks

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
        context_tensor: Device context tensor for RMA operations
        cur_rank: Current rank
        world_size: Total number of ranks
        BLOCK_SIZE_M: Block size for M dimension
        BLOCK_SIZE_N: Block size for N dimension
        BLOCK_SIZE_K: Block size for K dimension
        EVEN_K: Whether K is evenly divisible by BLOCK_SIZE_K
    """
    # Get program ID and compute which tile this program handles
    pid = tl.program_id(axis=0)
    num_tiles_n = tl.cdiv(N, BLOCK_SIZE_N)
    pid_m = pid // num_tiles_n
    pid_n = pid % num_tiles_n

    # ═══════════════════════════════════════════════════════════════════════
    # GEMM using tritonblas stages
    # ═══════════════════════════════════════════════════════════════════════
    tensorA = make_tensor_view(A, M, K, stride_am, stride_ak)
    tensorB = make_tensor_view(B, K, N, stride_bk, stride_bn)
    gemm_ctx = GemmContext(
        BLOCK_SIZE_M,
        BLOCK_SIZE_N,
        BLOCK_SIZE_K,
        num_sms=1,
        even_k=EVEN_K,
    )
    out_tile = Tile(pid_m, pid_n, BLOCK_SIZE_M, BLOCK_SIZE_N)
    acc = gemm_ctx.reduce_axis(tensorA, tensorB, out_tile)

    # Get row and column indices from tile (needed for one_shot/two_shot variants)
    rm, rn = out_tile.indices()

    # Convert to output dtype
    c = acc.to(C.type.element_ty)

    # Create views and context
    ctx = iris.DeviceContext.initialize(context_tensor, cur_rank, world_size)
    dst_view = iris.make_tensor_view(C, M, N, stride_cm, stride_cn)

    # Create tile object once for all variants
    tile_obj = iris.Tile(pid_m, pid_n, BLOCK_SIZE_M, BLOCK_SIZE_N, c)

    # Dispatch to appropriate all-reduce variant
    if VARIANT == "atomic":
        ctx.all_reduce_atomic(tile_obj, dst_view)
    elif VARIANT == "spinlock":
        ctx.all_reduce_spinlock(tile_obj, dst_view, locks)
    elif VARIANT == "one_shot" or VARIANT == "two_shot":
        tile_id = pid_m * num_tiles_n + pid_n
        if generation > 1:
            previous_generation = generation - 1
            # Previous readers and the two_shot owner's remote stores must finish
            # before this tile's auxiliary storage or output can be reused.
            for remote_rank in range(world_size):
                while (
                    ctx.atomic_cas(
                        completion_locks + tile_id,
                        previous_generation,
                        previous_generation,
                        to_rank=remote_rank,
                        sem="acquire",
                        scope="sys",
                    )
                    < previous_generation
                ):
                    pass
        # The auxiliary allocation is contiguous, independently of C's strides.
        temp_ptr = aux_buffer + rm[:, None] * N + rn[None, :]
        tl.store(temp_ptr, c, mask=(rm[:, None] < M) & (rn[None, :] < N), cache_modifier=".wt")
        tl.debug_barrier()
        tl.atomic_xchg(locks + tile_id, generation, sem="release", scope="sys")
        src_view = iris.make_tensor_view(aux_buffer, M, N, N, 1)

        if VARIANT == "one_shot":
            ctx.all_reduce_one_shot(tile_obj, src_view, dst_view, locks, generation=generation)
        elif VARIANT == "two_shot":
            ctx.all_reduce_two_shot(tile_obj, src_view, dst_view, locks, generation=generation)
        tl.debug_barrier()
        tl.atomic_xchg(completion_locks + tile_id, generation, sem="release", scope="sys")


def _workspace_matches(shmem, A, B, config, workspace):
    """Check allocation metadata and buffers, independently of prepared."""
    M, K = A.shape
    N = B.shape[1]
    variant = config.all_reduce_variant
    if workspace is None or workspace.owner is not shmem:
        return False
    if not workspace.allocation_matches("matmul_all_reduce", (M, N, K), A.dtype, shmem.get_num_ranks(), variant):
        return False
    if workspace.tile_shape != (config.block_size_m, config.block_size_n):
        return False
    total_tiles = triton.cdiv(M, config.block_size_m) * triton.cdiv(N, config.block_size_n)
    if variant in ("spinlock", "one_shot", "two_shot"):
        if workspace.locks is None or workspace.locks.numel() < total_tiles or workspace.locks.dtype != torch.int32:
            return False
    if variant in ("one_shot", "two_shot"):
        if (
            workspace.aux_buffer is None
            or workspace.aux_buffer.shape != (M, N)
            or workspace.aux_buffer.dtype != A.dtype
        ):
            return False
        if (
            workspace.completion_locks is None
            or workspace.completion_locks.numel() < total_tiles
            or workspace.completion_locks.dtype != torch.int32
        ):
            return False
        # Reallocate collectively before the signed int32 generation counter wraps.
        if not 0 <= workspace.generation < 2**31 - 1:
            return False
    return True


def _lock_capacity_error(shmem, A, B, config, workspace):
    """Preserve the error for undersized preallocated locks."""
    M, K = A.shape
    N = B.shape[1]
    if (
        workspace is None
        or workspace.owner is not shmem
        or not workspace.allocation_matches(
            "matmul_all_reduce", (M, N, K), A.dtype, shmem.get_num_ranks(), config.all_reduce_variant
        )
        or workspace.locks is None
    ):
        return None
    total_tiles = triton.cdiv(M, config.block_size_m) * triton.cdiv(N, config.block_size_n)
    if workspace.locks.numel() < total_tiles:
        return (
            f"Lock array too small: have {workspace.locks.numel()} but need {total_tiles}. "
            "Pre-allocate workspace with the smallest block sizes you intend to use."
        )
    return None


def _allocate_workspace(shmem, A, B, config, workspace):
    """Collectively allocate buffers without per-call output preparation."""
    M, K = A.shape
    N = B.shape[1]
    variant = config.all_reduce_variant
    total_tiles = triton.cdiv(M, config.block_size_m) * triton.cdiv(N, config.block_size_n)
    # Preserve the largest flag capacity this workspace already has.
    flag_capacity = total_tiles
    if workspace is not None and workspace.owner is shmem:
        for buffer in (workspace.locks, workspace.completion_locks):
            if buffer is not None:
                flag_capacity = max(flag_capacity, buffer.numel())
    # All ranks must allocate the same capacity.
    capacities = _gather_workspace_states(shmem, flag_capacity)
    flag_capacity = max(capacities)
    stream = torch.cuda.current_stream()
    # Finish previous remote accesses before replacing their buffers.
    shmem.barrier(stream=stream)
    locks = None
    aux_buffer = None
    completion_locks = None
    if variant in ("spinlock", "one_shot", "two_shot"):
        locks = shmem.zeros((flag_capacity,), dtype=torch.int32)
    if variant in ("one_shot", "two_shot"):
        aux_buffer = shmem.zeros((M, N), dtype=A.dtype)
        completion_locks = shmem.zeros((flag_capacity,), dtype=torch.int32)
    shmem.barrier(stream=stream)
    if workspace is None:
        workspace = FusedWorkspace()
    workspace.operation = "matmul_all_reduce"
    workspace.shape = (M, N, K)
    workspace.dtype = A.dtype
    workspace.world_size = shmem.get_num_ranks()
    workspace.variant = variant
    workspace.tile_shape = (config.block_size_m, config.block_size_n)
    workspace.owner = shmem
    workspace.locks = locks
    workspace.aux_buffer = aux_buffer
    workspace.completion_locks = completion_locks
    workspace.generation = 0
    workspace.prepared = False
    return workspace


def _gather_workspace_states(shmem, local_state):
    world_size = shmem.get_num_ranks()
    if world_size == 1:
        return [local_state]
    if not dist.is_initialized() or dist.get_world_size() != world_size:
        raise RuntimeError("Workspace agreement requires the Iris default process group.")
    states = [None] * world_size
    dist.all_gather_object(states, local_state)
    return states


def _workspace_offsets(shmem, workspace):
    """Compare symmetric heap offsets rather than GPU virtual addresses."""
    base = shmem.heap.allocator.get_base_address()
    return tuple(
        None if buffer is None else buffer.data_ptr() - base
        for buffer in (workspace.locks, workspace.aux_buffer, workspace.completion_locks)
    )


def _agree_workspace(shmem, A, B, config, workspace, *, check_capacity):
    """Make a common allocation decision before entering any collective allocation."""
    M, K = A.shape
    N = B.shape[1]
    needs_allocation = not _workspace_matches(shmem, A, B, config, workspace)
    parameters = (
        "launch" if check_capacity else "preamble",
        M,
        N,
        K,
        str(A.dtype),
        shmem.get_num_ranks(),
        config.all_reduce_variant,
        config.block_size_m,
        config.block_size_n,
        config.block_size_k,
    )
    local_state = {
        "parameters": parameters,
        "needs_allocation": needs_allocation,
        "offsets": None if needs_allocation else _workspace_offsets(shmem, workspace),
        "error": _lock_capacity_error(shmem, A, B, config, workspace) if check_capacity else None,
        "generation": None if needs_allocation else workspace.generation,
    }
    states = _gather_workspace_states(shmem, local_state)
    if any(state["parameters"] != parameters for state in states):
        raise ValueError("All ranks must use matching matmul_all_reduce parameters and call the same entry point.")
    for state in states:
        if state["error"] is not None:
            raise ValueError(state["error"])
    if any(state["needs_allocation"] for state in states):
        return True
    # Matching metadata does not guarantee that ranks chose the same allocation.
    if len({state["offsets"] for state in states}) != 1:
        return True
    if len({state["generation"] for state in states}) != 1:
        raise ValueError("Workspace generations differ across ranks.")
    return False


def _ensure_workspace(shmem, A, B, config, workspace, *, check_capacity=False):
    config.validate(world_size=shmem.get_num_ranks())
    if _agree_workspace(shmem, A, B, config, workspace, check_capacity=check_capacity):
        workspace = _allocate_workspace(shmem, A, B, config, workspace)
        offsets = _gather_workspace_states(shmem, _workspace_offsets(shmem, workspace))
        if len(set(offsets)) != 1:
            raise RuntimeError("Workspace allocations have different symmetric heap offsets.")
    return workspace


def _pre_kernel_sync(shmem, C, config, workspace):
    """Prepare accumulation variants; shot variants overwrite and version their buffers."""
    if config.all_reduce_variant in ("one_shot", "two_shot"):
        return
    stream = torch.cuda.current_stream()
    # Previous async calls may still access this rank's buffers remotely.
    shmem.barrier(stream=stream)
    if workspace.locks is not None:
        workspace.locks.zero_()
    C.zero_()
    shmem.barrier(stream=stream)
    workspace.prepared = True


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

    workspace = _ensure_workspace(shmem, A, B, config, workspace)
    _pre_kernel_sync(shmem, C, config, workspace)
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

    from iris.host.logging.logging import _log_rank

    _log_rank(
        logging.DEBUG,
        "matmul_all_reduce: shape=(%d,%d,%d) dtype=%s variant=%s rank=%d/%d",
        M,
        N,
        K,
        A.dtype,
        config.all_reduce_variant,
        rank,
        world_size,
        rank=rank,
        num_ranks=world_size,
    )

    workspace = _ensure_workspace(shmem, A, B, config, workspace, check_capacity=True)
    _pre_kernel_sync(shmem, C, config, workspace)

    # Get device context for RMA
    device_context = shmem.get_device_context()

    # Launch kernel
    num_pid_m = (M + config.block_size_m - 1) // config.block_size_m
    num_pid_n = (N + config.block_size_n - 1) // config.block_size_n
    total_tiles = num_pid_m * num_pid_n
    grid = (total_tiles,)

    even_k = K % config.block_size_k == 0
    generation = workspace.generation + 1

    iris_launch(
        _fused_matmul_all_reduce_kernel,
        grid,
        A,
        B,
        C,
        workspace.aux_buffer,
        workspace.locks,
        workspace.completion_locks,
        generation,
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
        even_k,
        config.all_reduce_variant,
        algorithm="matmul_all_reduce",
        rank=rank,
        dtype=A.dtype,
    )

    # Advance only after a successful launch; generations are runtime kernel arguments.
    workspace.generation = generation
    workspace.prepared = False

    # Barrier unless async
    if not async_op:
        shmem.barrier()

    return workspace
