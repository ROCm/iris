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
import triton
import triton.language as tl

from tritonblas.kernels.stages import GemmContext, ScheduleContext, make_input_view

from .config import FusedConfig
from .workspace import FusedWorkspace
import iris
from iris.host.tracing.kernel_artifacts import iris_launch


@triton.jit()
def _fused_matmul_all_reduce_kernel(
    A,
    B,
    C,
    aux_buffer,
    locks,
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
    GROUP_SIZE_M: tl.constexpr,
    NUM_SMS: tl.constexpr,
    NUM_XCDS: tl.constexpr,
    CHUNK_SIZE: tl.constexpr,
    EVEN_K: tl.constexpr,
    ALLOW_TF32: tl.constexpr,
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
    acc_dtype = tl.int32 if C.type.element_ty == tl.int8 else tl.float32
    gemm_ctx = GemmContext(
        BLOCK_SIZE_M,
        BLOCK_SIZE_N,
        BLOCK_SIZE_K,
        num_sms=NUM_SMS,
        num_xcds=NUM_XCDS,
        group_size_m=GROUP_SIZE_M,
        chunk_size=CHUNK_SIZE,
        acc_dtype=acc_dtype,
        even_k=EVEN_K,
        allow_tf32=ALLOW_TF32,
    )
    sched = ScheduleContext(M, N, K, gemm_ctx)
    tensorA = make_input_view(A, M, K, stride_am, stride_ak)
    tensorB = make_input_view(B, K, N, stride_bk, stride_bn)

    # Create views and context
    ctx = iris.DeviceContext.initialize(context_tensor, cur_rank, world_size)
    dst_view = iris.make_tensor_view(C, M, N, stride_cm, stride_cn)

    start, total, stride = sched.persistent_tile_range()
    for tile_idx in range(start, total, stride):
        out_tile = sched.get_tile_from_idx(tile_idx)
        acc = gemm_ctx.reduce_axis(tensorA, tensorB, out_tile)

        # Get row and column indices from tile (needed for one_shot/two_shot variants)
        rm, rn = out_tile.indices()

        # Convert to output dtype
        c = acc.to(C.type.element_ty)
        tile_obj = iris.Tile(out_tile.pid_m, out_tile.pid_n, BLOCK_SIZE_M, BLOCK_SIZE_N, c)

        # Dispatch to appropriate all-reduce variant
        if VARIANT == "atomic":
            ctx.all_reduce_atomic(tile_obj, dst_view)
        elif VARIANT == "spinlock":
            ctx.all_reduce_spinlock(tile_obj, dst_view, locks)
        elif VARIANT == "one_shot" or VARIANT == "two_shot":
            # For one_shot and two_shot: store tile to aux_buffer and signal ready with lock.
            temp_ptr = aux_buffer + rm[:, None] * stride_cm + rn[None, :] * stride_cn
            tl.store(temp_ptr, c, mask=(rm[:, None] < M) & (rn[None, :] < N), cache_modifier=".wt")
            tl.debug_barrier()

            # Locks are indexed by canonical tile coordinates so the protocol
            # stays independent of ScheduleContext's swizzled/persistent order.
            num_tiles_n = tl.cdiv(N, BLOCK_SIZE_N)
            tile_id = out_tile.pid_m * num_tiles_n + out_tile.pid_n
            lock_ptr = locks + tile_id
            tl.atomic_xchg(lock_ptr, 1, sem="release", scope="sys")

            src_view = iris.make_tensor_view(aux_buffer, M, N, stride_cm, stride_cn)
            if VARIANT == "one_shot":
                ctx.all_reduce_one_shot(tile_obj, src_view, dst_view, locks)
            elif VARIANT == "two_shot":
                ctx.all_reduce_two_shot(tile_obj, src_view, dst_view, locks)


_GEMM_CONFIG_FIELDS = (
    "block_size_m",
    "block_size_n",
    "block_size_k",
    "group_size_m",
    "num_sms",
    "num_xcds",
    "chunk_size",
    "cache_modifier_a",
    "cache_modifier_b",
    "allow_tf32",
)


def _config_uses_default_gemm_tuning(config: FusedConfig) -> bool:
    default = FusedConfig()
    return all(getattr(config, field) == getattr(default, field) for field in _GEMM_CONFIG_FIELDS)


def _default_chunk_size(total_tiles: int, group_size_m: int, num_xcds: int) -> int:
    chunk_size = group_size_m * group_size_m
    if num_xcds > 0:
        chunk_size = min(chunk_size, max(1, total_tiles // num_xcds))
    return max(1, chunk_size)


def _make_origami_selector(M: int, N: int, K: int, A: torch.Tensor, B: torch.Tensor, C):
    from tritonblas.matmul import _make_matmul_selector

    c_dtype = C.dtype if hasattr(C, "dtype") else C
    return _make_matmul_selector(
        M,
        N,
        K,
        A.dtype,
        B.dtype,
        c_dtype,
        A.device,
        streamk=False,
    )


def _selector_active_cus(selector, device: torch.device) -> int:
    active_cus = getattr(selector, "_ACTIVE_CU", None)
    if active_cus is None or active_cus <= 0:
        props = torch.cuda.get_device_properties(device)
        active_cus = props.multi_processor_count
    return int(active_cus)


def _matmul_all_reduce_launch_params(
    M: int,
    N: int,
    K: int,
    selector,
    device: torch.device,
    element_size: int,
    variant: str,
) -> dict:
    block_size_m = selector.block_m
    block_size_n = selector.block_n
    block_size_k = selector.block_k
    group_size_m = selector.group_m
    num_stages = getattr(selector, "num_stages", 2)
    selector_fallback = False

    # Origami's 256x256 tile is great when GEMM dominates, but one_shot also
    # does a full remote-rank reduction per output tile. For shallow K shapes,
    # keeping the old narrow-N tile avoids making each reduction work item too
    # large while still allowing the selector path for deeper GEMMs.
    if (
        variant == "one_shot"
        and K < 16 * 1024
        and block_size_m == 256
        and block_size_n == 256
        and block_size_k == 64
    ):
        block_size_n = 64
        group_size_m = 1
        num_stages = None
        selector_fallback = True

    # Atomic/spinlock variants can exceed the MI300 64 KiB LDS cap with the
    # common 256x256x64 Origami tile. Prefer the old narrow-N tile first; only
    # shrink M if a single-stage 256x64 tile still cannot fit.
    estimated_stage_count = num_stages if num_stages is not None else 2
    stage_bytes = (block_size_m * block_size_k + block_size_k * block_size_n) * element_size
    if variant in ("atomic", "spinlock") and stage_bytes * estimated_stage_count > 64 * 1024:
        block_size_n = min(block_size_n, 64)
        block_size_k = min(block_size_k, 64)
        group_size_m = 1
        num_stages = 1
        stage_bytes = (block_size_m * block_size_k + block_size_k * block_size_n) * element_size
        if stage_bytes > 64 * 1024:
            block_size_m = min(block_size_m, 128)
        selector_fallback = True

    # Origami calls this num_sms, but it is the XCD/chiplet workgroup mapping
    # count used by chiplet_transform_chunked, not the persistent launch grid.
    num_xcds = selector.num_sms
    if num_xcds <= 0:
        num_xcds = 1

    num_tiles_m = (M + block_size_m - 1) // block_size_m
    num_tiles_n = (N + block_size_n - 1) // block_size_n
    total_tiles = num_tiles_m * num_tiles_n
    num_sms = min(_selector_active_cus(selector, device), total_tiles)
    chunk_size = _default_chunk_size(num_sms, group_size_m, num_xcds)

    return {
        "block_size_m": block_size_m,
        "block_size_n": block_size_n,
        "block_size_k": block_size_k,
        "group_size_m": group_size_m,
        "num_xcds": num_xcds,
        "num_tiles_m": num_tiles_m,
        "num_tiles_n": num_tiles_n,
        "total_tiles": total_tiles,
        "num_sms": num_sms,
        "chunk_size": chunk_size,
        "num_warps": 8,
        "num_stages": num_stages,
        "matrix_instr_nonkdim": 16,
        "allow_tf32": torch.backends.cuda.matmul.allow_tf32,
        "selector_fallback": selector_fallback,
    }


def _config_launch_params(M: int, N: int, config: FusedConfig, device: torch.device) -> dict:
    num_tiles_m = (M + config.block_size_m - 1) // config.block_size_m
    num_tiles_n = (N + config.block_size_n - 1) // config.block_size_n
    total_tiles = num_tiles_m * num_tiles_n

    num_sms = config.num_sms
    if num_sms is None:
        props = torch.cuda.get_device_properties(device)
        num_sms = props.multi_processor_count
    num_sms = min(int(num_sms), total_tiles)

    num_xcds = config.num_xcds
    if num_xcds <= 0:
        num_xcds = 1

    return {
        "block_size_m": config.block_size_m,
        "block_size_n": config.block_size_n,
        "block_size_k": config.block_size_k,
        "group_size_m": config.group_size_m,
        "num_xcds": num_xcds,
        "num_tiles_m": num_tiles_m,
        "num_tiles_n": num_tiles_n,
        "total_tiles": total_tiles,
        "num_sms": num_sms,
        "chunk_size": max(1, config.chunk_size),
        "num_warps": 8,
        "num_stages": None,
        "matrix_instr_nonkdim": 16,
        "allow_tf32": config.allow_tf32,
        "selector_fallback": False,
    }


def matmul_all_reduce_preamble(
    shmem,
    C: torch.Tensor,
    A: torch.Tensor,
    B: torch.Tensor,
    config: Optional[FusedConfig] = None,
    workspace: Optional[FusedWorkspace] = None,
    selector=None,
    out_dtype: Optional[torch.dtype] = None,
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
        selector: Optional pre-built tritonBLAS Origami selector.
        out_dtype: Optional output dtype for selector construction.

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

    if selector is not None:
        launch = _matmul_all_reduce_launch_params(
            M, N, K, selector, A.device, A.element_size(), config.all_reduce_variant
        )
    elif _config_uses_default_gemm_tuning(config):
        c_dtype = dtype if out_dtype is None else out_dtype
        selector = _make_origami_selector(M, N, K, A, B, c_dtype)
        launch = _matmul_all_reduce_launch_params(
            M, N, K, selector, A.device, A.element_size(), config.all_reduce_variant
        )
    else:
        launch = _config_launch_params(M, N, config, A.device)

    if workspace is None:
        workspace = FusedWorkspace()

    workspace.operation = "matmul_all_reduce"
    workspace.shape = (M, N, K)
    workspace.dtype = dtype
    workspace.world_size = world_size
    workspace.variant = config.all_reduce_variant
    workspace.selector = selector
    workspace.config = config
    workspace.launch_params = launch
    workspace.selector_fallback = launch["selector_fallback"]
    workspace.prepared = False

    # Allocate locks for spinlock-based all-reduce
    total_tiles = launch["total_tiles"]

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
    selector=None,
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
        selector: Optional pre-built tritonBLAS Origami selector.

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

    config.validate(world_size=world_size)

    # Prepare workspace if needed
    needs_prepare = (
        workspace is None
        or selector is not None
        or getattr(workspace, "launch_params", None) is None
        or not workspace.matches("matmul_all_reduce", (M, N, K), A.dtype, world_size, config.all_reduce_variant)
    )

    if needs_prepare:
        workspace = matmul_all_reduce_preamble(
            shmem,
            C,
            A,
            B,
            config=config,
            workspace=workspace,
            selector=selector,
            out_dtype=C.dtype,
        )

    # Get device context for RMA
    device_context = shmem.get_device_context()

    config_launch_override = selector is None and config is not None and not _config_uses_default_gemm_tuning(config)
    if config_launch_override and not needs_prepare:
        launch = _config_launch_params(M, N, config, A.device)
    else:
        launch = workspace.launch_params

    block_size_m = launch["block_size_m"]
    block_size_n = launch["block_size_n"]
    block_size_k = launch["block_size_k"]
    total_tiles = launch["total_tiles"]

    if config_launch_override or getattr(workspace, "selector", None) is None:
        # Validate problem size against explicit FusedConfig block sizes.
        assert M >= block_size_m, f"M={M} too small for block_size_m={block_size_m}"
        assert K >= block_size_k, f"K={K} too small for block_size_k={block_size_k}"
        assert N >= block_size_n, f"N={N} too small for block_size_n={block_size_n}"

    # Validate that the pre-allocated lock array is large enough for the current tile count.
    # This can occur when the workspace was prepared with larger block sizes (fewer tiles)
    # and is then reused with smaller block sizes (more tiles).
    if workspace.locks is not None and workspace.locks.numel() < total_tiles:
        raise ValueError(
            f"Lock array too small: have {workspace.locks.numel()} but need {total_tiles}. "
            f"Pre-allocate workspace with the smallest block sizes you intend to use."
        )

    even_k = K % block_size_k == 0
    grid = (launch["num_sms"],)
    launch_kwargs = {
        "num_warps": launch["num_warps"],
        "matrix_instr_nonkdim": launch["matrix_instr_nonkdim"],
    }
    if launch["num_stages"] is not None:
        launch_kwargs["num_stages"] = launch["num_stages"]

    iris_launch(
        _fused_matmul_all_reduce_kernel,
        grid,
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
        block_size_m,
        block_size_n,
        block_size_k,
        launch["group_size_m"],
        launch["num_sms"],
        launch["num_xcds"],
        launch["chunk_size"],
        even_k,
        launch["allow_tf32"],
        config.all_reduce_variant,
        algorithm="matmul_all_reduce",
        rank=rank,
        dtype=A.dtype,
        **launch_kwargs,
    )

    # Mark workspace as used
    if workspace is not None:
        workspace.prepared = False

    # Barrier unless async
    if not async_op:
        shmem.barrier()

    return workspace
