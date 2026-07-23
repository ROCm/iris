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

import iris
from iris.host.tracing.kernel_artifacts import iris_launch
from tritonblas.kernels.stages import GemmContext, Tile, make_input_view
from tritonblas.matmul import _make_matmul_selector

from .config import FusedConfig
from .workspace import FusedWorkspace


_SUPPORTED_VARIANTS = ("one_shot", "two_shot")
_MAX_ONE_SHOT_INBOX_ELEMENTS = 2**31


@triton.jit()
def _matmul_all_reduce_gemm_publish_kernel(
    A,
    B,
    a_inbox,
    locks,
    M,
    N,
    K,
    stride_am,
    stride_ak,
    stride_bk,
    stride_bn,
    stride_inbox_m,
    stride_inbox_n,
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
    Partitioned-XCD GEMM that publishes partials directly into the rank-major inbox.

    one_shot:
      a_inbox[src_rank * M + row, col] on every rank.

    two_shot:
      on owner rank, a_inbox[src_rank * rows_per_rank + owner_local_row, col].
    """
    acc_dtype = tl.int32 if a_inbox.type.element_ty == tl.int8 else tl.float32
    gemm_ctx = GemmContext(
        BLOCK_SIZE_M,
        BLOCK_SIZE_N,
        BLOCK_SIZE_K,
        num_sms=NUM_SMS,
        num_xcds=NUM_XCDS,
        group_size_m=GROUP_SIZE_M,
        chunk_size=CHUNK_SIZE,
        cache_modifier_a=None,
        cache_modifier_b=None,
        acc_dtype=acc_dtype,
        even_k=EVEN_K,
        allow_tf32=ALLOW_TF32,
    )
    tensorA = make_input_view(A, M, K, stride_am, stride_ak)
    tensorB = make_input_view(B, K, N, stride_bk, stride_bn)
    ctx = iris.DeviceContext.initialize(context_tensor, cur_rank, world_size)

    if VARIANT == "one_shot":
        inbox_view = iris.make_tensor_view(a_inbox, M * world_size, N, stride_inbox_m, stride_inbox_n)

    launch_pid = tl.program_id(0)
    xcd_id = launch_pid % NUM_XCDS
    local_pid = launch_pid // NUM_XCDS
    sms_per_xcd = NUM_SMS // NUM_XCDS

    num_pid_n = tl.cdiv(N, BLOCK_SIZE_N)
    partition_rows = tl.cdiv(M, world_size)

    for partition_id in range(world_size):
        if partition_id % NUM_XCDS == xcd_id:
            partition_m_start = partition_id * partition_rows
            partition_m_end = tl.minimum(partition_m_start + partition_rows, M)
            first_pid_m = partition_m_start // BLOCK_SIZE_M
            last_pid_m = tl.cdiv(partition_m_end, BLOCK_SIZE_M)
            partition_tiles_m = last_pid_m - first_pid_m
            partition_tiles = partition_tiles_m * num_pid_n

            for local_tile_id in range(local_pid, partition_tiles, sms_per_xcd):
                num_pid_in_group = GROUP_SIZE_M * num_pid_n
                group_id = local_tile_id // num_pid_in_group
                first_group_pid_m = group_id * GROUP_SIZE_M
                group_size_m = tl.minimum(partition_tiles_m - first_group_pid_m, GROUP_SIZE_M)

                local_pid_m = first_group_pid_m + ((local_tile_id % num_pid_in_group) % group_size_m)
                pid_m = first_pid_m + local_pid_m
                pid_n = (local_tile_id % num_pid_in_group) // group_size_m

                tl.assume(pid_m >= 0)
                tl.assume(pid_n >= 0)

                out_tile = Tile(pid_m, pid_n, BLOCK_SIZE_M, BLOCK_SIZE_N)
                acc = gemm_ctx.reduce_axis(tensorA, tensorB, out_tile)
                rm, rn = out_tile.indices()
                mask = (
                    (rm[:, None] >= partition_m_start)
                    & (rm[:, None] < partition_m_end)
                    & (rm[:, None] < M)
                    & (rn[None, :] < N)
                )
                c = acc.to(a_inbox.type.element_ty)

                if VARIANT == "one_shot":
                    tile_obj = iris.Tile(pid_m, pid_n, BLOCK_SIZE_M, BLOCK_SIZE_N, c)
                    ctx.all_gather(tile_obj, inbox_view, dim=0, src_mask=mask)
                else:
                    rows_per_rank = M // world_size
                    local_rm = rm - partition_m_start
                    owner_row_mask = (rm >= partition_m_start) & (rm < partition_m_end)
                    inbox_rm = cur_rank * rows_per_rank + tl.where(owner_row_mask, local_rm, 0)
                    inbox_ptr = a_inbox + inbox_rm[:, None] * stride_inbox_m + rn[None, :] * stride_inbox_n
                    ctx.store(
                        inbox_ptr,
                        c,
                        to_rank=partition_id,
                        mask=mask & owner_row_mask[:, None],
                        hint=(1, BLOCK_SIZE_N),
                    )

            if local_pid < partition_tiles:
                tl.debug_barrier()
                if VARIANT == "one_shot":
                    for signal_rank in tl.static_range(0, world_size):
                        ctx.atomic_add(
                            locks,
                            1,
                            to_rank=signal_rank,
                            sem="release",
                            scope="sys",
                        )
                else:
                    ctx.atomic_add(
                        locks,
                        1,
                        to_rank=partition_id,
                        sem="release",
                        scope="sys",
                    )


@triton.jit()
def _matmul_all_reduce_reduce_scatter_kernel(
    C,
    a_inbox,
    locks,
    all_gather_local_arrivals,
    all_gather_rank_arrivals,
    expected_completion_value,
    barrier_generation,
    M,
    N,
    stride_inbox_m,
    stride_inbox_n,
    stride_cm,
    stride_cn,
    context_tensor: tl.tensor,
    cur_rank: tl.constexpr,
    world_size: tl.constexpr,
    REDUCE_BLOCK_SIZE_M: tl.constexpr,
    REDUCE_BLOCK_SIZE_N: tl.constexpr,
    GROUP_SIZE_M: tl.constexpr,
    NUM_SMS: tl.constexpr,
    REDUCE_NUM_PROGRAMS: tl.constexpr,
):
    """Reduce this rank's row shard, then all-gather it to every rank."""
    ctx = iris.DeviceContext.initialize(context_tensor, cur_rank, world_size)
    dst_view = iris.make_tensor_view(C, M, N, stride_cm, stride_cn)

    while tl.load(locks, cache_modifier=".cv", volatile=True) < expected_completion_value:
        pass

    pid = tl.program_id(0)
    rows_per_rank = M // world_size
    num_pid_m = tl.cdiv(rows_per_rank, REDUCE_BLOCK_SIZE_M)
    num_pid_n = tl.cdiv(N, REDUCE_BLOCK_SIZE_N)
    total_tiles = num_pid_m * num_pid_n
    acc_dtype = tl.int32 if C.type.element_ty == tl.int8 else tl.float32

    for tile_id in range(pid, total_tiles, NUM_SMS):
        num_pid_in_group = GROUP_SIZE_M * num_pid_n
        group_id = tile_id // num_pid_in_group
        first_pid_m = group_id * GROUP_SIZE_M
        group_size_m = tl.minimum(num_pid_m - first_pid_m, GROUP_SIZE_M)
        pid_m = first_pid_m + ((tile_id % num_pid_in_group) % group_size_m)
        pid_n = (tile_id % num_pid_in_group) // group_size_m

        tl.assume(pid_m >= 0)
        tl.assume(pid_n >= 0)

        rm_base = pid_m * REDUCE_BLOCK_SIZE_M
        rn_base = pid_n * REDUCE_BLOCK_SIZE_N

        rm = rm_base + tl.arange(0, REDUCE_BLOCK_SIZE_M)
        rm = tl.max_contiguous(tl.multiple_of(rm, REDUCE_BLOCK_SIZE_M), REDUCE_BLOCK_SIZE_M)
        rn = rn_base + tl.arange(0, REDUCE_BLOCK_SIZE_N)
        rn = tl.max_contiguous(tl.multiple_of(rn, REDUCE_BLOCK_SIZE_N), REDUCE_BLOCK_SIZE_N)
        mask = (rm[:, None] < rows_per_rank) & (rn[None, :] < N)

        acc = tl.zeros((REDUCE_BLOCK_SIZE_M, REDUCE_BLOCK_SIZE_N), dtype=acc_dtype)
        for reduce_src_rank in tl.static_range(0, world_size):
            src_rm = reduce_src_rank * rows_per_rank + rm
            src_ptr = a_inbox + src_rm[:, None] * stride_inbox_m + rn[None, :] * stride_inbox_n
            acc += tl.load(src_ptr, mask=mask, other=0.0).to(acc_dtype)

        tile_obj = iris.Tile(
            pid_m,
            pid_n,
            REDUCE_BLOCK_SIZE_M,
            REDUCE_BLOCK_SIZE_N,
            acc.to(C.type.element_ty),
        )
        ctx.all_gather(tile_obj, dst_view, dim=0, src_mask=mask)

    ctx.exit_barrier(
        all_gather_local_arrivals,
        all_gather_rank_arrivals,
        REDUCE_NUM_PROGRAMS,
        barrier_generation,
    )


@triton.jit()
def _matmul_all_reduce_local_reduce_kernel(
    C,
    a_inbox,
    locks,
    expected_publish_count,
    M,
    N,
    stride_inbox_m,
    stride_inbox_n,
    stride_cm,
    stride_cn,
    context_tensor: tl.tensor,
    cur_rank: tl.constexpr,
    world_size: tl.constexpr,
    REDUCE_BLOCK_SIZE_M: tl.constexpr,
    REDUCE_BLOCK_SIZE_N: tl.constexpr,
    GROUP_SIZE_M: tl.constexpr,
    NUM_SMS: tl.constexpr,
):
    """Local one-shot reduction after every rank has published its full partial output."""
    while tl.load(locks, cache_modifier=".cv", volatile=True) < expected_publish_count:
        pass

    pid = tl.program_id(0)
    num_pid_m = tl.cdiv(M, REDUCE_BLOCK_SIZE_M)
    num_pid_n = tl.cdiv(N, REDUCE_BLOCK_SIZE_N)
    total_tiles = num_pid_m * num_pid_n
    acc_dtype = tl.int32 if C.type.element_ty == tl.int8 else tl.float32

    for tile_id in range(pid, total_tiles, NUM_SMS):
        num_pid_in_group = GROUP_SIZE_M * num_pid_n
        group_id = tile_id // num_pid_in_group
        first_pid_m = group_id * GROUP_SIZE_M
        group_size_m = tl.minimum(num_pid_m - first_pid_m, GROUP_SIZE_M)
        pid_m = first_pid_m + ((tile_id % num_pid_in_group) % group_size_m)
        pid_n = (tile_id % num_pid_in_group) // group_size_m

        tl.assume(pid_m >= 0)
        tl.assume(pid_n >= 0)

        rm_base = pid_m * REDUCE_BLOCK_SIZE_M
        rn_base = pid_n * REDUCE_BLOCK_SIZE_N
        is_full = (rm_base + REDUCE_BLOCK_SIZE_M <= M) & (rn_base + REDUCE_BLOCK_SIZE_N <= N)

        rm = rm_base + tl.arange(0, REDUCE_BLOCK_SIZE_M)
        rm = tl.max_contiguous(tl.multiple_of(rm, REDUCE_BLOCK_SIZE_M), REDUCE_BLOCK_SIZE_M)
        rn = rn_base + tl.arange(0, REDUCE_BLOCK_SIZE_N)
        rn = tl.max_contiguous(tl.multiple_of(rn, REDUCE_BLOCK_SIZE_N), REDUCE_BLOCK_SIZE_N)

        out_ptr = C + rm[:, None] * stride_cm + rn[None, :] * stride_cn

        if is_full:
            acc = tl.zeros((REDUCE_BLOCK_SIZE_M, REDUCE_BLOCK_SIZE_N), dtype=acc_dtype)
            for reduce_src_rank in tl.static_range(0, world_size):
                src_rm = reduce_src_rank * M + rm
                src_ptr = a_inbox + src_rm[:, None] * stride_inbox_m + rn[None, :] * stride_inbox_n
                fast_src_ptr = tl.max_contiguous(
                    tl.multiple_of(src_ptr, (1, REDUCE_BLOCK_SIZE_N)),
                    (1, REDUCE_BLOCK_SIZE_N),
                )
                acc += tl.load(fast_src_ptr).to(acc_dtype)

            fast_out_ptr = tl.max_contiguous(
                tl.multiple_of(out_ptr, (1, REDUCE_BLOCK_SIZE_N)),
                (1, REDUCE_BLOCK_SIZE_N),
            )
            tl.store(fast_out_ptr, acc.to(C.type.element_ty))
        else:
            mask = (rm[:, None] < M) & (rn[None, :] < N)
            acc = tl.zeros((REDUCE_BLOCK_SIZE_M, REDUCE_BLOCK_SIZE_N), dtype=acc_dtype)
            for reduce_src_rank in tl.static_range(0, world_size):
                src_rm = reduce_src_rank * M + rm
                src_ptr = a_inbox + src_rm[:, None] * stride_inbox_m + rn[None, :] * stride_inbox_n
                acc += tl.load(src_ptr, mask=mask, other=0.0).to(acc_dtype)

            tl.store(out_ptr, acc.to(C.type.element_ty), mask=mask)


@triton.jit()
def _matmul_all_reduce_local_reduce_flat_kernel(
    C,
    a_inbox,
    locks,
    expected_publish_count,
    total_elements,
    context_tensor: tl.tensor,
    cur_rank: tl.constexpr,
    world_size: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
    NUM_SMS: tl.constexpr,
):
    """Fast one-shot local reduction for contiguous rank-major inbox buffers."""
    while tl.load(locks, cache_modifier=".cv", volatile=True) < expected_publish_count:
        pass

    pid = tl.program_id(0)
    total_blocks = total_elements // BLOCK_SIZE
    block_offsets = tl.arange(0, BLOCK_SIZE)
    block_offsets = tl.max_contiguous(tl.multiple_of(block_offsets, BLOCK_SIZE), BLOCK_SIZE)
    acc_dtype = tl.int32 if C.type.element_ty == tl.int8 else tl.float32

    for block_id in range(pid, total_blocks, NUM_SMS):
        linear_base = block_id * BLOCK_SIZE
        linear_offsets = linear_base + block_offsets
        acc = tl.zeros((BLOCK_SIZE,), dtype=acc_dtype)

        for reduce_src_rank in tl.static_range(0, world_size):
            src_offsets = reduce_src_rank * total_elements + linear_offsets
            src_ptr = a_inbox + src_offsets
            src_ptr = tl.max_contiguous(tl.multiple_of(src_ptr, BLOCK_SIZE), BLOCK_SIZE)
            acc += tl.load(src_ptr).to(acc_dtype)

        out_ptr = C + linear_offsets
        out_ptr = tl.max_contiguous(tl.multiple_of(out_ptr, BLOCK_SIZE), BLOCK_SIZE)
        tl.store(out_ptr, acc.to(C.type.element_ty))


def _validate_variant(variant: str):
    if variant not in _SUPPORTED_VARIANTS:
        raise ValueError(f"matmul_all_reduce supports only {_SUPPORTED_VARIANTS}, got {variant!r}")


def _validate_one_shot_inbox_size(M: int, N: int, world_size: int):
    inbox_elements = world_size * M * N
    if inbox_elements > _MAX_ONE_SHOT_INBOX_ELEMENTS:
        raise ValueError(
            "matmul_all_reduce one_shot requires world_size * M * N <= 2**31 elements "
            "because the current publish/reduce kernels use 32-bit element offsets; "
            f"got world_size={world_size}, M={M}, N={N}, elements={inbox_elements}."
        )


def _default_chunk_size(total_tiles: int, group_size_m: int, num_xcds: int) -> int:
    chunk_size = group_size_m * group_size_m
    if num_xcds > 0:
        chunk_size = min(chunk_size, max(1, total_tiles // num_xcds))
    return max(1, chunk_size)


def _round_up_to_multiple(value: int, multiple: int) -> int:
    if multiple <= 1:
        return max(1, value)
    return ((max(1, value) + multiple - 1) // multiple) * multiple


def _ceil_div(value: int, divisor: int) -> int:
    return (value + divisor - 1) // divisor


def _partitioned_xcd_gemm_num_sms(launch: dict, world_size: int) -> int:
    cached = launch.get("gemm_num_sms")
    if cached is not None:
        return int(cached)
    num_xcds = max(1, launch["num_xcds"])
    return _round_up_to_multiple(max(launch["num_sms"], min(num_xcds, world_size)), num_xcds)


def _partitioned_xcd_publish_counts(M: int, N: int, world_size: int, launch: dict) -> dict:
    block_size_m = launch["block_size_m"]
    block_size_n = launch["block_size_n"]
    num_xcds = max(1, launch["num_xcds"])
    gemm_num_sms = _partitioned_xcd_gemm_num_sms(launch, world_size)
    sms_per_xcd = max(1, gemm_num_sms // num_xcds)
    num_pid_n = _ceil_div(N, block_size_n)
    partition_rows = _ceil_div(M, world_size)

    publish_tiles = 0
    publish_programs_by_partition = []
    for partition_id in range(world_size):
        partition_m_start = partition_id * partition_rows
        partition_m_end = min(partition_m_start + partition_rows, M)
        first_pid_m = partition_m_start // block_size_m
        last_pid_m = _ceil_div(partition_m_end, block_size_m)
        partition_tiles_m = max(0, last_pid_m - first_pid_m)
        partition_tiles = partition_tiles_m * num_pid_n
        publish_tiles += partition_tiles
        publish_programs_by_partition.append(min(sms_per_xcd, partition_tiles))

    return {
        "tiles": publish_tiles,
        "programs": sum(publish_programs_by_partition),
        "programs_by_partition": publish_programs_by_partition,
    }


def _default_reduce_num_sms() -> int:
    return 64


def _reduce_block_size_n(N: int) -> int:
    return 64


def _make_origami_selector(M: int, N: int, K: int, A: torch.Tensor, B: torch.Tensor, C):
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


def _matmul_all_reduce_launch_params(M: int, N: int, K: int, selector, device: torch.device, variant: str) -> dict:
    block_size_m = selector.block_m
    block_size_n = selector.block_n
    block_size_k = selector.block_k
    group_size_m = selector.group_m
    num_stages = getattr(selector, "num_stages", 2)
    enable_selector_fallback = not getattr(selector, "disable_matmul_all_reduce_fallback", False)
    selector_fallback = False

    # Origami's 256x256 tile is good when GEMM dominates, but one_shot also
    # reduces every rank's full output. For shallow-K shapes, narrower N tiles
    # keep each reduction work item closer to the pre-refactor behavior.
    if (
        enable_selector_fallback
        and variant == "one_shot"
        and K < 16 * 1024
        and block_size_m == 256
        and block_size_n == 256
        and block_size_k <= 64
    ):
        block_size_n = 64
        block_size_k = 64
        if N == 8192 and 2048 < K < 3072:
            block_size_k = 32
            group_size_m = 1
        elif N >= 8 * K and K < 1024:
            group_size_m = 1
        num_stages = None
        selector_fallback = True

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
        "reduce_block_size_m": 32,
        "reduce_block_size_n": _reduce_block_size_n(N),
        "reduce_num_sms": _default_reduce_num_sms(),
        "selector_fallback": selector_fallback,
    }


def matmul_all_reduce_preamble(
    shmem,
    C: torch.Tensor,
    A: torch.Tensor,
    B: torch.Tensor,
    config: Optional[FusedConfig] = None,
    selector=None,
    out_dtype: Optional[torch.dtype] = None,
) -> FusedWorkspace:
    """Allocate and reset temporary buffers for matmul_all_reduce."""
    if config is None:
        config = FusedConfig()

    M, K = A.shape[:2]
    N = B.shape[1]
    dtype = A.dtype
    world_size = shmem.get_num_ranks()

    config.validate(world_size=world_size)
    _validate_variant(config.all_reduce_variant)

    if config.all_reduce_variant == "one_shot":
        _validate_one_shot_inbox_size(M, N, world_size)

    if config.all_reduce_variant == "two_shot" and M % world_size != 0:
        raise ValueError(
            "matmul_all_reduce two_shot requires M to be divisible by world_size "
            "because the final all-gather uses equal row shards."
        )

    if selector is None:
        c_dtype = dtype if out_dtype is None else out_dtype
        selector = _make_origami_selector(M, N, K, A, B, c_dtype)
    launch = _matmul_all_reduce_launch_params(M, N, K, selector, A.device, config.all_reduce_variant)
    launch["gemm_num_sms"] = _partitioned_xcd_gemm_num_sms(launch, world_size)
    publish_counts = _partitioned_xcd_publish_counts(M, N, world_size, launch)
    launch["publish_tiles"] = publish_counts["tiles"]
    launch["publish_programs"] = publish_counts["programs"]
    launch["publish_programs_by_partition"] = publish_counts["programs_by_partition"]
    if config.all_reduce_variant == "two_shot":
        launch["reduce_num_sms"] = _default_reduce_num_sms()
    else:
        launch["reduce_num_sms"] = launch["num_sms"]

    workspace = FusedWorkspace()

    workspace.operation = "matmul_all_reduce"
    workspace.shape = (M, N, K)
    workspace.dtype = dtype
    workspace.world_size = world_size
    workspace.variant = config.all_reduce_variant
    workspace.selector = selector
    workspace.config = config
    workspace.launch_params = launch

    inbox_rows = M * world_size if config.all_reduce_variant == "one_shot" else M
    workspace.a_inbox = shmem.zeros((inbox_rows, N), dtype=dtype)
    workspace.locks = shmem.zeros((1,), dtype=torch.int32)

    if config.all_reduce_variant == "two_shot":
        workspace.completion_signals = shmem.zeros((2,), dtype=torch.int32)
    else:
        workspace.completion_signals = None

    workspace.generation = 0
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
    Fused matrix multiplication and all-reduce.

    Computes C = all_reduce(A @ B) across all ranks. The one_shot variant
    publishes every rank's partial GEMM result to every rank and reduces
    locally. The two_shot variant stores row shards to owner ranks, reduces
    those shards, and stores the reduced result to every rank.
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

    rank = shmem.get_rank()
    world_size = shmem.get_num_ranks()

    config.validate(world_size=world_size)
    _validate_variant(config.all_reduce_variant)

    if config.all_reduce_variant == "one_shot":
        _validate_one_shot_inbox_size(M, N, world_size)

    if workspace is None:
        workspace = matmul_all_reduce_preamble(
            shmem,
            C,
            A,
            B,
            config=config,
            out_dtype=C.dtype,
        )

    launch = workspace.launch_params
    block_size_m = launch["block_size_m"]
    block_size_n = launch["block_size_n"]
    block_size_k = launch["block_size_k"]

    if config.all_reduce_variant == "two_shot" and M % world_size != 0:
        raise ValueError(
            "matmul_all_reduce two_shot requires M to be divisible by world_size "
            "because the final all-gather uses equal row shards."
        )
    generation = workspace.generation + 1
    workspace.generation = generation
    gemm_num_sms = _partitioned_xcd_gemm_num_sms(launch, world_size)
    launch["gemm_num_sms"] = gemm_num_sms
    if config.all_reduce_variant == "one_shot":
        if "publish_programs" not in launch:
            publish_counts = _partitioned_xcd_publish_counts(M, N, world_size, launch)
            launch["publish_tiles"] = publish_counts["tiles"]
            launch["publish_programs"] = publish_counts["programs"]
            launch["publish_programs_by_partition"] = publish_counts["programs_by_partition"]
        signals_per_generation = world_size * launch["publish_programs"]
    else:
        if "publish_programs_by_partition" not in launch:
            publish_counts = _partitioned_xcd_publish_counts(M, N, world_size, launch)
            launch["publish_tiles"] = publish_counts["tiles"]
            launch["publish_programs"] = publish_counts["programs"]
            launch["publish_programs_by_partition"] = publish_counts["programs_by_partition"]
        signals_per_generation = world_size * launch["publish_programs_by_partition"][rank]
    expected_publish_count = generation * signals_per_generation

    stride_am, stride_ak = A.stride()
    stride_bk, stride_bn = B.stride()
    stride_cm, stride_cn = C.stride()
    stride_inbox_m, stride_inbox_n = workspace.a_inbox.stride()

    even_k = K % block_size_k == 0
    launch_kwargs = {
        "num_warps": launch["num_warps"],
        "matrix_instr_nonkdim": launch["matrix_instr_nonkdim"],
    }
    if launch["num_stages"] is not None:
        launch_kwargs["num_stages"] = launch["num_stages"]

    iris_launch(
        _matmul_all_reduce_gemm_publish_kernel,
        (gemm_num_sms,),
        A,
        B,
        workspace.a_inbox,
        workspace.locks,
        M,
        N,
        K,
        stride_am,
        stride_ak,
        stride_bk,
        stride_bn,
        stride_inbox_m,
        stride_inbox_n,
        shmem.get_device_context(),
        rank,
        world_size,
        block_size_m,
        block_size_n,
        block_size_k,
        launch["group_size_m"],
        gemm_num_sms,
        launch["num_xcds"],
        launch["chunk_size"],
        even_k,
        launch["allow_tf32"],
        config.all_reduce_variant,
        algorithm="matmul_all_reduce_gemm_publish",
        rank=rank,
        dtype=A.dtype,
        **launch_kwargs,
    )

    if config.all_reduce_variant == "one_shot":
        reduce_block_size_m = 1
        reduce_block_size_n = 512
        num_sms = max(launch["reduce_num_sms"], launch["num_sms"] * 3)
        total_reduce_elements = M * N
        use_flat_local_reduce = (
            C.is_contiguous() and workspace.a_inbox.is_contiguous() and total_reduce_elements % reduce_block_size_n == 0
        )

        if use_flat_local_reduce:
            total_reduce_blocks = total_reduce_elements // reduce_block_size_n
            reduce_grid = (min(num_sms, total_reduce_blocks),)
            iris_launch(
                _matmul_all_reduce_local_reduce_flat_kernel,
                reduce_grid,
                C,
                workspace.a_inbox,
                workspace.locks,
                expected_publish_count,
                total_reduce_elements,
                shmem.get_device_context(),
                cur_rank=rank,
                world_size=world_size,
                BLOCK_SIZE=reduce_block_size_n,
                NUM_SMS=num_sms,
                algorithm="matmul_all_reduce_local_reduce_flat",
                rank=rank,
                dtype=A.dtype,
                num_warps=4,
                num_stages=2,
            )
        else:
            reduce_tiles_m = (M + reduce_block_size_m - 1) // reduce_block_size_m
            reduce_tiles_n = (N + reduce_block_size_n - 1) // reduce_block_size_n
            total_reduce_tiles = reduce_tiles_m * reduce_tiles_n
            reduce_grid = (min(num_sms, total_reduce_tiles),)
            iris_launch(
                _matmul_all_reduce_local_reduce_kernel,
                reduce_grid,
                C,
                workspace.a_inbox,
                workspace.locks,
                expected_publish_count,
                M,
                N,
                stride_inbox_m,
                stride_inbox_n,
                stride_cm,
                stride_cn,
                shmem.get_device_context(),
                cur_rank=rank,
                world_size=world_size,
                REDUCE_BLOCK_SIZE_M=reduce_block_size_m,
                REDUCE_BLOCK_SIZE_N=reduce_block_size_n,
                GROUP_SIZE_M=launch["group_size_m"],
                NUM_SMS=num_sms,
                algorithm="matmul_all_reduce_local_reduce",
                rank=rank,
                dtype=A.dtype,
                num_warps=4,
                num_stages=1,
            )
    else:
        rows_per_rank = M // world_size
        reduce_block_size_m = launch["reduce_block_size_m"]
        reduce_block_size_n = launch["reduce_block_size_n"]
        reduce_tiles_m = (rows_per_rank + reduce_block_size_m - 1) // reduce_block_size_m
        reduce_tiles_n = (N + reduce_block_size_n - 1) // reduce_block_size_n
        total_reduce_tiles = reduce_tiles_m * reduce_tiles_n
        num_sms = launch["reduce_num_sms"]
        reduce_grid = (min(num_sms, total_reduce_tiles),)
        all_gather_local_arrivals = workspace.completion_signals
        all_gather_rank_arrivals = workspace.completion_signals[1:]

        iris_launch(
            _matmul_all_reduce_reduce_scatter_kernel,
            reduce_grid,
            C,
            workspace.a_inbox,
            workspace.locks,
            all_gather_local_arrivals,
            all_gather_rank_arrivals,
            expected_publish_count,
            generation,
            M,
            N,
            stride_inbox_m,
            stride_inbox_n,
            stride_cm,
            stride_cn,
            shmem.get_device_context(),
            cur_rank=rank,
            world_size=world_size,
            REDUCE_BLOCK_SIZE_M=reduce_block_size_m,
            REDUCE_BLOCK_SIZE_N=reduce_block_size_n,
            GROUP_SIZE_M=launch["group_size_m"],
            NUM_SMS=num_sms,
            REDUCE_NUM_PROGRAMS=reduce_grid[0],
            algorithm="matmul_all_reduce_reduce_scatter",
            rank=rank,
            dtype=A.dtype,
            num_warps=4,
        )

    # Barrier unless async
    if not async_op:
        torch.cuda.synchronize()

    return workspace
