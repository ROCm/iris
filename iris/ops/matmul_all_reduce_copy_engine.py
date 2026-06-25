# SPDX-License-Identifier: MIT
# Copyright (c) 2025 Advanced Micro Devices, Inc. All rights reserved.

"""
High-level API for fused matrix multiplication and all-reduce.

This module provides a torch-like interface for GEMM+All-Reduce operations,
automatically inferring dimensions, strides, and hardware parameters.
"""

import logging
import time
from typing import Optional
import torch
import triton
import triton.language as tl
from xio import sdma_ep

from tritonblas.kernels.stages import GemmContext, ScheduleContext, make_input_view

from .config import FusedConfig
from .workspace import FusedWorkspace
import iris
from iris.host.tracing.kernel_artifacts import iris_launch
from .tritonblas_launch_wave_schedule import chiplet_transform_chunked, grouped_tile_coords


@triton.jit()
def _fused_matmul_all_reduce_copy_engine_kernel(
    A,
    B,
    C,
    local_aux_buffer,
    reduce_buffer,
    locks,
    M,
    N,
    K,
    stride_am,
    stride_ak,
    stride_bk,
    stride_bn,
    stride_local_aux_m,
    stride_local_aux_n,
    stride_reduce_m,
    stride_reduce_n,
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
        cache_modifier_a=None,
        cache_modifier_b=None,
        acc_dtype=acc_dtype,
        even_k=EVEN_K,
        allow_tf32=ALLOW_TF32,
    )
    sched = ScheduleContext(M, N, K, gemm_ctx)
    tensorA = make_input_view(A, M, K, stride_am, stride_ak)
    tensorB = make_input_view(B, K, N, stride_bk, stride_bn)

    start, total, stride = sched.persistent_tile_range()
    for tile_idx in range(start, total, stride):
        out_tile = sched.get_tile_from_idx(tile_idx)
        acc = gemm_ctx.reduce_axis(tensorA, tensorB, out_tile)
        rm, rn = out_tile.indices()
        c = acc.to(C.type.element_ty)
        mask = (rm[:, None] < M) & (rn[None, :] < N)

        if VARIANT == "two_shot":
            # Row-shard reduce-scatter phase. Split the GEMM tile by row shard
            # so the follow-up kernel can use ctx.all_gather(dim=0), which
            # assumes each rank owns dst_view.M // world_size rows.
            rows_per_rank = M // world_size
            local_ptr = local_aux_buffer + rm[:, None] * stride_local_aux_m + rn[None, :] * stride_local_aux_n
            tl.store(local_ptr, c, mask=mask, cache_modifier=".wt")

            for owner_rank in range(world_size):
                owner_start_m = owner_rank * rows_per_rank
                owner_end_m = owner_start_m + rows_per_rank
                owner_row_mask = (rm >= owner_start_m) & (rm < owner_end_m)
                if tl.max(owner_row_mask & (rm < M)):
                    local_rm = rm - owner_start_m
                    staged_rm = cur_rank * rows_per_rank + tl.where(owner_row_mask, local_rm, 0)
                    owner_mask = mask & owner_row_mask[:, None]

                    # The owner rank does not enqueue an SDMA self-copy. Put
                    # its local contribution directly into the receiver-side
                    # reduction buffer.
                    reduce_ptr = (
                        reduce_buffer
                        + staged_rm[:, None] * stride_reduce_m
                        + rn[None, :] * stride_reduce_n
                    )
                    tl.store(reduce_ptr, c, mask=owner_mask & (owner_rank == cur_rank), cache_modifier=".wt")

            # Release the persistent-wave flag after the local aux writes are
            # visible. Host-enqueued SDMA commands poll these flags, so the
            # copy engine can move completed waves without waiting for the
            # whole GEMM kernel to finish.
            tl.debug_barrier()
            wave_id = (tile_idx - start) // stride
            tl.atomic_add(locks + wave_id, 1, sem="release", scope="sys")
        else:
            # Matmul + aux publish stub for one_shot experiments.
            temp_ptr = local_aux_buffer + rm[:, None] * stride_local_aux_m + rn[None, :] * stride_local_aux_n
            tl.store(temp_ptr, c, mask=mask, cache_modifier=".wt")


@triton.jit()
def _matmul_all_reduce_copy_engine_wait_completion_kernel(
    completion_signals,
    expected_value,
    cur_rank: tl.constexpr,
    world_size: tl.constexpr,
):
    src_rank = tl.program_id(0)
    if src_rank >= world_size or src_rank == cur_rank:
        return
    while tl.load(completion_signals + src_rank, cache_modifier=".cv", volatile=True) < expected_value:
        pass


@triton.jit()
def _matmul_all_reduce_copy_engine_reduce_scatter_kernel(
    C,
    aux_buffer,
    completion_signals,
    expected_completion_value,
    M,
    N,
    stride_aux_m,
    stride_aux_n,
    stride_cm,
    stride_cn,
    context_tensor: tl.tensor,
    cur_rank: tl.constexpr,
    world_size: tl.constexpr,
    GEMM_BLOCK_SIZE_M: tl.constexpr,
    GEMM_BLOCK_SIZE_N: tl.constexpr,
    REDUCE_BLOCK_SIZE_M: tl.constexpr,
    REDUCE_BLOCK_SIZE_N: tl.constexpr,
    WAIT_FOR_COMPLETION: tl.constexpr,
):
    """
    Reduce this rank's row band, then all-gather it to every rank.

    Host-posted SDMA commands signal one completion slot per source rank. Each
    reduce program waits before reading so the remote rows it consumes are
    visible without requiring a separate kernel launch.
    """
    if WAIT_FOR_COMPLETION:
        for wait_src_rank in range(world_size):
            if wait_src_rank != cur_rank:
                while tl.load(
                    completion_signals + wait_src_rank,
                    cache_modifier=".cv",
                    volatile=True,
                ) < expected_completion_value:
                    pass

    rows_per_rank = M // world_size
    num_reduce_pid_n = tl.cdiv(N, REDUCE_BLOCK_SIZE_N)

    pid = tl.program_id(0)
    local_pid_m = pid // num_reduce_pid_n
    pid_n = pid - local_pid_m * num_reduce_pid_n

    rn_base = pid_n * REDUCE_BLOCK_SIZE_N
    rn = rn_base + tl.arange(0, REDUCE_BLOCK_SIZE_N)
    src_rank = tl.arange(0, world_size)

    # Keep the tail mask for now so N does not need to be an exact multiple of
    # REDUCE_BLOCK_SIZE_N. If we later restrict benchmark configs, this can be
    # switched to the unmasked fast path from the mapping note.
    mask = (local_pid_m < rows_per_rank) & (rn < N)

    acc_dtype = tl.int32 if C.type.element_ty == tl.int8 else tl.float32
    src_rm = src_rank[:, None] * rows_per_rank + local_pid_m
    src_ptr = aux_buffer + src_rm * stride_aux_m + rn[None, :] * stride_aux_n
    load_mask = (src_rank[:, None] < world_size) & mask[None, :]
    partials = tl.load(src_ptr, mask=load_mask, other=0.0, cache_modifier=".cg").to(acc_dtype)
    reduced = tl.sum(partials, axis=0)

    ctx = iris.DeviceContext.initialize(context_tensor, cur_rank, world_size)
    dst_view = iris.make_tensor_view(C, M, N, stride_cm, stride_cn)
    tile_obj = iris.Tile(
        local_pid_m,
        pid_n,
        REDUCE_BLOCK_SIZE_M,
        REDUCE_BLOCK_SIZE_N,
        tl.expand_dims(reduced.to(C.type.element_ty), 0),
    )
    ctx.all_gather(tile_obj, dst_view, dim=0)


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


def _reduce_block_size_n(N: int) -> int:
    return min(1024, 1 << (N - 1).bit_length())


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


def _build_transfer_plan(M: int, N: int, world_size: int, launch: dict, element_size: int) -> dict:
    block_size_m = launch["block_size_m"]
    block_size_n = launch["block_size_n"]
    rows_per_rank = M // world_size
    num_sms = launch["num_sms"]
    num_xcds = launch["num_xcds"]
    chunk_size = launch["chunk_size"]
    num_tiles_m = launch["num_tiles_m"]
    num_tiles_n = launch["num_tiles_n"]
    total_tiles = launch["total_tiles"]
    group_size_m = launch["group_size_m"]
    num_waves = (total_tiles + num_sms - 1) // num_sms

    transfers_by_owner_wave = [[[] for _ in range(num_waves)] for _ in range(world_size)]
    wave_tile_counts = [0 for _ in range(num_waves)]

    for launch_pid in range(num_sms):
        tile_id = chiplet_transform_chunked(launch_pid, num_sms, num_xcds, chunk_size)
        wave_id = 0
        while tile_id < total_tiles:
            pid_m, pid_n, _ = grouped_tile_coords(
                tile_id,
                num_tiles_m,
                num_tiles_n,
                group_size_m,
                num_xcds,
                chunk_size,
            )
            tile_m_start = pid_m * block_size_m
            tile_m_end = min(tile_m_start + block_size_m, M)
            col = pid_n * block_size_n
            width = min(block_size_n, N - col)
            wave_tile_counts[wave_id] += 1

            for owner_rank in range(world_size):
                owner_start = owner_rank * rows_per_rank
                owner_end = owner_start + rows_per_rank
                seg_start = max(tile_m_start, owner_start)
                seg_end = min(tile_m_end, owner_end)
                if seg_start < seg_end:
                    local_m = seg_start - owner_start
                    transfers_by_owner_wave[owner_rank][wave_id].append(
                        (local_m, col, width * element_size, seg_end - seg_start)
                    )
            tile_id += num_sms
            wave_id += 1

    wave_transfer_offsets = []
    wave_transfer_counts = []
    owner_last_wave = []
    transfer_row_offsets = []
    transfer_col_offsets = []
    transfer_width_bytes = []
    transfer_heights = []
    max_rects_per_owner_wave = 0
    running_offset = 0

    for owner_rank in range(world_size):
        last_wave = -1
        for wave_id, transfers in enumerate(transfers_by_owner_wave[owner_rank]):
            if transfers:
                last_wave = wave_id
        owner_last_wave.append(last_wave)

        for transfers in transfers_by_owner_wave[owner_rank]:
            wave_transfer_offsets.append(running_offset)
            wave_transfer_counts.append(len(transfers))
            max_rects_per_owner_wave = max(max_rects_per_owner_wave, len(transfers))
            for row, col, width_bytes, height in transfers:
                transfer_row_offsets.append(row)
                transfer_col_offsets.append(col)
                transfer_width_bytes.append(width_bytes)
                transfer_heights.append(height)
            running_offset += len(transfers)

    return {
        "num_transfer_waves": num_waves,
        "transfers_by_owner_wave": transfers_by_owner_wave,
        "wave_tile_counts": wave_tile_counts,
        "wave_transfer_offsets": wave_transfer_offsets,
        "wave_transfer_counts": wave_transfer_counts,
        "owner_last_wave": owner_last_wave,
        "transfer_row_offsets": transfer_row_offsets,
        "transfer_col_offsets": transfer_col_offsets,
        "transfer_width_bytes": transfer_width_bytes,
        "transfer_heights": transfer_heights,
        "max_rects_per_owner_wave": max(1, max_rects_per_owner_wave),
        "num_transfers": running_offset,
    }


def _ensure_transfer_workspace(
    shmem,
    workspace: FusedWorkspace,
    M: int,
    N: int,
    world_size: int,
    launch: dict,
    element_size: int,
    device: torch.device,
):
    plan_key = (
        M,
        N,
        world_size,
        launch["block_size_m"],
        launch["block_size_n"],
        launch["group_size_m"],
        launch["num_sms"],
        launch["num_xcds"],
        launch["chunk_size"],
    )
    if (
        getattr(workspace, "transfer_plan_key", None) != plan_key
        or getattr(workspace, "transfers_by_owner_wave", None) is None
    ):
        plan = _build_transfer_plan(M, N, world_size, launch, element_size)
        workspace.transfer_plan_key = plan_key
        workspace.num_transfer_waves = plan["num_transfer_waves"]
        workspace.max_rects_per_owner_wave = plan["max_rects_per_owner_wave"]
        workspace.num_transfers = plan["num_transfers"]
        workspace.transfers_by_owner_wave = plan["transfers_by_owner_wave"]
        workspace.wave_tile_counts_host = plan["wave_tile_counts"]
        workspace.owner_last_wave_host = plan["owner_last_wave"]
        workspace.wave_tile_counts = torch.tensor(plan["wave_tile_counts"], device=device, dtype=torch.int32)
        workspace.wave_transfer_offsets = torch.tensor(
            plan["wave_transfer_offsets"], device=device, dtype=torch.int32
        )
        workspace.wave_transfer_counts = torch.tensor(
            plan["wave_transfer_counts"], device=device, dtype=torch.int32
        )
        workspace.owner_last_wave = torch.tensor(plan["owner_last_wave"], device=device, dtype=torch.int32)
        workspace.transfer_row_offsets = torch.tensor(plan["transfer_row_offsets"], device=device, dtype=torch.int32)
        workspace.transfer_col_offsets = torch.tensor(plan["transfer_col_offsets"], device=device, dtype=torch.int32)
        workspace.transfer_width_bytes = torch.tensor(plan["transfer_width_bytes"], device=device, dtype=torch.int32)
        workspace.transfer_heights = torch.tensor(plan["transfer_heights"], device=device, dtype=torch.int32)

    if workspace.locks is None or workspace.locks.numel() != workspace.num_transfer_waves:
        workspace.locks = shmem.zeros((workspace.num_transfer_waves,), dtype=torch.int32)
    if getattr(workspace, "completion_signals", None) is None or workspace.completion_signals.numel() != world_size:
        workspace.completion_signals = shmem.zeros((world_size,), dtype=torch.int32)


def _post_host_copy_engine_transfers(
    shmem,
    workspace: FusedWorkspace,
    local_aux_buffer: torch.Tensor,
    reduce_buffer: torch.Tensor,
    rank: int,
    world_size: int,
    rows_per_rank: int,
    flag_iteration: int,
) -> float:
    """Queue host-side SDMA wait+copy packets for the two-shot reduce-scatter."""
    start = time.perf_counter()
    element_size = local_aux_buffer.element_size()
    stride_local_aux_m, stride_local_aux_n = local_aux_buffer.stride()
    stride_reduce_m, stride_reduce_n = reduce_buffer.stride()

    transfers_by_owner_wave = workspace.transfers_by_owner_wave
    wave_tile_counts = workspace.wave_tile_counts_host
    owner_last_wave = workspace.owner_last_wave_host

    local_aux_base = local_aux_buffer.data_ptr()
    reduce_base = reduce_buffer.data_ptr()
    signal_ptr_local = workspace.completion_signals.data_ptr() + rank * workspace.completion_signals.element_size()

    for dst_rank in range(world_size):
        if dst_rank == rank:
            continue

        dst_waves = transfers_by_owner_wave[dst_rank]
        for wave_id, wave_transfers in enumerate(dst_waves):
            if not wave_transfers:
                continue

            wait_value = (flag_iteration + 1) * wave_tile_counts[wave_id]
            wait_flag = workspace.locks.data_ptr() + wave_id * workspace.locks.element_size()
            signal_flag = None
            if wave_id == owner_last_wave[dst_rank]:
                signal_flag = shmem.heap.translate(signal_ptr_local, rank, dst_rank)

            tiles = []
            dst_ptrs = []
            dst_strides = []
            for row_offset, col_offset, width_bytes, height in wave_transfers:
                width_elems = width_bytes // element_size

                tile = sdma_ep.Tile()
                tile.pid_m = 0
                tile.pid_n = 0
                tile.block_m = height
                tile.block_n = width_elems
                tile.elem_size = element_size
                tile.src_stride = stride_local_aux_m * element_size

                src_offset = (
                    (dst_rank * rows_per_rank + row_offset) * stride_local_aux_m
                    + col_offset * stride_local_aux_n
                )
                tile.data = local_aux_base + src_offset * element_size

                dst_offset = (rank * rows_per_rank + row_offset) * stride_reduce_m + col_offset * stride_reduce_n
                dst_ptr_local = reduce_base + dst_offset * element_size

                tiles.append(tile)
                dst_ptrs.append(shmem.heap.translate(dst_ptr_local, rank, dst_rank))
                dst_strides.append(stride_reduce_m * element_size)

            shmem.put_tiles(
                tiles,
                dst_rank=dst_rank,
                dst_ptrs=dst_ptrs,
                dst_strides=dst_strides,
                wait_flag=wait_flag,
                wait_value=wait_value,
                signal_flag=signal_flag,
                signal_value=1,
                async_op=True,
                channel=0,
            )

    return (time.perf_counter() - start) * 1000.0


def matmul_all_reduce_copy_engine_prepost_transfers(
    shmem,
    A: torch.Tensor,
    B: torch.Tensor,
    workspace: FusedWorkspace,
    flag_iteration: int = 0,
) -> float:
    """Queue two_shot SDMA wait+copy packets before the timed GEMM launch."""
    if workspace is None:
        raise ValueError("workspace is required when preposting copy-engine transfers")
    if workspace.variant != "two_shot":
        return 0.0
    if workspace.aux_buffer is None or workspace.a_inbox is None:
        raise ValueError("two_shot workspace must have aux_buffer and a_inbox before preposting")

    M, _ = A.shape
    N = B.shape[1]
    world_size = shmem.get_num_ranks()
    launch = workspace.launch_params
    if launch is None:
        raise ValueError("workspace.launch_params must be initialized before preposting")

    _ensure_transfer_workspace(shmem, workspace, M, N, world_size, launch, A.element_size(), A.device)
    return _post_host_copy_engine_transfers(
        shmem,
        workspace,
        workspace.aux_buffer,
        workspace.a_inbox,
        shmem.get_rank(),
        world_size,
        M // world_size,
        flag_iteration,
    )


def _selector_active_cus(selector, device: torch.device) -> int:
    active_cus = getattr(selector, "_ACTIVE_CU", None)
    if active_cus is None or active_cus <= 0:
        props = torch.cuda.get_device_properties(device)
        active_cus = props.multi_processor_count
    return int(active_cus)


def _matmul_all_reduce_copy_engine_launch_params(
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
        "reduce_block_size_m": 1,
        "reduce_block_size_n": _reduce_block_size_n(N),
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
        "reduce_block_size_m": 1,
        "reduce_block_size_n": _reduce_block_size_n(N),
    }


def matmul_all_reduce_copy_engine_preamble(
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
    Allocate and reset temporary buffers for matmul_all_reduce_copy_engine.

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

    if config.all_reduce_variant == "two_shot" and M % world_size != 0:
        raise ValueError(
            "matmul_all_reduce_copy_engine two_shot requires M to be divisible by world_size "
            "because the final all-gather uses equal row shards."
        )

    if selector is not None:
        launch = _matmul_all_reduce_copy_engine_launch_params(
            M, N, K, selector, A.device, A.element_size(), config.all_reduce_variant
        )
    elif _config_uses_default_gemm_tuning(config):
        c_dtype = dtype if out_dtype is None else out_dtype
        selector = _make_origami_selector(M, N, K, A, B, c_dtype)
        launch = _matmul_all_reduce_copy_engine_launch_params(
            M, N, K, selector, A.device, A.element_size(), config.all_reduce_variant
        )
    else:
        launch = _config_launch_params(M, N, config, A.device)

    if workspace is None:
        workspace = FusedWorkspace()

    workspace.operation = "matmul_all_reduce_copy_engine"
    workspace.shape = (M, N, K)
    workspace.dtype = dtype
    workspace.world_size = world_size
    workspace.variant = config.all_reduce_variant
    workspace.selector = selector
    workspace.config = config
    workspace.launch_params = launch
    workspace.selector_fallback = launch["selector_fallback"]
    # workspace.prepared = False

    # Allocate auxiliary buffer for one_shot and two_shot to avoid race conditions
    # (GEMM results stored here, then reduced to final output)
    if config.all_reduce_variant in ["one_shot", "two_shot"]:
        aux_rows = M

        if workspace.aux_buffer is None or workspace.aux_buffer.shape != (aux_rows, N):
            workspace.aux_buffer = shmem.zeros((aux_rows, N), dtype=dtype)
        else:
            workspace.aux_buffer.zero_()

        if config.all_reduce_variant == "two_shot":
            if workspace.a_inbox is None or workspace.a_inbox.shape != (aux_rows, N):
                workspace.a_inbox = shmem.zeros((aux_rows, N), dtype=dtype)
            else:
                workspace.a_inbox.zero_()
            _ensure_transfer_workspace(shmem, workspace, M, N, world_size, launch, A.element_size(), A.device)
            workspace.locks.zero_()
            workspace.completion_signals.zero_()
        else:
            workspace.a_inbox = None
            if workspace.locks is None or workspace.locks.numel() != launch["total_tiles"]:
                workspace.locks = shmem.zeros((launch["total_tiles"],), dtype=torch.int32)
            else:
                workspace.locks.zero_()
    else:
        workspace.aux_buffer = None
        workspace.a_inbox = None
        workspace.locks = None
        workspace.completion_signals = None

    # Zero output tensor
    C.zero_()
    shmem.barrier()

    # workspace.prepared = True
    return workspace


def matmul_all_reduce_copy_engine(
    shmem,
    C: torch.Tensor,
    A: torch.Tensor,
    B: torch.Tensor,
    async_op: bool = False,
    config: Optional[FusedConfig] = None,
    workspace: Optional[FusedWorkspace] = None,
    selector=None,
    profile: Optional[dict] = None,
    flag_iteration: int = 0,
    copy_engine_transfers_preposted: bool = False,
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
        profile: Optional dict populated with GEMM and reduce/all-gather CUDA
            event timings plus host SDMA posting wall time. Enabling this forces
            event synchronization and is intended for instrumentation, not the
            hot benchmark path.
        flag_iteration: Launch generation for cumulative copy-engine wait and
            completion counters. Increment this when reusing a workspace without
            zeroing its synchronization buffers.
        copy_engine_transfers_preposted: If True, the two_shot SDMA wait+copy
            packets for this flag_iteration were already queued by the caller,
            usually from a benchmark preamble_fn outside the timed region.

    Returns:
        workspace: Updated workspace object (can be reused for subsequent calls)

    Example:
        >>> A = shmem.randn((1024, 512), dtype=torch.float16)
        >>> B = shmem.randn((512, 2048), dtype=torch.float16)
        >>> C = shmem.zeros((1024, 2048), dtype=torch.float16)
        >>> shmem.ops.matmul_all_reduce_copy_engine(C, A, B)
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
        "matmul_all_reduce_copy_engine: shape=(%d,%d,%d) dtype=%s variant=%s rank=%d/%d",
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
    if workspace is None:
        workspace = matmul_all_reduce_copy_engine_preamble(
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

    launch = workspace.launch_params

    block_size_m = launch["block_size_m"]
    block_size_n = launch["block_size_n"]
    block_size_k = launch["block_size_k"]
    total_tiles = launch["total_tiles"]

    # if config_launch_override or getattr(workspace, "selector", None) is None:
    #     # Validate problem size against explicit FusedConfig block sizes.
    #     assert M >= block_size_m, f"M={M} too small for block_size_m={block_size_m}"
    #     assert K >= block_size_k, f"K={K} too small for block_size_k={block_size_k}"
    #     assert N >= block_size_n, f"N={N} too small for block_size_n={block_size_n}"

    if config.all_reduce_variant == "two_shot":
        _ensure_transfer_workspace(shmem, workspace, M, N, world_size, launch, A.element_size(), A.device)
        required_locks = workspace.num_transfer_waves
    else:
        required_locks = total_tiles

    # Validate that the pre-allocated lock array is large enough for the current
    # synchronization scheme. The two_shot copy-engine path uses one flag per
    # persistent GEMM wave; the legacy paths use one flag per output tile.
    if workspace.locks is not None and workspace.locks.numel() < required_locks:
        raise ValueError(
            f"Lock array too small: have {workspace.locks.numel()} but need {required_locks}. "
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

    stride_local_aux_m, stride_local_aux_n = workspace.aux_buffer.stride()
    reduce_buffer = workspace.a_inbox if config.all_reduce_variant == "two_shot" else workspace.aux_buffer
    stride_reduce_m, stride_reduce_n = reduce_buffer.stride()

    if profile is not None:
        profile.clear()
        gemm_start = torch.cuda.Event(enable_timing=True)
        gemm_end = torch.cuda.Event(enable_timing=True)
        copy_wait_start = torch.cuda.Event(enable_timing=True)
        copy_wait_end = torch.cuda.Event(enable_timing=True)
        reduce_start = torch.cuda.Event(enable_timing=True)
        reduce_end = torch.cuda.Event(enable_timing=True)
    else:
        gemm_start = gemm_end = copy_wait_start = copy_wait_end = reduce_start = reduce_end = None

    if profile is not None:
        gemm_start.record()
    iris_launch(
        _fused_matmul_all_reduce_copy_engine_kernel,
        grid,
        A,
        B,
        C,
        workspace.aux_buffer,
        reduce_buffer,
        workspace.locks,
        M,
        N,
        K,
        stride_am,
        stride_ak,
        stride_bk,
        stride_bn,
        stride_local_aux_m,
        stride_local_aux_n,
        stride_reduce_m,
        stride_reduce_n,
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
        algorithm="matmul_all_reduce_copy_engine",
        rank=rank,
        dtype=A.dtype,
        **launch_kwargs,
    )

    if profile is not None:
        gemm_end.record()

    host_post_ms = 0.0
    if config.all_reduce_variant == "two_shot":
        if not copy_engine_transfers_preposted:
            host_post_ms = matmul_all_reduce_copy_engine_prepost_transfers(
                shmem,
                A,
                B,
                workspace,
                flag_iteration,
            )

        reduce_block_size_m = launch["reduce_block_size_m"]
        reduce_block_size_n = launch["reduce_block_size_n"]
        rows_per_rank = M // world_size
        reduce_tiles_m = (rows_per_rank + reduce_block_size_m - 1) // reduce_block_size_m
        reduce_tiles_n = (N + reduce_block_size_n - 1) // reduce_block_size_n
        reduce_grid = (reduce_tiles_m * reduce_tiles_n,)

        wait_in_reduce_kernel = profile is None
        if profile is not None:
            copy_wait_start.record()
            iris_launch(
                _matmul_all_reduce_copy_engine_wait_completion_kernel,
                (world_size,),
                workspace.completion_signals,
                flag_iteration + 1,
                rank,
                world_size,
                algorithm="matmul_all_reduce_copy_engine_wait_completion_profile",
                rank=rank,
                dtype=A.dtype,
                num_warps=8,
            )
            copy_wait_end.record()
            reduce_start.record()

        iris_launch(
            _matmul_all_reduce_copy_engine_reduce_scatter_kernel,
            reduce_grid,
            C,
            reduce_buffer,
            workspace.completion_signals,
            flag_iteration + 1,
            M,
            N,
            stride_reduce_m,
            stride_reduce_n,
            stride_cm,
            stride_cn,
            device_context,
            rank,
            world_size,
            block_size_m,
            block_size_n,
            reduce_block_size_m,
            reduce_block_size_n,
            wait_in_reduce_kernel,
            algorithm="matmul_all_reduce_copy_engine_reduce_scatter",
            rank=rank,
            dtype=A.dtype,
            num_warps=8,
        )

        if profile is not None:
            reduce_end.record()

    if profile is not None:
        if config.all_reduce_variant == "two_shot":
            reduce_end.synchronize()
            profile["host_post_ms"] = host_post_ms
            profile["copy_completion_wait_ms"] = copy_wait_start.elapsed_time(copy_wait_end)
            profile["reduce_allgather_no_wait_ms"] = reduce_start.elapsed_time(reduce_end)
            profile["reduce_allgather_ms"] = copy_wait_start.elapsed_time(reduce_end)
        else:
            gemm_end.synchronize()
            profile["host_post_ms"] = 0.0
            profile["copy_completion_wait_ms"] = 0.0
            profile["reduce_allgather_no_wait_ms"] = 0.0
            profile["reduce_allgather_ms"] = 0.0
        profile["gemm_ms"] = gemm_start.elapsed_time(gemm_end)
        profile["total_profiled_ms"] = profile["gemm_ms"] + profile["reduce_allgather_ms"]

    # Mark workspace as used
    # if workspace is not None:
    #     workspace.prepared = False

    # Barrier unless async
    if not async_op:
        shmem.barrier()

    return workspace
