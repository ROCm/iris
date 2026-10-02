# SPDX-License-Identifier: MIT
# Copyright (c) 2025 Advanced Micro Devices, Inc. All rights reserved.

"""
Fused GEMM + All-Gather operation using SDMA (copy engine) for scatter.

Each rank has a row-sharded input A_local (M_local x K) and computes C_local = A_local @ B.
Then scatters C_local tiles to form the full C (M x N) where M = world_size * M_local.

This variant uses SDMA hardware for data movement instead of compute shader scatter.
"""

from typing import Optional
import time
import torch
import triton
import triton.language as tl
import iris

from .workspace import FusedWorkspace
from tritonblas.matmul import persistent_matmul_lt, create_counter_config
from .tritonblas_launch_wave_schedule import build_launch_wave_plan


@triton.jit()
def wait_cnt():
    tl.inline_asm_elementwise("s_waitcnt vmcnt(0)", "=r", [], dtype=tl.int32, is_pure=False, pack=1)


@triton.jit()
def _launch_wave_wait_poster_kernel(
    C_local_base,
    flags,
    completion_signals,
    flag_iteration,
    wave_tile_counts,
    wave_transfer_offsets,
    wave_transfer_counts,
    transfer_row_offsets,
    transfer_col_offsets,
    transfer_width_bytes,
    transfer_heights,
    heap_bases: tl.tensor,
    copy_engine_ctx: tl.tensor,
    cur_rank: tl.constexpr,
    world_size: tl.constexpr,
    NUM_WAVES: tl.constexpr,
    MAX_RECTS_PER_WAVE: tl.constexpr,
    SRC_PITCH: tl.constexpr,
    DST_PITCH: tl.constexpr,
    STRIDE_N: tl.constexpr,
):
    dst_rank = tl.program_id(0)
    if dst_rank >= world_size or dst_rank == cur_rank:
        return

    ptr_dtype = C_local_base.dtype.element_ty
    if ptr_dtype == tl.float16 or ptr_dtype == tl.bfloat16:
        elem_size = 2
    elif ptr_dtype == tl.float32 or ptr_dtype == tl.int32:
        elem_size = 4
    elif ptr_dtype == tl.float64 or ptr_dtype == tl.int64:
        elem_size = 8
    else:
        elem_size = 4

    for wave_id in range(NUM_WAVES):
        transfer_start = tl.load(wave_transfer_offsets + wave_id)
        transfer_count = tl.load(wave_transfer_counts + wave_id)
        if transfer_count != 0:
            wait_value = (flag_iteration + 1) * tl.load(wave_tile_counts + wave_id)
            is_last_wave = wave_id == (NUM_WAVES - 1)

            if is_last_wave:
                iris.wait_then_put_signal_rects(
                    C_local_base,
                    C_local_base,
                    cur_rank,
                    dst_rank,
                    heap_bases,
                    copy_engine_ctx,
                    flags + wave_id,
                    wait_value,
                    completion_signals + cur_rank,
                    1,
                    transfer_row_offsets,
                    transfer_col_offsets,
                    transfer_width_bytes,
                    transfer_heights,
                    transfer_start,
                    transfer_count,
                    STRIDE_N * elem_size,
                    SRC_PITCH * elem_size,
                    DST_PITCH * elem_size,
                    MAX_RECTS_PER_WAVE,
                )
            else:
                iris.wait_then_put_rects(
                    C_local_base,
                    C_local_base,
                    cur_rank,
                    dst_rank,
                    heap_bases,
                    copy_engine_ctx,
                    flags + wave_id,
                    wait_value,
                    transfer_row_offsets,
                    transfer_col_offsets,
                    transfer_width_bytes,
                    transfer_heights,
                    transfer_start,
                    transfer_count,
                    STRIDE_N * elem_size,
                    SRC_PITCH * elem_size,
                    DST_PITCH * elem_size,
                    MAX_RECTS_PER_WAVE,
                )


@triton.jit()
def _wait_completion_signals_kernel(
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


def _selector_wave_size(selector, device: torch.device) -> int:
    wave_size = getattr(selector, "_ACTIVE_CU", None)
    if wave_size is None or wave_size <= 0:
        wave_size = torch.cuda.get_device_properties(device).multi_processor_count
    return int(wave_size)


def _ensure_tritonblas_launch_wave_workspace(
    shmem,
    workspace: FusedWorkspace,
    selector,
    device: torch.device,
    m_local: int,
    n: int,
    world_size: int,
    elem_size: int,
):
    num_tiles_m = triton.cdiv(m_local, selector.block_m)
    num_tiles_n = triton.cdiv(n, selector.block_n)
    total_tiles = num_tiles_m * num_tiles_n
    wave_size = _selector_wave_size(selector, device)
    num_xcds = max(1, int(getattr(selector, "num_sms", 1)))
    plan_key = (num_tiles_m, num_tiles_n, selector.group_m, total_tiles, wave_size, num_xcds)

    if getattr(workspace, "launch_wave_plan_key", None) != plan_key:
        plan = build_launch_wave_plan(
            num_tiles_m=num_tiles_m,
            num_tiles_n=num_tiles_n,
            group_size_m=selector.group_m,
            launch_grid=total_tiles,
            wave_size=wave_size,
            num_xcds=num_xcds,
        )
        workspace.launch_wave_plan = plan
        workspace.launch_wave_plan_key = plan_key
        workspace.locks = shmem.zeros((plan.num_waves,), dtype=torch.int32)
        transfers_by_wave = [[] for _ in range(plan.num_waves)]
        for transfer in plan.transfers:
            transfers_by_wave[transfer.wave_id].append(transfer)

        wave_transfer_offsets = []
        wave_transfer_counts = []
        transfer_row_offsets = []
        transfer_col_offsets = []
        transfer_width_bytes = []
        transfer_heights = []
        max_rects_per_wave = 0
        running_offset = 0

        for wave_transfers in transfers_by_wave:
            wave_transfer_offsets.append(running_offset)
            wave_transfer_counts.append(len(wave_transfers))
            max_rects_per_wave = max(max_rects_per_wave, len(wave_transfers))
            for transfer in wave_transfers:
                row_offset = transfer.m_tile_start * selector.block_m
                col_offset = transfer.n_tile_start * selector.block_n
                batch_height = min(transfer.m_tile_count * selector.block_m, m_local - row_offset)
                batch_width = min(transfer.n_tile_count * selector.block_n, n - col_offset)
                transfer_row_offsets.append(row_offset)
                transfer_col_offsets.append(col_offset)
                transfer_width_bytes.append(batch_width * elem_size)
                transfer_heights.append(batch_height)
            running_offset += len(wave_transfers)

        workspace.wave_transfer_offsets = torch.tensor(wave_transfer_offsets, device=device, dtype=torch.int32)
        workspace.wave_transfer_counts = torch.tensor(wave_transfer_counts, device=device, dtype=torch.int32)
        workspace.transfer_row_offsets = torch.tensor(transfer_row_offsets, device=device, dtype=torch.int32)
        workspace.transfer_col_offsets = torch.tensor(transfer_col_offsets, device=device, dtype=torch.int32)
        workspace.transfer_width_bytes = torch.tensor(transfer_width_bytes, device=device, dtype=torch.int32)
        workspace.transfer_heights = torch.tensor(transfer_heights, device=device, dtype=torch.int32)
        workspace.wave_tile_counts = torch.tensor(plan.wave_tile_counts, device=device, dtype=torch.int32)
        workspace.num_tiles_m = num_tiles_m
        workspace.num_tiles_n = num_tiles_n
        workspace.num_waves = plan.num_waves
        workspace.num_batches = plan.num_waves
        workspace.num_transfers = len(plan.transfers)
        workspace.max_rects_per_wave = max_rects_per_wave
    if getattr(workspace, "completion_signals", None) is None or workspace.completion_signals.numel() != world_size:
        workspace.completion_signals = shmem.zeros((world_size,), dtype=torch.int32)
    return workspace.launch_wave_plan


def _auto_m_tiles_per_batch(selector, M_local: int, N: int) -> int:
    """Auto-calculate optimal m_tiles_per_batch based on selector and shape."""
    num_tiles_m = (M_local + selector.block_m - 1) // selector.block_m
    num_tiles_n = (N + selector.block_n - 1) // selector.block_n
    active_cus = getattr(selector, "_ACTIVE_CU", None)
    if active_cus is None or active_cus <= 0:
        active_cus = torch.cuda.get_device_properties(torch.cuda.current_device()).multi_processor_count
    tiles_per_group = max(1, selector.group_m * num_tiles_n)
    groups_per_wave = max(1, int(active_cus) // tiles_per_group)
    return max(1, min(num_tiles_m, groups_per_wave * selector.group_m))


def matmul_all_gather_copy_engine_preamble(
    shmem,
    A: torch.Tensor,
    B: torch.Tensor,
    m_tiles_per_batch: Optional[int] = None,
    selector=None,
) -> FusedWorkspace:
    """Allocate workspace for matmul_all_gather_copy_engine including per-batch flags."""
    from tritonblas.matmul import _make_matmul_selector

    M_local, K = A.shape
    K2, N = B.shape
    world_size = shmem.get_num_ranks()

    assert K == K2, f"Inner dimensions must match: A has K={K}, B has K={K2}"

    M = M_local * world_size

    # Create selector for block size configuration if not provided
    if selector is None:
        selector = _make_matmul_selector(
            M_local,
            N,
            K,
            A.dtype,
            B.dtype,
            A.dtype,
            A.device,
            streamk=False,
        )

    # Auto-calculate optimal m_tiles_per_batch if not provided
    if m_tiles_per_batch is None:
        m_tiles_per_batch = _auto_m_tiles_per_batch(selector, M_local, N)

    # Calculate number of tiles based on selector
    num_tiles_m = (M_local + selector.block_m - 1) // selector.block_m
    num_tiles_n = (N + selector.block_n - 1) // selector.block_n

    num_tiles = num_tiles_m * num_tiles_n

    # Calculate number of batches
    num_batches = (num_tiles_m + m_tiles_per_batch - 1) // m_tiles_per_batch

    ws = FusedWorkspace(
        operation="matmul_all_gather_copy_engine",
        shape=(M, N, K),
        dtype=A.dtype,
        world_size=world_size,
        prepared=True,
    )

    # Allocate one readiness counter per M-batch.
    ws.locks = shmem.zeros((num_batches,), dtype=torch.int32)
    ws.completion_signals = shmem.zeros((world_size,), dtype=torch.int32)

    # Store metadata for later use
    ws.selector = selector
    ws.num_tiles_m = num_tiles_m
    ws.num_tiles_n = num_tiles_n
    ws.num_batches = num_batches
    ws.m_tiles_per_batch = m_tiles_per_batch

    return ws


def matmul_all_gather_copy_engine(
    shmem,
    output_tensor: torch.Tensor,
    A: torch.Tensor,
    B: torch.Tensor,
    bias: Optional[torch.Tensor] = None,
    async_op: bool = False,
    workspace: Optional[FusedWorkspace] = None,
    flag_iteration: int = 0,
    verbose: bool = False,
) -> FusedWorkspace:
    """
    Fused matrix multiplication and all-gather using SDMA (copy engine) for scatter.

    Computes: output = all_gather(A @ B + bias) along M dimension

    Each rank has A of shape (M_local, K) where M_local = M / world_size.
    The operation computes C_local = A @ B on each rank and uses SDMA hardware
    to scatter the tiles to all ranks (all-gather pattern).

    Args:
        shmem: Iris shmem context
        output_tensor: Output tensor C of shape (M, N) where M = M_local * world_size
        A: Input matrix A of shape (M_local, K)
        B: Input matrix B of shape (K, N)
        bias: Optional bias vector (M_local,)
        async_op: If False, performs barrier at end
        workspace: Optional pre-allocated workspace
        flag_iteration: Launch generation for cumulative batch counters.
                        Batch readiness counters are not reset each iteration;
                        the poster waits for the generation-adjusted target.
        verbose: If True, print poster/main/quiet timing breakdown in sync mode

    Returns:
        FusedWorkspace object
    """
    M_local, K = A.shape
    K2, N = B.shape
    world_size = shmem.get_num_ranks()
    rank = shmem.get_rank()

    assert K == K2, f"Inner dimensions must match: A has K={K}, B has K={K2}"

    M = M_local * world_size
    assert output_tensor.shape == (M, N), f"Output must be ({M}, {N}), got {output_tensor.shape}"

    timing_events = None
    cpu_timing = None
    if verbose:
        current_stream = torch.cuda.current_stream(device=A.device)
        timing_events = {
            "stream": current_stream,
            "poster_start": torch.cuda.Event(enable_timing=True),
            "poster_end": torch.cuda.Event(enable_timing=True),
            "main_start": torch.cuda.Event(enable_timing=True),
            "main_end": torch.cuda.Event(enable_timing=True),
            "quiet_start": torch.cuda.Event(enable_timing=True),
            "quiet_end": torch.cuda.Event(enable_timing=True),
        }
        cpu_timing = {
            "total_start": time.perf_counter(),
            "poster_launch_ms": 0.0,
            "main_launch_ms": 0.0,
            "quiet_cpu_ms": 0.0,
            "sync_wait_ms": 0.0,
        }

    # Allocate workspace if not provided
    if workspace is None:
        workspace = matmul_all_gather_copy_engine_preamble(shmem, A, B)

    stride_cm, stride_cn = output_tensor.stride()

    selector = workspace.selector

    launch_wave_plan = _ensure_tritonblas_launch_wave_workspace(
        shmem,
        workspace,
        selector,
        A.device,
        M_local,
        N,
        world_size,
        output_tensor.element_size(),
    )

    # Get metadata from workspace after any schedule planning.
    num_batches = workspace.num_batches

    if timing_events is not None:
        timing_events["poster_start"].record(timing_events["stream"])
    poster_launch_start = time.perf_counter() if cpu_timing is not None else None
    poster_grid = (world_size,)
    c_local_base = output_tensor[rank * M_local :, :]
    _launch_wave_wait_poster_kernel[poster_grid](
        c_local_base,
        workspace.locks,
        workspace.completion_signals,
        flag_iteration,
        workspace.wave_tile_counts,
        workspace.wave_transfer_offsets,
        workspace.wave_transfer_counts,
        workspace.transfer_row_offsets,
        workspace.transfer_col_offsets,
        workspace.transfer_width_bytes,
        workspace.transfer_heights,
        shmem.get_heap_bases(),
        shmem.get_copy_engine_ctx(),
        rank,
        world_size,
        workspace.num_waves,
        workspace.max_rects_per_wave,
        stride_cm,
        stride_cm,
        stride_cn,
    )
    if cpu_timing is not None:
        cpu_timing["poster_launch_ms"] = (time.perf_counter() - poster_launch_start) * 1000.0
    if timing_events is not None:
        timing_events["poster_end"].record(timing_events["stream"])

    # Launch GEMM after poster submission so SDMA can wait autonomously.
    if timing_events is not None:
        timing_events["main_start"].record(timing_events["stream"])
    main_launch_start = time.perf_counter() if cpu_timing is not None else None
    if bias is not None:
        import warnings

        warnings.warn(
            "Bias is not yet supported in the tritonBLAS SignalView path for "
            "matmul_all_gather_copy_engine. Ignoring bias for this launch."
        )

    counter_config = create_counter_config(
        workspace.locks,
        map_type="launch_wave",
        block_group_m=launch_wave_plan.wave_size,
    )
    c_local_view = output_tensor[rank * M_local : (rank + 1) * M_local, :]
    persistent_matmul_lt(
        A,
        B,
        c_local_view,
        selector,
        bias=None,
        work_stealing=False,
        counter_config=counter_config,
    )
    if cpu_timing is not None:
        cpu_timing["main_launch_ms"] = (time.perf_counter() - main_launch_start) * 1000.0
    if timing_events is not None:
        timing_events["main_end"].record(timing_events["stream"])

    if not async_op:
        wait_cpu_start = time.perf_counter() if cpu_timing is not None else None
        if timing_events is not None:
            timing_events["quiet_start"].record(timing_events["stream"])
        _wait_completion_signals_kernel[(world_size,)](
            workspace.completion_signals,
            flag_iteration + 1,
            rank,
            world_size,
        )
        if timing_events is not None:
            timing_events["quiet_end"].record(timing_events["stream"])
        sync_wait_start = time.perf_counter() if cpu_timing is not None else None
        torch.cuda.synchronize()
        if cpu_timing is not None:
            cpu_timing["sync_wait_ms"] = (time.perf_counter() - sync_wait_start) * 1000.0
            cpu_timing["quiet_cpu_ms"] = (time.perf_counter() - wait_cpu_start) * 1000.0

        if verbose and rank == 0:
            poster_ms = timing_events["poster_start"].elapsed_time(timing_events["poster_end"])
            main_ms = timing_events["main_start"].elapsed_time(timing_events["main_end"])
            quiet_ms = timing_events["quiet_start"].elapsed_time(timing_events["quiet_end"])
            gpu_total_ms = poster_ms + main_ms + quiet_ms
            cpu_launch_total_ms = cpu_timing["poster_launch_ms"] + cpu_timing["main_launch_ms"]
            cpu_total_ms = (time.perf_counter() - cpu_timing["total_start"]) * 1000.0
            tile_transfer_count = (world_size - 1) * getattr(workspace, "num_transfers", num_batches)
            shmem.info(
                f"[Rank {rank}] Device copy-engine GPU timing. "
                f"Poster: {poster_ms:.2f}ms, Main: {main_ms:.2f}ms, Wait: {quiet_ms:.2f}ms, "
                f"GPU total: {gpu_total_ms:.2f}ms"
            )
            shmem.info(
                f"[Rank {rank}] Device copy-engine CPU timing. "
                f"Poster launch: {cpu_timing['poster_launch_ms']:.2f}ms, "
                f"Main launch: {cpu_timing['main_launch_ms']:.2f}ms, "
                f"Launch total: {cpu_launch_total_ms:.2f}ms, "
                f"Wait: {cpu_timing['quiet_cpu_ms']:.2f}ms, "
                f"Barrier wait: {cpu_timing['sync_wait_ms']:.2f}ms, "
                f"CPU total: {cpu_total_ms:.2f}ms, "
                f"transfers={tile_transfer_count}"
            )

    return workspace
