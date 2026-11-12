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
from tritonblas.config import EpilogueConfig
from tritonblas.matmul import persistent_matmul_lt
from .tritonblas_tile_schedule import build_launch_wave_signal_plan


@triton.jit()
def wait_cnt():
    tl.inline_asm_elementwise("s_waitcnt vmcnt(0)", "=r", [], dtype=tl.int32, is_pure=False, pack=1)


@triton.jit()
def _signal_batch_wait_submit_kernel(
    C_local_base,
    flags,
    completion_signals,
    flag_iteration,
    batch_signal_ids,
    batch_producers,
    batch_transfer_offsets,
    batch_transfer_counts,
    transfer_row_offsets,
    transfer_col_offsets,
    transfer_width_bytes,
    transfer_heights,
    heap_bases: tl.tensor,
    copy_engine_ctx: tl.tensor,
    cur_rank: tl.constexpr,
    world_size: tl.constexpr,
    NUM_BATCHES: tl.constexpr,
    MAX_TRANSFERS_PER_BATCH: tl.constexpr,
    SRC_PITCH: tl.constexpr,
    DST_PITCH: tl.constexpr,
    STRIDE_N: tl.constexpr,
):
    """Submit one SDMA wait-then-copy(+signal) queue entry per signal batch to each peer.

    "Submit" means writing the packets into the SDMA queue and ringing the doorbell; the copy engine
    then waits on the batch's gate and copies autonomously, without a workgroup. Batches are queued in
    the readiness order tritonBLAS publishes, so no gate blocks the queue behind a later one.
    """
    dst_rank = tl.program_id(0)
    if dst_rank >= world_size or dst_rank == cur_rank:
        return

    ptr_dtype = C_local_base.dtype.element_ty
    elem_size = ptr_dtype.itemsize
    tl.static_assert(ptr_dtype.itemsize > 0, "the copy-engine ops do not support sub-byte element types")

    for batch_id in range(NUM_BATCHES):
        transfer_start = tl.load(batch_transfer_offsets + batch_id)
        transfer_count = tl.load(batch_transfer_counts + batch_id)
        if transfer_count != 0:
            signal_id = tl.load(batch_signal_ids + batch_id)
            wait_value = (flag_iteration + 1) * tl.load(batch_producers + batch_id)
            is_last_batch = batch_id == (NUM_BATCHES - 1)

            if is_last_batch:
                iris.wait_then_put_tile_batch_signal(
                    C_local_base,
                    C_local_base,
                    cur_rank,
                    dst_rank,
                    heap_bases,
                    copy_engine_ctx,
                    flags + signal_id,
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
                    MAX_TRANSFERS_PER_BATCH,
                )
            else:
                iris.wait_then_put_tile_batch(
                    C_local_base,
                    C_local_base,
                    cur_rank,
                    dst_rank,
                    heap_bases,
                    copy_engine_ctx,
                    flags + signal_id,
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
                    MAX_TRANSFERS_PER_BATCH,
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


def _build_signal_batch_workspace(
    workspace: FusedWorkspace,
    selector,
    device: torch.device,
    m_local: int,
    n: int,
    elem_size: int,
):
    """Plan the GEMM's tile signalling and flatten each batch's copies into device metadata arrays."""
    plan = build_launch_wave_signal_plan(m_local, n, selector, _selector_wave_size(selector, device))

    batch_transfer_offsets = []
    batch_transfer_counts = []
    transfer_row_offsets = []
    transfer_col_offsets = []
    transfer_width_bytes = []
    transfer_heights = []
    for batch in plan.batches:
        batch_transfer_offsets.append(len(transfer_row_offsets))
        batch_transfer_counts.append(len(batch.rects))
        for rect in batch.rects:
            row_offset = rect.m_tile_start * selector.block_m
            col_offset = rect.n_tile_start * selector.block_n
            transfer_row_offsets.append(row_offset)
            transfer_col_offsets.append(col_offset)
            transfer_width_bytes.append(min(rect.n_tile_count * selector.block_n, n - col_offset) * elem_size)
            transfer_heights.append(min(rect.m_tile_count * selector.block_m, m_local - row_offset))

    def _int32(values):
        return torch.tensor(values, device=device, dtype=torch.int32)

    workspace.signal_plan = plan
    workspace.batch_signal_ids = _int32([batch.signal_id for batch in plan.batches])
    workspace.batch_producers = _int32([batch.producers for batch in plan.batches])
    workspace.batch_transfer_offsets = _int32(batch_transfer_offsets)
    workspace.batch_transfer_counts = _int32(batch_transfer_counts)
    workspace.transfer_row_offsets = _int32(transfer_row_offsets)
    workspace.transfer_col_offsets = _int32(transfer_col_offsets)
    workspace.transfer_width_bytes = _int32(transfer_width_bytes)
    workspace.transfer_heights = _int32(transfer_heights)
    workspace.num_batches = len(plan.batches)
    workspace.num_transfers = len(transfer_row_offsets)
    workspace.max_transfers_per_batch = max(batch_transfer_counts)


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
    shmem._require_copy_engine()

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

    ws = FusedWorkspace(
        operation="matmul_all_gather_copy_engine",
        shape=(M, N, K),
        dtype=A.dtype,
        world_size=world_size,
        prepared=True,
    )

    ws.selector = selector
    ws.m_tiles_per_batch = m_tiles_per_batch
    _build_signal_batch_workspace(ws, selector, A.device, M_local, N, A.element_size())
    ws.num_tiles_m = ws.signal_plan.schedule.grid_m
    ws.num_tiles_n = ws.signal_plan.schedule.grid_n

    # One gate per signal, incremented by the GEMM epilogue as each of its tiles is stored.
    ws.locks = shmem.zeros((ws.signal_plan.num_signals,), dtype=torch.int32)
    ws.epilogue_config = EpilogueConfig(ws.locks, map_type="table", tile_to_slot=ws.signal_plan.signal_of_tile)
    ws.completion_signals = shmem.zeros((world_size,), dtype=torch.int32)

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
        flag_iteration: Launch generation for cumulative gate counters.
                        Gates are not reset each iteration; the submit kernel
                        waits for the generation-adjusted target.
        verbose: If True, print submit/main/quiet timing breakdown in sync mode

    Returns:
        FusedWorkspace object
    """
    # Fail fast: without the SDMA engine the GEMM launched below would wait forever on flags.
    shmem._require_copy_engine()

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
            "submit_start": torch.cuda.Event(enable_timing=True),
            "submit_end": torch.cuda.Event(enable_timing=True),
            "main_start": torch.cuda.Event(enable_timing=True),
            "main_end": torch.cuda.Event(enable_timing=True),
            "quiet_start": torch.cuda.Event(enable_timing=True),
            "quiet_end": torch.cuda.Event(enable_timing=True),
        }
        cpu_timing = {
            "total_start": time.perf_counter(),
            "submit_launch_ms": 0.0,
            "main_launch_ms": 0.0,
            "quiet_cpu_ms": 0.0,
            "sync_wait_ms": 0.0,
        }

    # Allocate workspace if not provided
    if workspace is None:
        workspace = matmul_all_gather_copy_engine_preamble(shmem, A, B)

    stride_cm, stride_cn = output_tensor.stride()

    selector = workspace.selector
    num_batches = workspace.num_batches

    if timing_events is not None:
        timing_events["submit_start"].record(timing_events["stream"])
    submit_launch_start = time.perf_counter() if cpu_timing is not None else None
    submit_grid = (world_size,)
    c_local_base = output_tensor[rank * M_local :, :]
    _signal_batch_wait_submit_kernel[submit_grid](
        c_local_base,
        workspace.locks,
        workspace.completion_signals,
        flag_iteration,
        workspace.batch_signal_ids,
        workspace.batch_producers,
        workspace.batch_transfer_offsets,
        workspace.batch_transfer_counts,
        workspace.transfer_row_offsets,
        workspace.transfer_col_offsets,
        workspace.transfer_width_bytes,
        workspace.transfer_heights,
        shmem.get_heap_bases(),
        shmem.get_copy_engine_ctx(),
        rank,
        world_size,
        num_batches,
        workspace.max_transfers_per_batch,
        stride_cm,
        stride_cm,
        stride_cn,
    )
    if cpu_timing is not None:
        cpu_timing["submit_launch_ms"] = (time.perf_counter() - submit_launch_start) * 1000.0
    if timing_events is not None:
        timing_events["submit_end"].record(timing_events["stream"])

    # Launch GEMM after submission so SDMA can wait autonomously.
    if timing_events is not None:
        timing_events["main_start"].record(timing_events["stream"])
    main_launch_start = time.perf_counter() if cpu_timing is not None else None
    if bias is not None:
        import warnings

        warnings.warn(
            "Bias is not yet supported in the tritonBLAS tile-signalling path for "
            "matmul_all_gather_copy_engine. Ignoring bias for this launch."
        )

    c_local_view = output_tensor[rank * M_local : (rank + 1) * M_local, :]
    persistent_matmul_lt(
        A,
        B,
        c_local_view,
        selector,
        bias=None,
        work_stealing=False,
        epilogue_config=workspace.epilogue_config,
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
            submit_ms = timing_events["submit_start"].elapsed_time(timing_events["submit_end"])
            main_ms = timing_events["main_start"].elapsed_time(timing_events["main_end"])
            quiet_ms = timing_events["quiet_start"].elapsed_time(timing_events["quiet_end"])
            gpu_total_ms = submit_ms + main_ms + quiet_ms
            cpu_launch_total_ms = cpu_timing["submit_launch_ms"] + cpu_timing["main_launch_ms"]
            cpu_total_ms = (time.perf_counter() - cpu_timing["total_start"]) * 1000.0
            tile_transfer_count = (world_size - 1) * workspace.num_transfers
            shmem.info(
                f"[Rank {rank}] Device copy-engine GPU timing. "
                f"Submit: {submit_ms:.2f}ms, Main: {main_ms:.2f}ms, Wait: {quiet_ms:.2f}ms, "
                f"GPU total: {gpu_total_ms:.2f}ms"
            )
            shmem.info(
                f"[Rank {rank}] Device copy-engine CPU timing. "
                f"Submit launch: {cpu_timing['submit_launch_ms']:.2f}ms, "
                f"Main launch: {cpu_timing['main_launch_ms']:.2f}ms, "
                f"Launch total: {cpu_launch_total_ms:.2f}ms, "
                f"Wait: {cpu_timing['quiet_cpu_ms']:.2f}ms, "
                f"Barrier wait: {cpu_timing['sync_wait_ms']:.2f}ms, "
                f"CPU total: {cpu_total_ms:.2f}ms, "
                f"transfers={tile_transfer_count}"
            )

    return workspace
