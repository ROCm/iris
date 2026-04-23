# SPDX-License-Identifier: MIT
# Copyright (c) 2025 Advanced Micro Devices, Inc. All rights reserved.

"""
Fused GEMM + All-Gather operation using host-initiated SDMA with POLL packets.

Each rank has a row-sharded input A_local (M_local x K) and computes C_local = A_local @ B.
The host pre-queues SDMA POLL+COPY packets for scatter, then the device kernel just stores
tiles to local HBM and sets flags to trigger the pre-queued transfers.

This is more efficient than device-initiated SDMA because:
- SDMA queue setup happens once on host (not per-tile)
- Device kernel is lightweight (store + set flag)
- SDMA hardware automatically performs scatter when flags are set

This implementation supports two backends:
- Custom Triton kernel (legacy, controlled by use_tritonblas=False)
- tritonBLAS with SignalView (default, use_tritonblas=True)
"""

from typing import Optional
import torch
import triton
import triton.language as tl

from .workspace import FusedWorkspace

# Import tritonBLAS
from tritonblas.matmul import persistent_matmul_lt
from tritonblas.matmul import create_counter_config
from tritonblas.matmul import _make_matmul_selector
from .tritonblas_launch_wave_schedule import build_launch_wave_plan

# Import Tile class from anvil module
try:
    import anvil

    Tile = anvil.Tile
except (ImportError, AttributeError):
    Tile = None  # Will raise error later if needed


@triton.jit()
def wait_cnt():
    tl.inline_asm_elementwise("s_waitcnt vmcnt(0)", "=r", [], dtype=tl.int32, is_pure=False, pack=1)


# Event IDs (must match iris.tracing.events.TraceEvent)
_WG_GEMM = 15


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
    # while tl.atomic_add(completion_signals + src_rank, 0, sem="acquire", scope="sys") < expected_value:
    while tl.load(completion_signals + src_rank, cache_modifier=".cv", volatile=True) < expected_value:
        pass


def _extract_wg_trace(shmem, grid_size, num_tiles, sdma_timestamps=None, **metadata):
    """Extract per-tile trace data from DeviceTracing events.

    For host-initiated SDMA:
    - Each tile generates trace events (not per workgroup)
    - SDMA timestamps captured by host via timestamp packets
    """
    import numpy as np

    bufs = shmem.tracing.trace_buffers
    n = min(shmem.tracing.trace_counter.item(), shmem.tracing.max_events)

    event_ids = bufs["event_id"][:n].cpu().numpy()
    pid_ms = bufs["pid_m"][:n].cpu().numpy()  # tile_id (not workgroup pid)
    timestamps = bufs["timestamp"][:n].cpu().numpy().astype(np.int64)
    end_ts = bufs["duration_cycles"][:n].cpu().numpy().astype(np.int64)
    xcc_ids = bufs["xcc_id"][:n].cpu().numpy().astype(np.int32)

    # Per-tile traces
    starts = torch.zeros(num_tiles, dtype=torch.int64)
    ends = torch.zeros(num_tiles, dtype=torch.int64)
    waits = torch.zeros(num_tiles, dtype=torch.int64)  # Not used but needed for plot
    xcds = torch.zeros(num_tiles, dtype=torch.int32)

    for i in range(n):
        eid = int(event_ids[i])
        tile_id = int(pid_ms[i])

        if eid == _WG_GEMM and tile_id < num_tiles:
            starts[tile_id] = int(timestamps[i])
            ends[tile_id] = int(end_ts[i])
            xcds[tile_id] = int(xcc_ids[i])

    result = {"start": starts, "end": ends, "wait": waits, "xcd": xcds, "grid_size": num_tiles, **metadata}

    # Add SDMA timestamps if available (world_size x 2: start/end per rank)
    if sdma_timestamps is not None:
        result["sdma_timestamps"] = sdma_timestamps.cpu()

    return result


def _auto_m_tiles_per_batch(selector, M_local: int, N: int) -> int:
    """Auto-calculate optimal m_tiles_per_batch based on selector and shape."""
    num_tiles_m = (M_local + selector.block_m - 1) // selector.block_m
    num_tiles_n = (N + selector.block_n - 1) // selector.block_n
    active_cus = getattr(selector, "_ACTIVE_CU", None)
    if active_cus is None or active_cus <= 0:
        import torch

        active_cus = torch.cuda.get_device_properties(torch.cuda.current_device()).multi_processor_count
    tiles_per_group = max(1, selector.group_m * num_tiles_n)
    groups_per_wave = max(1, int(active_cus) // tiles_per_group)
    return max(1, min(num_tiles_m, groups_per_wave * selector.group_m))


def matmul_all_gather_host_copy_engine_preamble(
    shmem,
    A: torch.Tensor,
    B: torch.Tensor,
    m_tiles_per_batch: Optional[int] = None,
    trace: bool = False,
    selector=None,
) -> FusedWorkspace:
    """Allocate workspace for matmul_all_gather_host_copy_engine."""
    M_local, K = A.shape
    K2, N = B.shape
    world_size = shmem.get_num_ranks()

    assert K == K2, f"Inner dimensions must match: A has K={K}, B has K={K2}"

    M = M_local * world_size

    ws = FusedWorkspace(
        operation="matmul_all_gather_host_copy_engine",
        shape=(M, N, K),
        dtype=A.dtype,
        world_size=world_size,
        prepared=True,
    )

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
    num_tiles_m = triton.cdiv(M_local, selector.block_m)
    num_tiles_n = triton.cdiv(N, selector.block_n)
    launch_wave_plan = build_launch_wave_plan(
        num_tiles_m=num_tiles_m,
        num_tiles_n=num_tiles_n,
        group_size_m=selector.group_m,
        launch_grid=num_tiles_m * num_tiles_n,
        wave_size=selector._ACTIVE_CU,
        num_xcds=selector.num_sms,
    )
    ws.selector = selector
    ws.launch_wave_plan = launch_wave_plan
    ws.locks = shmem.zeros((launch_wave_plan.num_waves,), dtype=torch.int32)
    ws.num_tiles_m = num_tiles_m
    ws.num_tiles_n = num_tiles_n
    ws.num_batches = launch_wave_plan.num_waves
    ws.num_waves = launch_wave_plan.num_waves
    ws.m_tiles_per_batch = m_tiles_per_batch

    ws.completion_signals = shmem.zeros((world_size,), dtype=torch.int32)

    return ws


def matmul_all_gather_host_copy_engine(
    shmem,
    output_tensor: torch.Tensor,
    A: torch.Tensor,
    B: torch.Tensor,
    bias: Optional[torch.Tensor] = None,
    async_op: bool = False,
    workspace: Optional[FusedWorkspace] = None,
    flag_iteration: int = 0,
    trace: bool = False,
    verbose: bool = False,
) -> FusedWorkspace:
    """
    Fused matrix multiplication and all-gather using host-initiated SDMA with POLL packets.

    Computes: output = all_gather(A @ B + bias) along M dimension

    Each rank has A of shape (M_local, K) where M_local = M / world_size.
    The host pre-queues SDMA POLL+COPY packets for all tiles and ranks.
    The device kernel computes tiles, stores to local HBM, then sets flags.
    SDMA hardware automatically performs scatter when flags are set.

    Args:
        shmem: Iris shmem context
        output_tensor: Output tensor C of shape (M, N) where M = M_local * world_size
        A: Input matrix A of shape (M_local, K)
        B: Input matrix B of shape (K, N)
        bias: Optional bias vector (M_local,)
        async_op: If False, performs barrier at end
        workspace: Optional pre-allocated workspace

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

    # Allocate workspace if not provided
    if workspace is None:
        workspace = matmul_all_gather_host_copy_engine_preamble(shmem, A, B, trace=trace)

    stride_cm, stride_cn = output_tensor.stride()
    device = A.device

    selector = workspace.selector
    launch_wave_plan = getattr(workspace, "launch_wave_plan", None)
    num_tiles_m = workspace.num_tiles_m
    num_tiles_n = workspace.num_tiles_n
    num_batches = workspace.num_batches
    if launch_wave_plan is None:
        raise ValueError("workspace.launch_wave_plan must be initialized in preamble")

    # ═══════════════════════════════════════════════════════════════════════
    # Device Phase: Compute GEMM + store + set flags
    # ═══════════════════════════════════════════════════════════════════════
    # tritonBLAS path with SignalView
    selector = workspace.selector

    # Create a view of output_tensor at this rank's position
    C_local_view = output_tensor[rank * M_local : (rank + 1) * M_local, :]

    counter_config = create_counter_config(
        workspace.locks,
        map_type="launch_wave",
        block_group_m=launch_wave_plan.wave_size,
    )

    # Use work-stealing if enabled
    tritonblas_config = None

    # Warn about bias
    if bias is not None:
        import warnings

        warnings.warn("Bias is not yet supported in tritonBLAS integration. Consider adding bias manually after GEMM.")

    # Launch tritonBLAS GEMM with SignalView
    persistent_matmul_lt(
        A,
        B,
        C_local_view,
        selector,
        config=tritonblas_config,
        bias=None,
        work_stealing=False,
        counter_config=counter_config,
    )

    # ═══════════════════════════════════════════════════════════════════════
    # Host Phase: Enqueue SDMA POLL+COPY packets for all tiles
    # (While kernel is running in parallel on device)
    # ═══════════════════════════════════════════════════════════════════════
    import time

    element_size = output_tensor.element_size()
    anvil_lib = shmem.copy_engines

    if verbose and rank == 0:
        shmem.info(
            f"[Rank {rank}] Starting SDMA loop (launch-wave transfers)... "
            f"num_m_tiles={num_tiles_m}, num_tiles_n={num_tiles_n}, "
            f"wave_size={launch_wave_plan.wave_size}"
        )
        shmem.info(
            f"[Rank {rank}] Will transfer in {launch_wave_plan.num_waves} waves "
            f"across {len(launch_wave_plan.transfers)} rects"
        )

    sdma_start_time = time.perf_counter()
    tile_transfer_count = 0

    # Get block size from workspace selector
    block_size_m = workspace.selector.block_m
    block_size_n = workspace.selector.block_n

    signal_ptr_local = workspace.completion_signals.data_ptr() + rank * workspace.completion_signals.element_size()
    transfers_by_wave = [[] for _ in range(launch_wave_plan.num_waves)]
    for transfer in launch_wave_plan.transfers:
        transfers_by_wave[transfer.wave_id].append(transfer)

    for wave_id, wave_transfers in enumerate(transfers_by_wave):
        if not wave_transfers:
            continue

        expected_flag_value = (flag_iteration + 1) * launch_wave_plan.wave_tile_counts[wave_id]
        wait_flag_ptr = workspace.locks.data_ptr() + wave_id * workspace.locks.element_size()
        is_last_wave = wave_id == (launch_wave_plan.num_waves - 1)

        tiles = []
        dst_ptrs_local = []
        dst_strides = []

        for transfer in wave_transfers:
            m_start = transfer.m_tile_start * block_size_m
            n_start = transfer.n_tile_start * block_size_n
            batch_height = min(transfer.m_tile_count * block_size_m, M_local - m_start)
            batch_width = min(transfer.n_tile_count * block_size_n, N - n_start)

            tile_obj = Tile()
            tile_obj.pid_m = 0
            tile_obj.pid_n = 0
            tile_obj.block_m = batch_height
            tile_obj.block_n = batch_width
            tile_obj.elem_size = element_size
            tile_obj.src_stride = stride_cm * element_size

            src_offset = (m_start + rank * M_local) * stride_cm + n_start * stride_cn
            tile_obj.data = output_tensor.data_ptr() + src_offset * element_size
            dst_offset_local = (m_start + rank * M_local) * stride_cm + n_start * stride_cn

            tiles.append(tile_obj)
            dst_ptrs_local.append(output_tensor.data_ptr() + dst_offset_local * element_size)
            dst_strides.append(stride_cm * element_size)

        for remote_rank in range(world_size):
            if remote_rank == rank:
                continue

            dst_ptrs_remote = [shmem.translate(dst_ptr_local, rank, remote_rank) for dst_ptr_local in dst_ptrs_local]
            signal_ptr_remote = None
            if is_last_wave:
                signal_ptr_remote = shmem.translate(signal_ptr_local, rank, remote_rank)

            shmem.put_tiles(
                tiles,
                dst_rank=remote_rank,
                dst_ptrs=dst_ptrs_remote,
                dst_strides=dst_strides,
                wait_flag=wait_flag_ptr,
                wait_value=expected_flag_value,
                signal_flag=signal_ptr_remote,
                signal_value=1,
                async_op=True,
                channel=0,
            )
            tile_transfer_count += len(wave_transfers)

            if verbose and wave_id < 2 and remote_rank == (rank + 1) % world_size:
                shmem.info(
                    f"[Rank {rank}→{remote_rank}] Queued wave={wave_id} "
                    f"transfers={len(wave_transfers)} tiles={launch_wave_plan.wave_tile_counts[wave_id]}"
                )

    sdma_end_post_time = time.perf_counter()

    if not async_op:
        _wait_completion_signals_kernel[(world_size,)](
            workspace.completion_signals,
            flag_iteration + 1,
            rank,
            world_size,
        )
        sdma_end_time = time.perf_counter()
        shmem.barrier()
        if verbose and rank == 0:
            post_ms = (sdma_end_post_time - sdma_start_time) * 1000.0
            quiet_ms = (sdma_end_time - sdma_end_post_time) * 1000.0
            total_ms = (sdma_end_time - sdma_start_time) * 1000.0
            shmem.info(
                f"[Rank {rank}] SDMA complete. "
                f"Post: {post_ms:.2f}ms, Wait: {quiet_ms:.2f}ms, Total: {total_ms:.2f}ms, "
                f"transfers={tile_transfer_count}"
            )

    return workspace
