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
import iris

from iris.tracing.events import TraceEvent
from .config import FusedConfig
from .workspace import FusedWorkspace

# Import tritonBLAS
from tritonblas.matmul import persistent_matmul_lt
from tritonblas.matmul import create_counter_config
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


@triton.jit()
def _fused_matmul_all_gather_host_copy_engine_kernel(
    A,  # (M_local, K) - each rank's local input
    B,  # (K, N) - replicated across ranks
    C_gathered,  # (M, N) - gathered output (M = M_local * world_size)
    bias_ptr,
    flags,  # Per-tile flags to trigger SDMA
    M_local,
    M,
    N,
    K,
    stride_am,
    stride_ak,
    stride_bk,
    stride_bn,
    stride_cm,
    stride_cn,
    stride_bias,
    context_tensor: tl.tensor,
    cur_rank: tl.constexpr,
    world_size: tl.constexpr,
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr,
    GROUP_SIZE_M: tl.constexpr,
    NUM_SMS: tl.constexpr,
    NUM_XCDS: tl.constexpr,
    NUM_M_TILES: tl.constexpr,
    NUM_TILES_N: tl.constexpr,
    NUM_K_BLOCKS: tl.constexpr,
    M_TILES_PER_BATCH: tl.constexpr,
    BIAS: tl.constexpr,
    EVEN_K: tl.constexpr,
    ALLOW_TF32: tl.constexpr,
    TRACE: tl.constexpr,
):
    """
    Fused GEMM + all-gather kernel using host-initiated SDMA with POLL packets.

    Computes local GEMM tile, stores to local HBM, then sets flag to trigger
    pre-queued SDMA transfers.
    """
    ctx = iris.DeviceContext.initialize(context_tensor, cur_rank, world_size, tracing=TRACE)

    pid = tl.program_id(0)

    # Persistent loop over local tiles using scheduler
    start = pid
    total = NUM_M_TILES * NUM_TILES_N
    stride = NUM_SMS

    for tile_id in range(start, total, stride):
        if TRACE:
            _trace_handle = ctx.tracing.record_event_start(
                event_id=TraceEvent().wg_gemm,
                target_rank=cur_rank,
                address=flags + tl.arange(0, 1),
                pid_m=tile_id,
                pid_n=0,
            )
        # Wave-aware tile assignment (similar to hbm_buffer's group-based assignment)
        num_pid_in_group = GROUP_SIZE_M * NUM_TILES_N
        group_id = tile_id // num_pid_in_group
        first_pid_m = group_id * GROUP_SIZE_M
        first_pid_m = min(first_pid_m, NUM_M_TILES - 1)
        group_sz = min(NUM_M_TILES - first_pid_m, GROUP_SIZE_M)
        pid_m = first_pid_m + ((tile_id % num_pid_in_group) % GROUP_SIZE_M)
        pid_n = (tile_id % num_pid_in_group) // GROUP_SIZE_M
        pid_m = min(pid_m, NUM_M_TILES - 1)

        # M and N tile indices
        rm = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
        rm = tl.max_contiguous(tl.multiple_of(rm, BLOCK_SIZE_M), BLOCK_SIZE_M)
        rn = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
        rn = tl.max_contiguous(tl.multiple_of(rn % N, BLOCK_SIZE_N), BLOCK_SIZE_N)

        # Initialize accumulator for this tile (must be inside the persistent loop!)
        acc_dtype = tl.int32 if C_gathered.type.element_ty == tl.int8 else tl.float32
        acc = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=acc_dtype)

        for k_block_idx in range(NUM_K_BLOCKS):
            # Load A from selected buffer
            rk = k_block_idx * BLOCK_SIZE_K + tl.arange(0, BLOCK_SIZE_K)
            rk = tl.max_contiguous(tl.multiple_of(rk, BLOCK_SIZE_K), BLOCK_SIZE_K)
            a_ptrs = A + rm.to(tl.int64)[:, None] * stride_am + rk[None, :] * stride_ak
            a = tl.load(a_ptrs)

            # Load B at global K position
            B_ptrs = B + rk[:, None] * stride_bk + rn[None, :] * stride_bn
            b = tl.load(B_ptrs)

            # Accumulate
            if ALLOW_TF32:
                acc = tl.dot(a, b, acc, allow_tf32=True)
            else:
                acc += tl.dot(a, b, allow_tf32=False)

        # ==================================================================
        # Write output
        # ==================================================================
        if BIAS:
            bias_val = tl.load(bias_ptr + rm * stride_bias, mask=rm < M_local, other=0.0)
            acc = acc + bias_val[:, None]

        c = acc.to(C_gathered.type.element_ty)

        global_offset = (rm + cur_rank * M_local)[:, None] * stride_cm + rn[None, :] * stride_cn
        mask = ((rm + cur_rank * M_local)[:, None] < M) & (rn[None, :] < N)

        # Store to local memory (SDMA will read from here when flag is set)
        tl.store(C_gathered + global_offset, c, mask=mask, cache_modifier=".wt")

        # C_ptrs = C_gathered + rm[:, None] * stride_cm + rn[None, :] * stride_cn
        # c_mask = (rm[:, None] < M) & (rn[None, :] < N)
        # tl.store(C_ptrs, c, mask=c_mask, cache_modifier=".wt")

        # TODO which one is better
        # wait_cnt()
        tl.debug_barrier()

        # ═══════════════════════════════════════════════════════════════════
        # Signal Phase: Set flag to trigger pre-queued SDMA transfers
        # ═══════════════════════════════════════════════════════════════════
        # Calculate which batch this M-tile belongs to
        batch_id = pid_m // M_TILES_PER_BATCH

        # Increment flag for this batch (one flag per batch, batching M_TILES_PER_BATCH M-rows × all N-tiles)
        # When flag reaches M_TILES_PER_BATCH * NUM_TILES_N, all tiles in this batch are complete
        tl.atomic_add(flags + batch_id, 1, scope="gpu", sem="release")

        if TRACE:
            ctx.tracing.record_event_end(_trace_handle)


def matmul_all_gather_host_copy_engine_preamble(
    shmem,
    A: torch.Tensor,
    B: torch.Tensor,
    config: Optional[FusedConfig] = None,
    m_tiles_per_batch: int = 1,
    trace: bool = False,
    use_tritonblas: bool = True,
) -> FusedWorkspace:
    """Allocate workspace for matmul_all_gather_host_copy_engine."""
    if config is None:
        config = FusedConfig()

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

    if use_tritonblas:
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
    else:
        num_tiles_m = (M_local + config.block_size_m - 1) // config.block_size_m
        num_tiles_n = (N + config.block_size_n - 1) // config.block_size_n
        num_batches = (num_tiles_m + m_tiles_per_batch - 1) // m_tiles_per_batch
        ws.locks = shmem.zeros((num_batches,), dtype=torch.int32)
        ws.num_tiles_m = num_tiles_m
        ws.num_tiles_n = num_tiles_n
        ws.num_batches = num_batches

    ws.completion_signals = shmem.zeros((world_size,), dtype=torch.int32)

    return ws


def matmul_all_gather_host_copy_engine(
    shmem,
    output_tensor: torch.Tensor,
    A: torch.Tensor,
    B: torch.Tensor,
    bias: Optional[torch.Tensor] = None,
    async_op: bool = False,
    config: Optional[FusedConfig] = None,
    workspace: Optional[FusedWorkspace] = None,
    flag_iteration: int = 0,
    m_tiles_per_batch: int = 1,
    trace: bool = False,
    verbose: bool = False,
    use_tritonblas: bool = True,
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
        config: Optional FusedConfig for tuning
        workspace: Optional pre-allocated workspace

    Returns:
        FusedWorkspace object
    """
    if config is None:
        config = FusedConfig()

    M_local, K = A.shape
    K2, N = B.shape
    world_size = shmem.get_num_ranks()
    rank = shmem.get_rank()

    assert K == K2, f"Inner dimensions must match: A has K={K}, B has K={K2}"

    M = M_local * world_size
    assert output_tensor.shape == (M, N), f"Output must be ({M}, {N}), got {output_tensor.shape}"

    # Allocate workspace if not provided
    if workspace is None:
        workspace = matmul_all_gather_host_copy_engine_preamble(
            shmem, A, B, config, m_tiles_per_batch, trace, use_tritonblas
        )

    stride_cm, stride_cn = output_tensor.stride()
    device = A.device

    selector = workspace.selector
    selector_shape = (
        selector.block_m,
        selector.block_n,
        selector.block_k,
        selector.group_m,
    )
    config_shape = (
        config.block_size_m,
        config.block_size_n,
        config.block_size_k,
        config.group_size_m,
    )
    if selector_shape != config_shape:
        raise ValueError(
            "all_gather_matmul_copy_engine requires selector/config geometry to match: "
            f"selector(M,N,K,G)=({selector.block_m},{selector.block_n},{selector.block_k},{selector.group_m}) "
            f"!= config(M,N,K,G)=({config.block_size_m},{config.block_size_n},{config.block_size_k},{config.group_size_m})"
        )
    launch_wave_plan = getattr(workspace, "launch_wave_plan", None)
    num_tiles_m = workspace.num_tiles_m
    num_tiles_n = workspace.num_tiles_n
    num_batches = workspace.num_batches
    if use_tritonblas and launch_wave_plan is None:
        raise ValueError("workspace.launch_wave_plan must be initialized in preamble for the tritonBLAS path")

    # ═══════════════════════════════════════════════════════════════════════
    # Device Phase: Compute GEMM + store + set flags
    # ═══════════════════════════════════════════════════════════════════════
    if use_tritonblas:
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

            warnings.warn(
                "Bias is not yet supported in tritonBLAS integration. Consider adding bias manually after GEMM."
            )

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

    else:
        # Legacy custom Triton kernel path
        stride_am, stride_ak = A.stride()
        stride_bk, stride_bn = B.stride()

        if bias is not None:
            assert bias.shape[0] == M_local
            bias_ptr = bias
            stride_bias = bias.stride()[0] if bias.dim() > 0 else 1
            use_bias = True
        else:
            bias_ptr = output_tensor
            stride_bias = 1
            use_bias = False

        num_sms = config.num_sms
        if num_sms is None:
            props = torch.cuda.get_device_properties(device)
            num_sms = props.multi_processor_count

        even_k = K % config.block_size_k == 0
        num_k_blocks = (K + config.block_size_k - 1) // config.block_size_k
        num_tiles = num_tiles_m * num_tiles_n

        # Setup tracing if requested
        if trace:
            max_trace_events = num_tiles * 2
            if not shmem.tracing.enabled:
                shmem.tracing.enable(max_events=max_trace_events)
            else:
                shmem.tracing.reset()

            sdma_timestamps = shmem.zeros((world_size, 2), dtype=torch.int64)

        context_tensor = shmem.get_device_context()

        grid = (num_sms,)
        _fused_matmul_all_gather_host_copy_engine_kernel[grid](
            A,
            B,
            output_tensor,
            bias_ptr,
            workspace.locks,
            M_local,
            M,
            N,
            K,
            stride_am,
            stride_ak,
            stride_bk,
            stride_bn,
            stride_cm,
            stride_cn,
            stride_bias,
            context_tensor,
            rank,
            world_size,
            config.block_size_m,
            config.block_size_n,
            config.block_size_k,
            config.group_size_m,
            num_sms,
            config.num_xcds,
            num_tiles_m,
            num_tiles_n,
            num_k_blocks,
            m_tiles_per_batch,
            use_bias,
            even_k,
            config.allow_tf32,
            trace,
        )

    # ═══════════════════════════════════════════════════════════════════════
    # Host Phase: Enqueue SDMA POLL+COPY packets for all tiles
    # (While kernel is running in parallel on device)
    # ═══════════════════════════════════════════════════════════════════════
    import time

    element_size = output_tensor.element_size()
    anvil_lib = shmem.copy_engines

    if verbose and rank == 0:
        if use_tritonblas:
            shmem.info(
                f"[Rank {rank}] Starting SDMA loop (launch-wave transfers)... "
                f"num_m_tiles={num_tiles_m}, num_tiles_n={num_tiles_n}, "
                f"wave_size={launch_wave_plan.wave_size}"
            )
            shmem.info(
                f"[Rank {rank}] Will transfer in {launch_wave_plan.num_waves} waves "
                f"across {len(launch_wave_plan.transfers)} rects"
            )
        else:
            shmem.info(
                f"[Rank {rank}] Starting SDMA loop (batched M-tile transfers)... "
                f"num_m_tiles={num_tiles_m}, num_tiles_n={num_tiles_n}, m_tiles_per_batch={m_tiles_per_batch}"
            )
            shmem.info(f"[Rank {rank}] Will transfer in {num_batches} batches")

    sdma_start_time = time.perf_counter()
    tile_transfer_count = 0

    # Submit start timestamp for each remote rank if tracing
    if trace:
        for remote_rank in range(world_size):
            if remote_rank != rank:
                timestamp_ptr = sdma_timestamps.data_ptr() + remote_rank * 2 * sdma_timestamps.element_size()
                anvil_lib.host_timestamp(rank, remote_rank, 0, timestamp_ptr)

    # Get block size (depends on backend)
    if use_tritonblas:
        block_size_m = workspace.selector.block_m
        block_size_n = workspace.selector.block_n
    else:
        block_size_m = config.block_size_m
        block_size_n = config.block_size_n

    signal_ptr_local = workspace.completion_signals.data_ptr() + rank * workspace.completion_signals.element_size()
    if use_tritonblas:
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

                dst_ptrs_remote = [
                    shmem.translate(dst_ptr_local, rank, remote_rank) for dst_ptr_local in dst_ptrs_local
                ]
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
    else:
        for batch_id in range(num_batches):
            m_tile_start = batch_id * m_tiles_per_batch
            m_tile_end = min(m_tile_start + m_tiles_per_batch, num_tiles_m)
            num_m_tiles_in_batch = m_tile_end - m_tile_start

            m_start = m_tile_start * block_size_m
            m_end = min(m_tile_end * block_size_m, M_local)
            batch_height = m_end - m_start
            batch_width = N
            expected_flag_value = (flag_iteration + 1) * num_m_tiles_in_batch * num_tiles_n
            wait_flag_ptr = workspace.locks.data_ptr() + batch_id * workspace.locks.element_size()

            tile_obj = Tile()
            tile_obj.pid_m = 0
            tile_obj.pid_n = 0
            tile_obj.block_m = batch_height
            tile_obj.block_n = batch_width
            tile_obj.elem_size = element_size
            tile_obj.src_stride = stride_cm * element_size
            src_offset = (m_start + rank * M_local) * stride_cm
            tile_obj.data = output_tensor.data_ptr() + src_offset * element_size

            dst_offset_local = (m_start + rank * M_local) * stride_cm
            dst_ptr_local = output_tensor.data_ptr() + dst_offset_local * element_size
            dst_stride = stride_cm * element_size

            for remote_rank in range(world_size):
                if remote_rank == rank:
                    continue
                dst_ptr_remote = shmem.translate(dst_ptr_local, rank, remote_rank)
                is_last_batch = batch_id == (num_batches - 1)

                if is_last_batch:
                    signal_ptr_remote = shmem.translate(signal_ptr_local, rank, remote_rank)
                    shmem.put_tile(
                        tile_obj,
                        dst_rank=remote_rank,
                        dst_ptr=dst_ptr_remote,
                        dst_stride=dst_stride,
                        wait_flag=wait_flag_ptr,
                        wait_value=expected_flag_value,
                        signal_flag=signal_ptr_remote,
                        signal_value=1,
                        async_op=True,
                        channel=0,
                    )
                else:
                    shmem.put_tile(
                        tile_obj,
                        dst_rank=remote_rank,
                        dst_ptr=dst_ptr_remote,
                        dst_stride=dst_stride,
                        wait_flag=wait_flag_ptr,
                        wait_value=expected_flag_value,
                        async_op=True,
                        channel=0,
                    )
                tile_transfer_count += 1

    sdma_end_post_time = time.perf_counter()

    # Submit end timestamp for each remote rank if tracing
    if trace:
        for remote_rank in range(world_size):
            if remote_rank != rank:
                timestamp_ptr = sdma_timestamps.data_ptr() + (remote_rank * 2 + 1) * sdma_timestamps.element_size()
                anvil_lib.host_timestamp(rank, remote_rank, 0, timestamp_ptr)

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

    # Extract trace data if tracing was enabled
    if trace:
        torch.cuda.synchronize()
        shmem.quiet()  # Wait for all SDMA operations to complete
        workspace.trace_data = _extract_wg_trace(shmem, num_sms, num_tiles, sdma_timestamps=sdma_timestamps)

    return workspace
