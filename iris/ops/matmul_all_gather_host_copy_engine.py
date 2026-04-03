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
"""

from typing import Optional
import torch
import triton
import triton.language as tl
import iris

from tritonblas.kernels.stages import GemmContext, ScheduleContext, make_tensor_view

from iris.device_utils import read_realtime
from iris.tracing.events import TraceEvent
from .config import FusedConfig
from .workspace import FusedWorkspace

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
) -> FusedWorkspace:
    """Allocate workspace for matmul_all_gather_host_copy_engine including per-batch flags."""
    if config is None:
        config = FusedConfig()

    M_local, K = A.shape
    K2, N = B.shape
    world_size = shmem.get_num_ranks()

    assert K == K2, f"Inner dimensions must match: A has K={K}, B has K={K2}"

    M = M_local * world_size

    # Calculate number of tiles
    num_tiles_m = (M_local + config.block_size_m - 1) // config.block_size_m
    num_tiles_n = (N + config.block_size_n - 1) // config.block_size_n
    num_tiles = num_tiles_m * num_tiles_n

    # Calculate number of batches
    num_batches = (num_tiles_m + m_tiles_per_batch - 1) // m_tiles_per_batch

    ws = FusedWorkspace(
        operation="matmul_all_gather_host_copy_engine",
        shape=(M, N, K),
        dtype=A.dtype,
        world_size=world_size,
        prepared=True,
    )

    # Allocate per-batch flags (one flag per batch, each batch contains m_tiles_per_batch M-rows)
    ws.locks = shmem.zeros((num_batches,), dtype=torch.int32)

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
    m_tiles_per_batch: int = 1,
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
        workspace = matmul_all_gather_host_copy_engine_preamble(shmem, A, B, config, m_tiles_per_batch, trace)

    stride_am, stride_ak = A.stride()
    stride_bk, stride_bn = B.stride()
    stride_cm, stride_cn = output_tensor.stride()

    if bias is not None:
        assert bias.shape[0] == M_local
        bias_ptr = bias
        stride_bias = bias.stride()[0] if bias.dim() > 0 else 1
        use_bias = True
    else:
        bias_ptr = output_tensor
        stride_bias = 1
        use_bias = False

    device = A.device
    num_sms = config.num_sms
    if num_sms is None:
        props = torch.cuda.get_device_properties(device)
        num_sms = props.multi_processor_count

    even_k = K % config.block_size_k == 0

    # Calculate number of tiles
    num_k_blocks = (K + config.block_size_k - 1) // config.block_size_k
    num_tiles_m = (M_local + config.block_size_m - 1) // config.block_size_m
    num_tiles_n = (N + config.block_size_n - 1) // config.block_size_n
    num_tiles = num_tiles_m * num_tiles_n

    # Setup tracing if requested
    if trace:
        # Each tile generates 1 event (start + end)
        max_trace_events = num_tiles * 2
        if not shmem.tracing.enabled:
            shmem.tracing.enable(max_events=max_trace_events)
        else:
            shmem.tracing.reset()

        # Allocate timestamp buffers if tracing (2 timestamps per rank: start and end)
        if trace:
            sdma_timestamps = shmem.zeros((world_size, 2), dtype=torch.int64)

    context_tensor = shmem.get_device_context()

    # Reset flags before kernel launch
    workspace.locks.zero_()
    torch.cuda.synchronize()
    shmem.barrier()

    # ═══════════════════════════════════════════════════════════════════════
    # Device Phase: Launch kernel to compute GEMM + store + set flags
    # ═══════════════════════════════════════════════════════════════════════
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

    # Calculate number of batches
    num_batches = (num_tiles_m + m_tiles_per_batch - 1) // m_tiles_per_batch

    if verbose and rank == 0:
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

    # Queue POLL+COPY packets for each batch (batching M_TILES_PER_BATCH M-rows × all N-tiles together)
    for batch_id in range(num_batches):
        # Calculate M-tile range for this batch
        m_tile_start = batch_id * m_tiles_per_batch
        m_tile_end = min(m_tile_start + m_tiles_per_batch, num_tiles_m)
        num_m_tiles_in_batch = m_tile_end - m_tile_start

        # Calculate batch bounds in rows
        m_start = m_tile_start * config.block_size_m
        m_end = min(m_tile_end * config.block_size_m, M_local)
        batch_height = m_end - m_start

        # Calculate total width (all N-tiles)
        batch_width = N  # Full N dimension

        # Create Tile object for 2D sub-window copy of entire batch
        tile_obj = Tile()
        tile_obj.pid_m = 0  # We'll handle offset in data pointer
        tile_obj.pid_n = 0
        tile_obj.block_m = batch_height
        tile_obj.block_n = batch_width
        tile_obj.elem_size = element_size
        tile_obj.src_stride = stride_cm * element_size  # Row stride in bytes

        # Source data pointer (output tensor at this rank's batch location, full N width)
        src_offset = (m_start + rank * M_local) * stride_cm
        tile_obj.data = output_tensor.data_ptr() + src_offset * element_size

        # For each remote rank, queue POLL+COPY
        for remote_rank in range(world_size):
            if remote_rank != rank:
                # Destination is the same logical position on remote rank
                dst_offset = (m_start + rank * M_local) * stride_cm
                dst_ptr_local = output_tensor.data_ptr() + dst_offset * element_size

                # Translate local pointer to remote rank's address space
                dst_ptr_remote = shmem.translate(dst_ptr_local, rank, remote_rank)
                dst_stride = stride_cm * element_size  # Row stride in bytes

                # Get flag pointer for this batch
                flag_ptr = workspace.locks.data_ptr() + batch_id * workspace.locks.element_size()

                # Wait for flag to reach num_m_tiles_in_batch * num_tiles_n (all tiles in this batch complete)
                expected_flag_value = num_m_tiles_in_batch * num_tiles_n

                # Use anvil host API to queue POLL+SUB_WINDOW_COPY for entire batch
                anvil_lib.host_wait_flag_then_put_tile(
                    rank,
                    remote_rank,
                    0,  # channel_idx
                    flag_ptr,
                    expected_flag_value,
                    tile_obj,
                    dst_ptr_remote,
                    dst_stride,
                )
                tile_transfer_count += 1

                if verbose and batch_id == 0 and remote_rank == (rank + 1) % world_size:
                    shmem.info(
                        f"[Rank {rank}→{remote_rank}] Queued batch={batch_id} "
                        f"({num_m_tiles_in_batch} M-tiles × {num_tiles_n} N-tiles, "
                        f"{batch_height}×{batch_width} elements)"
                    )

    sdma_end_post_time = time.perf_counter()

    # Submit end timestamp for each remote rank if tracing
    if trace:
        for remote_rank in range(world_size):
            if remote_rank != rank:
                timestamp_ptr = sdma_timestamps.data_ptr() + (remote_rank * 2 + 1) * sdma_timestamps.element_size()
                anvil_lib.host_timestamp(rank, remote_rank, 0, timestamp_ptr)

    # Wait for SDMA to complete (all flags have been set, SDMA transfers should finish)
    # Use anvil quiet to wait for SDMA completion
    # TODO part of async_op ?
    for remote_rank in range(world_size):
        if remote_rank != rank:
            anvil_lib.host_quiet(rank, remote_rank, 0)

    sdma_end_time = time.perf_counter()

    if verbose:
        post_ms = (sdma_end_post_time - sdma_start_time) * 1000.0
        quiet_ms = (sdma_end_time - sdma_end_post_time) * 1000.0
        total_ms = (sdma_end_time - sdma_start_time) * 1000.0
        shmem.info(
            f"[Rank {rank}] SDMA complete. "
            f"Post: {post_ms:.2f}ms, Quiet: {quiet_ms:.2f}ms, Total: {total_ms:.2f}ms, "
            f"transfers={tile_transfer_count}"
        )

    if not async_op:
        torch.cuda.synchronize()
        shmem.barrier()

    # Extract trace data if tracing was enabled
    if trace:
        torch.cuda.synchronize()
        # sdma_ts = workspace.sdma_timestamps if hasattr(workspace, 'sdma_timestamps') else None
        workspace.trace_data = _extract_wg_trace(shmem, num_sms, num_tiles, sdma_timestamps=sdma_timestamps)

    return workspace
