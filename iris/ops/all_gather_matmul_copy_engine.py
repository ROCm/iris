# SPDX-License-Identifier: MIT
# Copyright (c) 2025-2026 Advanced Micro Devices, Inc. All rights reserved.

"""
Fused All-Gather + GEMM using copy engine (SDMA) for data movement.

Key differences from HBM buffer variant:
- SMs only perform GEMM (no fetcher workgroups)
- Host orchestrates SDMA transfers of remote tiles to staged_a buffer
- GEMM processes local K-blocks first (from A_sharded), then remote K-blocks (from staged_a)
- Flags only track remote tiles, updated by copy engine via host_atomic_add_32
"""

from typing import Optional
import torch
import torch.distributed as dist
import triton
import triton.language as tl
import iris
import iris.x

from tritonblas.kernels.stages import GemmContext, ScheduleContext

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


@triton.jit
def _copy_engine_all_gather_matmul_kernel(
    A_sharded,
    B,
    C,
    bias_ptr,
    staged_a,
    flags_ptr,
    M,
    N,
    K,
    K_local,
    stride_am,
    stride_ak,
    stride_bk,
    stride_bn,
    stride_cm,
    stride_cn,
    stride_sa_m,  # staged_a stride in M dim
    stride_sa_k,  # staged_a stride in K dim
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
    NUM_K_BLOCKS_LOCAL: tl.constexpr,
    NUM_REMOTE_K_BLOCKS: tl.constexpr,
    K_PER_FLAG: tl.constexpr,
    BIAS: tl.constexpr,
    ALLOW_TF32: tl.constexpr,
    TRACE: tl.constexpr,
):
    """
    Persistent GEMM kernel with staged remote tiles.

    Grid: (num_sms,) - persistent scheduling
    Each SM processes multiple output tiles in a loop.
    """
    acc_dtype = tl.int32 if C.type.element_ty == tl.int8 else tl.float32
    zero = tl.program_id(0) * 0

    ctx = iris.DeviceContext.initialize(context_tensor, cur_rank, world_size, tracing=TRACE)

    # Create tritonblas context and scheduler for persistent GEMM
    # NOTE: For L2 cache reuse on MI300X, chunk_size should be divisible by group_size_m
    # since L2 is per-XCD. Recommended: chunk_size = SMs_per_XCD (e.g., 38 for MI300X)
    # and group_size_m divides chunk_size for proper grouping within XCD.
    gemm_ctx = GemmContext(
        BLOCK_SIZE_M,
        BLOCK_SIZE_N,
        BLOCK_SIZE_K,
        num_sms=NUM_SMS,
        num_xcds=NUM_XCDS,
        group_size_m=GROUP_SIZE_M,
        chunk_size=K_PER_FLAG,  # Reuse K_PER_FLAG parameter for chunk_size
        even_k=True,
        allow_tf32=ALLOW_TF32,
    )
    sched = ScheduleContext(M, N, K, gemm_ctx)

    # Persistent loop over output tiles
    start, total, stride = sched.persistent_tile_range()
    for tile_id in range(start, total, stride):
        # Get tile coordinates with swizzling from scheduler
        out_tile = sched.get_tile_from_idx(tile_id)
        pid_m = out_tile.pid_m
        pid_n = out_tile.pid_n

        # Initialize accumulator
        acc = gemm_ctx.init_accumulator()

        # M and N tile indices
        rm = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
        rm = tl.max_contiguous(tl.multiple_of(rm, BLOCK_SIZE_M), BLOCK_SIZE_M)
        rn = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
        rn = tl.max_contiguous(tl.multiple_of(rn % N, BLOCK_SIZE_N), BLOCK_SIZE_N)

        if TRACE:
            _trace_handle = ctx.tracing.record_event_start(
                event_id=TraceEvent().wg_gemm,
                target_rank=cur_rank,
                address=flags_ptr + tl.arange(0, 1),
                pid_m=tile_id,
                pid_n=zero,
            )
            _tile_wt = zero.to(tl.int64)  # Wait time for THIS tile only

        # ==================================================================
        # PHASE 1: Process LOCAL K-blocks from A_sharded (no synchronization)
        # ==================================================================
        for k_block_local in range(NUM_K_BLOCKS_LOCAL):
            # Local K indices
            rk_local = k_block_local * BLOCK_SIZE_K + tl.arange(0, BLOCK_SIZE_K)
            rk_local = tl.max_contiguous(tl.multiple_of(rk_local, BLOCK_SIZE_K), BLOCK_SIZE_K)

            # Load A from local A_sharded
            a_ptrs = A_sharded + rm.to(tl.int64)[:, None] * stride_am + rk_local[None, :] * stride_ak
            a = tl.load(a_ptrs)

            # Load B - map to global K coordinate for this rank
            k_block_global = cur_rank * NUM_K_BLOCKS_LOCAL + k_block_local
            rk_b = k_block_global * BLOCK_SIZE_K + tl.arange(0, BLOCK_SIZE_K)
            rk_b = tl.max_contiguous(tl.multiple_of(rk_b, BLOCK_SIZE_K), BLOCK_SIZE_K)
            B_ptrs = B + rk_b[:, None] * stride_bk + rn[None, :] * stride_bn
            b = tl.load(B_ptrs)

            # Accumulate
            if ALLOW_TF32:
                acc = tl.dot(a, b, acc, allow_tf32=True)
            else:
                acc += tl.dot(a, b, allow_tf32=False)

        # ==================================================================
        # PHASE 2: Process REMOTE K-blocks from staged_a (with per-M-tile sync)
        # ==================================================================
        # NEW: Wait once for ALL K-blocks of this M tile to be ready
        # This enables wave-aware copying on the host side

        # Measure wait time for this tile
        if TRACE:
            _ws = read_realtime()

        # Wait for this M tile's data to be fully copied
        # Note: if we've already synced this M tile, the flag will already be >= expected_flag_value
        flag_idx = pid_m
        expected_flag_value = world_size - 1  # All remote ranks must contribute
        while tl.atomic_add(flags_ptr + flag_idx, 0, sem="acquire", scope="sys") < expected_flag_value:
            pass

        if TRACE:
            # Measure wait time - will be ~0 if flag was already ready
            _tile_wt = read_realtime() - _ws

        # Now process all remote K-blocks for this M tile
        # NON-INTERLEAVED: each rank's K-blocks are contiguous in staged_a
        for k_block_local in range(NUM_K_BLOCKS_LOCAL):
            for src_rank_idx in range(world_size):
                if src_rank_idx != cur_rank:  # Process only remote ranks
                    # Load from staged_a using GLOBAL K indexing (non-interleaved)
                    k_block_global = src_rank_idx * NUM_K_BLOCKS_LOCAL + k_block_local
                    rk_staged = k_block_global * BLOCK_SIZE_K + tl.arange(0, BLOCK_SIZE_K)
                    rk_staged = tl.max_contiguous(tl.multiple_of(rk_staged, BLOCK_SIZE_K), BLOCK_SIZE_K)
                    a_ptrs = staged_a + rm.to(tl.int64)[:, None] * stride_sa_m + rk_staged[None, :] * stride_sa_k
                    a = tl.load(a_ptrs)

                    # Load B at global K position (src_rank's K)
                    rk_b = k_block_global * BLOCK_SIZE_K + tl.arange(0, BLOCK_SIZE_K)
                    rk_b = tl.max_contiguous(tl.multiple_of(rk_b, BLOCK_SIZE_K), BLOCK_SIZE_K)
                    B_ptrs = B + rk_b[:, None] * stride_bk + rn[None, :] * stride_bn
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
            bias_val = tl.load(bias_ptr + rm * stride_bias, mask=rm < M, other=0.0)
            acc = acc + bias_val[:, None]

        c = acc.to(C.type.element_ty)
        C_ptrs = C + rm[:, None] * stride_cm + rn[None, :] * stride_cn
        c_mask = (rm[:, None] < M) & (rn[None, :] < N)
        tl.store(C_ptrs, c, mask=c_mask, cache_modifier=".wt")

        if TRACE:
            ctx.tracing.record_event_end(_trace_handle)
            ctx.tracing.record_event_start(
                event_id=TraceEvent().wg_gemm_wait,
                target_rank=cur_rank,
                address=flags_ptr + tl.arange(0, 1),
                pid_m=tile_id,
                pid_n=_tile_wt.to(tl.int32),
            )


# ==========================================================================
# Python API
# ==========================================================================


def _get_wave_m_tile_schedule(num_sms, num_xcds, group_size_m, chunk_size, num_tiles_m, num_tiles_n):
    """
    Analyze which M tiles need to be copied before each wave.

    Returns:
        List of (wave_num, [m_tiles_to_copy]) tuples in execution order.
    """
    # Helper functions matching tritonblas ScheduleContext (pure Python versions)
    def chiplet_transform_chunked_py(pid, num_workgroups, num_xcds, chunk_size):
        if pid > (num_workgroups // (num_xcds * chunk_size)) * (num_xcds * chunk_size):
            return pid
        local_pid = pid // num_xcds
        chunk_idx = local_pid // chunk_size
        pos_in_chunk = local_pid % chunk_size
        xcd = pid % num_xcds
        return chunk_idx * num_xcds * chunk_size + xcd * chunk_size + pos_in_chunk

    def get_tile_from_idx_py(tile_id, num_tiles_m, num_tiles_n, group_size_m):
        num_pid_in_group = group_size_m * num_tiles_n
        group_id = tile_id // num_pid_in_group
        first_pid_m = group_id * group_size_m
        group_size_m_actual = min(num_tiles_m - first_pid_m, group_size_m)
        pid_m = first_pid_m + ((tile_id % num_pid_in_group) % group_size_m_actual)
        pid_n = (tile_id % num_pid_in_group) // group_size_m_actual
        return pid_m, pid_n

    total_tiles = num_tiles_m * num_tiles_n

    # Simulate persistent scheduling to determine which M tiles each wave accesses
    waves = {}
    for pid in range(num_sms):
        # Apply XCD transform
        transformed_pid = chiplet_transform_chunked_py(pid, num_sms, num_xcds, chunk_size)

        # Persistent loop
        for tile_id in range(transformed_pid, total_tiles, num_sms):
            wave_num = tile_id // num_sms
            tile_m, tile_n = get_tile_from_idx_py(tile_id, num_tiles_m, num_tiles_n, group_size_m)

            if wave_num not in waves:
                waves[wave_num] = set()
            waves[wave_num].add(tile_m)

    # Convert to ordered copy schedule (only copy new M tiles each wave)
    copied = set()
    schedule = []
    for wave_num in sorted(waves.keys()):
        new_m_tiles = sorted(waves[wave_num] - copied)
        if new_m_tiles:
            schedule.append((wave_num, new_m_tiles))
        copied.update(waves[wave_num])

    return schedule


def all_gather_matmul_copy_engine_preamble(
    shmem,
    A_sharded: torch.Tensor,
    B: torch.Tensor,
    config: Optional[FusedConfig] = None,
    k_per_flag: int = 1,
    staged_a_layout: str = "k_contiguous",
) -> FusedWorkspace:
    """
    Allocate workspace for copy engine variant.

    Args:
        staged_a_layout: "k_contiguous" (default, row-major (M,K)) or
                         "m_contiguous" (col-major, stored as (K,M) transposed).
    """
    if config is None:
        config = FusedConfig()

    M, K_local = A_sharded.shape
    K, N = B.shape
    world_size = shmem.get_num_ranks()

    assert world_size * K_local == K
    assert K_local % config.block_size_k == 0
    assert K % config.block_size_k == 0
    assert M % config.block_size_m == 0

    num_m_tiles = M // config.block_size_m
    num_k_blocks = K // config.block_size_k
    num_k_blocks_local = K_local // config.block_size_k
    num_remote_k_blocks = num_k_blocks - num_k_blocks_local

    # NEW: Use per-M-tile flags for wave-aware copying
    # Each flag tracks when all K-blocks for an M tile are ready from all remote ranks
    num_flags = num_m_tiles

    ws = FusedWorkspace(
        operation="all_gather_matmul_copy_engine",
        shape=(M, N, K),
        dtype=A_sharded.dtype,
        world_size=world_size,
        variant=f"copy_engine_{staged_a_layout}_wave_aware_noninterleaved",
        prepared=True,
    )

    # Allocate staged_a - full K dimension (NON-INTERLEAVED like HBM buffer)
    # Each rank's K-blocks are stored contiguously for efficient bulk SDMA
    if staged_a_layout == "m_contiguous":
        storage = shmem.zeros((K, M), dtype=A_sharded.dtype)
        ws.aux_buffer = storage.T  # (M, K) with M-contiguous
    else:
        ws.aux_buffer = shmem.zeros((M, K), dtype=A_sharded.dtype)

    # Allocate per-M-tile flags
    ws.locks = shmem.zeros((num_flags,), dtype=torch.int32)

    total_tiles_tracked = num_m_tiles * num_k_blocks_local * (world_size - 1)
    shmem.info(
        f"Allocated {num_flags} per-M-tile flags for {total_tiles_tracked} tiles, "
        f"flags buffer at 0x{ws.locks.data_ptr():x}"
    )

    # Share pointers across ranks for SDMA addressing
    # Need: A_sharded (source), staged_a (destination), flags (signaling)
    A_sharded_ptr_tensor = torch.tensor([A_sharded.data_ptr()], dtype=torch.int64, device='cuda')
    A_sharded_ptrs = [torch.zeros(1, dtype=torch.int64, device='cuda') for _ in range(world_size)]
    dist.all_gather(A_sharded_ptrs, A_sharded_ptr_tensor)

    staged_a_ptr_tensor = torch.tensor([ws.aux_buffer.data_ptr()], dtype=torch.int64, device='cuda')
    staged_a_ptrs = [torch.zeros(1, dtype=torch.int64, device='cuda') for _ in range(world_size)]
    dist.all_gather(staged_a_ptrs, staged_a_ptr_tensor)

    flags_ptr_tensor = torch.tensor([ws.locks.data_ptr()], dtype=torch.int64, device='cuda')
    flags_ptrs = [torch.zeros(1, dtype=torch.int64, device='cuda') for _ in range(world_size)]
    dist.all_gather(flags_ptrs, flags_ptr_tensor)

    # Store all remote pointers in workspace
    ws.remote_pointers = {
        'A_sharded': [ptr.item() for ptr in A_sharded_ptrs],
        'staged_a': [ptr.item() for ptr in staged_a_ptrs],
        'flags': [ptr.item() for ptr in flags_ptrs],
    }

    # Note: heap_bases are already cached in shmem.heap_bases_cpu (done in iris.py __init__)

    buffer_mb = M * K * A_sharded.element_size() / (1024**2)
    sa_stride_m, sa_stride_k = ws.aux_buffer.stride()
    shmem.info(
        f"Copy Engine: staged_a=({M},{K}) [{buffer_mb:.1f} MB] "
        f"layout={staged_a_layout} strides=({sa_stride_m},{sa_stride_k}), "
        f"NON-INTERLEAVED: each rank's K-blocks contiguous"
    )

    shmem.barrier()
    return ws


_WG_GEMM = 15
_WG_GEMM_WAIT = 16


def _extract_wg_trace(shmem, grid_size, num_tiles, **metadata):
    """Reconstruct per-tile trace arrays from DeviceTracing events.

    For persistent scheduling, grid_size is the number of workgroups (304),
    but num_tiles is the total number of output tiles (e.g., 8192).
    We trace each tile individually.
    """
    import numpy as np

    bufs = shmem.tracing.trace_buffers
    n = min(shmem.tracing.trace_counter.item(), shmem.tracing.max_events)

    event_ids = bufs["event_id"][:n].cpu().numpy()
    pid_ms = bufs["pid_m"][:n].cpu().numpy()  # tile_id is stored in pid_m
    timestamps = bufs["timestamp"][:n].cpu().numpy().astype(np.int64)
    end_ts = bufs["duration_cycles"][:n].cpu().numpy().astype(np.int64)
    xcc_ids = bufs["xcc_id"][:n].cpu().numpy().astype(np.int32)
    pid_ns = bufs["pid_n"][:n].cpu().numpy()

    starts = torch.zeros(num_tiles, dtype=torch.int64)
    ends = torch.zeros(num_tiles, dtype=torch.int64)
    waits = torch.zeros(num_tiles, dtype=torch.int64)
    xcds = torch.zeros(num_tiles, dtype=torch.int32)

    for i in range(n):
        eid = int(event_ids[i])
        tile_id = int(pid_ms[i])  # Use tile_id as the key
        if tile_id >= num_tiles:
            continue
        if eid == _WG_GEMM:
            starts[tile_id] = int(timestamps[i])
            ends[tile_id] = int(end_ts[i])
            xcds[tile_id] = int(xcc_ids[i])
        elif eid == _WG_GEMM_WAIT:
            waits[tile_id] = int(pid_ns[i])

    return {"start": starts, "end": ends, "wait": waits, "xcd": xcds, "grid_size": num_tiles, **metadata}


def all_gather_matmul_copy_engine(
    shmem,
    output_tensor: torch.Tensor,
    A_sharded: torch.Tensor,
    B: torch.Tensor,
    bias: Optional[torch.Tensor] = None,
    async_op: bool = False,
    config: Optional[FusedConfig] = None,
    workspace: Optional[FusedWorkspace] = None,
    k_per_flag: int = 1,
    staged_a_layout: str = "k_contiguous",
    num_warps: Optional[int] = None,
    num_stages: Optional[int] = None,
    trace: bool = False,
    verbose: bool = True,
) -> FusedWorkspace:
    """
    All-gather + matmul with copy engine orchestrating remote tile transfers.

    Key differences from HBM buffer:
    - No fetcher workgroups (only GEMM)
    - Host uses SDMA to copy remote tiles
    - GEMM processes local tiles first, then remote tiles

    Args:
        staged_a_layout: Buffer layout for gathered A.
            "k_contiguous" — (M,K) row-major, K is fast dim.
            "m_contiguous" — (M,K) with M as fast dim.
    """
    if config is None:
        config = FusedConfig()

    M, K_local = A_sharded.shape
    K, N = B.shape
    world_size = shmem.get_num_ranks()
    rank = shmem.get_rank()

    assert world_size * K_local == K
    assert output_tensor.shape == (M, N)
    assert M % config.block_size_m == 0
    assert K % config.block_size_k == 0
    assert K_local % config.block_size_k == 0

    num_k_blocks = K // config.block_size_k
    num_k_blocks_local = K_local // config.block_size_k
    num_remote_k_blocks = num_k_blocks - num_k_blocks_local

    # Allow non-multiples of k_per_flag (last batch will have fewer tiles)
    # assert num_remote_k_blocks % k_per_flag == 0

    if workspace is None:
        workspace = all_gather_matmul_copy_engine_preamble(shmem, A_sharded, B, config, k_per_flag, staged_a_layout)

    workspace.locks.zero_()

    stride_am, stride_ak = A_sharded.stride()
    stride_bk, stride_bn = B.stride()
    stride_cm, stride_cn = output_tensor.stride()
    stride_sa_m, stride_sa_k = workspace.aux_buffer.stride()

    if bias is not None:
        assert bias.shape[0] == M
        bias_ptr = bias
        stride_bias = bias.stride()[0] if bias.dim() > 0 else 1
        use_bias = True
    else:
        bias_ptr = output_tensor
        stride_bias = 1
        use_bias = False

    num_m_tiles = M // config.block_size_m
    num_tiles_n = (N + config.block_size_n - 1) // config.block_size_n
    total_tiles = num_m_tiles * num_tiles_n

    # Grid: persistent GEMM - auto-detect SM count for proper wave execution
    grid_size = config.num_sms
    if grid_size is None:
        import torch.cuda
        props = torch.cuda.get_device_properties(torch.cuda.current_device())
        grid_size = props.multi_processor_count

    if trace:
        # With persistent scheduling, we trace all tiles (not just workgroups)
        # Each tile generates 2 events (start + wait)
        max_trace_events = total_tiles * 2
        if not shmem.tracing.enabled:
            shmem.tracing.enable(max_events=max_trace_events)
        else:
            shmem.tracing.reset()

    launch_kwargs = {"matrix_instr_nonkdim": 16}
    if num_warps is not None:
        launch_kwargs["num_warps"] = num_warps
    if num_stages is not None:
        launch_kwargs["num_stages"] = num_stages

    # ======================================================================
    # Host orchestration: SDMA copy setup
    # ======================================================================
    anvil_lib = shmem.copy_engines
    torch.cuda.current_device()  # Initialize CUDA context

    # SDMA queues already connected during iris init
    if verbose and rank == 0:
        shmem.info(f"[Rank {rank}] Copy engines connected, launching kernel...")
        shmem.info(
            f"Kernel params: grid_size={grid_size}, num_m_tiles={num_m_tiles}, "
            f"num_tiles_n={num_tiles_n}, num_k_blocks={num_k_blocks}, "
            f"num_k_blocks_local={num_k_blocks_local}"
        )
        shmem.info(
            f"Pointers: A_sharded=0x{A_sharded.data_ptr():x}, "
            f"B=0x{B.data_ptr():x}, C=0x{output_tensor.data_ptr():x}, "
            f"bias_ptr=0x{bias_ptr.data_ptr():x}, "
            f"staged_a=0x{workspace.aux_buffer.data_ptr():x}, "
            f"flags=0x{workspace.locks.data_ptr():x} (n={workspace.locks.numel()})"
        )

    # Launch kernel (non-blocking initially for parallel execution)
    _copy_engine_all_gather_matmul_kernel[(grid_size,)](
        A_sharded,
        B,
        output_tensor,
        bias_ptr,
        workspace.aux_buffer,
        workspace.locks,
        M,
        N,
        K,
        K_local,
        stride_am,
        stride_ak,
        stride_bk,
        stride_bn,
        stride_cm,
        stride_cn,
        stride_sa_m,
        stride_sa_k,
        stride_bias,
        shmem.get_device_context(),
        rank,
        world_size,
        config.block_size_m,
        config.block_size_n,
        config.block_size_k,
        config.group_size_m,
        grid_size,
        config.num_xcds,
        num_m_tiles,
        num_tiles_n,
        num_k_blocks,
        num_k_blocks_local,
        num_remote_k_blocks,
        config.chunk_size,  # Pass chunk_size for XCD transform
        use_bias,
        config.allow_tf32,
        trace,
        **launch_kwargs,
    )

    # ======================================================================
    # SDMA transfer loop (WAVE-AWARE PUSH: copy M tiles in wave order)
    # ======================================================================
    # NEW: Copy tiles in the order they're consumed by waves for better overlap

    import time
    sdma_start_time = time.perf_counter()

    # Get wave schedule to determine which M tiles to copy for each wave
    num_tiles_n = (N + config.block_size_n - 1) // config.block_size_n
    wave_schedule = _get_wave_m_tile_schedule(
        grid_size, config.num_xcds, config.group_size_m,
        config.chunk_size,
        num_m_tiles, num_tiles_n
    )

    # Always show first 3 waves for debugging overlap issues
    if rank == 0:
        shmem.info(f"Wave schedule: {len(wave_schedule)} waves total")
        for wave_num, m_tiles in wave_schedule[:3]:
            shmem.info(f"  Wave {wave_num}: M tiles {m_tiles}")

    if verbose and rank == 0:
        # Debug: Check which M tiles are in the schedule
        all_scheduled_m_tiles = set()
        for _, m_tiles in wave_schedule:
            all_scheduled_m_tiles.update(m_tiles)

        shmem.info(
            f"[Rank {rank}] Starting WAVE-AWARE PUSH loop... "
            f"num_m_tiles={num_m_tiles}, num_k_blocks_local={num_k_blocks_local}, "
            f"num_waves={len(wave_schedule)}"
        )
        shmem.info(f"  M tiles in schedule: {sorted(all_scheduled_m_tiles)} ({len(all_scheduled_m_tiles)}/{num_m_tiles})")
        if len(all_scheduled_m_tiles) < num_m_tiles:
            missing = set(range(num_m_tiles)) - all_scheduled_m_tiles
            shmem.info(f"  WARNING: Missing M tiles: {sorted(missing)}")

    elem_size = A_sharded.element_size()
    staged_a_base_addr = workspace.aux_buffer.data_ptr()
    flags_base_addr = workspace.locks.data_ptr()
    tile_transfer_count = 0
    flag_update_count = 0

    # Track which M tiles we've already copied (to avoid duplicates across waves)
    copied_m_tiles = set()

    # Copy tiles in wave order
    for wave_num, m_tiles_to_copy in wave_schedule:
        # Only copy M tiles we haven't copied yet
        new_m_tiles = [m for m in m_tiles_to_copy if m not in copied_m_tiles]

        if not new_m_tiles:
            continue  # Nothing new to copy for this wave

        # For each destination rank, copy the new M tiles
        for dst_rank in range(world_size):
            if dst_rank == rank:
                continue  # Don't push to ourselves

            # Copy each new M tile in ONE SDMA call (non-interleaved layout)
            for m_tile in new_m_tiles:
                # Source: entire M tile from A_sharded (256 x K_local)
                m_start = m_tile * config.block_size_m

                # Destination: contiguous block in staged_a at this rank's K offset
                # NON-INTERLEAVED: staged_a[m, rank*K_local:(rank+1)*K_local]
                k_offset_global = rank * num_k_blocks_local * config.block_size_k
                dst_offset_bytes = (m_start * stride_sa_m + k_offset_global * stride_sa_k) * elem_size
                dst_ptr_local = staged_a_base_addr + dst_offset_bytes
                dst_ptr_remote = shmem.translate(dst_ptr_local, rank, dst_rank)

                # Create Tile struct for 2D transfer - ENTIRE M tile!
                tile = Tile()
                tile.data = A_sharded.data_ptr()
                tile.pid_m = m_tile
                tile.pid_n = 0  # Start of K dimension
                tile.block_m = config.block_size_m
                tile.block_n = K_local  # ENTIRE K_local in one call!
                tile.elem_size = elem_size
                tile.src_stride = stride_am * elem_size

                # Signal flag address
                flag_idx = m_tile
                flag_addr_local = flags_base_addr + flag_idx * 4
                flag_addr_remote = shmem.translate(flag_addr_local, rank, dst_rank)

                # Perform 2D SDMA transfer + atomic signal in ONE submission!
                anvil_lib.host_put_tile_signal(
                    rank, dst_rank, 0, tile, dst_ptr_remote, stride_sa_m * elem_size, flag_addr_remote, 1
                )
                tile_transfer_count += 1
                flag_update_count += 1

        # Mark these M tiles as copied
        copied_m_tiles.update(new_m_tiles)

        if verbose and rank == 0 and wave_num < 3:
            shmem.info(
                f"  Wave {wave_num}: copied {len(new_m_tiles)} M tiles "
                f"(total {len(copied_m_tiles)}/{num_m_tiles})"
            )

    sdma_end_post_time = time.perf_counter()
    # Ensure all SDMA operations complete
    # for dst_rank in range(world_size):
    #     if rank == dst_rank:
    #         continue
    #     # Wait for all SDMA operations to this destination to complete
    #     anvil_lib.host_quiet(rank, dst_rank, 0)

    sdma_end_time = time.perf_counter()
    sdma_elapsed_ms = (sdma_end_time - sdma_start_time) * 1000.0
    sdma_post_elapsed_ms = (sdma_end_post_time - sdma_start_time) * 1000.0

    if verbose and rank == 0:
        shmem.info(
            f"[Rank {rank}] PUSH complete. SDMA time: {sdma_elapsed_ms:.2f} ms "
            f"(time to post SDMA commands: {sdma_post_elapsed_ms:.2f} ms) "
            f"transfers={tile_transfer_count}, flags={flag_update_count}"
        )

    # ======================================================================
    # Synchronize
    # ======================================================================
    if not async_op:
        torch.cuda.synchronize()  # Wait for kernel completion
        shmem.barrier()

    if trace:
        torch.cuda.synchronize()
        total_tiles = num_m_tiles * num_tiles_n
        workspace.trace_data = _extract_wg_trace(
            shmem,
            grid_size,
            total_tiles,
            num_m_tiles=num_m_tiles,
            num_tiles_n=num_tiles_n,
        )
