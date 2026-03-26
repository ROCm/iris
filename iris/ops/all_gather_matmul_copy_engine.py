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
    M_TILES_PER_FLAG: tl.constexpr,
    BIAS: tl.constexpr,
    ALLOW_TF32: tl.constexpr,
    TRACE: tl.constexpr,
):
    """
    Non-persistent GEMM kernel with whole M-tile transfers.

    Grid: (total_tiles,) - one workgroup per output tile
    Each workgroup waits for its M-tile-group, then processes all K-blocks.

    All K-blocks (local and remote) are loaded from staged_a - no branches.
    Host pre-copies local K-blocks to staged_a before kernel launch.
    """
    acc_dtype = tl.int32 if C.type.element_ty == tl.int8 else tl.float32
    zero = tl.program_id(0) * 0

    ctx = iris.DeviceContext.initialize(context_tensor, cur_rank, world_size, tracing=TRACE)

    # Non-persistent GEMM: one workgroup per output tile (like hbm_buffer)
    # Use wave-aware tile assignment with swizzling
    pid = tl.program_id(0)

    # Wave-aware tile assignment (similar to hbm_buffer's group-based assignment)
    num_pid_in_group = GROUP_SIZE_M * NUM_TILES_N
    group_id = pid // num_pid_in_group
    first_pid_m = group_id * GROUP_SIZE_M
    first_pid_m = min(first_pid_m, NUM_M_TILES - 1)
    group_sz = min(NUM_M_TILES - first_pid_m, GROUP_SIZE_M)
    pid_m = first_pid_m + ((pid % num_pid_in_group) % group_sz)
    pid_n = (pid % num_pid_in_group) // group_sz
    pid_m = min(pid_m, NUM_M_TILES - 1)

    # Initialize accumulator
    acc_dtype = tl.int32 if C.type.element_ty == tl.int8 else tl.float32
    acc = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=acc_dtype)

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
            pid_m=pid,
            pid_n=zero,
        )
        _tile_wt = zero.to(tl.int64)  # Wait time for THIS tile only

    # ==================================================================
    # PHASE: Wait for entire M-tile-group, then process ALL K-blocks
    # ==================================================================
    # Wait for all remote ranks to transfer their K-blocks for this M-tile's group

    if TRACE:
        _ws = read_realtime()

    # Single wait for entire M-tile group (all remote K-blocks)
    m_tile_group_id = pid_m // M_TILES_PER_FLAG
    flag_idx = m_tile_group_id
    while tl.atomic_add(flags_ptr + flag_idx, 0, sem="acquire", scope="sys") < (world_size - 1):
        pass

    if TRACE:
        _tile_wt = (read_realtime() - _ws)

    # Process ALL K-blocks in global order (all from staged_a - no branch)
    for k_block_global in range(NUM_K_BLOCKS):
        # Load from staged_a (includes both local and remote K-blocks)
        rk = k_block_global * BLOCK_SIZE_K + tl.arange(0, BLOCK_SIZE_K)
        rk = tl.max_contiguous(tl.multiple_of(rk, BLOCK_SIZE_K), BLOCK_SIZE_K)

        a_ptrs = staged_a + rm.to(tl.int64)[:, None] * stride_sa_m + rk[None, :] * stride_sa_k
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
            pid_m=pid,
            pid_n=_tile_wt.to(tl.int32),
        )


# ==========================================================================
# Python API
# ==========================================================================


def _get_wave_m_tile_schedule(num_sms, num_tiles_m, num_tiles_n, group_size_m):
    """
    Analyze which M tiles need to be transferred before each wave (non-persistent scheduling).

    Returns:
        List of (wave_num, [m_tiles_to_copy]) tuples in execution order.
    """
    # Non-persistent: wave-aware tile assignment (matches kernel logic)
    total_tiles = num_tiles_m * num_tiles_n
    num_waves = (total_tiles + num_sms - 1) // num_sms

    waves = {}
    for pid in range(min(total_tiles, num_sms * num_waves)):
        wave_num = pid // num_sms

        # Wave-aware assignment (matches kernel)
        num_pid_in_group = group_size_m * num_tiles_n
        group_id = pid // num_pid_in_group
        first_pid_m = group_id * group_size_m
        first_pid_m = min(first_pid_m, num_tiles_m - 1)
        group_sz = min(num_tiles_m - first_pid_m, group_size_m)
        pid_m = first_pid_m + ((pid % num_pid_in_group) % group_sz)
        pid_m = min(pid_m, num_tiles_m - 1)

        if wave_num not in waves:
            waves[wave_num] = set()
        waves[wave_num].add(pid_m)

    # Convert to ordered schedule (only transfer new M tiles each wave)
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
    m_tiles_per_flag: int = 1,
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

    # Per-M-tile-group flags for batched M-tile transfers
    # Each flag tracks when all remote K-blocks are ready for a group of M-tiles
    num_m_tile_groups = (num_m_tiles + m_tiles_per_flag - 1) // m_tiles_per_flag
    num_flags = num_m_tile_groups

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

    shmem.info(
        f"Allocated {num_flags} per-M-tile-group flags "
        f"({num_m_tiles} M-tiles, {m_tiles_per_flag} M-tiles per flag) "
        f"flags buffer at 0x{ws.locks.data_ptr():x}"
    )

    # Copy local K-blocks into staged_a (one-time setup in preamble)
    # This allows kernel to always load from staged_a without branches
    rank = shmem.get_rank()
    k_start = rank * K_local
    k_end = (rank + 1) * K_local
    ws.aux_buffer[:, k_start:k_end].copy_(A_sharded)

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
    m_tiles_per_flag: int = 1,
    staged_a_layout: str = "k_contiguous",
    num_warps: Optional[int] = None,
    num_stages: Optional[int] = None,
    trace: bool = False,
    verbose: bool = True,
    debug: bool = False,
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
        workspace = all_gather_matmul_copy_engine_preamble(shmem, A_sharded, B, config, m_tiles_per_flag, staged_a_layout)

    workspace.locks.zero_()

    # Note: Local K-blocks already copied to staged_a in preamble
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

    # Calculate M-tile groups for flag structure
    num_m_tile_groups = (num_m_tiles + m_tiles_per_flag - 1) // m_tiles_per_flag

    # Grid: non-persistent GEMM - one workgroup per output tile (like hbm_buffer)
    grid_size = total_tiles

    if trace:
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
            f"num_k_blocks_local={num_k_blocks_local}, group_size_m={config.group_size_m}, "
            f"m_tiles_per_flag={m_tiles_per_flag}"
        )
        shmem.info(
            f"Pointers: A_sharded=0x{A_sharded.data_ptr():x}, "
            f"B=0x{B.data_ptr():x}, C=0x{output_tensor.data_ptr():x}, "
            f"bias_ptr=0x{bias_ptr.data_ptr():x}, "
            f"staged_a=0x{workspace.aux_buffer.data_ptr():x}, "
            f"flags=0x{workspace.locks.data_ptr():x} (n={workspace.locks.numel()})"
        )

    # Launch kernel (non-blocking to allow parallel SDMA posting)
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
        m_tiles_per_flag,
        use_bias,
        config.allow_tf32,
        trace,
        **launch_kwargs,
    )

    # ======================================================================
    # SDMA transfer loop: Global K-block ordering (like hbm_buffer)
    # ======================================================================

    import time
    sdma_start_time = time.perf_counter()

    # Analyze wave schedule to understand M-tile access patterns
    props = torch.cuda.get_device_properties(torch.cuda.current_device())
    num_sms = props.multi_processor_count
    wave_schedule = _get_wave_m_tile_schedule(num_sms, num_m_tiles, num_tiles_n, config.group_size_m)

    if verbose and rank == 0:
        shmem.info(
            f"[Rank {rank}] Starting SDMA loop (batched M-tile transfers)... "
            f"num_m_tiles={num_m_tiles}, m_tiles_per_flag={m_tiles_per_flag}, num_groups={num_m_tile_groups}"
        )
        shmem.info(f"[Rank {rank}] Wave schedule: {len(wave_schedule)} waves")
        for wave_num, m_tiles in wave_schedule[:3]:  # Show first 3 waves
            shmem.info(f"  Wave {wave_num}: needs {len(m_tiles)} new M-tiles: {m_tiles[:10]}{'...' if len(m_tiles) > 10 else ''}")

    elem_size = A_sharded.element_size()
    staged_a_base_addr = workspace.aux_buffer.data_ptr()
    flags_base_addr = workspace.locks.data_ptr()
    tile_transfer_count = 0

    # For each M-tile group, batch transfer all M-tiles to all remote ranks
    for m_tile_group_id in range(num_m_tile_groups):
        m_tile_start = m_tile_group_id * m_tiles_per_flag
        m_tile_end = min(m_tile_start + m_tiles_per_flag, num_m_tiles)
        num_m_tiles_in_group = m_tile_end - m_tile_start

        for dst_rank in range(world_size):
            if dst_rank == rank:
                continue

            # Batch transfer all M-tiles in this group as one 2D transfer
            # Source: A_sharded[m_tile_start:m_tile_end, :]
            # Shape: (num_m_tiles_in_group * block_size_m, num_k_blocks_local * block_size_k)
            k_block_start_global = rank * num_k_blocks_local

            tile = Tile()
            tile.pid_m = m_tile_start
            tile.pid_n = 0  # Our K-blocks start at column 0 in A_sharded
            tile.block_m = num_m_tiles_in_group * config.block_size_m  # Batch multiple M-tiles
            tile.block_n = num_k_blocks_local * config.block_size_k  # Entire width
            tile.elem_size = elem_size
            tile.src_stride = stride_am * elem_size
            tile.data = A_sharded.data_ptr()

            # Destination offset: m_tile_start * M + k_block_global_start * K
            dst_offset_bytes = (m_tile_start * config.block_size_m * stride_sa_m + k_block_start_global * config.block_size_k * stride_sa_k) * elem_size
            dst_ptr_local = staged_a_base_addr + dst_offset_bytes
            dst_ptr_remote = shmem.translate(dst_ptr_local, rank, dst_rank)

            # Get flag address
            flag_idx = m_tile_group_id
            flag_addr_local = flags_base_addr + flag_idx * 4
            flag_addr_remote = shmem.translate(flag_addr_local, rank, dst_rank)

            # Batched transfer + signal in one SDMA command
            anvil_lib.host_put_tile_signal(rank, dst_rank, 0, tile, dst_ptr_remote, stride_sa_m * elem_size, flag_addr_remote, 1)
            tile_transfer_count += 1

            if debug and m_tile_group_id < 2:
                shmem.info(
                    f"[Rank {rank}] Transferred M-tile-group={m_tile_group_id} (M-tiles {m_tile_start}-{m_tile_end-1}) "
                    f"to dst_rank={dst_rank}, flag_idx={flag_idx}, size={tile.block_m}×{tile.block_n}"
                )

    sdma_end_post_time = time.perf_counter()

    if debug:
        shmem.info(f"[Rank {rank}] Posted all SDMA, now calling host_quiet...")

    # Ensure all SDMA operations complete
    for dst_rank in range(world_size):
        if dst_rank == rank:
            continue
        if debug:
            shmem.info(f"[Rank {rank}] Calling host_quiet for dst_rank={dst_rank}")
        anvil_lib.host_quiet(rank, dst_rank, 0)
        if debug:
            shmem.info(f"[Rank {rank}] host_quiet complete for dst_rank={dst_rank}")

    sdma_end_time = time.perf_counter()

    if debug:
        shmem.info(f"[Rank {rank}] All SDMA complete")
    sdma_elapsed_ms = (sdma_end_time - sdma_start_time) * 1000.0
    sdma_post_elapsed_ms = (sdma_end_post_time - sdma_start_time) * 1000.0

    if verbose:
        shmem.info(
            f"[Rank {rank}] PUSH complete. SDMA time: {sdma_elapsed_ms:.2f} ms "
            f"(time to post SDMA commands: {sdma_post_elapsed_ms:.2f} ms) "
            f"transfers={tile_transfer_count}"
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
