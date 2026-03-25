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
    gemm_ctx = GemmContext(
        BLOCK_SIZE_M,
        BLOCK_SIZE_N,
        BLOCK_SIZE_K,
        num_sms=NUM_SMS,
        num_xcds=NUM_XCDS,
        group_size_m=GROUP_SIZE_M,
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
            _wt = zero.to(tl.int64)

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
        # PHASE 2: Process REMOTE K-blocks from staged_a (with 2D flag sync)
        # ==================================================================
        # Process by k_flag_group (like HBM buffer), waiting once per group
        # for ALL src_ranks to contribute their tiles for that k-range
        NUM_K_FLAG_GROUPS: tl.constexpr = (NUM_K_BLOCKS_LOCAL + K_PER_FLAG - 1) // K_PER_FLAG

        for k_flag_group in range(NUM_K_FLAG_GROUPS):
            # Wait for ALL remote ranks to finish transferring this k_flag_group
            if TRACE:
                _ws = read_realtime()

            flag_idx = pid_m * NUM_K_FLAG_GROUPS + k_flag_group
            expected_flag_value = world_size - 1
            while tl.atomic_add(flags_ptr + flag_idx, 0, sem="acquire", scope="sys") < expected_flag_value:
                pass

            if TRACE:
                _wt = _wt + (read_realtime() - _ws)

            # Process k_blocks in this group from all remote ranks
            k_block_start = k_flag_group * K_PER_FLAG
            k_block_end = min(k_block_start + K_PER_FLAG, NUM_K_BLOCKS_LOCAL)

            for k_block_local in range(k_block_start, k_block_end):
                for src_rank_idx in range(world_size):
                    if src_rank_idx != cur_rank:  # Process only remote ranks
                        # Load from staged_a using interleaved addressing
                        remote_rank_offset = src_rank_idx if src_rank_idx < cur_rank else src_rank_idx - 1
                        interleaved_k_idx = k_block_local * (world_size - 1) + remote_rank_offset

                        rk_staged = interleaved_k_idx * BLOCK_SIZE_K + tl.arange(0, BLOCK_SIZE_K)
                        rk_staged = tl.max_contiguous(tl.multiple_of(rk_staged, BLOCK_SIZE_K), BLOCK_SIZE_K)
                        a_ptrs = staged_a + rm.to(tl.int64)[:, None] * stride_sa_m + rk_staged[None, :] * stride_sa_k
                        a = tl.load(a_ptrs)

                        # Load B at global K position (src_rank's K)
                        k_block_global = src_rank_idx * NUM_K_BLOCKS_LOCAL + k_block_local
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
                pid_n=_wt.to(tl.int32),
            )


# ==========================================================================
# Python API
# ==========================================================================


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

    # Calculate flags with batching: 2D array [m_tile, k_flag_group]
    # Each k_flag_group covers k_per_flag K-blocks across ALL source ranks
    num_k_flag_groups = (num_k_blocks_local + k_per_flag - 1) // k_per_flag
    num_flags = num_m_tiles * num_k_flag_groups

    ws = FusedWorkspace(
        operation="all_gather_matmul_copy_engine",
        shape=(M, N, K),
        dtype=A_sharded.dtype,
        world_size=world_size,
        variant=f"copy_engine_{staged_a_layout}",
        prepared=True,
    )

    # Allocate staged_a - only needs to store REMOTE K-blocks (interleaved)
    K_remote = num_remote_k_blocks * config.block_size_k

    if staged_a_layout == "m_contiguous":
        storage = shmem.zeros((K_remote, M), dtype=A_sharded.dtype)
        ws.aux_buffer = storage.T  # (M, K_remote) with M-contiguous
    else:
        ws.aux_buffer = shmem.zeros((M, K_remote), dtype=A_sharded.dtype)

    # Allocate flags (batched - k_per_flag tiles per flag)
    ws.locks = shmem.zeros((num_flags,), dtype=torch.int32)

    total_tiles_tracked = num_m_tiles * num_k_blocks_local * (world_size - 1)
    shmem.info(
        f"Allocated {num_flags} flags ({num_m_tiles}×{num_k_flag_groups}) for {total_tiles_tracked} tiles "
        f"(batch size {k_per_flag}), flags buffer at 0x{ws.locks.data_ptr():x}"
    )

    # Share pointers across ranks for SDMA addressing
    # Need: A_sharded (source), staged_a (destination), flags (signaling)
    A_sharded_ptr_tensor = torch.tensor([A_sharded.data_ptr()], dtype=torch.int64, device="cuda")
    A_sharded_ptrs = [torch.zeros(1, dtype=torch.int64, device="cuda") for _ in range(world_size)]
    dist.all_gather(A_sharded_ptrs, A_sharded_ptr_tensor)

    staged_a_ptr_tensor = torch.tensor([ws.aux_buffer.data_ptr()], dtype=torch.int64, device="cuda")
    staged_a_ptrs = [torch.zeros(1, dtype=torch.int64, device="cuda") for _ in range(world_size)]
    dist.all_gather(staged_a_ptrs, staged_a_ptr_tensor)

    flags_ptr_tensor = torch.tensor([ws.locks.data_ptr()], dtype=torch.int64, device="cuda")
    flags_ptrs = [torch.zeros(1, dtype=torch.int64, device="cuda") for _ in range(world_size)]
    dist.all_gather(flags_ptrs, flags_ptr_tensor)

    # Store all remote pointers in workspace
    ws.remote_pointers = {
        "A_sharded": [ptr.item() for ptr in A_sharded_ptrs],
        "staged_a": [ptr.item() for ptr in staged_a_ptrs],
        "flags": [ptr.item() for ptr in flags_ptrs],
    }

    # Note: heap_bases are already cached in shmem.heap_bases_cpu (done in iris.py __init__)

    buffer_mb = M * K_remote * A_sharded.element_size() / (1024**2)
    sa_stride_m, sa_stride_k = ws.aux_buffer.stride()
    shmem.info(
        f"Copy Engine: staged_a=({M},{K_remote}) [{buffer_mb:.1f} MB] "
        f"layout={staged_a_layout} strides=({sa_stride_m},{sa_stride_k}), "
        f"interleaved by K-block across {world_size - 1} remote ranks"
    )

    shmem.barrier()
    return ws


_WG_GEMM = 15
_WG_GEMM_WAIT = 16


def _extract_wg_trace(shmem, grid_size, **metadata):
    """Reconstruct per-workgroup trace arrays from DeviceTracing events."""
    import numpy as np

    bufs = shmem.tracing.trace_buffers
    n = min(shmem.tracing.trace_counter.item(), shmem.tracing.max_events)

    event_ids = bufs["event_id"][:n].cpu().numpy()
    pids = bufs["pid"][:n].cpu().numpy()
    timestamps = bufs["timestamp"][:n].cpu().numpy().astype(np.int64)
    end_ts = bufs["duration_cycles"][:n].cpu().numpy().astype(np.int64)
    xcc_ids = bufs["xcc_id"][:n].cpu().numpy().astype(np.int32)
    pid_ns = bufs["pid_n"][:n].cpu().numpy()

    starts = torch.zeros(grid_size, dtype=torch.int64)
    ends = torch.zeros(grid_size, dtype=torch.int64)
    waits = torch.zeros(grid_size, dtype=torch.int64)
    xcds = torch.zeros(grid_size, dtype=torch.int32)

    for i in range(n):
        eid = int(event_ids[i])
        wg = int(pids[i])
        if wg >= grid_size:
            continue
        if eid == _WG_GEMM:
            starts[wg] = int(timestamps[i])
            ends[wg] = int(end_ts[i])
            xcds[wg] = int(xcc_ids[i])
        elif eid == _WG_GEMM_WAIT:
            waits[wg] = int(pid_ns[i])

    return {"start": starts, "end": ends, "wait": waits, "xcd": xcds, "grid_size": grid_size, **metadata}


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

    # Grid: persistent GEMM (typically NUM_SMS)
    grid_size = config.num_sms if config.num_sms is not None else num_m_tiles * num_tiles_n

    if trace:
        max_trace_events = grid_size * 4
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
    # Host orchestration: Connect copy engines and prepare for SDMA
    # ======================================================================
    anvil_lib = shmem.copy_engines
    torch.cuda.current_device()  # Initialize CUDA context

    # Connect to all remote ranks
    for remote_rank in range(world_size):
        if remote_rank != rank:
            anvil_lib.connect(rank, remote_rank, num_channels=1, allocate_on_host=True)

    if rank == 0:
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
        k_per_flag,
        use_bias,
        config.allow_tf32,
        trace,
        **launch_kwargs,
    )

    # ======================================================================
    # SDMA transfer loop (PUSH model: each rank pushes to remote staged_a)
    # ======================================================================
    # Each rank pushes its LOCAL A_sharded tiles to all OTHER ranks' staged_a buffers
    # in interleaved order, batching flag updates every k_per_flag tiles

    import time

    sdma_start_time = time.perf_counter()

    if rank == 0:
        shmem.info(
            f"[Rank {rank}] Starting PUSH loop with interleaving... "
            f"num_m_tiles={num_m_tiles}, num_k_blocks_local={num_k_blocks_local}"
        )

    elem_size = A_sharded.element_size()
    staged_a_base_addr = workspace.aux_buffer.data_ptr()
    flags_base_addr = workspace.locks.data_ptr()
    flag_update_count = 0

    # Calculate number of flags (same as preamble calculation)
    num_k_flag_groups = (num_k_blocks_local + k_per_flag - 1) // k_per_flag
    num_flags = num_m_tiles * num_k_flag_groups

    # Transfer tiles in interleaved order to each destination rank
    for dst_rank in range(world_size):
        if dst_rank == rank:
            continue  # Don't push to ourselves

        tile_transfer_count = 0
        flags_to_set = set()  # Track which flags we need to set for this destination

        # if rank == 0 and dst_rank == 1:
        #     shmem.info(f"[Rank {rank}] Starting transfers to dst_rank {dst_rank}")

        # Transfer tiles in the order they appear in dst_rank's staged_a buffer (interleaved)
        for m_tile in range(num_m_tiles):
            for k_block_local in range(num_k_blocks_local):
                for remote_rank_offset in range(world_size - 1):
                    # Determine which source rank this position is for (from dst_rank's perspective)
                    if remote_rank_offset < dst_rank:
                        src_rank = remote_rank_offset
                    else:
                        src_rank = remote_rank_offset + 1

                    if src_rank != rank:
                        continue  # Not our tile to push

                    # Calculate flag using 2D indexing: flag[m_tile, k_flag_group]
                    k_flag_group = k_block_local // k_per_flag
                    flag_idx = m_tile * num_k_flag_groups + k_flag_group
                    flags_to_set.add(flag_idx)  # Track this flag

                    # Source tile from A_sharded - use base pointer and pid to specify offset
                    m_start = m_tile * config.block_size_m
                    k_start = k_block_local * config.block_size_k

                    # Destination offset in staged_a (interleaved)
                    interleaved_k_idx = k_block_local * (world_size - 1) + remote_rank_offset

                    # Calculate destination pointer (base of interleaved position)
                    dst_offset_m = m_start
                    dst_offset_k = interleaved_k_idx * config.block_size_k
                    dst_offset_bytes = (dst_offset_m * stride_sa_m + dst_offset_k * stride_sa_k) * elem_size
                    dst_ptr_local = staged_a_base_addr + dst_offset_bytes
                    dst_ptr_remote = shmem.translate(dst_ptr_local, rank, dst_rank)

                    # Create Tile struct for 2D transfer
                    tile = Tile()
                    tile.data = A_sharded.data_ptr()  # Base pointer to A_sharded
                    tile.pid_m = m_tile  # Tile row coordinate
                    tile.pid_n = k_block_local  # Tile column coordinate (in K dimension)
                    tile.block_m = config.block_size_m
                    tile.block_n = config.block_size_k
                    tile.elem_size = elem_size
                    tile.src_stride = stride_am * elem_size  # A_sharded row stride in bytes

                    # Debug: verify tile parameters
                    if rank == 0 and dst_rank == 1 and m_tile == 0 and k_block_local == 0 and tile_transfer_count < 3:
                        shmem.info(
                            f"[Rank {rank}→{dst_rank}] Tile: m={tile.pid_m} n={tile.pid_n} "
                            f"block_m={tile.block_m} block_n={tile.block_n} "
                            f"src_stride={tile.src_stride} elem_size={elem_size} "
                            f"dst_offset_bytes={dst_offset_bytes} "
                            f"interleaved_k_idx={interleaved_k_idx}"
                        )

                    # Perform 2D SDMA transfer using sub-window copy
                    anvil_lib.host_put_tile(rank, dst_rank, 0, tile, dst_ptr_remote, stride_sa_m * elem_size)

                    tile_transfer_count += 1

                    # if rank == 0 and dst_rank == 1 and len(flags_set) <= 3:
                    #     shmem.info(
                    #         f"[Rank {rank}] Set flag {flag_batch_idx} at global_tile_id={global_tile_id} "
                    #         f"(m={m_tile}, remote_counter={remote_tile_counter})"
                    #     )

        # Wait for all SDMA transfers to complete
        # anvil_lib.host_quiet(rank, dst_rank, 0)

        # Set only the flags for the tile batches we transferred
        # Each rank increments only the flags it contributed tiles to
        for flag_idx in sorted(flags_to_set):
            flag_addr_local = flags_base_addr + flag_idx * 4
            flag_addr_remote = shmem.translate(flag_addr_local, rank, dst_rank)
            anvil_lib.host_atomic_add_32(rank, dst_rank, 0, flag_addr_remote, 1)
            flag_update_count += 1

        # Ensure flag atomics complete before proceeding
        # This guarantees memory ordering: data writes visible before flags
        # anvil_lib.host_quiet(rank, dst_rank, 0)

        # if rank == 0 and dst_rank == 1:
        #     shmem.info(
        #         f"[Rank {rank}] Completed {tile_transfer_count} tile transfers "
        #         f"to dst_rank {dst_rank}, set {len(flags_set)} unique flags"
        #     )

    for dst_rank in range(world_size):
        if rank == dst_rank:
            continue
        # Wait for all SDMA operations to this destination to complete
        anvil_lib.host_quiet(rank, dst_rank, 0)

    sdma_end_time = time.perf_counter()
    sdma_elapsed_ms = (sdma_end_time - sdma_start_time) * 1000.0

    if rank == 0:
        shmem.info(
            f"[Rank {rank}] PUSH complete. SDMA time: {sdma_elapsed_ms:.2f} ms "
            f"({tile_transfer_count * (world_size - 1)} total transfers across all ranks)"
            f"({flag_update_count} total flag updates)"
        )

    # ======================================================================
    # Synchronize
    # ======================================================================
    if not async_op:
        torch.cuda.synchronize()  # Wait for kernel completion
        shmem.barrier()

    if trace:
        torch.cuda.synchronize()
        workspace.trace_data = _extract_wg_trace(
            shmem,
            grid_size,
            num_m_tiles=num_m_tiles,
            num_tiles_n=num_tiles_n,
        )
