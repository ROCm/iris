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
from tritonblas.matmul import persistent_matmul_lt
from tritonblas.config import PrologueConfig

from .workspace import FusedWorkspace


@triton.jit
def _batch_poster_kernel(
    A_sharded,
    staged_a,
    flags_ptr,
    flag_iteration,
    M,
    K_local,
    stride_am,
    stride_sa_m,
    stride_sa_k,
    context_tensor: tl.tensor,
    heap_bases_ptr: tl.tensor,
    copy_engine_ctx: tl.tensor,
    cur_rank: tl.constexpr,
    world_size: tl.constexpr,
    BLOCK_SIZE_M: tl.constexpr,
    NUM_M_TILES: tl.constexpr,
    M_TILES_PER_BATCH: tl.constexpr,
):
    """Post one SDMA transfer per (batch, rank) including local copy."""
    ctx = iris.DeviceContext.initialize(context_tensor, cur_rank, world_size)

    pid = tl.program_id(0)
    dst_rank = pid

    ptr_dtype = A_sharded.dtype.element_ty
    if ptr_dtype == tl.float16 or ptr_dtype == tl.bfloat16:
        elem_size = 2
    elif ptr_dtype == tl.float32 or ptr_dtype == tl.int32:
        elem_size = 4
    elif ptr_dtype == tl.float64 or ptr_dtype == tl.int64:
        elem_size = 8
    else:
        elem_size = 4

    num_batches = (NUM_M_TILES + M_TILES_PER_BATCH - 1) // M_TILES_PER_BATCH
    rows_per_batch = M_TILES_PER_BATCH * BLOCK_SIZE_M

    for batch_id in range(num_batches):
        src_m_offset = batch_id * rows_per_batch
        remaining_rows = M - src_m_offset
        tile_height = tl.minimum(remaining_rows, rows_per_batch)

        src_m_offset_i64 = (src_m_offset + 0 * stride_am).to(tl.int64)
        stride_am_i64 = (stride_am + 0 * src_m_offset).to(tl.int64)
        src_ptr = A_sharded + src_m_offset_i64 * stride_am_i64

        dst_m_offset = src_m_offset
        dst_k_offset = cur_rank * K_local
        dst_m_offset_i64 = (dst_m_offset + 0 * stride_sa_m).to(tl.int64)
        dst_k_offset_i64 = (dst_k_offset + 0 * stride_sa_k).to(tl.int64)
        stride_sa_m_i64 = (stride_sa_m + 0 * dst_m_offset).to(tl.int64)
        stride_sa_k_i64 = (stride_sa_k + 0 * dst_m_offset).to(tl.int64)
        dst_ptr = staged_a + (dst_m_offset_i64 * stride_sa_m_i64 + dst_k_offset_i64 * stride_sa_k_i64)

        tile_width_bytes = K_local * elem_size
        src_pitch_bytes = stride_am * elem_size
        dst_pitch_bytes = stride_sa_m * elem_size

        iris.put_signal_rect(
            src_ptr,
            dst_ptr,
            cur_rank,
            dst_rank,
            heap_bases_ptr,
            copy_engine_ctx,
            flags_ptr + batch_id,
            1,
            width_bytes=tile_width_bytes,
            height=tile_height,
            src_pitch=src_pitch_bytes,
            dst_pitch=dst_pitch_bytes,
        )


def all_gather_matmul_copy_engine_preamble(
    shmem,
    A_sharded: torch.Tensor,
    B: torch.Tensor,
    selector=None,
    m_tiles_per_batch: Optional[int] = None,
    staged_a_layout: str = "k_contiguous",
) -> FusedWorkspace:
    """
    Allocate workspace for copy engine variant.

    Args:
        staged_a_layout: "k_contiguous" (default, row-major (M,K)) or
                         "m_contiguous" (col-major, stored as (K,M) transposed).
    """
    shmem._require_copy_engine()

    from tritonblas.matmul import _make_matmul_selector

    M, K_local = A_sharded.shape
    K, N = B.shape
    world_size = shmem.get_num_ranks()

    assert world_size * K_local == K

    if selector is None:
        selector = _make_matmul_selector(
            M, N, K, A_sharded.dtype, B.dtype, A_sharded.dtype, A_sharded.device, streamk=False
        )

    assert K_local % selector.block_k == 0
    assert K % selector.block_k == 0
    assert M % selector.block_m == 0

    num_m_tiles = M // selector.block_m
    num_tiles_n = (N + selector.block_n - 1) // selector.block_n
    total_tiles = num_m_tiles * num_tiles_n

    if m_tiles_per_batch is None:
        # Auto-calculate optimal m_tiles_per_batch
        active_cus = getattr(selector, "_ACTIVE_CU", None)
        if active_cus is None or active_cus <= 0:
            active_cus = torch.cuda.get_device_properties(torch.cuda.current_device()).multi_processor_count
        tiles_per_group = max(1, selector.group_m * num_tiles_n)
        groups_per_wave = max(1, int(active_cus) // tiles_per_group)
        m_tiles_per_batch = max(1, min(num_m_tiles, groups_per_wave * selector.group_m))

    num_batches = (num_m_tiles + m_tiles_per_batch - 1) // m_tiles_per_batch
    num_flags = num_batches
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

    # Allocate per-batch flags
    ws.locks = shmem.zeros((num_flags,), dtype=torch.int32)
    ws.wait_expected = shmem.zeros((total_tiles,), dtype=torch.int32)

    # Store metadata
    ws.selector = selector
    ws.m_tiles_per_batch = m_tiles_per_batch
    ws.num_m_tiles = num_m_tiles
    ws.num_batches = num_batches

    shmem.info(
        f"Allocated {num_flags} per-batch flags "
        f"(tiles={num_m_tiles}, "
        f"{m_tiles_per_batch} M-tiles per batch) "
        f"flags buffer at 0x{ws.locks.data_ptr():x}"
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

    buffer_mb = M * K * A_sharded.element_size() / (1024**2)
    sa_stride_m, sa_stride_k = ws.aux_buffer.stride()
    shmem.info(
        f"Copy Engine: staged_a=({M},{K}) [{buffer_mb:.1f} MB] "
        f"layout={staged_a_layout} strides=({sa_stride_m},{sa_stride_k}), "
        f"NON-INTERLEAVED: each rank's K-blocks contiguous"
    )

    shmem.barrier()
    return ws


def all_gather_matmul_copy_engine(
    shmem,
    output_tensor: torch.Tensor,
    A_sharded: torch.Tensor,
    B: torch.Tensor,
    bias: Optional[torch.Tensor] = None,
    async_op: bool = False,
    workspace: Optional[FusedWorkspace] = None,
    flag_iteration: int = 0,
    staged_a_layout: str = "k_contiguous",
    num_warps: Optional[int] = None,
    num_stages: Optional[int] = None,
    trace: bool = False,
    verbose: bool = False,
    device_initiated: bool = False,
) -> FusedWorkspace:
    """
    All-gather + matmul with copy engine orchestrating remote tile transfers.

    Key differences from HBM buffer:
    - No fetcher workgroups (only GEMM)
    - Host uses SDMA to copy remote tiles (default) OR device WGs initiate SDMA (device_initiated=True)
    - GEMM processes local tiles first, then remote tiles

    Args:
        staged_a_layout: Buffer layout for gathered A.
            "k_contiguous" — (M,K) row-major, K is fast dim.
            "m_contiguous" — (M,K) with M as fast dim.
        device_initiated: If True, use device-side WGs to initiate SDMA transfers instead of host.
        trace: Not supported yet; must be False.
        k_per_flag: Retained for call compatibility; ignored by the current per-batch design.
    """
    # Fail fast: without the SDMA engine the GEMM launched below would wait forever on flags.
    shmem._require_copy_engine()
    if trace:
        raise NotImplementedError("trace=True is not supported for all_gather_matmul_copy_engine yet")

    M, K_local = A_sharded.shape
    K, N = B.shape
    world_size = shmem.get_num_ranks()
    rank = shmem.get_rank()

    assert world_size * K_local == K
    assert output_tensor.shape == (M, N)

    if workspace is None:
        workspace = all_gather_matmul_copy_engine_preamble(shmem, A_sharded, B, staged_a_layout=staged_a_layout)

    selector = workspace.selector
    m_tiles_per_batch = workspace.m_tiles_per_batch

    assert M % selector.block_m == 0
    assert K % selector.block_k == 0
    assert K_local % selector.block_k == 0

    num_k_blocks_local = K_local // selector.block_k
    num_m_tiles = M // selector.block_m
    num_tiles_n = (N + selector.block_n - 1) // selector.block_n

    # Local K-blocks will be copied via SDMA (host or device initiated)
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

    # num_m_tiles, num_tiles_n already calculated above
    total_tiles = num_m_tiles * num_tiles_n

    # Auto-detect num_sms from device if not specified
    num_comm_wgs = selector._hardware.NUM_XCD if device_initiated else 0
    gemm_tiles = total_tiles

    # m_tiles_per_batch was already set above before calling preamble
    # Calculate number of batches
    num_batches = (num_m_tiles + m_tiles_per_batch - 1) // m_tiles_per_batch

    launch_kwargs = {"matrix_instr_nonkdim": 16}
    if num_warps is not None:
        launch_kwargs["num_warps"] = num_warps
    if num_stages is not None:
        launch_kwargs["num_stages"] = num_stages

    # ======================================================================
    # Host orchestration: SDMA copy setup
    # ======================================================================
    torch.cuda.current_device()  # Initialize CUDA context

    # SDMA queues already connected during iris init
    if verbose and rank == 0:
        shmem.info(f"[Rank {rank}] Copy engines connected, launching kernel...")
        shmem.info(
            f"Kernel params: num_m_tiles={num_m_tiles}, "
            f"num_tiles_n={num_tiles_n}, num_k_blocks_local={num_k_blocks_local}, "
            f"group_size_m={selector.group_m}, m_tiles_per_batch={m_tiles_per_batch}"
        )
        shmem.info(
            f"Pointers: A_sharded=0x{A_sharded.data_ptr():x}, "
            f"B=0x{B.data_ptr():x}, C=0x{output_tensor.data_ptr():x}, "
            f"bias_ptr=0x{bias_ptr.data_ptr():x}, "
            f"staged_a=0x{workspace.aux_buffer.data_ptr():x}, "
            f"flags=0x{workspace.locks.data_ptr():x} (n={workspace.locks.numel()})"
        )

    if verbose and rank == 0:
        shmem.info(
            f"Launching kernel: gemm_tiles={gemm_tiles}, device_initiated={device_initiated}, sdma_wgs={num_comm_wgs}"
        )

    tb_block_m = selector.block_m
    prologue_config = PrologueConfig(
        wait_buffer=workspace.locks,
        expected_buffer=workspace.wait_expected,
        expected_inc=world_size,
        map_type="block",
        block_group_m=m_tiles_per_batch,
        block_group_n=num_tiles_n,
    )

    if use_bias:
        import warnings

        warnings.warn(
            "Bias is not yet supported in the persistent tritonBLAS path for all_gather_matmul_copy_engine. "
            "Ignoring bias for this launch."
        )

    # ======================================================================
    # Launch kernel and orchestrate SDMA transfers
    # ======================================================================

    import time

    sdma_start_time = time.perf_counter()

    if device_initiated:
        # Device-initiated: keep the known-good split path in production.
        # The combined COMM_WGS+GEMM kernel remains in this file as an
        # experimental option, but we do not use it by default.
        poster_grid = world_size
        _batch_poster_kernel[(poster_grid,)](
            A_sharded,
            workspace.aux_buffer,
            workspace.locks,
            flag_iteration,
            M,
            K_local,
            stride_am,
            stride_sa_m,
            stride_sa_k,
            shmem.get_device_context(),
            shmem.get_heap_bases(),
            shmem.get_copy_engine_ctx(),
            rank,
            world_size,
            selector.block_m,
            num_m_tiles,
            m_tiles_per_batch,
        )
        persistent_matmul_lt(
            workspace.aux_buffer,
            B,
            output_tensor,
            selector,
            bias=None,
            prologue_config=prologue_config,
        )
    else:
        # Host-initiated:
        # Launch GEMM first, then post batches to overlap SDMA.
        elem_size = A_sharded.element_size()
        staged_a_base_addr = workspace.aux_buffer.data_ptr()
        flags_base_addr = workspace.locks.data_ptr()
        tile_transfer_count = 0

        def post_host_batch(batch_id: int, m_tile_start: int, num_m_tiles_in_batch: int) -> None:
            nonlocal tile_transfer_count

            for dst_rank in range(world_size):
                flag_idx = batch_id
                flag_addr_local = flags_base_addr + flag_idx * 4
                flag_addr_remote = shmem.heap.translate(flag_addr_local, rank, dst_rank)

                # Source is the local shard, so batches only advance in M.
                src_offset_bytes = (m_tile_start * selector.block_m * stride_am) * elem_size
                src_ptr = A_sharded.data_ptr() + src_offset_bytes

                # Destination is this rank's global-K slot inside staged_a.
                dst_offset_bytes = (
                    m_tile_start * selector.block_m * stride_sa_m + rank * K_local * stride_sa_k
                ) * elem_size
                dst_ptr_local = staged_a_base_addr + dst_offset_bytes
                dst_ptr_remote = shmem.heap.translate(dst_ptr_local, rank, dst_rank)

                # Use shmem.put_tile method with signal
                shmem.put_tile(
                    global_addr=src_ptr,
                    to_rank=dst_rank,
                    dst_addr=dst_ptr_remote,
                    data_size=elem_size,
                    tile_height=num_m_tiles_in_batch * selector.block_m,
                    tile_width=K_local,
                    src_stride=stride_am,
                    dst_stride=stride_sa_m,
                    signal_flag=flag_addr_remote,
                    signal_value=1,
                    async_op=True,
                    channel=0,
                )
                tile_transfer_count += 1

                if verbose and batch_id == 0 and dst_rank == (rank + 1) % world_size:
                    shmem.info(
                        f"[Rank {rank}→{dst_rank}] Signaled batch={batch_id} flag_idx={flag_idx} "
                        f"({num_m_tiles_in_batch} rows × full local K shard)"
                    )

        if verbose and rank == 0:
            num_batches_calc = workspace.locks.numel()
            shmem.info(
                f"[Rank {rank}] Starting SDMA loop (batched M-tile transfers)... "
                f"num_tiles_m={num_m_tiles}, num_k_blocks_local={num_k_blocks_local}, "
                f"m_tiles_per_batch={m_tiles_per_batch}, tb_block_m={tb_block_m}"
            )
            shmem.info(f"[Rank {rank}] Will transfer in {num_batches_calc} batches of {m_tiles_per_batch} M-tiles each")

        # TODO not always faster
        # Prime batch 0 before GEMM launch so the first released tile-group can
        # start immediately instead of stalling on an empty wait queue.
        # TODO issue with time measurement
        # first_batch_tiles = min(m_tiles_per_batch, num_m_tiles)
        # post_host_batch(0, 0, first_batch_tiles)

        if verbose and rank == 0:
            shmem.info(f"[Rank {rank}] Launching GEMM kernel after pre-posting batch 0...")

        persistent_matmul_lt(
            workspace.aux_buffer,
            B,
            output_tensor,
            selector,
            bias=None,
            prologue_config=prologue_config,
        )

        # Post the remaining batches while GEMM is already running.
        batch_id = 0
        for m_tile_start in range(0, num_m_tiles, m_tiles_per_batch):
            m_tile_end = min(m_tile_start + m_tiles_per_batch, num_m_tiles)
            num_m_tiles_in_batch = m_tile_end - m_tile_start
            post_host_batch(batch_id, m_tile_start, num_m_tiles_in_batch)
            batch_id += 1

        sdma_end_post_time = time.perf_counter()

        if verbose:
            # Ensure all SDMA operations complete
            for dst_rank in range(world_size):
                shmem.quiet(to_rank=dst_rank, channel=0)
            sdma_end_time = time.perf_counter()

            post_ms = (sdma_end_post_time - sdma_start_time) * 1000.0
            quiet_ms = (sdma_end_time - sdma_end_post_time) * 1000.0
            total_ms = (sdma_end_time - sdma_start_time) * 1000.0
            shmem.info(
                f"[Rank {rank}] SDMA complete. "
                f"Post: {post_ms:.2f}ms, Quiet: {quiet_ms:.2f}ms, Total: {total_ms:.2f}ms, "
                f"transfers={tile_transfer_count}"
            )
            sample_count = min(8, workspace.locks.numel())
            sample_flags = workspace.locks[:sample_count].cpu().tolist()
            shmem.info(
                f"[Rank {rank}] Flag sample after SDMA quiet: "
                f"expected_inc={world_size}, flags[:{sample_count}]={sample_flags}"
            )

    # ======================================================================
    # Synchronize
    # ======================================================================
    if not async_op:
        torch.cuda.synchronize()  # Wait for kernel completion
        shmem.barrier()

    return workspace
