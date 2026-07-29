# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.

"""
Fused GEMM + Ring Reduce-Scatter.

Architecture:
  - GEMM WGs compute partial C = A_local @ B, store to staged_c (symmetric heap)
  - Scatter WGs do ring RS: ws-1 steps, each step reads 1 neighbor, accumulates
  - Per-tile flags for GEMM→scatter sync, per-step flags for ring step sync
  - O(ws-1) XGMI hops total — same bandwidth as RCCL ring

Ring RS algorithm:
  - staged_c split into ws chunks along M (chunk_r = rows r*M_local..(r+1)*M_local)
  - Step 0: rank r reads chunk (r-1)%ws from rank (r-1)%ws, adds to own chunk (r-1)%ws
  - Step s: rank r reads chunk (r-s-1)%ws from rank (r-1)%ws (which has s+1 accumulated partials)
  - After ws-1 steps: rank r's chunk r is fully reduced

Data flow:
  A (M, K_local) x B (K_local, N) -> staged_c (M, N) -> ring RS -> C (M_local, N)
"""

from typing import Optional
import torch
import triton
import triton.language as tl
import iris

from tritonblas.kernels.stages import GemmContext, make_tensor_view, Tile

from .config import FusedConfig
from .workspace import FusedWorkspace


@triton.jit
def _fused_ring_reduce_scatter_kernel(
    A,
    B,
    C,
    staged_c,
    gemm_flags_ptr,
    ring_flags_ptr,
    M,
    N,
    K_local,
    M_local,
    stride_am,
    stride_ak,
    stride_bk,
    stride_bn,
    stride_cm,
    stride_cn,
    stride_sc_m,
    stride_sc_n,
    heap_bases: tl.tensor,
    context_tensor: tl.tensor,
    cur_rank: tl.constexpr,
    world_size: tl.constexpr,
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr,
    GROUP_SIZE_M: tl.constexpr,
    NUM_SCATTER_SMS: tl.constexpr,
    NUM_M_TILES: tl.constexpr,
    NUM_TILES_N: tl.constexpr,
    NUM_M_TILES_LOCAL: tl.constexpr,
    TOTAL_LOCAL_TILES: tl.constexpr,
    GEMM_TILES_PER_STAGE: tl.constexpr,
    ALLOW_TF32: tl.constexpr,
    EVEN_K: tl.constexpr,
):
    pid = tl.program_id(0)
    acc_dtype = tl.float32

    ctx = iris.DeviceContext.initialize(context_tensor, cur_rank, world_size)

    if pid < GEMM_TILES_PER_STAGE:
        # ==============================================================
        # GEMM PHASE — compute partial C = A_local @ B, store to staged_c
        # ==============================================================
        gemm_pid = pid

        num_pid_in_group = GROUP_SIZE_M * NUM_TILES_N
        group_id = gemm_pid // num_pid_in_group
        first_pid_m = group_id * GROUP_SIZE_M
        first_pid_m = min(first_pid_m, NUM_M_TILES - 1)
        group_sz = min(NUM_M_TILES - first_pid_m, GROUP_SIZE_M)
        pid_m = first_pid_m + ((gemm_pid % num_pid_in_group) % group_sz)
        pid_n = (gemm_pid % num_pid_in_group) // group_sz
        pid_m = min(pid_m, NUM_M_TILES - 1)

        tensorA = make_tensor_view(A, M, K_local, stride_am, stride_ak)
        tensorB = make_tensor_view(B, K_local, N, stride_bk, stride_bn)
        gemm_ctx = GemmContext(
            BLOCK_SIZE_M, BLOCK_SIZE_N, BLOCK_SIZE_K,
            num_sms=1, even_k=EVEN_K,
        )
        out_tile = Tile(pid_m, pid_n, BLOCK_SIZE_M, BLOCK_SIZE_N)
        acc = gemm_ctx.reduce_axis(tensorA, tensorB, out_tile)

        rm, rn = out_tile.indices()
        c = acc.to(staged_c.type.element_ty)

        rm_base = pid_m * BLOCK_SIZE_M
        rn_base = pid_n * BLOCK_SIZE_N
        is_full = (rm_base + BLOCK_SIZE_M <= M) & (rn_base + BLOCK_SIZE_N <= N)

        sc_ptrs = staged_c + rm[:, None] * stride_sc_m + rn[None, :] * stride_sc_n
        if is_full:
            tl.store(sc_ptrs, c, cache_modifier=".wt")
        else:
            sc_mask = (rm[:, None] < M) & (rn[None, :] < N)
            tl.store(sc_ptrs, c, mask=sc_mask, cache_modifier=".wt")

        tile_id = pid_m * NUM_TILES_N + pid_n
        tl.debug_barrier()
        tl.atomic_xchg(gemm_flags_ptr + tile_id, 1, sem="release", scope="sys")

    else:
        # ==============================================================
        # SCATTER PHASE — ring reduce-scatter
        # ==============================================================
        scatter_pid = pid - GEMM_TILES_PER_STAGE

        local_base = tl.load(heap_bases + cur_rank).to(tl.uint64)

        # Ring RS: ws-1 steps
        # Step s: rank r processes chunk (r - s - 1) % ws
        #   - reads that chunk from rank (r-1) % ws (prev neighbor)
        #   - adds to own staged_c for that chunk
        #   - signals completion via ring_flags

        prev_rank = (cur_rank + world_size - 1) % world_size
        prev_base = tl.load(heap_bases + prev_rank).to(tl.uint64)

        # Offset of staged_c in the symmetric heap
        staged_c_offset = staged_c.to(tl.uint64) - local_base

        for tile_offset in range(scatter_pid, TOTAL_LOCAL_TILES, NUM_SCATTER_SMS):
            pid_n = tile_offset % NUM_TILES_N
            local_pid_m = tile_offset // NUM_TILES_N

            rn = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
            rn = tl.max_contiguous(tl.multiple_of(rn % N, BLOCK_SIZE_N), BLOCK_SIZE_N)

            rn_base = pid_n * BLOCK_SIZE_N
            is_full_n = rn_base + BLOCK_SIZE_N <= N

            for step in tl.static_range(world_size - 1):
                # Which chunk (in global M-tile space) are we processing this step?
                chunk_rank = (cur_rank - step - 1 + world_size) % world_size
                global_pid_m = chunk_rank * NUM_M_TILES_LOCAL + local_pid_m
                tile_id = global_pid_m * NUM_TILES_N + pid_n

                rm = global_pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
                rm = tl.max_contiguous(tl.multiple_of(rm, BLOCK_SIZE_M), BLOCK_SIZE_M)

                rm_base = global_pid_m * BLOCK_SIZE_M
                is_full = (rm_base + BLOCK_SIZE_M <= M) & is_full_n

                if step == 0:
                    # Step 0: wait for OWN GEMM to finish this tile, load own partial
                    gemm_flag_offset = gemm_flags_ptr.to(tl.uint64) - local_base + tile_id
                    while tl.atomic_add(gemm_flags_ptr + tile_id, 0, sem="acquire", scope="gpu") == 0:
                        pass

                    # Also wait for prev rank's GEMM to finish this tile
                    prev_gemm_flag_ptr = (prev_base + (gemm_flags_ptr.to(tl.uint64) - local_base) + tile_id).to(tl.pointer_type(tl.int32))
                    while tl.atomic_add(prev_gemm_flag_ptr, 0, sem="acquire", scope="sys") == 0:
                        pass
                else:
                    # Step s>0: wait for prev rank to finish step s-1 for this tile
                    prev_ring_flag_offset = (step - 1) * TOTAL_LOCAL_TILES + tile_offset
                    prev_ring_flag_ptr = (prev_base + (ring_flags_ptr.to(tl.uint64) - local_base) + prev_ring_flag_offset).to(tl.pointer_type(tl.int32))
                    while tl.atomic_add(prev_ring_flag_ptr, 0, sem="acquire", scope="sys") == 0:
                        pass

                # Read prev rank's staged_c for this tile
                sc_offset = rm[:, None] * stride_sc_m + rn[None, :] * stride_sc_n
                prev_sc_ptrs = (prev_base + staged_c_offset).to(staged_c.type) + sc_offset
                mask = (rm[:, None] < M) & (rn[None, :] < N)

                if is_full:
                    prev_tile = tl.load(prev_sc_ptrs, cache_modifier=".cv")
                    own_tile = tl.load(staged_c + sc_offset)
                    result = prev_tile.to(acc_dtype) + own_tile.to(acc_dtype)
                    result_cast = result.to(staged_c.type.element_ty)
                    tl.store(staged_c + sc_offset, result_cast, cache_modifier=".wt")
                else:
                    prev_tile = tl.load(prev_sc_ptrs, mask=mask, other=0.0, cache_modifier=".cv")
                    own_tile = tl.load(staged_c + sc_offset, mask=mask, other=0.0)
                    result = prev_tile.to(acc_dtype) + own_tile.to(acc_dtype)
                    result_cast = result.to(staged_c.type.element_ty)
                    tl.store(staged_c + sc_offset, result_cast, mask=mask, cache_modifier=".wt")

                # Signal this step is done
                ring_flag_offset = step * TOTAL_LOCAL_TILES + tile_offset
                tl.debug_barrier()
                tl.atomic_xchg(ring_flags_ptr + ring_flag_offset, 1, sem="release", scope="sys")

            # After ws-1 steps, our chunk (cur_rank) is fully reduced in staged_c
            # Copy to output C
            own_global_pid_m = cur_rank * NUM_M_TILES_LOCAL + local_pid_m
            own_rm = own_global_pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
            own_rm = tl.max_contiguous(tl.multiple_of(own_rm, BLOCK_SIZE_M), BLOCK_SIZE_M)

            out_rm = local_pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
            out_rm = tl.max_contiguous(tl.multiple_of(out_rm, BLOCK_SIZE_M), BLOCK_SIZE_M)

            sc_offset_own = own_rm[:, None] * stride_sc_m + rn[None, :] * stride_sc_n

            own_is_full = (own_global_pid_m * BLOCK_SIZE_M + BLOCK_SIZE_M <= M) & is_full_n
            if own_is_full:
                final = tl.load(staged_c + sc_offset_own)
                out_ptrs = C + out_rm[:, None] * stride_cm + rn[None, :] * stride_cn
                tl.store(out_ptrs, final, cache_modifier=".wt")
            else:
                out_mask = (out_rm[:, None] < M_local) & (rn[None, :] < N)
                final = tl.load(staged_c + sc_offset_own, mask=(own_rm[:, None] < M) & (rn[None, :] < N), other=0.0)
                out_ptrs = C + out_rm[:, None] * stride_cm + rn[None, :] * stride_cn
                tl.store(out_ptrs, final, mask=out_mask, cache_modifier=".wt")


def matmul_reduce_scatter_ring_preamble(
    ctx,
    A: torch.Tensor,
    B: torch.Tensor,
    config: Optional[FusedConfig] = None,
) -> FusedWorkspace:
    if config is None:
        config = FusedConfig(block_size_m=128, block_size_n=64, block_size_k=64, group_size_m=4)

    M, K_local = A.shape
    _, N = B.shape
    world_size = ctx.get_num_ranks()

    assert M % config.block_size_m == 0
    assert M % world_size == 0

    num_m_tiles = M // config.block_size_m
    num_tiles_n = (N + config.block_size_n - 1) // config.block_size_n
    total_tiles = num_m_tiles * num_tiles_n
    num_m_tiles_local = (M // world_size) // config.block_size_m
    total_local_tiles = num_m_tiles_local * num_tiles_n

    ws = FusedWorkspace(
        operation="matmul_reduce_scatter_ring",
        shape=(M, N, K_local),
        dtype=A.dtype,
        world_size=world_size,
        variant="ring",
        prepared=True,
    )

    ws.aux_buffer = ctx.zeros((M, N), dtype=A.dtype)
    ws.locks = ctx.zeros((total_tiles,), dtype=torch.int32)
    # Ring flags: (ws-1) steps × total_local_tiles
    ws.ring_flags = ctx.zeros(((world_size - 1) * total_local_tiles,), dtype=torch.int32)

    ctx.barrier()
    return ws


def matmul_reduce_scatter_ring(
    ctx,
    output_tensor: torch.Tensor,
    A: torch.Tensor,
    B: torch.Tensor,
    async_op: bool = False,
    config: Optional[FusedConfig] = None,
    workspace: Optional[FusedWorkspace] = None,
    num_scatter_sms: int = 32,
    num_warps: Optional[int] = 8,
    num_stages: Optional[int] = 2,
) -> FusedWorkspace:
    if config is None:
        config = FusedConfig(block_size_m=128, block_size_n=64, block_size_k=64, group_size_m=4)

    M, K_local = A.shape
    _, N = B.shape
    rank = ctx.get_rank()
    world_size = ctx.get_num_ranks()
    M_local = M // world_size

    assert output_tensor.shape == (M_local, N)
    assert M % config.block_size_m == 0
    assert M % world_size == 0

    if workspace is None:
        workspace = matmul_reduce_scatter_ring_preamble(ctx, A, B, config)

    workspace.locks.zero_()
    workspace.ring_flags.zero_()

    num_m_tiles = M // config.block_size_m
    num_tiles_n = (N + config.block_size_n - 1) // config.block_size_n
    gemm_tiles = num_m_tiles * num_tiles_n
    num_m_tiles_local = M_local // config.block_size_m
    total_local_tiles = num_m_tiles_local * num_tiles_n

    grid_size = gemm_tiles + num_scatter_sms

    launch_kwargs = {}
    if getattr(torch.version, "hip", None):
        launch_kwargs["matrix_instr_nonkdim"] = 16
    if num_warps is not None:
        launch_kwargs["num_warps"] = num_warps
    if num_stages is not None:
        launch_kwargs["num_stages"] = num_stages

    _fused_ring_reduce_scatter_kernel[(grid_size,)](
        A,
        B,
        output_tensor,
        workspace.aux_buffer,
        workspace.locks,
        workspace.ring_flags,
        M,
        N,
        K_local,
        M_local,
        A.stride(0),
        A.stride(1),
        B.stride(0),
        B.stride(1),
        output_tensor.stride(0),
        output_tensor.stride(1),
        workspace.aux_buffer.stride(0),
        workspace.aux_buffer.stride(1),
        ctx.get_heap_bases(),
        ctx.get_device_context(),
        rank,
        world_size,
        config.block_size_m,
        config.block_size_n,
        config.block_size_k,
        config.group_size_m,
        num_scatter_sms,
        num_m_tiles,
        num_tiles_n,
        num_m_tiles_local,
        total_local_tiles,
        gemm_tiles,
        config.allow_tf32,
        K_local % config.block_size_k == 0,
        **launch_kwargs,
    )

    if not async_op:
        ctx.barrier()

    return workspace
