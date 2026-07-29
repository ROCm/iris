# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.

"""
Fused GEMM + ReduceScatter — single kernel, 2-way WG specialization.

Architecture (mirrors ex22 but with iris.load pull instead of iris.atomic_add push):
  - GEMM WGs: compute partial C → staged_c (.wt), signal via monotonic counter
  - Comm WGs: poll per-tile counter on ALL peers → iris.load + accumulate in fp32 → store

Cross-rank sync uses monotonic counters (never zeroed, increment per call).
Same pattern as _per_block_barrier from allreduce graph capture work.
"""

from typing import Optional
import torch
import triton
import triton.language as tl
import iris

from .config import FusedConfig
from .workspace import FusedWorkspace


@triton.jit
def _translate_ptr(ptr, from_rank, to_rank, heap_bases):
    from_base = tl.load(heap_bases + from_rank)
    to_base = tl.load(heap_bases + to_rank)
    offset = ptr.to(tl.uint64) - from_base
    return (to_base + offset).to(ptr.type)


@triton.jit
def _fused_gemm_rs_kernel(
    A,
    B,
    C_staged,
    C_out,
    tile_flags,
    M,
    N,
    K,
    M_local,
    stride_am,
    stride_ak,
    stride_bk,
    stride_bn,
    stride_sc_m,
    stride_sc_n,
    stride_out_m,
    stride_out_n,
    heap_bases: tl.tensor,
    iteration: tl.constexpr,
    cur_rank: tl.constexpr,
    world_size: tl.constexpr,
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr,
    GROUP_SIZE_M: tl.constexpr,
    GEMM_SMS: tl.constexpr,
    NUM_SMS: tl.constexpr,
    EVEN_K: tl.constexpr,
):
    pid = tl.program_id(0)
    acc_dtype = tl.float32

    num_pid_m = tl.cdiv(M, BLOCK_SIZE_M)
    num_pid_n = tl.cdiv(N, BLOCK_SIZE_N)
    total_tiles = num_pid_m * num_pid_n
    M_per_rank = M // world_size
    num_local_m_tiles = M_per_rank // BLOCK_SIZE_M
    total_local_tiles = num_local_m_tiles * num_pid_n

    if pid < GEMM_SMS:
        # ==============================================================
        # GEMM PHASE — persistent, compute all tiles
        # ==============================================================
        for tile_id in range(pid, total_tiles, GEMM_SMS):
            num_pid_in_group = GROUP_SIZE_M * num_pid_n
            group_id = tile_id // num_pid_in_group
            first_pid_m = group_id * GROUP_SIZE_M
            group_size_m = min(num_pid_m - first_pid_m, GROUP_SIZE_M)
            pid_m = first_pid_m + ((tile_id % num_pid_in_group) % group_size_m)
            pid_n = (tile_id % num_pid_in_group) // group_size_m

            rm = (pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)) % M
            rn = (pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)) % N
            rm = tl.max_contiguous(tl.multiple_of(rm, BLOCK_SIZE_M), BLOCK_SIZE_M)
            rn = tl.max_contiguous(tl.multiple_of(rn, BLOCK_SIZE_N), BLOCK_SIZE_N)

            rk = tl.arange(0, BLOCK_SIZE_K)
            A_BASE = A + rm[:, None] * stride_am + rk[None, :] * stride_ak
            B_BASE = B + rk[:, None] * stride_bk + rn[None, :] * stride_bn

            loop_k = tl.cdiv(K, BLOCK_SIZE_K)
            if not EVEN_K:
                loop_k -= 1

            acc = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=acc_dtype)
            for k in range(0, loop_k):
                a = tl.load(tl.multiple_of(A_BASE, (1, 16)))
                b = tl.load(tl.multiple_of(B_BASE, (16, 1)))
                acc += tl.dot(a, b)
                A_BASE += BLOCK_SIZE_K * stride_ak
                B_BASE += BLOCK_SIZE_K * stride_bk

            if not EVEN_K:
                rk2 = loop_k * BLOCK_SIZE_K + tl.arange(0, BLOCK_SIZE_K)
                A_LAST = A + rm[:, None] * stride_am + rk2[None, :] * stride_ak
                B_LAST = B + rk2[:, None] * stride_bk + rn[None, :] * stride_bn
                a = tl.load(A_LAST, mask=rk2[None, :] < K, other=0.0)
                b = tl.load(B_LAST, mask=rk2[:, None] < K, other=0.0)
                acc += tl.dot(a, b)

            c = acc.to(C_staged.type.element_ty)
            sc_offset = rm[:, None] * stride_sc_m + rn[None, :] * stride_sc_n
            tl.store(C_staged + sc_offset, c, cache_modifier=".wt")

            # Monotonic counter signal — increment on ALL ranks (including self)
            tl.debug_barrier()
            flag_ptr = tile_flags + tile_id
            my_flag = _translate_ptr(flag_ptr, cur_rank, cur_rank, heap_bases)
            tl.atomic_add(my_flag, 1, sem="release", scope="sys")

            for peer in tl.static_range(world_size):
                peer_rank = peer
                if peer_rank != cur_rank:
                    peer_flag = _translate_ptr(flag_ptr, cur_rank, peer_rank, heap_bases)
                    tl.atomic_add(peer_flag, 1, sem="release", scope="sys")

    else:
        # ==============================================================
        # COMM PHASE — pull from all peers + reduce in registers
        # ==============================================================
        COMM_SMS = NUM_SMS - GEMM_SMS
        comm_pid = pid - GEMM_SMS

        m_offset = cur_rank * num_local_m_tiles
        target = iteration

        for tile_id in range(comm_pid, total_local_tiles, COMM_SMS):
            local_pid_m = tile_id // num_pid_n
            pid_n = tile_id % num_pid_n
            global_pid_m = m_offset + local_pid_m
            global_tile_id = global_pid_m * num_pid_n + pid_n

            # Wait for ALL peers to signal this tile (monotonic: wait until >= target)
            for peer in tl.static_range(world_size):
                poll_ptr = tile_flags + global_tile_id
                poll_local = _translate_ptr(poll_ptr, cur_rank, cur_rank, heap_bases)
                while tl.atomic_cas(poll_local, target, target, sem="acquire", scope="sys") < target:
                    pass

            # Pull from all peers and accumulate in registers
            rm = global_pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
            rm = tl.max_contiguous(tl.multiple_of(rm, BLOCK_SIZE_M), BLOCK_SIZE_M)
            rn = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
            rn = tl.max_contiguous(tl.multiple_of(rn % N, BLOCK_SIZE_N), BLOCK_SIZE_N)

            sc_offset = rm[:, None] * stride_sc_m + rn[None, :] * stride_sc_n
            base_ptr = C_staged + sc_offset
            is_full = (global_pid_m * BLOCK_SIZE_M + BLOCK_SIZE_M <= M) & (
                pid_n * BLOCK_SIZE_N + BLOCK_SIZE_N <= N
            )

            if is_full:
                start_rank = comm_pid % world_size
                acc = iris.load(base_ptr, cur_rank, start_rank, heap_bases, hint=(1, BLOCK_SIZE_N)).to(acc_dtype)
                for i in tl.static_range(1, world_size):
                    r = (start_rank + i) % world_size
                    acc += iris.load(base_ptr, cur_rank, r, heap_bases, hint=(1, BLOCK_SIZE_N)).to(acc_dtype)

                out_rm = local_pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
                out_rm = tl.max_contiguous(tl.multiple_of(out_rm, BLOCK_SIZE_M), BLOCK_SIZE_M)
                tl.store(C_out + out_rm[:, None] * stride_out_m + rn[None, :] * stride_out_n,
                         acc.to(C_out.type.element_ty))
            else:
                mask = (rm[:, None] < M) & (rn[None, :] < N)
                start_rank = comm_pid % world_size
                acc = iris.load(base_ptr, cur_rank, start_rank, heap_bases, mask=mask, hint=(1, BLOCK_SIZE_N)).to(acc_dtype)
                for i in tl.static_range(1, world_size):
                    r = (start_rank + i) % world_size
                    acc += iris.load(base_ptr, cur_rank, r, heap_bases, mask=mask, hint=(1, BLOCK_SIZE_N)).to(acc_dtype)

                out_rm = local_pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
                out_mask = (out_rm[:, None] < M_local) & (rn[None, :] < N)
                tl.store(C_out + out_rm[:, None] * stride_out_m + rn[None, :] * stride_out_n,
                         acc.to(C_out.type.element_ty), mask=out_mask)


def matmul_reduce_scatter_fused(
    ctx,
    output_tensor: torch.Tensor,
    A: torch.Tensor,
    B: torch.Tensor,
    config: Optional[FusedConfig] = None,
    workspace: Optional[FusedWorkspace] = None,
    gemm_sms: int = 240,
    num_sms: int = 304,
    num_warps: int = 8,
):
    """
    Single-kernel fused GEMM+RS with 2-way WG specialization.

    Monotonic counter flags for cross-rank sync. No flag reset needed.
    """
    if config is None:
        config = FusedConfig(block_size_m=128, block_size_n=64, block_size_k=64, group_size_m=4)

    M, K = A.shape
    _, N = B.shape
    rank = ctx.get_rank()
    world_size = ctx.get_num_ranks()
    M_local = M // world_size

    assert output_tensor.shape == (M_local, N)

    num_m_tiles = M // config.block_size_m
    num_n_tiles = (N + config.block_size_n - 1) // config.block_size_n
    total_tiles = num_m_tiles * num_n_tiles

    if workspace is None:
        workspace = FusedWorkspace(
            operation="matmul_reduce_scatter_fused",
            shape=(M, N, K),
            dtype=A.dtype,
            world_size=world_size,
            variant="fused_monotonic",
            prepared=True,
        )
        workspace.aux_buffer = ctx.zeros((M, N), dtype=A.dtype)
        # tile_flags: one int32 per tile, monotonic counters, in symmetric heap
        workspace.locks = ctx.zeros((total_tiles,), dtype=torch.int32)
        workspace._iteration = 0
        ctx.barrier()

    workspace._iteration += 1

    launch_kwargs = {}
    if getattr(torch.version, "hip", None):
        launch_kwargs["matrix_instr_nonkdim"] = 16

    _fused_gemm_rs_kernel[(num_sms,)](
        A, B,
        workspace.aux_buffer,
        output_tensor,
        workspace.locks,
        M, N, K, M_local,
        A.stride(0), A.stride(1),
        B.stride(0), B.stride(1),
        workspace.aux_buffer.stride(0), workspace.aux_buffer.stride(1),
        output_tensor.stride(0), output_tensor.stride(1),
        ctx.get_heap_bases(),
        workspace._iteration,
        rank, world_size,
        config.block_size_m, config.block_size_n, config.block_size_k,
        config.group_size_m,
        gemm_sms, num_sms,
        K % config.block_size_k == 0,
        num_warps=num_warps, num_stages=2,
        **launch_kwargs,
    )

    return workspace
