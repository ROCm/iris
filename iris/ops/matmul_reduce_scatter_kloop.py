# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.

"""
Fused GEMM + RS with K-loop interleaved push.

After each K-block of GEMM, push the partial accumulation to the owner.
The owner accumulates across both K-blocks and ranks simultaneously.
Uses atomic_add on the owner's output buffer.

Key insight: XGMI stores are hidden behind MFMA compute for the next
K-block. At K_local=2048 with bk=64, that's 32 K-iterations — each
pushes a partial and the next MFMA hides the XGMI latency.

Simpler than WG specialization — every WG does everything.
No staging buffers — accumulate directly into owner's output via atomics.
"""

from typing import Optional
import torch
import triton
import triton.language as tl
import iris


@triton.jit
def _kloop_push_rs_kernel(
    A, B,
    C_out,
    M, N, K, M_local,
    stride_am, stride_ak,
    stride_bk, stride_bn,
    stride_out_m, stride_out_n,
    heap_bases: tl.tensor,
    cur_rank: tl.constexpr,
    world_size: tl.constexpr,
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr,
    GROUP_SIZE_M: tl.constexpr,
    NUM_SMS: tl.constexpr,
    EVEN_K: tl.constexpr,
    NUM_M_TILES: tl.constexpr,
    NUM_N_TILES: tl.constexpr,
    TOTAL_TILES: tl.constexpr,
):
    """
    Each WG: for each tile, do full GEMM K-loop, then push final result
    to owner via iris.atomic_add. Owner accumulates across all ranks.

    This is simpler than K-block-level push — pushes once per tile, not
    per K-block. The GEMM K-loop runs at full speed, then one XGMI push.
    """
    pid = tl.program_id(0)
    acc_dtype = tl.float32

    for tile_id in range(pid, TOTAL_TILES, NUM_SMS):
        num_pid_in_group = GROUP_SIZE_M * NUM_N_TILES
        group_id = tile_id // num_pid_in_group
        first_pid_m = group_id * GROUP_SIZE_M
        group_size_m = min(NUM_M_TILES - first_pid_m, GROUP_SIZE_M)
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

        c = acc.to(C_out.type.element_ty)

        # Push result to owner via iris.atomic_add
        tile_m_start = pid_m * BLOCK_SIZE_M
        owner_rank = tile_m_start // M_local

        local_m = tile_m_start - owner_rank * M_local
        out_rm = local_m + tl.arange(0, BLOCK_SIZE_M)
        out_rn = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
        out_offset = out_rm[:, None] * stride_out_m + out_rn[None, :] * stride_out_n
        out_mask = (out_rm[:, None] < M_local) & (out_rn[None, :] < N)

        if owner_rank == cur_rank:
            tl.atomic_add(C_out + out_offset, c, mask=out_mask)
        else:
            iris.atomic_add(
                C_out + out_offset, c,
                cur_rank, owner_rank, heap_bases,
                mask=out_mask,
                sem="relaxed", scope="sys",
            )


def matmul_reduce_scatter_kloop(
    ctx,
    output_tensor: torch.Tensor,
    A: torch.Tensor,
    B: torch.Tensor,
    block_m: int = 128,
    block_n: int = 64,
    block_k: int = 64,
    group_m: int = 4,
    num_sms: int = 304,
    num_warps: int = 8,
):
    """
    Fused GEMM+RS: full GEMM then push final tile to owner via atomic_add.

    Same as example 22 but without WG specialization — all WGs do compute+push.
    Simplest possible single-kernel fusion.
    """
    M, K = A.shape
    _, N = B.shape
    rank = ctx.get_rank()
    world_size = ctx.get_num_ranks()
    M_local = M // world_size

    assert output_tensor.shape == (M_local, N)
    assert M % (world_size * block_m) == 0

    num_m_tiles = M // block_m
    num_n_tiles = (N + block_n - 1) // block_n
    total_tiles = num_m_tiles * num_n_tiles

    output_tensor.zero_()
    heap_bases = ctx.get_heap_bases()
    ctx.barrier()

    launch_kwargs = {}
    if getattr(torch.version, "hip", None):
        launch_kwargs["matrix_instr_nonkdim"] = 16

    _kloop_push_rs_kernel[(num_sms,)](
        A, B, output_tensor,
        M, N, K, M_local,
        A.stride(0), A.stride(1),
        B.stride(0), B.stride(1),
        output_tensor.stride(0), output_tensor.stride(1),
        heap_bases, rank, world_size,
        block_m, block_n, block_k, group_m,
        num_sms, K % block_k == 0,
        num_m_tiles, num_n_tiles, total_tiles,
        num_warps=num_warps, num_stages=2,
        **launch_kwargs,
    )

    ctx.barrier()
