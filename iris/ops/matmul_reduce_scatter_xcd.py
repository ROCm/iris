# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.

"""
XCD-aware fused GEMM + ReduceScatter.

Key insight from Fleet: co-locate GEMM and comm WGs on the same XCD.
The GEMM→comm flag handoff becomes intra-XCD (device-scope, nearly free)
instead of sys-scope (0.06ms overhead).

Architecture:
  - Each XCD handles a subset of tiles
  - Within each XCD: first N WGs do GEMM, rest do comm
  - GEMM→comm sync is device-scope (cheap)
  - Comm WGs do iris.load from peers (cross-rank, sys-scope for data only)
  - XCD ID discovered at runtime via HW_REG_XCC_ID

MI355X: 8 XCDs × 38 CUs = 304 CUs total.
"""

from typing import Optional
import torch
import triton
import triton.language as tl
import iris


@triton.jit
def _get_xcd_id():
    """Read current XCD ID from hardware register (MI300/MI350/MI355X)."""
    return tl.inline_asm_elementwise(
        "s_getreg_b32 $0, hwreg(HW_REG_XCC_ID, 0, 16)",
        "=s", [], dtype=tl.int32, is_pure=True, pack=1,
    )


@triton.jit
def _xcd_aware_gemm_rs_kernel(
    A, B,
    C_staged,
    C_out,
    tile_flags,
    M, N, K, M_local,
    stride_am, stride_ak,
    stride_bk, stride_bn,
    stride_sc_m, stride_sc_n,
    stride_out_m, stride_out_n,
    heap_bases: tl.tensor,
    cur_rank: tl.constexpr,
    world_size: tl.constexpr,
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr,
    GROUP_SIZE_M: tl.constexpr,
    GEMM_SMS_PER_XCD: tl.constexpr,
    NUM_XCDS: tl.constexpr,
    CUS_PER_XCD: tl.constexpr,
    NUM_SMS: tl.constexpr,
    EVEN_K: tl.constexpr,
    NUM_M_TILES: tl.constexpr,
    NUM_N_TILES: tl.constexpr,
    NUM_LOCAL_M_TILES: tl.constexpr,
    TOTAL_TILES: tl.constexpr,
    TOTAL_LOCAL_TILES: tl.constexpr,
):
    pid = tl.program_id(0)
    acc_dtype = tl.float32

    # XCD-aware PID mapping: group consecutive PIDs onto same XCD
    # pid -> (xcd_id, local_pid_within_xcd)
    xcd_id = pid % NUM_XCDS
    local_pid = pid // NUM_XCDS

    # Within each XCD: first GEMM_SMS_PER_XCD WGs do GEMM, rest do comm
    COMM_SMS_PER_XCD: tl.constexpr = CUS_PER_XCD - GEMM_SMS_PER_XCD

    if local_pid < GEMM_SMS_PER_XCD:
        # ==============================================================
        # GEMM PHASE — this WG computes GEMM tiles
        # ==============================================================
        gemm_pid = xcd_id * GEMM_SMS_PER_XCD + local_pid
        total_gemm_sms = NUM_XCDS * GEMM_SMS_PER_XCD

        for tile_id in range(gemm_pid, TOTAL_TILES, total_gemm_sms):
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

            c = acc.to(C_staged.type.element_ty)
            sc_offset = rm[:, None] * stride_sc_m + rn[None, :] * stride_sc_n
            tl.store(C_staged + sc_offset, c, cache_modifier=".wt")

            # Signal tile done using atomic_add counter:
            # Each rank adds 1 to the flag on ALL ranks (including self).
            # Comm WGs wait until flag == world_size (all ranks done).
            # Local add is device-scope (cheap).
            # Remote adds are sys-scope (fire-and-forget, 1 per peer per tile).
            tl.debug_barrier()
            tl.atomic_add(tile_flags + tile_id, 1, sem="release", scope="gpu")
            for peer in tl.static_range(world_size):
                if peer != cur_rank:
                    iris.atomic_add(
                        tile_flags + tile_id, 1,
                        cur_rank, peer, heap_bases,
                        sem="release", scope="sys",
                    )

    elif local_pid < CUS_PER_XCD:
        # ==============================================================
        # COMM PHASE — iris.load from peers + reduce in registers
        # ==============================================================
        comm_local = local_pid - GEMM_SMS_PER_XCD
        comm_pid = xcd_id * COMM_SMS_PER_XCD + comm_local
        total_comm_sms = NUM_XCDS * COMM_SMS_PER_XCD

        m_offset = cur_rank * NUM_LOCAL_M_TILES

        for tile_id in range(comm_pid, TOTAL_LOCAL_TILES, total_comm_sms):
            local_pid_m = tile_id // NUM_N_TILES
            pid_n = tile_id % NUM_N_TILES
            global_pid_m = m_offset + local_pid_m
            global_tile_id = global_pid_m * NUM_N_TILES + pid_n

            # Wait until ALL ranks have signaled this tile.
            # Each rank adds 1 to our local flag → wait for flag >= world_size.
            # Poll is DEVICE-scope (cheap!) — peers pushed via fire-and-forget sys-scope writes.
            while tl.atomic_add(tile_flags + global_tile_id, 0, sem="acquire", scope="gpu") < world_size:
                pass

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


def matmul_reduce_scatter_xcd(
    ctx,
    output_tensor: torch.Tensor,
    A: torch.Tensor,
    B: torch.Tensor,
    block_m: int = 128,
    block_n: int = 64,
    block_k: int = 64,
    group_m: int = 4,
    gemm_sms_per_xcd: int = 30,
    num_xcds: int = 8,
    cus_per_xcd: int = 38,
    num_warps: int = 8,
):
    """
    XCD-aware fused GEMM+RS.

    Co-locates GEMM and comm WGs on the same XCD for cheap device-scope
    flag handoff. Cross-rank data reads via iris.load (sys scope for data only).
    """
    M, K = A.shape
    _, N = B.shape
    rank = ctx.get_rank()
    world_size = ctx.get_num_ranks()
    M_local = M // world_size

    assert output_tensor.shape == (M_local, N)
    assert M % (world_size * block_m) == 0
    assert gemm_sms_per_xcd < cus_per_xcd

    num_m_tiles = M // block_m
    num_n_tiles = (N + block_n - 1) // block_n
    total_tiles = num_m_tiles * num_n_tiles
    num_local_m_tiles = M_local // block_m
    total_local_tiles = num_local_m_tiles * num_n_tiles

    staged_c = ctx.zeros((M, N), dtype=A.dtype)
    tile_flags = torch.zeros(total_tiles, dtype=torch.int32, device=f"cuda:{rank}")
    heap_bases = ctx.get_heap_bases()

    num_sms = num_xcds * cus_per_xcd

    # No host barrier — cross-rank sync via per-tile atomic counters.
    # GEMM WGs push flags to all peers. Comm WGs poll locally.

    launch_kwargs = {}
    if getattr(torch.version, "hip", None):
        launch_kwargs["matrix_instr_nonkdim"] = 16

    _xcd_aware_gemm_rs_kernel[(num_sms,)](
        A, B, staged_c, output_tensor, tile_flags,
        M, N, K, M_local,
        A.stride(0), A.stride(1),
        B.stride(0), B.stride(1),
        staged_c.stride(0), staged_c.stride(1),
        output_tensor.stride(0), output_tensor.stride(1),
        heap_bases, rank, world_size,
        block_m, block_n, block_k, group_m,
        gemm_sms_per_xcd, num_xcds, cus_per_xcd, num_sms,
        K % block_k == 0,
        num_m_tiles, num_n_tiles, num_local_m_tiles,
        total_tiles, total_local_tiles,
        num_warps=num_warps, num_stages=2,
        **launch_kwargs,
    )

    ctx.barrier()
