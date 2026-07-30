# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.

"""
Full fusion with batched scatter — no CU split, pipelined XGMI.

Every WG uses ALL 304 CUs for both GEMM and scatter (no specialization).
The problem with naive full fusion (ex07): each WG does
  compute tile (0.04us) -> XGMI push (3us round-trip) -> repeat
so the CU idles ~98% of the time waiting on the network.

Fix: compute TILES_PER_BATCH tiles into registers, THEN issue all their
scatters back-to-back so the XGMI ops pipeline instead of serializing.

Register budget: a bm x bn fp32 accumulator is bm*bn*4 bytes.
  bm=32 bn=64  ->  8 KB/tile -> can batch several
  bm=128 bn=256 -> 128 KB/tile -> can only hold one
So this needs SMALL tiles to allow deep batching.
"""

from typing import Optional
import torch
import triton
import triton.language as tl
import iris


@triton.jit
def _batched_fusion_kernel(
    A, B,
    C_out,
    M, N, K, M_local,
    stride_am, stride_ak,
    stride_bk, stride_bn,
    stride_out_m, stride_out_n,
    stride_slot,
    heap_bases: tl.tensor,
    cur_rank: tl.constexpr,
    world_size: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
    NUM_SMS: tl.constexpr,
    BATCH: tl.constexpr,      # tiles computed before scattering
    USE_SLOTS: tl.constexpr,  # bulk store to slots vs element-wise atomics
    EVEN_K: tl.constexpr,
    NUM_M_TILES: tl.constexpr,
    NUM_N_TILES: tl.constexpr,
    TOTAL_TILES: tl.constexpr,
):
    """
    Each WG: compute BATCH tiles into registers, then scatter all BATCH
    back-to-back so the XGMI pushes pipeline.

    Uses iris.atomic_add to accumulate directly into the owner's output.
    """
    pid = tl.program_id(0)
    acc_dtype = tl.float32

    # Stride by NUM_SMS*BATCH so each WG owns BATCH consecutive tiles per step
    for batch_base in range(pid * BATCH, TOTAL_TILES, NUM_SMS * BATCH):

        # ---- Phase A: compute BATCH tiles, keep in registers ----
        # Triton unrolls this static loop; each iteration's `acc` is a
        # separate register block, so all BATCH results are live at once.
        for b in tl.static_range(BATCH):
            tile_id = batch_base + b
            if tile_id < TOTAL_TILES:
                pid_m = tile_id // NUM_N_TILES
                pid_n = tile_id % NUM_N_TILES

                rm = (pid_m * BLOCK_M + tl.arange(0, BLOCK_M)) % M
                rn = (pid_n * BLOCK_N + tl.arange(0, BLOCK_N)) % N
                rm = tl.max_contiguous(tl.multiple_of(rm, BLOCK_M), BLOCK_M)
                rn = tl.max_contiguous(tl.multiple_of(rn, BLOCK_N), BLOCK_N)

                rk = tl.arange(0, BLOCK_K)
                A_BASE = A + rm[:, None] * stride_am + rk[None, :] * stride_ak
                B_BASE = B + rk[:, None] * stride_bk + rn[None, :] * stride_bn

                loop_k = tl.cdiv(K, BLOCK_K)
                if not EVEN_K:
                    loop_k -= 1

                acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=acc_dtype)
                for k in range(0, loop_k):
                    a = tl.load(tl.multiple_of(A_BASE, (1, 16)))
                    bb = tl.load(tl.multiple_of(B_BASE, (16, 1)))
                    acc += tl.dot(a, bb)
                    A_BASE += BLOCK_K * stride_ak
                    B_BASE += BLOCK_K * stride_bk

                if not EVEN_K:
                    rk2 = loop_k * BLOCK_K + tl.arange(0, BLOCK_K)
                    A_L = A + rm[:, None] * stride_am + rk2[None, :] * stride_ak
                    B_L = B + rk2[:, None] * stride_bk + rn[None, :] * stride_bn
                    a = tl.load(A_L, mask=rk2[None, :] < K, other=0.0)
                    bb = tl.load(B_L, mask=rk2[:, None] < K, other=0.0)
                    acc += tl.dot(a, bb)

                # ---- Phase B: scatter straight from registers ----
                # Owner rank for this tile's M-range
                tile_m_start = pid_m * BLOCK_M
                owner = tile_m_start // M_local
                local_m = tile_m_start - owner * M_local

                out_rm = local_m + tl.arange(0, BLOCK_M)
                out_rn = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
                out_off = out_rm[:, None] * stride_out_m + out_rn[None, :] * stride_out_n
                out_msk = (out_rm[:, None] < M_local) & (out_rn[None, :] < N)

                c = acc.to(C_out.type.element_ty)
                if USE_SLOTS:
                    # Bulk store into the owner's per-source slot.
                    # C_out is [world_size, M_local, N]; we own slot cur_rank.
                    # No atomics — a plain vectorized store. A second pass
                    # sums the ws slots locally.
                    slot_off = cur_rank * stride_slot + out_off
                    if owner == cur_rank:
                        tl.store(C_out + slot_off, c, mask=out_msk)
                    else:
                        iris.store(C_out + slot_off, c,
                                   cur_rank, owner, heap_bases, mask=out_msk)
                else:
                    # Element-wise remote atomics (BLOCK_M*BLOCK_N per tile)
                    if owner == cur_rank:
                        tl.atomic_add(C_out + out_off, c, mask=out_msk)
                    else:
                        iris.atomic_add(
                            C_out + out_off, c,
                            cur_rank, owner, heap_bases,
                            mask=out_msk, sem="relaxed", scope="sys",
                        )


@triton.jit
def _local_slot_reduce(
    slots, out,
    n_elem,
    stride_slot,
    world_size: tl.constexpr,
    BLOCK: tl.constexpr,
    NUM_SMS: tl.constexpr,
):
    """Sum the ws per-source slots into the final output. Pure local HBM."""
    pid = tl.program_id(0)
    for base in range(pid * BLOCK, n_elem, NUM_SMS * BLOCK):
        offs = base + tl.arange(0, BLOCK)
        msk = offs < n_elem
        acc = tl.zeros((BLOCK,), dtype=tl.float32)
        for r in tl.static_range(world_size):
            acc += tl.load(slots + r * stride_slot + offs, mask=msk, other=0.0).to(tl.float32)
        tl.store(out + offs, acc.to(out.type.element_ty), mask=msk)


def matmul_reduce_scatter_batched(
    ctx,
    output_tensor: torch.Tensor,
    A: torch.Tensor,
    B: torch.Tensor,
    block_m: int = 32,
    block_n: int = 64,
    block_k: int = 64,
    batch: int = 4,
    num_sms: int = 304,
    num_warps: int = 4,
    mfma: int = 32,
    use_slots: bool = False,
    slots_buf: Optional[torch.Tensor] = None,
):
    """
    Full fusion, no CU split. All WGs compute `batch` tiles into registers,
    then pipeline `batch` XGMI scatters.

    output_tensor MUST be in the symmetric heap (peers atomic_add into it).
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

    hb = ctx.get_heap_bases()

    if use_slots:
        # Per-source slots: [world_size, M_local, N] in the symmetric heap.
        if slots_buf is None:
            slots_buf = ctx.zeros((world_size, M_local, N), dtype=A.dtype)
        target = slots_buf
        stride_slot = slots_buf.stride(0)
        s_m, s_n = slots_buf.stride(1), slots_buf.stride(2)
    else:
        output_tensor.zero_()
        target = output_tensor
        stride_slot = 0
        s_m, s_n = output_tensor.stride(0), output_tensor.stride(1)

    ctx.barrier()

    kw = {"num_warps": num_warps, "num_stages": 2}
    if getattr(torch.version, "hip", None):
        kw["matrix_instr_nonkdim"] = mfma

    _batched_fusion_kernel[(num_sms,)](
        A, B, target,
        M, N, K, M_local,
        A.stride(0), A.stride(1),
        B.stride(0), B.stride(1),
        s_m, s_n, stride_slot,
        hb, rank, world_size,
        block_m, block_n, block_k,
        num_sms, batch, use_slots,
        K % block_k == 0,
        num_m_tiles, num_n_tiles, total_tiles,
        **kw,
    )

    ctx.barrier()

    if use_slots:
        # Second pass: sum the ws slots locally (pure HBM, no XGMI).
        n_elem = M_local * N
        _local_slot_reduce[(num_sms,)](
            slots_buf, output_tensor, n_elem, stride_slot,
            world_size, 4096, num_sms,
            num_warps=4,
        )

    return slots_buf
