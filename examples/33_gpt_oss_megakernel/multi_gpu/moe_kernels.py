# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
"""MoE-rank kernel: compute one selected expert (gate-up -> SwiGLU -> down) for the
current token from the FP8 expert input delivered by the attention rank, and write
the gate-weighted result vector to a local output buffer. Reuses the single-GPU
device ops in common/. One persistent launch per layer; the grid barrier separates
the gate-up / SwiGLU / down phases (down's contraction spans the full SwiGLU output,
so the phases cannot overlap)."""

import triton
import triton.language as tl
import iris

from common.barrier import _barrier
from common.gemv_fp4 import _gemv_fp4_scaled
from common.swiglu import _swiglu_quant_fp8


@triton.jit
def moe_expert_kernel(
    # this rank's full expert table (all E experts x all layers), MXFP4
    gu_blk_p,
    gu_scl_p,
    gu_b_p,
    dn_blk_p,
    dn_scl_p,
    dn_b_p,
    # inbox (local heap): FP8 expert input + meta delivered by the attention rank
    nfp8_p,  # [H] e4m3
    nfp8_scl_p,  # [GU_NB] e8m0
    meta_p,  # [1] int32 expert id
    gw_p,  # [1] fp32 gate weight
    # scratch + output (local heap)
    gu_p,  # [2*I] fp32 gate-up
    afp8_p,  # [I] e4m3 swiglu output
    afp8_scl_p,  # [DN_NB] e8m0
    out_p,  # [H] fp32 gate-weighted expert output -> read back by the attn rank
    bar_p,  # [1] int32 grid barrier counter (local)
    bar_base,  # running barrier target base (this layer's first target / NWG)
    layer,
    alpha,
    limit,
    NWG: tl.constexpr,
    E: tl.constexpr,
    H: tl.constexpr,
    I: tl.constexpr,
    GU_NB: tl.constexpr,
    DN_NB: tl.constexpr,
    BLOCK_NQ: tl.constexpr,
    BLOCK_ND: tl.constexpr,
    BLOCK_KQ: tl.constexpr,
    MTILE: tl.constexpr,
    NSTAGES: tl.constexpr,
):
    pid = tl.program_id(0)
    e_id = tl.load(meta_p)
    gwv = tl.load(gw_p).to(tl.float32)
    eidx = (layer * E + e_id).to(tl.int64)  # int64: expert blobs overflow int32

    # --- gate-up: [2I, H] FP4 x FP8 -> gu_p ---
    gu_blk = gu_blk_p + eidx * (2 * I) * (H // 2)
    gu_scl = gu_scl_p + eidx * (2 * I) * GU_NB
    gu_b = gu_b_p + eidx * (2 * I)
    _gemv_fp4_scaled(
        gu_blk, gu_scl, nfp8_p, nfp8_scl_p, gu_p, gu_b, True, 2 * I, H, GU_NB, pid, 1.0, False,
        BLOCK_NQ, BLOCK_KQ, MTILE,
    )
    _barrier(bar_p, (bar_base + 1) * NWG)

    # --- SwiGLU(gate-up) -> FP8 activation (producer==consumer per 32-block) ---
    _swiglu_quant_fp8(gu_p, afp8_p, afp8_scl_p, DN_NB, pid, alpha, limit)
    _barrier(bar_p, (bar_base + 2) * NWG)

    # --- down: [H, I] FP4 x FP8 -> out_p, scaled by the gate weight ---
    dn_blk = dn_blk_p + eidx * H * (I // 2)
    dn_scl = dn_scl_p + eidx * H * DN_NB
    dn_b = dn_b_p + eidx * H
    _gemv_fp4_scaled(
        dn_blk, dn_scl, afp8_p, afp8_scl_p, out_p, dn_b, True, H, I, DN_NB, pid, gwv, False,
        BLOCK_ND, BLOCK_KQ, MTILE, NSTAGES,
    )


@triton.jit
def scatter_back_kernel(
    out_p,  # [H] fp32 local gate-weighted expert output
    r_res_p,  # attn rank's res buffer base ([TOPK, H]); slot offset baked into pointer by host
    H: tl.constexpr,
    slot: tl.constexpr,
    moe_rank: tl.constexpr,
    attn_rank: tl.constexpr,
    BLOCK: tl.constexpr,
    heap_bases,
):
    """Copy this expert's result vector back to the attention rank's per-slot inbox.
    An all-rank shmem.barrier() after this launch makes the write visible to rank 0."""
    pid = tl.program_id(0)
    off = pid * BLOCK + tl.arange(0, BLOCK)
    m = off < H
    v = iris.load(out_p + off, moe_rank, moe_rank, heap_bases, mask=m)
    iris.store(r_res_p + slot * H + off, v, moe_rank, attn_rank, heap_bases, mask=m)
