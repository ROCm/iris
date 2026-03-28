# SPDX-License-Identifier: MIT
# Copyright (c) 2025 Advanced Micro Devices, Inc. All rights reserved.

"""
Fused AllReduce + Residual Add + RMSNorm collective operation for Iris.

Composes three ops that appear together in every LLM transformer layer:
  1. AllReduce (sum partials across all ranks)
  2. Residual add (residual += reduced)
  3. RMSNorm (normalize with learnable weight)

Uses a two-phase approach:
  Phase 1: AllReduce via the existing fast two-shot kernel (proven, optimized).
           After this, all ranks have the identical reduced tensor.
  Phase 2: Local fused residual add + RMSNorm kernel (no communication needed).
           Since all ranks have the same reduced tensor, and the residual and
           weight are already replicated (tensor parallelism invariant), each
           rank computes the same result independently.
"""

from typing import Optional

import triton
import triton.language as tl
import torch
import iris
from .config import Config
from .all_reduce import all_reduce, all_reduce_preamble
from .utils import extract_group_info


@triton.jit
def _residual_rmsnorm_kernel(
    reduced_ptr,
    residual_ptr,
    weight_ptr,
    norm_out_ptr,
    tokens,
    hidden,
    stride_red_t,
    stride_red_h,
    stride_res_t,
    stride_res_h,
    stride_out_t,
    stride_out_h,
    eps,
    BLOCK_HIDDEN: tl.constexpr,
    BLOCK_TOKENS: tl.constexpr,
):
    """
    Local fused residual add + RMSNorm kernel. No communication.

    Grid: (cdiv(tokens, BLOCK_TOKENS),)
    Each CTA processes BLOCK_TOKENS rows. BLOCK_HIDDEN covers the full hidden dim.
    """
    pid = tl.program_id(0)
    row_start = pid * BLOCK_TOKENS

    col_offsets = tl.arange(0, BLOCK_HIDDEN)
    col_offsets = tl.max_contiguous(tl.multiple_of(col_offsets, BLOCK_HIDDEN), BLOCK_HIDDEN)
    col_mask = col_offsets < hidden
    is_full = BLOCK_HIDDEN <= hidden

    # Load weight once per CTA (shared across all rows)
    if is_full:
        w = tl.load(weight_ptr + col_offsets).to(tl.float32)
    else:
        w = tl.load(weight_ptr + col_offsets, mask=col_mask, other=0.0).to(tl.float32)

    for row_idx in range(BLOCK_TOKENS):
        row = row_start + row_idx
        if row < tokens:
            if is_full:
                # Fast path: no masks
                red_offset = row * stride_red_t + col_offsets * stride_red_h
                red = tl.load(reduced_ptr + red_offset).to(tl.float32)

                res_offset = row * stride_res_t + col_offsets * stride_res_h
                res_ptrs = residual_ptr + res_offset
                res = tl.load(res_ptrs).to(tl.float32)
                res = res + red

                # Store updated residual
                tl.store(res_ptrs, res.to(residual_ptr.type.element_ty))

                # RMSNorm
                row_var = tl.sum(res * res, axis=0)
                rms = tl.rsqrt(row_var / hidden + eps)
                norm = res * rms * w

                # Store norm output
                out_offset = row * stride_out_t + col_offsets * stride_out_h
                tl.store(norm_out_ptr + out_offset, norm.to(norm_out_ptr.type.element_ty))

            else:
                # Slow path: masked
                red_offset = row * stride_red_t + col_offsets * stride_red_h
                red = tl.load(reduced_ptr + red_offset, mask=col_mask, other=0.0).to(tl.float32)

                res_offset = row * stride_res_t + col_offsets * stride_res_h
                res_ptrs = residual_ptr + res_offset
                res = tl.load(res_ptrs, mask=col_mask, other=0.0).to(tl.float32)
                res = res + red

                tl.store(res_ptrs, res.to(residual_ptr.type.element_ty), mask=col_mask)

                sq = tl.where(col_mask, res * res, 0.0)
                row_var = tl.sum(sq, axis=0)
                rms = tl.rsqrt(row_var / hidden + eps)
                norm = res * rms * w

                out_offset = row * stride_out_t + col_offsets * stride_out_h
                tl.store(norm_out_ptr + out_offset, norm.to(norm_out_ptr.type.element_ty), mask=col_mask)


def _next_power_of_2(n: int) -> int:
    """Return the smallest power of 2 >= n."""
    if n <= 0:
        return 1
    n -= 1
    n |= n >> 1
    n |= n >> 2
    n |= n >> 4
    n |= n >> 8
    n |= n >> 16
    return n + 1


def all_reduce_rmsnorm(
    partial: torch.Tensor,
    residual: torch.Tensor,
    weight: torch.Tensor,
    ctx,
    eps: float = 1e-6,
    group=None,
    async_op: bool = False,
    config: Optional[Config] = None,
) -> torch.Tensor:
    """
    Fused AllReduce + Residual Add + RMSNorm.

    Each rank has a partial GEMM output. This operation:
      1. Sums partials across all ranks (AllReduce)
      2. Adds the reduced result to the residual (in-place)
      3. Applies RMSNorm with the given weight

    After allreduce, all ranks have identical reduced data, so the
    residual add and RMSNorm are purely local (no communication).

    Args:
        partial: [tokens, hidden] -- each rank's partial GEMM output (on symmetric heap).
        residual: [tokens, hidden] -- residual connection, updated IN-PLACE (on symmetric heap).
        weight: [hidden] -- RMSNorm gamma (replicated across ranks).
        ctx: Iris ctx context.
        eps: RMSNorm epsilon. Default: 1e-6.
        group: ProcessGroup or None. Default: None.
        async_op: If False, barrier at end. Default: False.
        config: Optional Config instance. Default: None (uses defaults).

    Returns:
        norm_out: [tokens, hidden] -- normalized output (on symmetric heap).
    """
    if config is None:
        config = Config(block_size_m=32, block_size_n=64, all_reduce_distribution=1)

    if config.use_gluon:
        raise ValueError("fused_ar_rmsnorm does not support use_gluon=True.")

    # Validate shapes
    if partial.dim() != 2:
        raise ValueError(f"partial must be 2D [tokens, hidden], got {partial.dim()}D")
    if residual.dim() != 2:
        raise ValueError(f"residual must be 2D [tokens, hidden], got {residual.dim()}D")
    if weight.dim() != 1:
        raise ValueError(f"weight must be 1D [hidden], got {weight.dim()}D")

    tokens, hidden = partial.shape
    if residual.shape != (tokens, hidden):
        raise ValueError(f"residual shape {residual.shape} doesn't match partial shape {partial.shape}")
    if weight.shape[0] != hidden:
        raise ValueError(f"weight size {weight.shape[0]} doesn't match hidden dimension {hidden}")

    rank_in_group, rank_global, world_size, rank_start, rank_stride = extract_group_info(group, ctx)

    # Phase 1: AllReduce via existing fast two-shot kernel
    # After this, every rank has the identical reduced result.
    reduced = ctx.zeros((tokens, hidden), dtype=partial.dtype)
    ar_config = Config(
        all_reduce_variant="two_shot",
        all_reduce_distribution=config.all_reduce_distribution,
        comm_sms=config.comm_sms,
        block_size_m=config.block_size_m,
        block_size_n=config.block_size_n,
    )
    workspace = all_reduce_preamble(reduced, partial, ctx, config=ar_config)
    all_reduce(reduced, partial, ctx, group=group, async_op=True, config=ar_config, workspace=workspace)

    # Phase 2: Local fused residual add + RMSNorm (no communication)
    # The residual is replicated across ranks, reduced is identical on all ranks,
    # so each rank computes the same result independently.
    norm_out = ctx.zeros((tokens, hidden), dtype=partial.dtype)

    BLOCK_HIDDEN = _next_power_of_2(hidden)
    BLOCK_TOKENS = 1  # Process one row per CTA iteration
    grid = (tokens,)  # One CTA per row

    _residual_rmsnorm_kernel[grid](
        reduced,
        residual,
        weight,
        norm_out,
        tokens,
        hidden,
        reduced.stride(0),
        reduced.stride(1),
        residual.stride(0),
        residual.stride(1),
        norm_out.stride(0),
        norm_out.stride(1),
        eps,
        BLOCK_HIDDEN,
        BLOCK_TOKENS,
        num_warps=max(1, min(BLOCK_HIDDEN // 256, 8)),
        num_stages=1,
    )

    if not async_op:
        ctx.barrier()

    return norm_out
