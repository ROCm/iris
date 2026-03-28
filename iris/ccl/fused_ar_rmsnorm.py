# SPDX-License-Identifier: MIT
# Copyright (c) 2025 Advanced Micro Devices, Inc. All rights reserved.

"""
Fused AllReduce + Residual Add + RMSNorm collective operation for Iris.

Composes three ops that appear together in every LLM transformer layer:
  1. AllReduce (sum partials across all ranks)
  2. Residual add (residual += reduced)
  3. RMSNorm (normalize with learnable weight)

Uses a two-phase approach:
  Phase 1: AllReduce via the one-shot kernel (each rank reads all peers,
           no broadcast, no inter-rank barrier needed).
  Phase 2: Local fused residual add + RMSNorm kernel (no communication).

The one-shot variant is used instead of two-shot because it writes only
to local memory (no iris.store to other ranks), so no device_barrier is
needed between phases — GPU stream ordering guarantees the local store
completes before the RMSNorm kernel reads.

Buffers are allocated once via all_reduce_rmsnorm_preamble() and reused
across calls. This avoids the expensive symmetric heap allocation
(which includes dist.barrier + DMA-BUF re-import) on the hot path.
"""

from dataclasses import dataclass
from typing import Optional

import triton
import triton.language as tl
import torch
from .config import Config
from .all_reduce import all_reduce, all_reduce_preamble, AllReduceWorkspace


@dataclass
class AllReduceRMSNormWorkspace:
    """Pre-allocated buffers for all_reduce_rmsnorm."""

    reduced: Optional[torch.Tensor] = None
    norm_out: Optional[torch.Tensor] = None
    ar_workspace: Optional[AllReduceWorkspace] = None
    ar_config: Optional[Config] = None


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

    Grid: (tokens,)
    Each CTA processes one row. BLOCK_HIDDEN covers the full hidden dim.
    """
    pid = tl.program_id(0)
    row_start = pid * BLOCK_TOKENS

    col_offsets = tl.arange(0, BLOCK_HIDDEN)
    col_offsets = tl.max_contiguous(tl.multiple_of(col_offsets, BLOCK_HIDDEN), BLOCK_HIDDEN)
    col_mask = col_offsets < hidden
    is_full = BLOCK_HIDDEN <= hidden

    # Load weight once per CTA
    if is_full:
        w = tl.load(weight_ptr + col_offsets).to(tl.float32)
    else:
        w = tl.load(weight_ptr + col_offsets, mask=col_mask, other=0.0).to(tl.float32)

    for row_idx in range(BLOCK_TOKENS):
        row = row_start + row_idx
        if row < tokens:
            if is_full:
                red_offset = row * stride_red_t + col_offsets * stride_red_h
                red = tl.load(reduced_ptr + red_offset).to(tl.float32)

                res_offset = row * stride_res_t + col_offsets * stride_res_h
                res_ptrs = residual_ptr + res_offset
                res = tl.load(res_ptrs).to(tl.float32)
                res = res + red

                tl.store(res_ptrs, res.to(residual_ptr.type.element_ty))

                row_var = tl.sum(res * res, axis=0)
                rms = tl.rsqrt(row_var / hidden + eps)
                norm = res * rms * w

                out_offset = row * stride_out_t + col_offsets * stride_out_h
                tl.store(norm_out_ptr + out_offset, norm.to(norm_out_ptr.type.element_ty))
            else:
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


def all_reduce_rmsnorm_preamble(
    partial: torch.Tensor,
    residual: torch.Tensor,
    ctx,
    config: Optional[Config] = None,
    workspace: Optional[AllReduceRMSNormWorkspace] = None,
) -> AllReduceRMSNormWorkspace:
    """
    Pre-allocate buffers for all_reduce_rmsnorm. Call once, reuse workspace.

    This avoids the expensive symmetric heap allocation (dist.barrier +
    DMA-BUF re-import) on the hot path.

    Args:
        partial: [tokens, hidden] -- shape template for allocation.
        residual: [tokens, hidden] -- shape template (unused, for API symmetry).
        ctx: Iris ctx context.
        config: Optional Config instance. Default: None.
        workspace: Optional existing workspace to reuse. Default: None.

    Returns:
        AllReduceRMSNormWorkspace with pre-allocated buffers.
    """
    if config is None:
        config = Config(block_size_m=32, block_size_n=64)

    tokens, hidden = partial.shape

    if workspace is None:
        workspace = AllReduceRMSNormWorkspace()

    # Allocate reduced buffer
    if (
        workspace.reduced is None
        or workspace.reduced.shape != (tokens, hidden)
        or workspace.reduced.dtype != partial.dtype
    ):
        workspace.reduced = ctx.zeros((tokens, hidden), dtype=partial.dtype)

    # Allocate norm_out buffer
    if (
        workspace.norm_out is None
        or workspace.norm_out.shape != (tokens, hidden)
        or workspace.norm_out.dtype != partial.dtype
    ):
        workspace.norm_out = ctx.zeros((tokens, hidden), dtype=partial.dtype)

    # Prepare allreduce workspace — use one_shot variant.
    # One-shot reads from all peers and writes locally only (no iris.store),
    # so no device_barrier is needed between allreduce and RMSNorm.
    ar_config = Config(
        all_reduce_variant="one_shot",
        comm_sms=config.comm_sms,
        block_size_m=config.block_size_m,
        block_size_n=config.block_size_n,
    )
    workspace.ar_config = ar_config
    workspace.ar_workspace = all_reduce_preamble(
        workspace.reduced, partial, ctx, config=ar_config, workspace=workspace.ar_workspace
    )

    return workspace


def all_reduce_rmsnorm(
    partial: torch.Tensor,
    residual: torch.Tensor,
    weight: torch.Tensor,
    ctx,
    eps: float = 1e-6,
    group=None,
    async_op: bool = False,
    config: Optional[Config] = None,
    workspace: Optional[AllReduceRMSNormWorkspace] = None,
) -> torch.Tensor:
    """
    Fused AllReduce + Residual Add + RMSNorm.

    For best performance, call all_reduce_rmsnorm_preamble() once to
    pre-allocate buffers, then pass the workspace to this function.

    Args:
        partial: [tokens, hidden] -- each rank's partial (on symmetric heap).
        residual: [tokens, hidden] -- residual, updated IN-PLACE (on symmetric heap).
        weight: [hidden] -- RMSNorm gamma (replicated across ranks).
        ctx: Iris ctx context.
        eps: RMSNorm epsilon. Default: 1e-6.
        group: ProcessGroup or None. Default: None.
        async_op: If False, barrier at end. Default: False.
        config: Optional Config instance. Default: None.
        workspace: Optional pre-allocated workspace. Default: None.

    Returns:
        norm_out: [tokens, hidden] -- normalized output (on symmetric heap).
    """
    if config is None:
        config = Config(block_size_m=32, block_size_n=64)

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

    # Allocate if no workspace provided (slow path — allocates on every call)
    if workspace is None:
        workspace = all_reduce_rmsnorm_preamble(partial, residual, ctx, config=config)

    reduced = workspace.reduced
    norm_out = workspace.norm_out
    ar_config = workspace.ar_config
    ar_workspace = workspace.ar_workspace

    # Phase 1: AllReduce (one-shot)
    # One-shot reads from all peers via iris.load and writes the reduced
    # result to the local output buffer only — no iris.store to other ranks.
    # This means no inter-phase barrier is needed: GPU stream ordering
    # guarantees the local tl.store completes before the next kernel reads.
    all_reduce(reduced, partial, ctx, group=group, async_op=True, config=ar_config, workspace=ar_workspace)

    # Phase 2: Local fused residual add + RMSNorm
    BLOCK_HIDDEN = _next_power_of_2(hidden)
    BLOCK_TOKENS = 1
    grid = (tokens,)

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

    # Both phases are local-only from this rank's perspective (one-shot
    # allreduce reads remotely but writes locally; RMSNorm is fully local).
    # No device_barrier is needed for correctness. Provide one only if
    # async_op=False for callers that need completion guarantees.
    if not async_op:
        ctx.device_barrier(group=group)

    return norm_out
