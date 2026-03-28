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
  Phase 2: Fused residual add + RMSNorm + broadcast to all peers.

Phase 2 fuses the residual add with RMSNorm to avoid an extra HBM round-trip.
The broadcast propagates both the updated residual and norm output to all ranks
via iris.store, eliminating the need for a separate allgather.
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
def _residual_rmsnorm_broadcast_kernel(
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
    heap_bases: tl.tensor,
    group_rank: tl.constexpr,
    iris_rank: tl.constexpr,
    world_size: tl.constexpr,
    rank_start: tl.constexpr,
    rank_stride: tl.constexpr,
    BLOCK_HIDDEN: tl.constexpr,
    COMM_SMS: tl.constexpr,
    DISTRIBUTION: tl.constexpr,
):
    """
    Fused residual add + RMSNorm + broadcast for assigned rows.

    After allreduce completes, each rank already has the full reduced tensor.
    This kernel:
    1. Adds the reduced value to the residual (in-place).
    2. Computes RMSNorm of the updated residual.
    3. Broadcasts both residual and norm_out to all peers.

    Grid: (COMM_SMS,)
    BLOCK_HIDDEN covers the entire hidden dimension (padded to power of 2).
    Each CTA persistently processes its assigned rows.
    """
    pid = tl.program_id(0)

    total_rows = tokens
    rows_per_rank = tl.cdiv(total_rows, world_size)

    if DISTRIBUTION == 0:
        start_row = group_rank
        row_stride_val = world_size
        remaining = total_rows - start_row
        remaining = tl.maximum(remaining, 0)
        max_row_offset = tl.cdiv(remaining, row_stride_val)
    else:
        start_row = group_rank * rows_per_rank
        row_stride_val = 1
        remaining = total_rows - start_row
        remaining = tl.maximum(remaining, 0)
        max_row_offset = tl.minimum(rows_per_rank, remaining)

    col_offsets = tl.arange(0, BLOCK_HIDDEN)
    col_offsets = tl.max_contiguous(tl.multiple_of(col_offsets, BLOCK_HIDDEN), BLOCK_HIDDEN)
    col_mask = col_offsets < hidden

    is_full = BLOCK_HIDDEN <= hidden

    for row_offset in range(pid, max_row_offset, COMM_SMS):
        row = start_row + row_offset * row_stride_val

        if row < total_rows:
            start_rank_idx = pid % world_size

            if is_full:
                # ---- Fast path: no masks ----
                # Load reduced and residual
                red_offset = row * stride_red_t + col_offsets * stride_red_h
                red = tl.load(reduced_ptr + red_offset).to(tl.float32)

                res_offset = row * stride_res_t + col_offsets * stride_res_h
                res_ptrs = residual_ptr + res_offset
                res = tl.load(res_ptrs).to(tl.float32)
                res = res + red

                # Store updated residual + broadcast
                tl.store(res_ptrs, res.to(residual_ptr.type.element_ty), cache_modifier=".wt")
                for i in tl.static_range(0, world_size):
                    remote_rank_idx = (start_rank_idx + i) % world_size
                    remote_rank = rank_start + remote_rank_idx * rank_stride
                    if remote_rank_idx != group_rank:
                        iris.store(
                            res_ptrs,
                            res.to(residual_ptr.type.element_ty),
                            iris_rank,
                            remote_rank,
                            heap_bases,
                            hint=BLOCK_HIDDEN,
                        )

                # RMSNorm
                row_var = tl.sum(res * res, axis=0)
                rms = tl.rsqrt(row_var / hidden + eps)
                w = tl.load(weight_ptr + col_offsets).to(tl.float32)
                norm = res * rms * w

                # Store norm output + broadcast
                out_offset = row * stride_out_t + col_offsets * stride_out_h
                out_ptrs = norm_out_ptr + out_offset
                tl.store(out_ptrs, norm.to(norm_out_ptr.type.element_ty), cache_modifier=".wt")
                for i in tl.static_range(0, world_size):
                    remote_rank_idx = (start_rank_idx + i) % world_size
                    remote_rank = rank_start + remote_rank_idx * rank_stride
                    if remote_rank_idx != group_rank:
                        iris.store(
                            out_ptrs,
                            norm.to(norm_out_ptr.type.element_ty),
                            iris_rank,
                            remote_rank,
                            heap_bases,
                            hint=BLOCK_HIDDEN,
                        )

            else:
                # ---- Slow path: masked ----
                red_offset = row * stride_red_t + col_offsets * stride_red_h
                red = tl.load(reduced_ptr + red_offset, mask=col_mask, other=0.0).to(tl.float32)

                res_offset = row * stride_res_t + col_offsets * stride_res_h
                res_ptrs = residual_ptr + res_offset
                res = tl.load(res_ptrs, mask=col_mask, other=0.0).to(tl.float32)
                res = res + red

                tl.store(
                    res_ptrs,
                    res.to(residual_ptr.type.element_ty),
                    mask=col_mask,
                    cache_modifier=".wt",
                )
                for i in tl.static_range(0, world_size):
                    remote_rank_idx = (start_rank_idx + i) % world_size
                    remote_rank = rank_start + remote_rank_idx * rank_stride
                    if remote_rank_idx != group_rank:
                        iris.store(
                            res_ptrs,
                            res.to(residual_ptr.type.element_ty),
                            iris_rank,
                            remote_rank,
                            heap_bases,
                            mask=col_mask,
                            hint=BLOCK_HIDDEN,
                        )

                sq = tl.where(col_mask, res * res, 0.0)
                row_var = tl.sum(sq, axis=0)
                rms = tl.rsqrt(row_var / hidden + eps)
                w = tl.load(weight_ptr + col_offsets, mask=col_mask, other=0.0).to(tl.float32)
                norm = res * rms * w

                out_offset = row * stride_out_t + col_offsets * stride_out_h
                out_ptrs = norm_out_ptr + out_offset
                tl.store(
                    out_ptrs,
                    norm.to(norm_out_ptr.type.element_ty),
                    mask=col_mask,
                    cache_modifier=".wt",
                )
                for i in tl.static_range(0, world_size):
                    remote_rank_idx = (start_rank_idx + i) % world_size
                    remote_rank = rank_start + remote_rank_idx * rank_stride
                    if remote_rank_idx != group_rank:
                        iris.store(
                            out_ptrs,
                            norm.to(norm_out_ptr.type.element_ty),
                            iris_rank,
                            remote_rank,
                            heap_bases,
                            mask=col_mask,
                            hint=BLOCK_HIDDEN,
                        )


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

    Uses two phases:
      Phase 1: AllReduce via existing optimized two-shot kernel.
      Phase 2: Fused residual add + RMSNorm + broadcast.

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

    heap_bases = ctx.get_heap_bases()

    # Phase 1: AllReduce via existing fast two-shot kernel
    reduced = ctx.zeros((tokens, hidden), dtype=partial.dtype)
    ar_config = Config(
        all_reduce_variant="two_shot",
        all_reduce_distribution=config.all_reduce_distribution,
        comm_sms=config.comm_sms,
        block_size_m=config.block_size_m,
        block_size_n=config.block_size_n,
    )
    workspace = all_reduce_preamble(reduced, partial, ctx, config=ar_config)
    all_reduce(reduced, partial, ctx, group=group, async_op=False, config=ar_config, workspace=workspace)

    # Phase 2: Fused residual add + RMSNorm + broadcast
    norm_out = ctx.zeros((tokens, hidden), dtype=partial.dtype)

    BLOCK_HIDDEN = _next_power_of_2(hidden)

    _residual_rmsnorm_broadcast_kernel[(config.comm_sms,)](
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
        heap_bases,
        rank_in_group,
        rank_global,
        world_size,
        rank_start,
        rank_stride,
        BLOCK_HIDDEN,
        config.comm_sms,
        config.all_reduce_distribution,
        num_warps=8,
        num_stages=1,
        waves_per_eu=1,
    )

    if not async_op:
        ctx.barrier()

    return norm_out
