# SPDX-License-Identifier: MIT
# Copyright (c) 2025 Advanced Micro Devices, Inc. All rights reserved.

"""
Fused AllReduce + Residual Add + RMSNorm collective operation for Iris.

Eliminates two global memory round-trips by fusing three ops into one kernel:
  1. AllReduce (sum partials across all ranks)
  2. Residual add (residual += reduced)
  3. RMSNorm (normalize with learnable weight)

This is the critical fusion in every LLM transformer layer, appearing after both
the attention projection and MLP down-projection.

The kernel uses a tiled two-pass approach per row:
  Pass 1: For each tile of the hidden dimension, allreduce + residual add + broadcast
           residual, while accumulating the sum-of-squares for RMSNorm.
  Pass 2: Compute rms, then for each tile, reload residual from L2, apply normalization,
           store and broadcast norm_out.

This avoids loading the entire hidden dimension into registers at once, which would
cause register spilling for typical LLM hidden sizes (4096-8192).
"""

from typing import Optional

import triton
import triton.language as tl
import torch
import iris
from .config import Config
from .utils import extract_group_info


@triton.jit
def _fused_ar_rmsnorm_two_shot_kernel(
    partial_ptr,
    residual_ptr,
    weight_ptr,
    norm_out_ptr,
    tokens,
    hidden,
    stride_partial_t,
    stride_partial_h,
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
    NUM_TILES: tl.constexpr,
    BLOCK_H: tl.constexpr,
    COMM_SMS: tl.constexpr,
    DISTRIBUTION: tl.constexpr,
):
    """
    Two-shot fused AllReduce + Residual + RMSNorm with tiled hidden dimension.

    Grid: (COMM_SMS,)
    Each CTA processes one row at a time, iterating persistently across assigned rows.
    The hidden dimension is tiled into NUM_TILES tiles of BLOCK_H elements each.

    For each row:
      Pass 1 (tile loop): allreduce partial, residual += reduced, broadcast residual,
                           accumulate sum-of-squares for variance.
      Pass 2 (tile loop): compute rms, apply normalization, broadcast norm_out.
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

    # Rotate starting rank to distribute load across XGMI links
    start_rank_idx = pid % world_size

    # Persistent loop: each CTA handles multiple rows
    for row_offset in range(pid, max_row_offset, COMM_SMS):
        row = start_row + row_offset * row_stride_val

        if row < total_rows:
            row_var: tl.float32 = 0.0

            # === Pass 1: AllReduce + Residual + Broadcast + Variance ===
            for tile in tl.static_range(0, NUM_TILES):
                col_base = tile * BLOCK_H
                col_offsets = col_base + tl.arange(0, BLOCK_H)
                col_mask = col_offsets < hidden

                partial_offset = row * stride_partial_t + col_offsets * stride_partial_h
                partial_ptrs = partial_ptr + partial_offset

                # Reduce partials from all ranks
                start_rank_global = rank_start + start_rank_idx * rank_stride
                acc = iris.load(partial_ptrs, iris_rank, start_rank_global, heap_bases, mask=col_mask).to(tl.float32)

                for i in tl.static_range(1, world_size):
                    remote_rank_idx = (start_rank_idx + i) % world_size
                    remote_rank = rank_start + remote_rank_idx * rank_stride
                    acc += iris.load(partial_ptrs, iris_rank, remote_rank, heap_bases, mask=col_mask).to(tl.float32)

                # Residual add
                res_offset = row * stride_res_t + col_offsets * stride_res_h
                res_ptrs = residual_ptr + res_offset
                res = tl.load(res_ptrs, mask=col_mask, other=0.0).to(tl.float32)
                res = res + acc

                # Store updated residual locally + broadcast
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
                            hint=BLOCK_H,
                        )

                # Accumulate sum-of-squares for RMSNorm variance
                sq = tl.where(col_mask, res * res, 0.0)
                row_var += tl.sum(sq, axis=0)

            # Compute RMS normalization factor
            rms = tl.rsqrt(row_var / hidden + eps)

            # === Pass 2: Apply normalization + Broadcast ===
            for tile in tl.static_range(0, NUM_TILES):
                col_base = tile * BLOCK_H
                col_offsets = col_base + tl.arange(0, BLOCK_H)
                col_mask = col_offsets < hidden

                # Reload residual (should be in L2 from pass 1)
                res_offset = row * stride_res_t + col_offsets * stride_res_h
                res_ptrs = residual_ptr + res_offset
                res = tl.load(res_ptrs, mask=col_mask, other=0.0).to(tl.float32)

                # Apply normalization
                w = tl.load(weight_ptr + col_offsets, mask=col_mask, other=0.0).to(tl.float32)
                norm = res * rms * w

                # Store norm output locally + broadcast
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
                            hint=BLOCK_H,
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

    Args:
        partial: [tokens, hidden] — each rank's partial GEMM output (on symmetric heap).
        residual: [tokens, hidden] — residual connection, updated IN-PLACE (on symmetric heap).
        weight: [hidden] — RMSNorm gamma (replicated across ranks).
        ctx: Iris ctx context.
        eps: RMSNorm epsilon. Default: 1e-6.
        group: ProcessGroup or None. Default: None.
        async_op: If False, barrier at end. Default: False.
        config: Optional Config instance. Default: None (uses defaults).

    Returns:
        norm_out: [tokens, hidden] — normalized output (on symmetric heap).
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

    # Allocate output on symmetric heap
    norm_out = ctx.zeros((tokens, hidden), dtype=partial.dtype)

    # Tile size for hidden dimension — balance between register pressure and tile count
    BLOCK_H = min(_next_power_of_2(hidden), config.block_size_n if hasattr(config, 'block_size_n') else 256)
    # Ensure BLOCK_H is reasonable (not too small, not too large)
    BLOCK_H = max(BLOCK_H, 64)
    BLOCK_H = min(BLOCK_H, 1024)
    NUM_TILES = (hidden + BLOCK_H - 1) // BLOCK_H

    _fused_ar_rmsnorm_two_shot_kernel[(config.comm_sms,)](
        partial,
        residual,
        weight,
        norm_out,
        tokens,
        hidden,
        partial.stride(0),
        partial.stride(1),
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
        NUM_TILES,
        BLOCK_H,
        config.comm_sms,
        config.all_reduce_distribution,
        num_warps=8,
        num_stages=1,
        waves_per_eu=1,
    )

    if not async_op:
        ctx.barrier()

    return norm_out
