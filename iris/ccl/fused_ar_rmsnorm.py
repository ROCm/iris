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
    BLOCK_HIDDEN: tl.constexpr,
    COMM_SMS: tl.constexpr,
    DISTRIBUTION: tl.constexpr,
):
    """
    Two-shot fused AllReduce + Residual + RMSNorm.

    Grid: (COMM_SMS,)
    Each CTA processes one row at a time, iterating persistently across assigned rows.
    BLOCK_HIDDEN covers the entire hidden dimension (padded to power of 2).

    Phase 1: Each rank reduces its assigned partition of rows from all ranks.
    Phase 2: Fused residual add + RMSNorm + broadcast to all peers.

    This follows the two-shot pattern from all_reduce.py:
    - Each rank is assigned a subset of rows (block or striding distribution).
    - Phase 1: reduce partial[row, :] across all ranks for assigned rows.
    - Phase 2: residual += reduced, RMSNorm, store+broadcast both residual and norm_out.
    """
    pid = tl.program_id(0)

    total_rows = tokens
    rows_per_rank = tl.cdiv(total_rows, world_size)

    if DISTRIBUTION == 0:
        # Striding: rank handles rows [group_rank, group_rank + world_size, ...]
        start_row = group_rank
        row_stride_val = world_size
        remaining = total_rows - start_row
        remaining = tl.maximum(remaining, 0)
        max_row_offset = tl.cdiv(remaining, row_stride_val)
    else:
        # Block: rank handles rows [group_rank * rows_per_rank, ...)
        start_row = group_rank * rows_per_rank
        row_stride_val = 1
        remaining = total_rows - start_row
        remaining = tl.maximum(remaining, 0)
        max_row_offset = tl.minimum(rows_per_rank, remaining)

    col_offsets = tl.arange(0, BLOCK_HIDDEN)
    col_mask = col_offsets < hidden

    # Unmasked fast path check: BLOCK_HIDDEN == hidden (power-of-2 hidden dims)
    is_full = BLOCK_HIDDEN <= hidden

    # Persistent loop: each CTA handles multiple rows
    for row_offset in range(pid, max_row_offset, COMM_SMS):
        row = start_row + row_offset * row_stride_val

        if row < total_rows:
            # Phase 1: Reduce partials from all ranks for this row
            partial_offset = row * stride_partial_t + col_offsets * stride_partial_h
            partial_ptrs = partial_ptr + partial_offset

            # Rotate starting rank to distribute load across XGMI links
            start_rank_idx = pid % world_size
            start_rank_global = rank_start + start_rank_idx * rank_stride

            if is_full:
                # ---- Fast path: no masks (BLOCK_HIDDEN == hidden) ----
                acc = iris.load(partial_ptrs, iris_rank, start_rank_global, heap_bases).to(tl.float32)

                for i in tl.static_range(1, world_size):
                    remote_rank_idx = (start_rank_idx + i) % world_size
                    remote_rank = rank_start + remote_rank_idx * rank_stride
                    acc += iris.load(partial_ptrs, iris_rank, remote_rank, heap_bases).to(tl.float32)

                # Phase 2: Residual add + RMSNorm
                res_offset = row * stride_res_t + col_offsets * stride_res_h
                res_ptrs = residual_ptr + res_offset
                res = tl.load(res_ptrs).to(tl.float32)
                res = res + acc

                # Store updated residual locally + broadcast
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
                            hint=(1, BLOCK_HIDDEN),
                        )

                # RMS normalization
                row_var = tl.sum(res * res, axis=0)
                rms = tl.rsqrt(row_var / hidden + eps)
                w = tl.load(weight_ptr + col_offsets).to(tl.float32)
                norm = res * rms * w

                # Store norm output locally + broadcast
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
                            hint=(1, BLOCK_HIDDEN),
                        )

            else:
                # ---- Slow path: masked (BLOCK_HIDDEN > hidden) ----
                acc = iris.load(partial_ptrs, iris_rank, start_rank_global, heap_bases, mask=col_mask).to(tl.float32)

                for i in tl.static_range(1, world_size):
                    remote_rank_idx = (start_rank_idx + i) % world_size
                    remote_rank = rank_start + remote_rank_idx * rank_stride
                    acc += iris.load(partial_ptrs, iris_rank, remote_rank, heap_bases, mask=col_mask).to(tl.float32)

                # Phase 2: Residual add + RMSNorm
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
                            hint=(1, BLOCK_HIDDEN),
                        )

                # RMS normalization (zero out padding before sum)
                sq = tl.where(col_mask, res * res, 0.0)
                row_var = tl.sum(sq, axis=0)
                rms = tl.rsqrt(row_var / hidden + eps)
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
                            hint=(1, BLOCK_HIDDEN),
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
    shmem,
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
        shmem: Iris shmem context.
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

    rank_in_group, rank_global, world_size, rank_start, rank_stride = extract_group_info(group, shmem)

    heap_bases = shmem.get_heap_bases()

    # Allocate output on symmetric heap
    norm_out = shmem.zeros((tokens, hidden), dtype=partial.dtype)

    # BLOCK_HIDDEN must cover the entire hidden dimension for RMSNorm row reduction
    BLOCK_HIDDEN = _next_power_of_2(hidden)

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
        BLOCK_HIDDEN,
        config.comm_sms,
        config.all_reduce_distribution,
        num_warps=8,
        num_stages=1,
        waves_per_eu=1,
    )

    if not async_op:
        shmem.barrier()

    return norm_out
