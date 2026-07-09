# SPDX-License-Identifier: MIT
# Copyright (c) 2025 Advanced Micro Devices, Inc. All rights reserved.

"""
Triton kernels for reduce-scatter collective communication.

Three variants:
- two_shot: Serial accumulation chain, general-purpose (stock)
- inreg: In-register reduction, all peer loads independent. Best at <=2MB.
- twophase: Two-phase decomposition (readall + local reduce). Best at 2-8MB.
"""

from dataclasses import dataclass
from typing import Optional

import torch
import triton
import triton.language as tl
import iris
from iris.host.tracing.kernel_artifacts import iris_launch
from ..utils import chiplet_transform_chunked

# Variant constants
VARIANT_TWO_SHOT = "two_shot"
VARIANT_INREG = "inreg"
VARIANT_TWOPHASE = "twophase"


@dataclass
class ReduceScatterWorkspace:
    """Workspace for twophase reduce-scatter (scratch buffer)."""

    variant: str = ""
    scratch: Optional[torch.Tensor] = None
    chunk_per_rank: int = 0
    prepared: bool = False


@triton.jit()
def persistent_reduce_scatter_two_shot(
    input_ptr,
    output_ptr,
    M,
    N,
    stride_in_m,
    stride_in_n,
    stride_out_m,
    stride_out_n,
    heap_bases: tl.tensor,
    group_rank: tl.constexpr,
    iris_rank: tl.constexpr,
    world_size: tl.constexpr,
    rank_start: tl.constexpr,
    rank_stride: tl.constexpr,
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
    GROUP_SIZE_M: tl.constexpr,
    COMM_SMS: tl.constexpr,
    NUM_XCDS: tl.constexpr,
    CHUNK_SIZE: tl.constexpr,
    DISTRIBUTION: tl.constexpr,
):
    """
    Reduce-scatter using two-shot approach.

    Each rank reduces its assigned tiles from all ranks and stores the result
    only to its own output (no broadcast to other ranks).
    """
    pid = tl.program_id(0)

    if NUM_XCDS != 1:
        pid = chiplet_transform_chunked(pid, COMM_SMS, NUM_XCDS, CHUNK_SIZE)

    num_pid_m = tl.cdiv(M, BLOCK_SIZE_M)
    num_pid_n = tl.cdiv(N, BLOCK_SIZE_N)
    total_tiles = num_pid_m * num_pid_n

    acc_dtype = tl.float32 if output_ptr.type.element_ty != tl.int8 else tl.int32

    tiles_per_rank = tl.cdiv(total_tiles, world_size)
    if DISTRIBUTION == 0:
        start_tile = group_rank
        stride = world_size
        remaining = total_tiles - start_tile
        remaining = tl.maximum(remaining, 0)
        max_tile_offset = tl.cdiv(remaining, stride)
    else:
        start_tile = group_rank * tiles_per_rank
        stride = 1
        remaining = total_tiles - start_tile
        remaining = tl.maximum(remaining, 0)
        max_tile_offset = tl.minimum(tiles_per_rank, remaining)

    for tile_offset in range(pid, max_tile_offset, COMM_SMS):
        tile_id = start_tile + tile_offset * stride

        num_pid_in_group = GROUP_SIZE_M * num_pid_n
        group_id = tile_id // num_pid_in_group
        first_pid_m = group_id * GROUP_SIZE_M
        group_size_m = min(num_pid_m - first_pid_m, GROUP_SIZE_M)
        pid_m = first_pid_m + ((tile_id % num_pid_in_group) % group_size_m)
        pid_n = (tile_id % num_pid_in_group) // group_size_m

        tl.assume(pid_m >= 0)
        tl.assume(pid_n >= 0)

        rm_base = pid_m * BLOCK_SIZE_M
        rn_base = pid_n * BLOCK_SIZE_N

        is_full = (rm_base + BLOCK_SIZE_M <= M) & (rn_base + BLOCK_SIZE_N <= N)

        # Build indices (used by both paths)
        rm = rm_base + tl.arange(0, BLOCK_SIZE_M)
        rn = rn_base + tl.arange(0, BLOCK_SIZE_N)

        rm = tl.max_contiguous(tl.multiple_of(rm, BLOCK_SIZE_M), BLOCK_SIZE_M)
        rn = tl.max_contiguous(tl.multiple_of(rn, BLOCK_SIZE_N), BLOCK_SIZE_N)

        input_offset = rm[:, None] * stride_in_m + rn[None, :] * stride_in_n
        output_offset = rm[:, None] * stride_out_m + rn[None, :] * stride_out_n

        base_ptr = input_ptr + input_offset
        out_ptr = output_ptr + output_offset

        # Fast path: NO MASKS (full tiles)
        if is_full:
            start_rank_idx = pid % world_size
            start_rank_global = rank_start + start_rank_idx * rank_stride
            acc = iris.load(base_ptr, iris_rank, start_rank_global, heap_bases, hint=(1, BLOCK_SIZE_N)).to(acc_dtype)
            for i in tl.static_range(1, world_size):
                remote_rank_idx = (start_rank_idx + i) % world_size
                remote_rank = rank_start + remote_rank_idx * rank_stride
                acc += iris.load(base_ptr, iris_rank, remote_rank, heap_bases, hint=(1, BLOCK_SIZE_N)).to(acc_dtype)

            reduced = acc.to(output_ptr.type.element_ty)

            # Store only to own rank (no broadcast)
            tl.store(out_ptr, reduced, cache_modifier=".wt")

        # Slow path: MASKED (only boundary tiles land here)
        else:
            mask = (rm[:, None] < M) & (rn[None, :] < N)

            start_rank_idx = pid % world_size
            start_rank_global = rank_start + start_rank_idx * rank_stride
            acc = iris.load(base_ptr, iris_rank, start_rank_global, heap_bases, mask=mask, hint=(1, BLOCK_SIZE_N)).to(
                acc_dtype
            )
            for i in tl.static_range(1, world_size):
                remote_rank_idx = (start_rank_idx + i) % world_size
                remote_rank = rank_start + remote_rank_idx * rank_stride
                acc += iris.load(base_ptr, iris_rank, remote_rank, heap_bases, mask=mask, hint=(1, BLOCK_SIZE_N)).to(
                    acc_dtype
                )

            reduced = acc.to(output_ptr.type.element_ty)

            # Store only to own rank (no broadcast)
            tl.store(out_ptr, reduced, mask=mask, cache_modifier=".wt")


@triton.jit()
def persistent_reduce_scatter_inreg(
    input_ptr,
    output_ptr,
    chunk_per_rank,
    chunk_offset,
    heap_bases: tl.tensor,
    iris_rank: tl.constexpr,
    world_size: tl.constexpr,
    rank_start: tl.constexpr,
    rank_stride: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
    COMM_SMS: tl.constexpr,
):
    """
    In-register reduce-scatter: load all peers independently, reduce in registers.

    All 8 peer loads are issued independently (no accumulation dependency chain),
    then reduced in a single expression. No scratch buffer needed. Best at <=2MB
    where register pressure is manageable and XGMI latency hiding matters most.
    """
    pid = tl.program_id(0)
    num_tiles = tl.cdiv(chunk_per_rank, BLOCK_SIZE_N)
    acc_dtype = tl.float32

    for tile_id in range(pid, num_tiles, COMM_SMS):
        rn_base = tile_id * BLOCK_SIZE_N
        rn = rn_base + tl.arange(0, BLOCK_SIZE_N)
        rn = tl.max_contiguous(tl.multiple_of(rn, BLOCK_SIZE_N), BLOCK_SIZE_N)
        mask = rn < chunk_per_rank

        base_ptr = input_ptr + (chunk_offset + rn)

        # Load all 8 peers independently -- no accumulation dependency
        r0 = rank_start + 0 * rank_stride
        d0 = iris.load(base_ptr, iris_rank, r0, heap_bases, mask=mask, hint=(BLOCK_SIZE_N,)).to(acc_dtype)
        r1 = rank_start + 1 * rank_stride
        d1 = iris.load(base_ptr, iris_rank, r1, heap_bases, mask=mask, hint=(BLOCK_SIZE_N,)).to(acc_dtype)
        r2 = rank_start + 2 * rank_stride
        d2 = iris.load(base_ptr, iris_rank, r2, heap_bases, mask=mask, hint=(BLOCK_SIZE_N,)).to(acc_dtype)
        r3 = rank_start + 3 * rank_stride
        d3 = iris.load(base_ptr, iris_rank, r3, heap_bases, mask=mask, hint=(BLOCK_SIZE_N,)).to(acc_dtype)
        r4 = rank_start + 4 * rank_stride
        d4 = iris.load(base_ptr, iris_rank, r4, heap_bases, mask=mask, hint=(BLOCK_SIZE_N,)).to(acc_dtype)
        r5 = rank_start + 5 * rank_stride
        d5 = iris.load(base_ptr, iris_rank, r5, heap_bases, mask=mask, hint=(BLOCK_SIZE_N,)).to(acc_dtype)
        r6 = rank_start + 6 * rank_stride
        d6 = iris.load(base_ptr, iris_rank, r6, heap_bases, mask=mask, hint=(BLOCK_SIZE_N,)).to(acc_dtype)
        r7 = rank_start + 7 * rank_stride
        d7 = iris.load(base_ptr, iris_rank, r7, heap_bases, mask=mask, hint=(BLOCK_SIZE_N,)).to(acc_dtype)

        # Reduce in registers -- all loads already issued
        acc = d0 + d1 + d2 + d3 + d4 + d5 + d6 + d7

        out = acc.to(output_ptr.type.element_ty)
        tl.store(output_ptr + rn, out, mask=mask, cache_modifier=".wt")


@triton.jit()
def rs_readall_kernel(
    input_ptr,
    scratch_ptr,
    chunk_per_rank,
    chunk_offset,
    heap_bases: tl.tensor,
    iris_rank: tl.constexpr,
    world_size: tl.constexpr,
    rank_start: tl.constexpr,
    rank_stride: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
    COMM_SMS: tl.constexpr,
):
    """
    Phase 1 of two-phase reduce-scatter: read assigned chunk from all peers
    into scratch buffer. All reads are independent (no accumulation chain).
    """
    pid = tl.program_id(0)
    num_tiles = tl.cdiv(chunk_per_rank, BLOCK_SIZE_N)
    total_work = num_tiles * world_size

    for work_id in range(pid, total_work, COMM_SMS):
        tile_id = work_id // world_size
        peer_idx = work_id % world_size

        rn_base = tile_id * BLOCK_SIZE_N
        rn = rn_base + tl.arange(0, BLOCK_SIZE_N)
        rn = tl.max_contiguous(tl.multiple_of(rn, BLOCK_SIZE_N), BLOCK_SIZE_N)
        mask = rn < chunk_per_rank

        remote_rank = rank_start + peer_idx * rank_stride
        base_ptr = input_ptr + (chunk_offset + rn)
        data = iris.load(
            base_ptr,
            iris_rank,
            remote_rank,
            heap_bases,
            mask=mask,
            hint=(BLOCK_SIZE_N,),
        )

        scratch_offset = peer_idx * chunk_per_rank + rn
        tl.store(scratch_ptr + scratch_offset, data, mask=mask)


@triton.jit()
def rs_reduce_kernel(
    scratch_ptr,
    output_ptr,
    chunk_per_rank,
    world_size: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
    COMM_SMS: tl.constexpr,
):
    """
    Phase 2 of two-phase reduce-scatter: reduce across scratch buffers into
    output. Pure local HBM -- no XGMI traffic.
    """
    pid = tl.program_id(0)
    num_tiles = tl.cdiv(chunk_per_rank, BLOCK_SIZE_N)
    acc_dtype = tl.float32

    for tile_id in range(pid, num_tiles, COMM_SMS):
        rn_base = tile_id * BLOCK_SIZE_N
        rn = rn_base + tl.arange(0, BLOCK_SIZE_N)
        rn = tl.max_contiguous(tl.multiple_of(rn, BLOCK_SIZE_N), BLOCK_SIZE_N)
        mask = rn < chunk_per_rank

        acc = tl.load(scratch_ptr + rn, mask=mask).to(acc_dtype)
        for i in tl.static_range(1, 8):  # max world_size=8
            if i < world_size:
                val = tl.load(scratch_ptr + i * chunk_per_rank + rn, mask=mask).to(acc_dtype)
                acc += val

        out = acc.to(output_ptr.type.element_ty)
        tl.store(output_ptr + rn, out, mask=mask, cache_modifier=".wt")


def _select_variant(size_bytes, world_size):
    """Auto-select best RS variant based on message size.

    Based on benchmarks on 8x MI300X:
    - two_shot: Best at all standalone RS sizes (2.37-2.74x vs RCCL at <=1MB).
      Benefits from 2D tiling and unmasked fast path.
    - inreg: Best when used as RS phase of RS+AG all-reduce decomposition.
      Independent loads hide XGMI latency at <=2MB.
    - twophase: Best when used as RS phase at 2-8MB. Separates XGMI reads
      from local reduction.

    For standalone reduce_scatter, two_shot wins across the board.
    The inreg/twophase variants are primarily useful when called directly
    for RS+AG all-reduce decomposition.
    """
    return VARIANT_TWO_SHOT


def launch(
    output_tensor,
    input_tensor,
    ctx,
    rank_in_group,
    rank_global,
    world_size,
    rank_start,
    rank_stride,
    config,
    workspace=None,
):
    """Launch the Triton reduce-scatter kernel."""
    M, N = input_tensor.shape[:2]

    variant = config.reduce_scatter_variant.lower()
    if variant == "auto":
        element_size = input_tensor.element_size()
        size_bytes = M * N * element_size
        variant = _select_variant(size_bytes, world_size)

    heap_bases = ctx.get_heap_bases()

    if variant == VARIANT_TWO_SHOT:
        stride_in_m, stride_in_n = input_tensor.stride(0), input_tensor.stride(1)
        stride_out_m, stride_out_n = output_tensor.stride(0), output_tensor.stride(1)
        distribution = config.all_reduce_distribution

        iris_launch(
            persistent_reduce_scatter_two_shot,
            (config.comm_sms,),
            input_tensor,
            output_tensor,
            M,
            N,
            stride_in_m,
            stride_in_n,
            stride_out_m,
            stride_out_n,
            heap_bases,
            rank_in_group,
            rank_global,
            world_size,
            rank_start,
            rank_stride,
            config.block_size_m,
            config.block_size_n,
            config.swizzle_size,
            config.comm_sms,
            config.num_xcds,
            config.chunk_size,
            distribution,
            num_stages=config.num_stages,
            num_warps=config.num_warps,
            waves_per_eu=config.waves_per_eu,
            algorithm="reduce_scatter",
            rank=rank_global,
            dtype=input_tensor.dtype,
        )

    elif variant == VARIANT_INREG:
        # Flatten to 1D for the inreg kernel
        numel = M * N
        chunk_per_rank = numel // world_size
        chunk_offset = rank_in_group * chunk_per_rank

        # Use 1D view
        input_flat = input_tensor.view(-1)
        output_flat = output_tensor.view(-1)[chunk_offset : chunk_offset + chunk_per_rank]

        block_n = min(config.block_size_n, chunk_per_rank)
        # Ensure block_n is power of 2 and >= 64
        block_n = max(64, block_n)

        iris_launch(
            persistent_reduce_scatter_inreg,
            (config.comm_sms,),
            input_flat,
            output_flat,
            chunk_per_rank,
            chunk_offset,
            heap_bases,
            rank_global,
            world_size,
            rank_start,
            rank_stride,
            block_n,
            config.comm_sms,
            num_warps=8,
            num_stages=1,
            waves_per_eu=1,
            algorithm="reduce_scatter",
            rank=rank_global,
            dtype=input_tensor.dtype,
        )

    elif variant == VARIANT_TWOPHASE:
        # Flatten to 1D for the twophase kernels
        numel = M * N
        chunk_per_rank = numel // world_size
        chunk_offset = rank_in_group * chunk_per_rank

        input_flat = input_tensor.view(-1)
        output_flat = output_tensor.view(-1)[chunk_offset : chunk_offset + chunk_per_rank]

        block_n = min(config.block_size_n, chunk_per_rank)
        block_n = max(64, block_n)

        # Allocate or reuse scratch buffer
        if workspace is None or workspace.scratch is None or workspace.chunk_per_rank != chunk_per_rank:
            scratch = ctx.zeros((world_size * chunk_per_rank,), dtype=input_tensor.dtype)
            if workspace is None:
                workspace = ReduceScatterWorkspace()
            workspace.scratch = scratch
            workspace.chunk_per_rank = chunk_per_rank
            workspace.variant = VARIANT_TWOPHASE
            workspace.prepared = True

        # Phase 1: read all peers into scratch
        iris_launch(
            rs_readall_kernel,
            (config.comm_sms,),
            input_flat,
            workspace.scratch,
            chunk_per_rank,
            chunk_offset,
            heap_bases,
            rank_global,
            world_size,
            rank_start,
            rank_stride,
            block_n,
            config.comm_sms,
            num_warps=8,
            num_stages=1,
            waves_per_eu=1,
            algorithm="reduce_scatter",
            rank=rank_global,
            dtype=input_tensor.dtype,
        )

        # Phase 2: local HBM reduce
        iris_launch(
            rs_reduce_kernel,
            (config.comm_sms,),
            workspace.scratch,
            output_flat,
            chunk_per_rank,
            world_size,
            block_n,
            config.comm_sms,
            num_warps=8,
            num_stages=1,
            waves_per_eu=1,
            algorithm="reduce_scatter",
            rank=rank_global,
            dtype=input_tensor.dtype,
        )

    return workspace
