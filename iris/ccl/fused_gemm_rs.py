# SPDX-License-Identifier: MIT
# Copyright (c) 2025 Advanced Micro Devices, Inc. All rights reserved.

"""
Fused GEMM + Reduce-Scatter collective for row-parallel tensor parallelism.

In row-parallel TP each rank computes partial = input[:, rank*H_shard:(rank+1)*H_shard] @ weight_shard
and the results are reduce-scattered so each rank ends with 1/N of the reduced output columns.

This kernel fuses the GEMM with the scatter: each output tile is atomically added to its
destination rank's output buffer immediately as the GEMM computes it, overlapping compute
and communication.

Design:
- Float32 output buffer on symmetric heap (GPU atomics don't support bf16/fp16).
- No aux_buffer or locks — atomics provide implicit synchronization.
- Tile swizzling spreads atomic destinations across ranks to reduce contention.
"""

from dataclasses import dataclass
from typing import Optional

import torch
import triton
import triton.language as tl
import iris
from .config import Config
from .utils import extract_group_info


@dataclass
class GemmReduceScatterWorkspace:
    """Reusable workspace for fused GEMM + reduce-scatter.

    Attributes:
        output_f32: [tokens, shard_size] fp32 buffer on the symmetric heap.
                    Each rank accumulates its output shard here via atomics.
    """

    output_f32: Optional[torch.Tensor] = None


@triton.jit()
def _fused_gemm_rs_kernel(
    # Tensor pointers
    input_ptr,
    weight_ptr,
    output_ptr,
    # Dimensions
    tokens,
    K,
    H_shard,
    shard_size,
    # Input strides
    stride_input_m,
    stride_input_k,
    # Weight strides
    stride_weight_h,
    stride_weight_n,
    # Output strides
    stride_out_m,
    stride_out_n,
    # Iris
    heap_bases: tl.tensor,
    group_rank: tl.constexpr,
    iris_rank: tl.constexpr,
    world_size: tl.constexpr,
    rank_start: tl.constexpr,
    rank_stride: tl.constexpr,
    # Tile sizes
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    """
    Fused GEMM + reduce-scatter kernel.

    Grid: (cdiv(tokens, BLOCK_M) * cdiv(K, BLOCK_N),)

    Each program computes one (BLOCK_M x BLOCK_N) output tile of the GEMM, then
    atomically adds it to the correct destination rank's output buffer.

    The n-coordinate is swizzled so different ranks start at different column offsets,
    spreading atomic traffic across destination ranks.
    """
    pid = tl.program_id(0)

    num_pid_m = tl.cdiv(tokens, BLOCK_M)
    num_pid_n = tl.cdiv(K, BLOCK_N)
    total_tiles = num_pid_m * num_pid_n

    if pid >= total_tiles:
        return

    # Decode pid -> (pid_m, pid_n) with simple row-major
    pid_m = pid // num_pid_n
    pid_n_raw = pid % num_pid_n

    # Swizzle n-coordinate: different ranks start at different offsets
    # This spreads atomic-add destinations across ranks, reducing contention
    tiles_per_shard = tl.cdiv(shard_size, BLOCK_N)
    pid_n = (pid_n_raw + group_rank * tiles_per_shard) % num_pid_n

    # --- GEMM k-loop ---
    # A = input[:, rank*H_shard : (rank+1)*H_shard], shape [tokens, H_shard]
    # B = weight_shard, shape [H_shard, K]
    # This rank reads input columns offset by rank * H_shard (row-parallel sharding)

    rm = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    rn = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    num_k_tiles = tl.cdiv(H_shard, BLOCK_K)
    for k_tile in range(num_k_tiles):
        rk = k_tile * BLOCK_K + tl.arange(0, BLOCK_K)

        # Load A tile: input[rm, rk] (rk indexes within H_shard)
        a_ptrs = input_ptr + rm[:, None] * stride_input_m + rk[None, :] * stride_input_k
        a_mask = (rm[:, None] < tokens) & (rk[None, :] < H_shard)
        a = tl.load(a_ptrs, mask=a_mask, other=0.0)

        # Load B tile: weight[rk, rn]
        b_ptrs = weight_ptr + rk[:, None] * stride_weight_h + rn[None, :] * stride_weight_n
        b_mask = (rk[:, None] < H_shard) & (rn[None, :] < K)
        b = tl.load(b_ptrs, mask=b_mask, other=0.0)

        acc += tl.dot(a, b)

    # --- Epilogue: atomic scatter to destination rank ---
    # n_start is the column index in the full output [tokens, K]
    n_start = pid_n * BLOCK_N
    # Which rank owns this column shard?
    dest_rank_in_group = n_start // shard_size
    # Local column offset within destination rank's shard
    local_n = n_start % shard_size

    local_n_offsets = local_n + tl.arange(0, BLOCK_N)
    out_offsets = rm[:, None] * stride_out_m + local_n_offsets[None, :] * stride_out_n
    mask = (rm[:, None] < tokens) & (local_n_offsets[None, :] < shard_size)

    # Compute destination global rank
    dest_rank_global = rank_start + dest_rank_in_group * rank_stride

    if dest_rank_in_group == group_rank:
        # Local atomic add
        tl.atomic_add(output_ptr + out_offsets, acc, mask=mask)
    else:
        # Remote atomic add via iris RMA
        iris.atomic_add(
            output_ptr + out_offsets,
            acc,
            iris_rank,
            dest_rank_global,
            heap_bases,
            mask=mask,
        )

    tl.debug_barrier()


def gemm_reduce_scatter_preamble(
    input_tensor,
    weight_shard,
    shmem,
    config: Optional[Config] = None,
    workspace: Optional[GemmReduceScatterWorkspace] = None,
):
    """
    Allocate workspace for fused GEMM + reduce-scatter.

    Allocates a float32 output buffer on the symmetric heap. Re-allocates if
    the shape changes.

    Args:
        input_tensor: [tokens, H] activation tensor (used for shape inference).
        weight_shard: [H_shard, K] this rank's weight shard (used for shape inference).
        shmem: Iris shmem context.
        config: Optional Config instance.
        workspace: Optional existing workspace to reuse.

    Returns:
        GemmReduceScatterWorkspace ready for gemm_reduce_scatter().
    """
    if workspace is None:
        workspace = GemmReduceScatterWorkspace()

    tokens = input_tensor.shape[0]
    K = weight_shard.shape[1]
    world_size = shmem.get_num_ranks()
    shard_size = K // world_size

    needed_shape = (tokens, shard_size)

    if workspace.output_f32 is None or workspace.output_f32.shape != needed_shape:
        workspace.output_f32 = shmem.zeros(needed_shape, dtype=torch.float32)

    return workspace


def gemm_reduce_scatter(
    input_tensor,
    weight_shard,
    shmem,
    group=None,
    async_op: bool = False,
    config: Optional[Config] = None,
    workspace: Optional[GemmReduceScatterWorkspace] = None,
):
    """
    Fused GEMM + reduce-scatter collective.

    Computes partial = input @ weight_shard and atomically scatters each output
    tile to its destination rank, so each rank ends with its 1/world_size shard
    of the reduced output.

    Args:
        input_tensor: [tokens, H_shard] this rank's input shard. Each rank reads
                      its own columns of the full activation matrix.
        weight_shard: [H_shard, K] this rank's weight shard. K is the full output dim.
        shmem: Iris shmem context.
        group: ProcessGroup or None.
        async_op: If False, barrier after kernel. If True, return immediately.
        config: Optional Config instance.
        workspace: Optional pre-allocated workspace from gemm_reduce_scatter_preamble().

    Returns:
        torch.Tensor: [tokens, shard_size] in input_tensor's dtype.
    """
    if config is None:
        config = Config(block_size_m=128, block_size_n=128)

    rank_in_group, rank_global, world_size, rank_start, rank_stride_val = extract_group_info(group, shmem)

    tokens = input_tensor.shape[0]
    H_shard = input_tensor.shape[1]
    K = weight_shard.shape[1]

    if weight_shard.shape[0] != H_shard:
        raise ValueError(
            f"weight_shard.shape[0] ({weight_shard.shape[0]}) must match "
            f"input_tensor.shape[1] ({H_shard}): inner dimension mismatch."
        )
    if K % world_size != 0:
        raise ValueError(f"weight output dim K ({K}) must be divisible by world_size ({world_size}).")

    shard_size = K // world_size

    # Allocate workspace if not provided
    if workspace is None or workspace.output_f32 is None or workspace.output_f32.shape != (tokens, shard_size):
        workspace = gemm_reduce_scatter_preamble(input_tensor, weight_shard, shmem, config=config, workspace=workspace)

    # Zero the fp32 accumulation buffer — all ranks will atomically add into it
    workspace.output_f32.zero_()
    shmem.barrier()

    heap_bases = shmem.get_heap_bases()

    BLOCK_M = config.block_size_m
    BLOCK_N = config.block_size_n
    BLOCK_K = 32  # Inner-dim tile size for GEMM k-loop

    num_pid_m = triton.cdiv(tokens, BLOCK_M)
    num_pid_n = triton.cdiv(K, BLOCK_N)
    grid = (num_pid_m * num_pid_n,)

    _fused_gemm_rs_kernel[grid](
        input_tensor,
        weight_shard,
        workspace.output_f32,
        tokens,
        K,
        H_shard,
        shard_size,
        input_tensor.stride(0),
        input_tensor.stride(1),
        weight_shard.stride(0),
        weight_shard.stride(1),
        workspace.output_f32.stride(0),
        workspace.output_f32.stride(1),
        heap_bases,
        rank_in_group,
        rank_global,
        world_size,
        rank_start,
        rank_stride_val,
        BLOCK_M,
        BLOCK_N,
        BLOCK_K,
        num_warps=8,
        num_stages=2,
    )

    if not async_op:
        shmem.barrier()

    # Convert fp32 accumulation buffer to original dtype
    output = workspace.output_f32.to(input_tensor.dtype)

    return output
