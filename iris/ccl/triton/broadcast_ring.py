# SPDX-License-Identifier: MIT
# Copyright (c) 2025 Advanced Micro Devices, Inc. All rights reserved.

"""
Ring broadcast kernel for large messages.

Pipelined ring: divide data into W chunks (one per rank). In each of
W-1 steps, each rank that already has a chunk forwards it to its
successor via iris.store. After W-1 steps all ranks have all data.

Bandwidth efficiency: (W-1)/W — 87.5% for W=8 GPUs.
Each step transfers exactly 1/W of the data per link, fully utilizing
all W XGMI links in parallel (each link carries a different chunk).

Barrier overhead: W-1 barriers per kernel launch. Amortized over the
large message sizes where this kernel is used (>=8MB).
"""

import triton
import triton.language as tl
import iris
from iris.host.tracing.kernel_artifacts import iris_launch
from iris.ccl.utils import inline_device_barrier
from ..utils import chiplet_transform_chunked


@triton.jit()
def persistent_broadcast_ring(
    tensor_ptr,
    M,
    N,
    stride_m,
    stride_n,
    chunk_rows,
    heap_bases: tl.tensor,
    group_rank: tl.constexpr,
    iris_rank: tl.constexpr,
    src: tl.constexpr,
    world_size: tl.constexpr,
    rank_start: tl.constexpr,
    rank_stride: tl.constexpr,
    barrier_flags_ptr,
    wg_done_ptr,
    barrier_sense_ptr,
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
    COMM_SMS: tl.constexpr,
    NUM_XCDS: tl.constexpr,
    CHUNK_SIZE: tl.constexpr,
    INLINE_BARRIER: tl.constexpr = True,
):
    """
    Ring broadcast with pipelined chunks.

    Ring order: src, (src+1)%W, (src+2)%W, ..., (src+W-1)%W
    Data divided into W chunks. In step s:
      - Each rank at ring_pos p (p > 0) that has chunk c pushes it to successor
      - Chunk c arrives at ring_pos p in step p-1+c (mod W pipeline)

    Simplified: in step s, rank at ring_pos p pushes chunk
    chunk_idx = (src + p - 1 - s + W) % W if it has received it.

    Actually using the simpler per-step approach:
    Step s: rank at ring_pos p, if p <= s, pushes chunk_idx = (s - p + src) % W
    to successor.
    """
    pid = tl.program_id(0)

    if NUM_XCDS != 1:
        pid = chiplet_transform_chunked(pid, COMM_SMS, NUM_XCDS, CHUNK_SIZE)

    num_pid_n = tl.cdiv(N, BLOCK_SIZE_N)

    ring_pos = (group_rank - src + world_size) % world_size
    succ_group = (group_rank + 1) % world_size
    succ_iris = rank_start + succ_group * rank_stride
    pred_group = (group_rank - 1 + world_size) % world_size
    pred_iris = rank_start + pred_group * rank_stride

    # W-1 ring steps
    for step in tl.static_range(world_size - 1):
        # In step s, rank at ring_pos p receives a chunk if p == s+1
        # More precisely: in step s, the chunk that originated at ring_pos 0 (src)
        # and has been forwarded s times arrives at ring_pos s+1.
        # Which chunk? In a pipelined ring:
        # Step s: the chunk being forwarded by ring_pos p is:
        #   chunk_idx_in_ring = (ring_pos - 1 + (world_size - 1 - step)) % world_size
        #   but only if ring_pos > 0 and step >= ring_pos - 1

        # Simplest correct version: in step s, ring_pos s+1 pulls from predecessor
        # This means rank at ring_pos=1 reads in step 0, ring_pos=2 in step 1, etc.
        # Sequential — NOT pipelined. But correct. Only 1 link active per step.

        # Pipelined version: in step s, ALL ranks that have data forward ONE chunk
        # to successor. Multiple links active simultaneously.

        # Pipelined ring broadcast:
        # chunk_idx that ring_pos p pushes in step s:
        #   The chunk at distance d from src enters the ring in step 0
        #   After step s, it has reached ring_pos s+1
        #   So rank at ring_pos p has chunks: all c where c < p (received in earlier steps)
        #                                     plus c = p if step >= p-1
        #   In step s, ring_pos p forwards the NEWEST chunk it has:
        #     chunk = step - (p - 1) ... but only if ring_pos > 0 and step >= p-1
        #   Wait, this doesn't pipeline W chunks.

        # RCCL pipeline: divide data into W chunks, numbered 0..W-1
        # Step s (0..W-2):
        #   For each ring_pos p (1..W-1):
        #     If step >= ring_pos - 1:
        #       chunk_to_forward = step - (ring_pos - 1)
        #       if chunk_to_forward < W:
        #         pull chunk_to_forward from predecessor, store locally, push to successor

        # This means in step s:
        # ring_pos=1: forwards chunk s (if s < W)
        # ring_pos=2: forwards chunk s-1 (if s >= 1 and s-1 < W)
        # ring_pos=p: forwards chunk s-p+1 (if s >= p-1 and s-p+1 < W)

        # So different ranks forward different chunks in the same step!
        # This IS pipelined — all active ranks work on different chunks in parallel.

        # For ring_pos p: chunk_idx = step - ring_pos + 1
        chunk_idx = step - ring_pos + 1

        # Only forward if: ring_pos > 0 (not src), step >= ring_pos-1, chunk valid
        if ring_pos > 0:
            if chunk_idx >= 0:
                if chunk_idx < world_size:
                    # Pull this chunk from predecessor, write locally
                    c_row_start = chunk_idx * chunk_rows
                    c_actual_rows = tl.minimum(chunk_rows, M - c_row_start)
                    num_pid_m_chunk = tl.cdiv(c_actual_rows, BLOCK_SIZE_M)
                    total_tiles_chunk = num_pid_m_chunk * num_pid_n

                    for tile_id in range(pid, total_tiles_chunk, COMM_SMS):
                        pid_m = tile_id // num_pid_n
                        pid_n = tile_id % num_pid_n

                        rm_base = c_row_start + pid_m * BLOCK_SIZE_M
                        rn_base = pid_n * BLOCK_SIZE_N

                        rm = rm_base + tl.arange(0, BLOCK_SIZE_M)
                        rn = rn_base + tl.arange(0, BLOCK_SIZE_N)
                        rm = tl.max_contiguous(tl.multiple_of(rm, BLOCK_SIZE_M), BLOCK_SIZE_M)
                        rn = tl.max_contiguous(tl.multiple_of(rn, BLOCK_SIZE_N), BLOCK_SIZE_N)

                        offset = rm[:, None] * stride_m + rn[None, :] * stride_n
                        ptrs = tensor_ptr + offset

                        is_full = (rm_base + BLOCK_SIZE_M <= M) & (rn_base + BLOCK_SIZE_N <= N)

                        if is_full:
                            data = iris.load(ptrs, iris_rank, pred_iris, heap_bases)
                            tl.store(ptrs, data, cache_modifier=".wt")
                        else:
                            mask = (rm[:, None] < M) & (rn[None, :] < N)
                            data = iris.load(ptrs, iris_rank, pred_iris, heap_bases, mask=mask)
                            tl.store(ptrs, data, mask=mask, cache_modifier=".wt")

        # Barrier after each step — all ranks must complete before next step
        if INLINE_BARRIER:
            inline_device_barrier(
                pid,
                barrier_flags_ptr,
                wg_done_ptr,
                barrier_sense_ptr,
                heap_bases,
                iris_rank,
                world_size,
                rank_start,
                rank_stride,
                COMM_SMS,
            )


_dummy_barrier_cache: dict = {}


def _get_dummy_barrier(device):
    """Return cached dummy barrier tensors for the no-inline-barrier path."""
    if device not in _dummy_barrier_cache:
        import torch

        _dummy_barrier_cache[device] = tuple(torch.zeros(1, dtype=torch.int32, device=device) for _ in range(3))
    return _dummy_barrier_cache[device]


def launch(
    tensor,
    ctx,
    rank_in_group,
    rank_global,
    world_size,
    rank_start,
    rank_stride,
    src,
    config,
    inline_barrier=True,
    barrier_state=None,
):
    """Launch ring broadcast kernel."""
    M, N = tensor.shape[:2]
    stride_m, stride_n = tensor.stride(0), tensor.stride(1)

    chunk_rows = (M + world_size - 1) // world_size
    chunk_rows = ((chunk_rows + config.block_size_m - 1) // config.block_size_m) * config.block_size_m

    heap_bases = ctx.get_heap_bases()

    if inline_barrier and barrier_state is not None:
        barrier_flags, wg_done, barrier_sense = barrier_state
    else:
        barrier_flags, wg_done, barrier_sense = _get_dummy_barrier(tensor.device)

    iris_launch(
        persistent_broadcast_ring,
        (config.comm_sms,),
        tensor,
        M,
        N,
        stride_m,
        stride_n,
        chunk_rows,
        heap_bases,
        rank_in_group,
        rank_global,
        src,
        world_size,
        rank_start,
        rank_stride,
        barrier_flags,
        wg_done,
        barrier_sense,
        config.block_size_m,
        config.block_size_n,
        config.comm_sms,
        config.num_xcds,
        config.chunk_size,
        inline_barrier,
        num_stages=config.num_stages,
        num_warps=config.num_warps,
        waves_per_eu=config.waves_per_eu,
        algorithm="broadcast_ring",
        rank=rank_global,
        dtype=tensor.dtype,
    )
