# SPDX-License-Identifier: MIT
# Copyright (c) 2025 Advanced Micro Devices, Inc. All rights reserved.

"""
Ring reduce kernel for large messages.

Reverse of ring broadcast: data flows toward root with reduction at
each hop. Divide data into W chunks. In each of W-1 steps, each rank
pulls a chunk from its ring successor, reduces with local data, and
stores locally. After W-1 steps, root has the fully reduced result.

Ring order (reverse): dst <- (dst-1+W)%W <- (dst-2+W)%W <- ... <- (dst+1)%W
Equivalently: successor in reduce ring = predecessor in broadcast ring.

Bandwidth efficiency: (W-1)/W — 87.5% for W=8 GPUs.
"""

import triton
import triton.language as tl
import iris
from iris.host.tracing.kernel_artifacts import iris_launch
from iris.ccl.utils import inline_device_barrier
from ..utils import chiplet_transform_chunked


@triton.jit()
def persistent_reduce_ring(
    input_ptr,
    output_ptr,
    M,
    N,
    stride_in_m,
    stride_in_n,
    stride_out_m,
    stride_out_n,
    chunk_rows,
    heap_bases: tl.tensor,
    group_rank: tl.constexpr,
    iris_rank: tl.constexpr,
    dst: tl.constexpr,
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
    Ring reduce with pipelined chunks.

    Ring flows toward dst. Each rank pulls from successor in ring order
    and reduces with its own data. The ring is:
      farthest_rank -> ... -> dst+2 -> dst+1 -> dst

    ring_pos: 0 = dst (root), W-1 = farthest rank
    In the reduce ring, data flows from high ring_pos to low ring_pos.
    successor_in_ring = rank at ring_pos + 1 = rank that sends TO this rank.

    Step s: rank at ring_pos p (p < W-1) pulls chunk c from its ring successor
    where c = step - (W - 2 - ring_pos).
    Only active if step >= W - 2 - ring_pos and c < W.
    """
    pid = tl.program_id(0)

    if NUM_XCDS != 1:
        pid = chiplet_transform_chunked(pid, COMM_SMS, NUM_XCDS, CHUNK_SIZE)

    num_pid_n = tl.cdiv(N, BLOCK_SIZE_N)
    acc_dtype = tl.float32 if output_ptr.type.element_ty != tl.int8 else tl.int32

    # Ring position: 0 = dst (root), increasing = farther from root
    ring_pos = (group_rank - dst + world_size) % world_size

    # Ring successor (the rank that sends data TO us)
    succ_group = (group_rank + 1) % world_size
    succ_iris = rank_start + succ_group * rank_stride

    # Step 0: every rank copies input to output (needed for in-place accumulation)
    # But we'll do accumulation on-the-fly during ring steps.

    # First: copy local input to output for all ranks
    # (output will be accumulated into during ring steps)
    num_pid_m_full = tl.cdiv(M, BLOCK_SIZE_M)
    total_tiles_full = num_pid_m_full * num_pid_n

    for tile_id in range(pid, total_tiles_full, COMM_SMS):
        pid_m = tile_id // num_pid_n
        pid_n = tile_id % num_pid_n

        rm_base = pid_m * BLOCK_SIZE_M
        rn_base = pid_n * BLOCK_SIZE_N

        rm = rm_base + tl.arange(0, BLOCK_SIZE_M)
        rn = rn_base + tl.arange(0, BLOCK_SIZE_N)
        rm = tl.max_contiguous(tl.multiple_of(rm, BLOCK_SIZE_M), BLOCK_SIZE_M)
        rn = tl.max_contiguous(tl.multiple_of(rn, BLOCK_SIZE_N), BLOCK_SIZE_N)

        in_offset = rm[:, None] * stride_in_m + rn[None, :] * stride_in_n
        out_offset = rm[:, None] * stride_out_m + rn[None, :] * stride_out_n

        is_full = (rm_base + BLOCK_SIZE_M <= M) & (rn_base + BLOCK_SIZE_N <= N)

        if is_full:
            data = tl.load(input_ptr + in_offset)
            tl.store(output_ptr + out_offset, data, cache_modifier=".wt")
        else:
            mask = (rm[:, None] < M) & (rn[None, :] < N)
            data = tl.load(input_ptr + in_offset, mask=mask, other=0.0)
            tl.store(output_ptr + out_offset, data, mask=mask, cache_modifier=".wt")

    # Barrier: ensure all ranks have written their local input to output
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

    # Ring reduce steps: W-1 steps
    # In each step, ranks pull from successor and accumulate
    # Pipelined: in step s, rank at ring_pos p processes chunk c
    # where c = s - (W - 2 - p) = s - W + 2 + p
    # Active if c >= 0 and c < W and ring_pos < W-1
    for step in tl.static_range(world_size - 1):
        chunk_idx = step - (world_size - 2 - ring_pos)

        if ring_pos < world_size - 1:
            if chunk_idx >= 0:
                if chunk_idx < world_size:
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

                        out_offset = rm[:, None] * stride_out_m + rn[None, :] * stride_out_n

                        is_full = (rm_base + BLOCK_SIZE_M <= M) & (rn_base + BLOCK_SIZE_N <= N)

                        if is_full:
                            local = tl.load(output_ptr + out_offset).to(acc_dtype)
                            remote = iris.load(
                                output_ptr + out_offset, iris_rank, succ_iris, heap_bases
                            ).to(acc_dtype)
                            acc = local + remote
                            tl.store(
                                output_ptr + out_offset,
                                acc.to(output_ptr.type.element_ty),
                                cache_modifier=".wt",
                            )
                        else:
                            mask = (rm[:, None] < M) & (rn[None, :] < N)
                            local = tl.load(output_ptr + out_offset, mask=mask, other=0.0).to(acc_dtype)
                            remote = iris.load(
                                output_ptr + out_offset, iris_rank, succ_iris, heap_bases, mask=mask
                            ).to(acc_dtype)
                            acc = local + remote
                            tl.store(
                                output_ptr + out_offset,
                                acc.to(output_ptr.type.element_ty),
                                mask=mask,
                                cache_modifier=".wt",
                            )

        # Barrier after each step
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
    output_tensor,
    input_tensor,
    ctx,
    rank_in_group,
    rank_global,
    dst,
    world_size,
    rank_start,
    rank_stride,
    config,
    inline_barrier=True,
    barrier_state=None,
):
    """Launch ring reduce kernel."""
    M, N = input_tensor.shape[:2]
    stride_in_m, stride_in_n = input_tensor.stride(0), input_tensor.stride(1)
    stride_out_m, stride_out_n = output_tensor.stride(0), output_tensor.stride(1)

    chunk_rows = (M + world_size - 1) // world_size
    chunk_rows = ((chunk_rows + config.block_size_m - 1) // config.block_size_m) * config.block_size_m

    heap_bases = ctx.get_heap_bases()

    if inline_barrier and barrier_state is not None:
        barrier_flags, wg_done, barrier_sense = barrier_state
    else:
        barrier_flags, wg_done, barrier_sense = _get_dummy_barrier(input_tensor.device)

    iris_launch(
        persistent_reduce_ring,
        (config.comm_sms,),
        input_tensor,
        output_tensor,
        M,
        N,
        stride_in_m,
        stride_in_n,
        stride_out_m,
        stride_out_n,
        chunk_rows,
        heap_bases,
        rank_in_group,
        rank_global,
        dst,
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
        num_warps=8,
        num_stages=1,
        waves_per_eu=1,
        algorithm="reduce_ring",
        rank=rank_global,
        dtype=input_tensor.dtype,
    )
