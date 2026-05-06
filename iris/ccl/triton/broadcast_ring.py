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

Uses point-to-point step counters (not global barriers) between ring
steps. Each rank signals its successor after writing; successor polls
predecessor before reading. ~3-5us per step vs ~17us for global barrier.
"""

import triton
import triton.language as tl
import iris
from iris.host.tracing.kernel_artifacts import iris_launch
from iris.ccl.utils import inline_device_barrier
from iris.host.distributed.helpers import _translate_ptr
from ..utils import chiplet_transform_chunked


@triton.jit()
def _p2p_signal(
    step_flags_ptr,
    iris_rank: tl.constexpr,
    heap_bases: tl.tensor,
    COMM_SMS: tl.constexpr,
):
    """Signal completion: pid 0 increments own step counter after all CTAs finish writing."""
    pid = tl.program_id(0)
    tl.debug_barrier()
    if pid == 0:
        own_ptr = step_flags_ptr + iris_rank
        own_translated = _translate_ptr(own_ptr, iris_rank, iris_rank, heap_bases)
        tl.atomic_add(own_translated, 1, sem="release", scope="sys")


@triton.jit()
def _p2p_wait(
    step_flags_ptr,
    target,
    remote_iris_rank,
    iris_rank: tl.constexpr,
    heap_bases: tl.tensor,
):
    """Wait for a specific remote rank's step counter to reach target."""
    pid = tl.program_id(0)
    if pid == 0:
        remote_ptr = step_flags_ptr + remote_iris_rank
        remote_translated = _translate_ptr(remote_ptr, iris_rank, remote_iris_rank, heap_bases)
        while tl.atomic_cas(remote_translated, target, target, sem="acquire", scope="sys") < target:
            pass


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
    step_flags_ptr,
    step_base,
    barrier_flags_ptr,
    wg_done_ptr,
    barrier_sense_ptr,
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
    COMM_SMS: tl.constexpr,
    NUM_XCDS: tl.constexpr,
    CHUNK_SIZE: tl.constexpr,
    USE_P2P: tl.constexpr = True,
    INLINE_BARRIER: tl.constexpr = True,
):
    """
    Ring broadcast with pipelined chunks and point-to-point synchronization.

    Instead of a global barrier after each step, uses per-rank step counters:
    - After writing chunk data, rank signals successor via atomic_add on own flag
    - Before reading chunk data, rank polls predecessor's flag via atomic_cas
    - Only 1 rank waits on 1 other rank per step (not all-to-all sync)
    """
    pid = tl.program_id(0)

    if NUM_XCDS != 1:
        pid = chiplet_transform_chunked(pid, COMM_SMS, NUM_XCDS, CHUNK_SIZE)

    num_pid_n = tl.cdiv(N, BLOCK_SIZE_N)

    ring_pos = (group_rank - src + world_size) % world_size
    pred_group = (group_rank - 1 + world_size) % world_size
    pred_iris = rank_start + pred_group * rank_stride

    for step in tl.static_range(world_size - 1):
        chunk_idx = step - ring_pos + 1

        if ring_pos > 0:
            if chunk_idx >= 0:
                if chunk_idx < world_size:
                    if USE_P2P:
                        # Wait for predecessor to finish writing this chunk
                        target = step_base + step + 1
                        _p2p_wait(step_flags_ptr, target, pred_iris, iris_rank, heap_bases)
                        # Broadcast to all pids after pid 0 confirms
                        tl.debug_barrier()

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

        if USE_P2P:
            # Signal successor that this step's data is written
            _p2p_signal(step_flags_ptr, iris_rank, heap_bases, COMM_SMS)
        elif INLINE_BARRIER:
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


_dummy_flags_cache: dict = {}


def _get_dummy_flags(device):
    """Return cached dummy flags tensor."""
    if device not in _dummy_flags_cache:
        import torch

        _dummy_flags_cache[device] = torch.zeros(1, dtype=torch.int32, device=device)
    return _dummy_flags_cache[device]


_step_flags_cache: dict = {}


def _get_step_flags(ctx, group=None):
    """Get or create point-to-point step flags on symmetric heap. Monotonic, never reset."""
    key = group
    if key not in _step_flags_cache:
        _step_flags_cache[key] = ctx.zeros((ctx.num_ranks,), dtype=__import__("torch").int32)
        ctx.device_barrier(group)
    return _step_flags_cache[key]


_step_base_cache: dict = {}


def _get_step_base(group=None):
    """Track step base for monotonic step counters."""
    key = group
    if key not in _step_base_cache:
        _step_base_cache[key] = 0
    return _step_base_cache[key]


def _advance_step_base(world_size, group=None):
    """Advance step base by world_size-1 (number of ring steps per invocation)."""
    key = group
    if key not in _step_base_cache:
        _step_base_cache[key] = 0
    old = _step_base_cache[key]
    _step_base_cache[key] = old + (world_size - 1)
    return old


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
    group=None,
    use_p2p=True,
):
    """Launch ring broadcast kernel."""
    M, N = tensor.shape[:2]
    stride_m, stride_n = tensor.stride(0), tensor.stride(1)

    chunk_rows = (M + world_size - 1) // world_size
    chunk_rows = ((chunk_rows + config.block_size_m - 1) // config.block_size_m) * config.block_size_m

    heap_bases = ctx.get_heap_bases()

    if use_p2p:
        step_flags = _get_step_flags(ctx, group)
        step_base = _advance_step_base(world_size, group)
    else:
        step_flags = _get_dummy_flags(tensor.device)
        step_base = 0

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
        step_flags,
        step_base,
        barrier_flags,
        wg_done,
        barrier_sense,
        config.block_size_m,
        config.block_size_n,
        config.comm_sms,
        config.num_xcds,
        config.chunk_size,
        use_p2p,
        inline_barrier,
        num_stages=config.num_stages,
        num_warps=config.num_warps,
        waves_per_eu=config.waves_per_eu,
        algorithm="broadcast_ring",
        rank=rank_global,
        dtype=tensor.dtype,
    )
