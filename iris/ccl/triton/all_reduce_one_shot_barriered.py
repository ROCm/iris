# SPDX-License-Identifier: MIT
# Copyright (c) 2025 Advanced Micro Devices, Inc. All rights reserved.

"""
Triton one-shot allreduce with vLLM-style in-kernel barriers.

Matches vLLM's CustomAllReduce barrier protocol:
  1. Read persistent flag counter from memory: flag = _flag[block] + 1
  2. Store flag to each PEER's sync slot (system scope)
  3. Poll OWN sync slots until peers write >= flag (acquire load)
  4. Persist flag counter: _flag[block] = flag

Key difference from the gluon barrier:
  - Uses store(flag) + poll(< flag), not atomic_add + poll(>= target)
  - Persistent flag counter read from MEMORY each call (fresh in graph replay)
  - iris.store for writes, iris.load(.cv) for polls (acquire semantics)
"""

import torch
import triton
import triton.language as tl
import iris


@triton.jit
def _barrier(
    flag_counter_ptr,
    sync_flags_ptr,
    heap_bases: tl.tensor,
    group_rank: tl.constexpr,
    iris_rank: tl.constexpr,
    world_size: tl.constexpr,
    rank_start: tl.constexpr,
    rank_stride: tl.constexpr,
):
    """vLLM-style in-kernel barrier for graph-capture-safe allreduce."""
    pid = tl.program_id(0)

    # Read persistent flag counter and increment
    flag = tl.load(flag_counter_ptr + pid) + 1

    # Write flag to each peer's sync slot
    for i in range(world_size):
        remote_rank = rank_start + i * rank_stride
        if remote_rank != iris_rank:
            slot = sync_flags_ptr + pid * world_size + group_rank
            iris.store(slot, flag, iris_rank, remote_rank, heap_bases)

    # Poll own sync slots until peers write >= flag
    for i in range(world_size):
        remote_rank = rank_start + i * rank_stride
        if remote_rank != iris_rank:
            slot = sync_flags_ptr + pid * world_size + i
            while iris.load(slot, iris_rank, iris_rank, heap_bases, cache_modifier=".cv") < flag:
                pass

    tl.debug_barrier()

    # Persist flag counter
    if tl.program_id(0) == pid:
        tl.store(flag_counter_ptr + pid, flag)


@triton.jit()
def one_shot_all_reduce_triton_barriered(
    input_ptr,
    output_ptr,
    N_ELEMENTS,
    heap_bases: tl.tensor,
    group_rank: tl.constexpr,
    iris_rank: tl.constexpr,
    world_size: tl.constexpr,
    rank_start: tl.constexpr,
    rank_stride: tl.constexpr,
    flag_counter_ptr,
    start_sync_ptr,
    end_sync_ptr,
    BLOCK_SIZE: tl.constexpr,
    COMM_SMS: tl.constexpr,
    SINGLE_BARRIER: tl.constexpr,
):
    """One-shot allreduce with vLLM-style in-kernel barriers."""
    pid = tl.program_id(0)

    # Start barrier
    _barrier(
        flag_counter_ptr, start_sync_ptr, heap_bases,
        group_rank, iris_rank, world_size,
        rank_start, rank_stride,
    )

    # Reduction
    total_tiles = tl.cdiv(N_ELEMENTS, BLOCK_SIZE)
    for tile_id in range(pid, total_tiles, COMM_SMS):
        base_offset = tile_id * BLOCK_SIZE
        offsets = base_offset + tl.arange(0, BLOCK_SIZE)
        mask = offsets < N_ELEMENTS

        acc = tl.zeros((BLOCK_SIZE,), dtype=tl.float32)
        for i in range(world_size):
            remote_rank = rank_start + i * rank_stride
            partial = iris.load(input_ptr + offsets, iris_rank, remote_rank, heap_bases, mask=mask)
            acc += partial.to(tl.float32)

        tl.store(output_ptr + offsets, acc.to(output_ptr.type.element_ty), mask=mask)

    # End barrier
    if not SINGLE_BARRIER:
        _barrier(
            flag_counter_ptr, end_sync_ptr, heap_bases,
            group_rank, iris_rank, world_size,
            rank_start, rank_stride,
        )


class _BarrieredWorkspace:
    def __init__(self, ctx, world_size, max_blocks=16):
        self.flag_counter = ctx.zeros((max_blocks,), dtype=torch.int32)
        self.start_sync = ctx.zeros((max_blocks * world_size,), dtype=torch.int32)
        self.end_sync = ctx.zeros((max_blocks * world_size,), dtype=torch.int32)
        self.prepared = True


def launch(
    output_tensor, input_tensor, ctx,
    rank_in_group, rank_global, world_size,
    rank_start, rank_stride, config,
    workspace=None, group=None,
):
    numel = input_tensor.numel()
    flat_input = input_tensor.contiguous().view(-1)
    flat_output = output_tensor.contiguous().view(-1)

    block_size = 2048
    num_sms = min(16, (numel + block_size - 1) // block_size)
    if numel <= 8192:
        num_sms = 1
    elif numel <= 32768:
        num_sms = min(4, num_sms)

    if workspace is None or not hasattr(workspace, 'flag_counter'):
        workspace = _BarrieredWorkspace(ctx, world_size)

    capturing = torch.cuda.is_current_stream_capturing()
    heap_bases = ctx.get_heap_bases()

    one_shot_all_reduce_triton_barriered[(num_sms,)](
        flat_input, flat_output, numel, heap_bases,
        rank_in_group, rank_global, world_size,
        rank_start, rank_stride,
        workspace.flag_counter,
        workspace.start_sync,
        workspace.end_sync,
        block_size, num_sms, capturing,
    )
    return workspace


def all_reduce_preamble(output_tensor, input_tensor, ctx, config=None, workspace=None):
    world_size = ctx.get_num_ranks()
    if workspace is None or not hasattr(workspace, 'flag_counter'):
        workspace = _BarrieredWorkspace(ctx, world_size)
    return workspace
