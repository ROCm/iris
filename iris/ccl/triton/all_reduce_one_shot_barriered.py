# SPDX-License-Identifier: MIT
# Copyright (c) 2025 Advanced Micro Devices, Inc. All rights reserved.

"""
Triton one-shot allreduce with vLLM-style in-kernel barriers.

Pre-computes all translated flag addresses during init so the
barrier kernel uses raw pointers — no runtime address translation,
fully graph-capturable.
"""

import torch
import triton
import triton.language as tl
import iris


@triton.jit
def _barrier_precomputed(
    peer_sync_ptrs,
    self_sync_ptr,
    flag_counter_ptr,
    group_rank: tl.constexpr,
    world_size: tl.constexpr,
):
    """vLLM-style barrier using pre-computed translated pointers.

    peer_sync_ptrs: [world_size] int64 tensor of pre-translated base pointers
    self_sync_ptr: our own sync array (regular triton pointer)
    flag_counter_ptr: persistent per-block flag counter
    """
    pid = tl.program_id(0)

    # Read persistent flag and increment
    flag = tl.load(flag_counter_ptr + pid) + 1

    # Write flag to each peer's sync slot
    for i in range(world_size):
        if i != group_rank:
            # Get pre-translated pointer for peer i
            peer_base = tl.load(peer_sync_ptrs + i).to(tl.pointer_type(tl.int32))
            peer_slot = peer_base + pid * world_size + group_rank
            tl.store(peer_slot, flag)

    # Poll own sync slots
    for i in range(world_size):
        if i != group_rank:
            my_slot = self_sync_ptr + pid * world_size + i
            while tl.load(my_slot, cache_modifier=".cv") < flag:
                pass

    tl.debug_barrier()

    # Persist flag counter
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
    start_peer_ptrs,
    start_self_ptr,
    end_peer_ptrs,
    end_self_ptr,
    BLOCK_SIZE: tl.constexpr,
    COMM_SMS: tl.constexpr,
    SINGLE_BARRIER: tl.constexpr,
):
    """One-shot allreduce with pre-computed barrier pointers."""
    pid = tl.program_id(0)

    _barrier_precomputed(
        start_peer_ptrs, start_self_ptr, flag_counter_ptr,
        group_rank, world_size,
    )

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

    if not SINGLE_BARRIER:
        _barrier_precomputed(
            end_peer_ptrs, end_self_ptr, flag_counter_ptr,
            group_rank, world_size,
        )


class _BarrieredWorkspace:
    def __init__(self, ctx, world_size, rank, max_blocks=16):
        self.flag_counter = ctx.zeros((max_blocks,), dtype=torch.int32)
        self.start_sync = ctx.zeros((max_blocks * world_size,), dtype=torch.int32)
        self.end_sync = ctx.zeros((max_blocks * world_size,), dtype=torch.int32)

        # Pre-compute translated pointers for each peer
        heap_bases = ctx.get_heap_bases()
        local_base = heap_bases[rank].item()

        start_ptrs = []
        end_ptrs = []
        for peer in range(world_size):
            if peer == rank:
                start_ptrs.append(self.start_sync.data_ptr())
                end_ptrs.append(self.end_sync.data_ptr())
            else:
                peer_base = heap_bases[peer].item()
                start_offset = self.start_sync.data_ptr() - local_base
                start_ptrs.append(peer_base + start_offset)
                end_offset = self.end_sync.data_ptr() - local_base
                end_ptrs.append(peer_base + end_offset)

        self.start_peer_ptrs = torch.tensor(start_ptrs, dtype=torch.int64, device=f"cuda:{rank}")
        self.end_peer_ptrs = torch.tensor(end_ptrs, dtype=torch.int64, device=f"cuda:{rank}")
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
        workspace = _BarrieredWorkspace(ctx, world_size, rank_global)

    capturing = torch.cuda.is_current_stream_capturing()
    heap_bases = ctx.get_heap_bases()

    one_shot_all_reduce_triton_barriered[(num_sms,)](
        flat_input, flat_output, numel, heap_bases,
        rank_in_group, rank_global, world_size,
        rank_start, rank_stride,
        workspace.flag_counter,
        workspace.start_peer_ptrs,
        workspace.start_sync,
        workspace.end_peer_ptrs,
        workspace.end_sync,
        block_size, num_sms, capturing,
    )
    return workspace


def all_reduce_preamble(output_tensor, input_tensor, ctx, config=None, workspace=None):
    world_size = ctx.get_num_ranks()
    rank = ctx.get_rank()
    if workspace is None or not hasattr(workspace, 'flag_counter'):
        workspace = _BarrieredWorkspace(ctx, world_size, rank)
    return workspace
