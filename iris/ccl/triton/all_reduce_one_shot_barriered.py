# SPDX-License-Identifier: MIT
# Copyright (c) 2025 Advanced Micro Devices, Inc. All rights reserved.

"""
Triton one-shot allreduce with IN-KERNEL barriers.

Same algorithm as gluon one-shot but compiled with @triton.jit.
Uses iris.atomic_add for barrier signals and iris.load with
cache_modifier=".cv" for poll reads (emits sc0 sc1 = acquire).

This is the graph-capture-safe version: the poll load has acquire
semantics so it reads fresh flag values from remote GPUs.
"""

import torch
import triton
import triton.language as tl
import iris


@triton.jit
def _triton_barrier(
    flags_ptr,
    heap_bases: tl.tensor,
    group_rank: tl.constexpr,
    iris_rank: tl.constexpr,
    world_size: tl.constexpr,
    rank_start: tl.constexpr,
    rank_stride: tl.constexpr,
):
    """In-kernel barrier with acquire/release semantics.

    Protocol:
      1. buffer_wbl2 sc0 sc1 (flush dirty L2 — release)
      2. iris.atomic_add on own flag (release, system scope)
      3. iris.atomic_add on remote flags (relaxed, signal peers)
      4. Poll peer flags with iris.load(cache_modifier=".cv")
         → compiles to buffer_load sc0 sc1 (acquire)
      5. buffer_inv sc0 sc1 (invalidate L2 after barrier)
    """
    # Flush dirty L2 before signaling
    tl.inline_asm_elementwise(
        "buffer_wbl2 sc0 sc1",
        "=r",
        args=[],
        dtype=tl.int32,
        is_pure=False,
        pack=1,
    )

    # Atomic add on own flag (release)
    old = iris.atomic_add(
        flags_ptr + group_rank,
        1,
        iris_rank,
        iris_rank,
        heap_bases,
        sem="release",
        scope="sys",
    )
    target = old + 1

    # Signal remote flags (relaxed)
    for i in range(world_size):
        remote_rank = rank_start + i * rank_stride
        if remote_rank != iris_rank:
            iris.atomic_add(
                flags_ptr + group_rank,
                1,
                iris_rank,
                remote_rank,
                heap_bases,
                sem="relaxed",
                scope="sys",
            )

    # Poll peer flags with acquire semantics (.cv → sc0 sc1)
    for i in range(world_size):
        remote_rank = rank_start + i * rank_stride
        if remote_rank != iris_rank:
            while (
                iris.load(
                    flags_ptr + i,
                    iris_rank,
                    iris_rank,
                    heap_bases,
                    cache_modifier=".cv",
                )
                < target
            ):
                pass

    # Invalidate L2 after barrier
    tl.inline_asm_elementwise(
        "buffer_inv sc0 sc1",
        "=r",
        args=[],
        dtype=tl.int32,
        is_pure=False,
        pack=1,
    )


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
    start_flags_ptr,
    end_flags_ptr,
    BLOCK_SIZE: tl.constexpr,
    COMM_SMS: tl.constexpr,
    SINGLE_BARRIER: tl.constexpr,
):
    """One-shot allreduce with in-kernel barriers (triton + acquire/release).

    Same algorithm as gluon one_shot_all_reduce_gluon but compiled with
    @triton.jit using cache_modifier=".cv" for acquire loads.
    """
    pid = tl.program_id(0)

    # Start barrier: all ranks must have written their input
    _triton_barrier(
        start_flags_ptr,
        heap_bases,
        group_rank,
        iris_rank,
        world_size,
        rank_start,
        rank_stride,
    )

    # Reduction: read all peers, accumulate in FP32
    total_tiles = tl.cdiv(N_ELEMENTS, BLOCK_SIZE)

    for tile_id in range(pid, total_tiles, COMM_SMS):
        base_offset = tile_id * BLOCK_SIZE
        offsets = base_offset + tl.arange(0, BLOCK_SIZE)
        mask = offsets < N_ELEMENTS

        acc = tl.zeros((BLOCK_SIZE,), dtype=tl.float32)
        for i in range(world_size):
            remote_rank = rank_start + i * rank_stride
            partial = iris.load(
                input_ptr + offsets,
                iris_rank,
                remote_rank,
                heap_bases,
                mask=mask,
            )
            acc += partial.to(tl.float32)

        tl.store(
            output_ptr + offsets,
            acc.to(output_ptr.type.element_ty),
            mask=mask,
        )

    # Optional end barrier
    if not SINGLE_BARRIER:
        _triton_barrier(
            end_flags_ptr,
            heap_bases,
            group_rank,
            iris_rank,
            world_size,
            rank_start,
            rank_stride,
        )


class _BarrieredWorkspace:
    def __init__(self, ctx, world_size):
        self.start_flags = ctx.zeros((world_size,), dtype=torch.int32)
        self.end_flags = ctx.zeros((world_size,), dtype=torch.int32)
        self.prepared = True


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
    group=None,
):
    """Launch the triton one-shot allreduce with in-kernel barriers."""
    numel = input_tensor.numel()
    flat_input = input_tensor.contiguous().view(-1)
    flat_output = output_tensor.contiguous().view(-1)

    block_size = 2048
    num_sms = min(16, (numel + block_size - 1) // block_size)
    if numel <= 8192:
        num_sms = 1
    elif numel <= 32768:
        num_sms = min(4, num_sms)

    if workspace is None or not hasattr(workspace, 'start_flags'):
        workspace = _BarrieredWorkspace(ctx, world_size)

    capturing = torch.cuda.is_current_stream_capturing()
    if not capturing:
        workspace.start_flags.zero_()
        workspace.end_flags.zero_()

    heap_bases = ctx.get_heap_bases()

    one_shot_all_reduce_triton_barriered[(num_sms,)](
        flat_input,
        flat_output,
        numel,
        heap_bases,
        rank_in_group,
        rank_global,
        world_size,
        rank_start,
        rank_stride,
        workspace.start_flags,
        workspace.end_flags,
        block_size,
        num_sms,
        capturing,
    )

    return workspace


def all_reduce_preamble(output_tensor, input_tensor, ctx, config=None, workspace=None):
    world_size = ctx.get_num_ranks()
    if workspace is None or not hasattr(workspace, 'start_flags'):
        workspace = _BarrieredWorkspace(ctx, world_size)
    return workspace
