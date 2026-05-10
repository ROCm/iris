# SPDX-License-Identifier: MIT
# Copyright (c) 2025 Advanced Micro Devices, Inc. All rights reserved.

"""
Gluon kernel for one-shot all-reduce collective communication.

This module is lazily imported only when config.use_gluon=True.
If gluon is not installed, the import itself raises ValueError.

One-shot all-reduce: each rank reads all remote inputs via XGMI,
accumulates in FP32, and writes the reduced result locally. No
staging buffers, no atomics, no write-to-remote.

The gluon barrier uses 2 fences per barrier (1 release atomic_add +
1 inline buffer_inv) vs Triton's ~15. For graph-captured workloads,
the end barrier can be elided (SINGLE_BARRIER=True) because the
start barrier of the next replay guarantees all prior writes completed.

Beats RCCL by 1.5-2.3x on 8x MI300X for 1K-128K BF16 elements.
"""

try:
    import triton.language as tl
    from triton.experimental import gluon
    from triton.experimental.gluon import language as gl
except ImportError as e:
    raise ValueError("Gluon is not available. Install Triton with Gluon support or set use_gluon=False.") from e

from iris.mem.gluon.context import Context as IrisDeviceCtx
from iris.host.tracing.kernel_artifacts import iris_launch


@gluon.jit
def _gluon_barrier(
    ctx_obj,
    flags_ptr,
    group_rank: gl.constexpr,
    world_size: gl.constexpr,
    rank_start: gl.constexpr,
    rank_stride: gl.constexpr,
):
    """
    GPU-side barrier using gluon atomics.

    Fence budget: 1 buffer_wbl2 (release atomic) + 1 buffer_inv (inline asm)
    = 2 fences total, vs ~15 for Triton's barrier.

    Protocol:
      1. Release atomic_add on local flag (buffer_wbl2 fence)
      2. Relaxed atomic_add to all remote flags (0 fences)
      3. Poll local copies of all remote flags with .cv loads (0 fences)
      4. Inline buffer_inv to invalidate L2 (1 fence)
    """
    cur_rank = ctx_obj.cur_rank

    my_flag = flags_ptr + group_rank

    old = ctx_obj.atomic_add(my_flag, 1, to_rank=cur_rank, sem="release", scope="sys")
    target = old + 1

    for i in range(world_size):
        remote_rank = rank_start + i * rank_stride
        if remote_rank != cur_rank:
            ctx_obj.atomic_add(my_flag, 1, to_rank=remote_rank, sem="relaxed", scope="sys")

    for i in range(world_size):
        remote_rank = rank_start + i * rank_stride
        if remote_rank != cur_rank:
            poll_ptr = flags_ptr + i
            poll_translated = ctx_obj._translate(poll_ptr, cur_rank, cur_rank)
            while gl.load(poll_translated, cache_modifier=".cv") < target:
                pass

    tl.inline_asm_elementwise(
        "buffer_inv sc1",
        "=r",
        args=[],
        dtype=tl.int32,
        is_pure=False,
        pack=1,
    )


@gluon.jit
def one_shot_all_reduce_gluon(
    IrisDeviceCtx: gl.constexpr,
    context_tensor,
    input_ptr,
    output_ptr,
    N_ELEMENTS,
    group_rank: gl.constexpr,
    iris_rank: gl.constexpr,
    world_size: gl.constexpr,
    rank_start: gl.constexpr,
    rank_stride: gl.constexpr,
    start_flags_ptr,
    end_flags_ptr,
    BLOCK_SIZE: gl.constexpr,
    COMM_SMS: gl.constexpr,
    SINGLE_BARRIER: gl.constexpr,
    THREADS_PER_WARP: gl.constexpr,
    WARPS_PER_CTA: gl.constexpr,
    TRACING: gl.constexpr = False,
):
    """
    One-shot all-reduce using gluon with flat 1D tiling.

    Each CTA reads all remote inputs via heap_bases pointer translation,
    accumulates in FP32, and writes the reduced result locally. The read-
    from-remote pattern means no rank writes to another rank's memory
    during the data phase — only the barrier touches remote memory.

    When SINGLE_BARRIER=True (graph replay mode):
      start_barrier → reduce → done
      The start_barrier of replay N+1 guarantees all ranks finished
      writing output in replay N. End barrier is redundant.

    When SINGLE_BARRIER=False (standalone mode):
      start_barrier → reduce → end_barrier

    Memory layout (BlockedLayout):
        A 1D BlockedLayout distributes BLOCK_SIZE elements across the
        thread hierarchy:
            ELEMS_PER_THREAD = BLOCK_SIZE // (THREADS_PER_WARP * WARPS_PER_CTA)

    Args:
        IrisDeviceCtx: Gluon device context class for remote memory operations.
        context_tensor: Opaque tensor holding IrisDeviceCtx state.
        input_ptr: Pointer to local input tensor (flattened, N_ELEMENTS).
        output_ptr: Pointer to output tensor (flattened, N_ELEMENTS).
        N_ELEMENTS: Total number of elements in the tensor.
        group_rank: This rank's index within the ProcessGroup (0..world_size-1).
        iris_rank: This rank's global index in the iris context (for RMA addressing).
        world_size: Total number of ranks in the group.
        rank_start: First iris rank in the group (for RMA target computation).
        rank_stride: Stride between consecutive iris ranks in the group.
        start_flags_ptr: Pointer to start barrier flags (world_size int32s on symmetric heap).
        end_flags_ptr: Pointer to end barrier flags (world_size int32s on symmetric heap).
        BLOCK_SIZE: Elements per tile (must be multiple of THREADS_PER_WARP * WARPS_PER_CTA).
        COMM_SMS: Number of CUs for persistent scheduling.
        SINGLE_BARRIER: If True, skip end barrier (safe for graph replay).
        THREADS_PER_WARP: Threads per warp/wavefront (64 for AMD).
        WARPS_PER_CTA: Number of warps per workgroup.
    """
    ctx = IrisDeviceCtx.initialize(context_tensor, tracing=TRACING)
    pid = gl.program_id(0)

    _gluon_barrier(ctx, start_flags_ptr, group_rank, world_size, rank_start, rank_stride)

    # Hoist local heap base: avoids redundant gl.load(heap_bases) in inner loop
    local_base = gl.load(ctx.heap_bases + iris_rank)
    input_int = tl.cast(input_ptr, gl.uint64)
    input_offset = input_int - local_base

    total_tiles = gl.cdiv(N_ELEMENTS, BLOCK_SIZE)

    ELEMS_PER_THREAD: gl.constexpr = BLOCK_SIZE // (THREADS_PER_WARP * WARPS_PER_CTA)
    flat_layout: gl.constexpr = gl.BlockedLayout([ELEMS_PER_THREAD], [THREADS_PER_WARP], [WARPS_PER_CTA], [0])

    for tile_id in range(pid, total_tiles, COMM_SMS):
        base_offset = tile_id * BLOCK_SIZE
        offsets = base_offset + gl.arange(0, BLOCK_SIZE, layout=flat_layout)
        is_full = base_offset + BLOCK_SIZE <= N_ELEMENTS

        if is_full:
            r0_base = gl.load(ctx.heap_bases + rank_start)
            r0_ptr = tl.cast(tl.cast(r0_base, gl.pointer_type(gl.int8)) + input_offset, input_ptr.dtype)
            acc = gl.load(r0_ptr + offsets).to(gl.float32)

            for i in range(1, world_size):
                remote_rank = rank_start + i * rank_stride
                ri_base = gl.load(ctx.heap_bases + remote_rank)
                ri_ptr = tl.cast(tl.cast(ri_base, gl.pointer_type(gl.int8)) + input_offset, input_ptr.dtype)
                acc += gl.load(ri_ptr + offsets).to(gl.float32)

            gl.store(output_ptr + offsets, acc.to(output_ptr.type.element_ty))
        else:
            mask = offsets < N_ELEMENTS
            r0_base = gl.load(ctx.heap_bases + rank_start)
            r0_ptr = tl.cast(tl.cast(r0_base, gl.pointer_type(gl.int8)) + input_offset, input_ptr.dtype)
            acc = gl.load(r0_ptr + offsets, mask=mask, other=0.0).to(gl.float32)

            for i in range(1, world_size):
                remote_rank = rank_start + i * rank_stride
                ri_base = gl.load(ctx.heap_bases + remote_rank)
                ri_ptr = tl.cast(tl.cast(ri_base, gl.pointer_type(gl.int8)) + input_offset, input_ptr.dtype)
                acc += gl.load(ri_ptr + offsets, mask=mask, other=0.0).to(gl.float32)

            gl.store(output_ptr + offsets, acc.to(output_ptr.type.element_ty), mask=mask)

    if not SINGLE_BARRIER:
        _gluon_barrier(ctx, end_flags_ptr, group_rank, world_size, rank_start, rank_stride)


def _get_num_sms(numel):
    """Select CTA count based on message size."""
    BLOCK_SIZE = 2048
    total_tiles = (numel + BLOCK_SIZE - 1) // BLOCK_SIZE
    if numel <= 8192:
        return 1
    elif numel <= 32768:
        return min(total_tiles, 4)
    else:
        return min(total_tiles, 16)


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
    """Launch the Gluon one-shot all-reduce kernel."""

    numel = input_tensor.numel()
    flat_input = input_tensor.contiguous().view(-1)
    flat_output = output_tensor.contiguous().view(-1)

    block_size = 2048
    num_warps = 8
    num_sms = _get_num_sms(numel)

    context_tensor = ctx.get_device_context()
    tracing = getattr(ctx, "tracing", None)
    tracing_enabled = bool(tracing and getattr(tracing, "enabled", False))

    if workspace is None or not hasattr(workspace, "start_flags"):
        import torch
        if torch.cuda.is_current_stream_capturing():
            raise RuntimeError(
                "iris gluon all_reduce: workspace must be pre-created before graph capture. "
                "Call all_reduce or all_reduce_preamble once outside capture first."
            )
        workspace = _GluonAllReduceWorkspace(ctx, world_size)

    iris_launch(
        one_shot_all_reduce_gluon,
        (num_sms,),
        IrisDeviceCtx,
        context_tensor,
        flat_input,
        flat_output,
        numel,
        rank_in_group,
        rank_global,
        world_size,
        rank_start,
        rank_stride,
        workspace.start_flags,
        workspace.end_flags,
        block_size,
        num_sms,
        getattr(config, "all_reduce_single_barrier", False),
        config.threads_per_warp,
        num_warps,
        tracing_enabled,
        num_stages=config.num_stages,
        num_warps=num_warps,
        waves_per_eu=config.waves_per_eu,
        algorithm="all_reduce",
        rank=rank_global,
        dtype=input_tensor.dtype,
    )

    return workspace


def all_reduce_preamble(output_tensor, input_tensor, ctx, config=None, workspace=None):
    """Allocate barrier flag tensors for the gluon one-shot all-reduce."""
    world_size = ctx.get_num_ranks()
    if workspace is None or not hasattr(workspace, "start_flags"):
        workspace = _GluonAllReduceWorkspace(ctx, world_size)
    workspace.prepared = True
    return workspace


class _GluonAllReduceWorkspace:
    """Holds barrier flag tensors for the gluon one-shot all-reduce."""

    def __init__(self, ctx, world_size):
        import torch

        self.start_flags = ctx.zeros((world_size,), dtype=torch.int32)
        self.end_flags = ctx.zeros((world_size,), dtype=torch.int32)
        self.prepared = True
