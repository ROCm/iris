# SPDX-License-Identifier: MIT
# Copyright (c) 2025 Advanced Micro Devices, Inc. All rights reserved.

"""
Gluon kernel for broadcast collective communication.

Only the root rank does work: it loads tiles from its local tensor and
stores them to every other rank via remote pointer translation (and
locally via gl.store). Non-root ranks exit immediately.

Uses flat-2D tiling: a single 1D arange over BLOCK_SIZE_M * BLOCK_SIZE_N
elements with div/mod to compute 2D row/col indices, staying within
gluon's 1D BlockedLayout framework.
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
def persistent_broadcast_gluon(
    IrisDeviceCtx: gl.constexpr,
    context_tensor,
    tensor_ptr,
    M,
    N,
    stride_m,
    stride_n,
    group_rank: gl.constexpr,
    iris_rank: gl.constexpr,
    world_size: gl.constexpr,
    rank_start: gl.constexpr,
    rank_stride: gl.constexpr,
    src_rank: gl.constexpr,
    BLOCK_SIZE_M: gl.constexpr,
    BLOCK_SIZE_N: gl.constexpr,
    GROUP_SIZE_M: gl.constexpr,
    COMM_SMS: gl.constexpr,
    THREADS_PER_WARP: gl.constexpr,
    WARPS_PER_CTA: gl.constexpr,
    TRACING: gl.constexpr = False,
):
    """
    Persistent broadcast kernel using Gluon with flat-2D tiling.

    Only the root rank (src_rank) does work. It loads tiles from its local
    tensor and stores them to all ranks via remote pointer translation.
    Non-root ranks exit immediately.
    """
    # Non-root ranks have nothing to do
    if group_rank != src_rank:
        return

    ctx = IrisDeviceCtx.initialize(context_tensor, tracing=TRACING)

    pid = gl.program_id(0)

    num_pid_m = gl.cdiv(M, BLOCK_SIZE_M)
    num_pid_n = gl.cdiv(N, BLOCK_SIZE_N)
    total_tiles = num_pid_m * num_pid_n

    # Flat 1D layout covering BLOCK_SIZE_M * BLOCK_SIZE_N elements
    TOTAL_ELEMS: gl.constexpr = BLOCK_SIZE_M * BLOCK_SIZE_N
    ELEMS_PER_THREAD: gl.constexpr = TOTAL_ELEMS // (THREADS_PER_WARP * WARPS_PER_CTA)
    flat_layout: gl.constexpr = gl.BlockedLayout([ELEMS_PER_THREAD], [THREADS_PER_WARP], [WARPS_PER_CTA], [0])

    # Hoist local heap base outside the tile loop: eliminates redundant
    # gl.load(heap_bases) calls in the inner store loop.
    local_base = gl.load(ctx.heap_bases + iris_rank)

    for tile_id in range(pid, total_tiles, COMM_SMS):
        # Swizzled tile index computation for better L2 locality
        num_pid_in_group = GROUP_SIZE_M * num_pid_n
        group_id = tile_id // num_pid_in_group
        first_pid_m = group_id * GROUP_SIZE_M
        group_size_m = min(num_pid_m - first_pid_m, GROUP_SIZE_M)
        pid_m = first_pid_m + ((tile_id % num_pid_in_group) % group_size_m)
        pid_n = (tile_id % num_pid_in_group) // group_size_m

        # Flat index -> 2D row/col within tile
        flat_idx = gl.arange(0, TOTAL_ELEMS, layout=flat_layout)
        row_local = flat_idx // BLOCK_SIZE_N
        col_local = flat_idx % BLOCK_SIZE_N

        # Global row/col
        row = pid_m * BLOCK_SIZE_M + row_local
        col = pid_n * BLOCK_SIZE_N + col_local

        mask = (row < M) & (col < N)

        offsets = row * stride_m + col * stride_n
        src_ptrs = tensor_ptr + offsets

        # Load tile from root's local tensor
        data = gl.load(src_ptrs, mask=mask, other=0.0)

        # Traffic-shaped stores to all ranks: stagger write order per rank
        # so each rank writes to a different target at any given moment,
        # avoiding memory controller contention on the receiver side.
        for rank_idx in range(world_size):
            dest_idx = (group_rank + rank_idx) % world_size
            target_iris_rank = rank_start + dest_idx * rank_stride

            if dest_idx == group_rank:
                gl.store(src_ptrs, data, mask=mask, cache_modifier=".wt")
            else:
                # Hoisted translation: compute ptr_delta from pre-loaded
                # local_base rather than calling ctx.store() which would
                # do 2x gl.load(heap_bases) per call.
                target_base = gl.load(ctx.heap_bases + target_iris_rank)
                ptr_delta = target_base - local_base
                ptrs_int = tl.cast(src_ptrs, gl.uint64)
                remote_ptrs_int = ptrs_int + ptr_delta
                remote_ptrs = tl.cast(remote_ptrs_int, src_ptrs.dtype)
                gl.store(remote_ptrs, data, mask=mask)


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
):
    """Launch the Gluon broadcast kernel."""
    M, N = tensor.shape[:2]
    stride_m, stride_n = tensor.stride(0), tensor.stride(1)

    # Apply optimal defaults for gluon flat-2D kernel when user hasn't
    # overridden block sizes from the Config defaults (32x64).
    block_size_m = config.block_size_m
    block_size_n = config.block_size_n
    if block_size_m == 32 and block_size_n == 64:
        # User didn't override — use optimal flat-2D tile: 8x256
        block_size_m = 8
        block_size_n = 256

    # Validate flat-2D layout constraints.
    total_elems = block_size_m * block_size_n
    threads_per_cta = config.threads_per_warp * config.num_warps
    if total_elems < threads_per_cta:
        raise ValueError(
            f"Gluon broadcast requires block_size_m * block_size_n >= "
            f"threads_per_warp * num_warps ({threads_per_cta}), "
            f"got {block_size_m} * {block_size_n} = {total_elems}."
        )
    if total_elems % threads_per_cta != 0:
        raise ValueError(
            f"Gluon broadcast requires block_size_m * block_size_n to be a "
            f"multiple of threads_per_warp * num_warps ({threads_per_cta}), "
            f"got {block_size_m} * {block_size_n} = {total_elems}. "
            f"Recommended: block_size_m=8, block_size_n=256."
        )

    context_tensor = ctx.get_device_context()
    tracing = getattr(ctx, "tracing", None)
    tracing_enabled = bool(tracing and getattr(tracing, "enabled", False))

    iris_launch(
        persistent_broadcast_gluon,
        (config.comm_sms,),
        IrisDeviceCtx,
        context_tensor,
        tensor,
        M,
        N,
        stride_m,
        stride_n,
        rank_in_group,
        rank_global,
        world_size,
        rank_start,
        rank_stride,
        src,
        block_size_m,
        block_size_n,
        config.swizzle_size,
        config.comm_sms,
        config.threads_per_warp,
        config.num_warps,
        tracing_enabled,
        num_stages=config.num_stages,
        num_warps=config.num_warps,
        waves_per_eu=config.waves_per_eu,
        algorithm="broadcast",
        rank=rank_global,
        dtype=tensor.dtype,
    )
