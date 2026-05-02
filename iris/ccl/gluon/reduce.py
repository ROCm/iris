# SPDX-License-Identifier: MIT
# Copyright (c) 2025 Advanced Micro Devices, Inc. All rights reserved.

"""
Gluon kernel for reduce collective communication.

Only the root rank does work: it gathers all partials via remote loads,
accumulates in float32, and stores locally. Non-root ranks are no-ops.

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
def persistent_reduce_gluon(
    IrisDeviceCtx: gl.constexpr,
    context_tensor,
    input_ptr,
    output_ptr,
    M,
    N,
    stride_in_m,
    stride_in_n,
    stride_out_m,
    stride_out_n,
    group_rank: gl.constexpr,
    iris_rank: gl.constexpr,
    world_size: gl.constexpr,
    rank_start: gl.constexpr,
    rank_stride: gl.constexpr,
    dst_rank: gl.constexpr,
    BLOCK_SIZE_M: gl.constexpr,
    BLOCK_SIZE_N: gl.constexpr,
    GROUP_SIZE_M: gl.constexpr,
    COMM_SMS: gl.constexpr,
    THREADS_PER_WARP: gl.constexpr,
    WARPS_PER_CTA: gl.constexpr,
    TRACING: gl.constexpr = False,
):
    """
    Persistent reduce kernel using Gluon with flat-2D tiling.

    Only the root rank (dst_rank) does work. It loads partials from every
    rank via remote memory access, accumulates in float32, and stores the
    result locally. Non-root ranks exit immediately.
    """
    # Non-root ranks have nothing to do
    if group_rank != dst_rank:
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

    # Hoist local heap base outside the tile loop
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

        input_offsets = row * stride_in_m + col * stride_in_n
        output_offsets = row * stride_out_m + col * stride_out_n

        # Accumulate partials from all ranks
        # Start with rank 0's data, then add the rest
        first_target = rank_start
        first_base = gl.load(ctx.heap_bases + first_target)
        first_delta = first_base - local_base
        first_ptrs_int = tl.cast(input_ptr + input_offsets, gl.uint64) + first_delta
        first_ptrs = tl.cast(first_ptrs_int, input_ptr.dtype)
        acc = gl.load(first_ptrs, mask=mask, other=0.0).to(gl.float32)

        for i in range(1, world_size):
            target_iris_rank = rank_start + i * rank_stride
            if target_iris_rank == iris_rank:
                # Local load — no pointer translation needed
                partial = gl.load(input_ptr + input_offsets, mask=mask, other=0.0)
            else:
                target_base = gl.load(ctx.heap_bases + target_iris_rank)
                ptr_delta = target_base - local_base
                remote_ptrs_int = tl.cast(input_ptr + input_offsets, gl.uint64) + ptr_delta
                remote_ptrs = tl.cast(remote_ptrs_int, input_ptr.dtype)
                partial = gl.load(remote_ptrs, mask=mask, other=0.0)
            acc += partial.to(gl.float32)

        # Store result locally (root rank only)
        gl.store(output_ptr + output_offsets, acc.to(output_ptr.dtype.element_ty), mask=mask, cache_modifier=".wt")


def launch(
    output_tensor,
    input_tensor,
    ctx,
    rank_in_group,
    rank_global,
    world_size,
    rank_start,
    rank_stride,
    dst,
    config,
):
    """Launch the Gluon reduce kernel."""
    M, N = input_tensor.shape[:2]
    stride_in_m, stride_in_n = input_tensor.stride(0), input_tensor.stride(1)
    stride_out_m, stride_out_n = output_tensor.stride(0), output_tensor.stride(1)

    # Apply optimal defaults for gluon flat-2D kernel when user hasn't
    # overridden block sizes from the Config defaults (32x64).
    block_size_m = config.block_size_m
    block_size_n = config.block_size_n
    if block_size_m == 32 and block_size_n == 64:
        block_size_m = 8
        block_size_n = 256

    # Validate flat-2D layout constraints.
    total_elems = block_size_m * block_size_n
    threads_per_cta = config.threads_per_warp * config.num_warps
    if total_elems < threads_per_cta:
        raise ValueError(
            f"Gluon reduce requires block_size_m * block_size_n >= "
            f"threads_per_warp * num_warps ({threads_per_cta}), "
            f"got {block_size_m} * {block_size_n} = {total_elems}."
        )
    if total_elems % threads_per_cta != 0:
        raise ValueError(
            f"Gluon reduce requires block_size_m * block_size_n to be a "
            f"multiple of threads_per_warp * num_warps ({threads_per_cta}), "
            f"got {block_size_m} * {block_size_n} = {total_elems}. "
            f"Recommended: block_size_m=8, block_size_n=256."
        )

    context_tensor = ctx.get_device_context()
    tracing = getattr(ctx, "tracing", None)
    tracing_enabled = bool(tracing and getattr(tracing, "enabled", False))

    iris_launch(
        persistent_reduce_gluon,
        (config.comm_sms,),
        IrisDeviceCtx,
        context_tensor,
        input_tensor,
        output_tensor,
        M,
        N,
        stride_in_m,
        stride_in_n,
        stride_out_m,
        stride_out_n,
        rank_in_group,
        rank_global,
        world_size,
        rank_start,
        rank_stride,
        dst,
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
        algorithm="reduce",
        rank=rank_global,
        dtype=input_tensor.dtype,
    )
