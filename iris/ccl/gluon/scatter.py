# SPDX-License-Identifier: MIT
# Copyright (c) 2025 Advanced Micro Devices, Inc. All rights reserved.

"""
Gluon kernel for scatter collective communication.

Uses flat-2D tiling: a single 1D arange over BLOCK_SIZE_M * BLOCK_SIZE_N elements
with div/mod to compute 2D row/col indices. Only the root rank does work,
loading each destination rank's chunk from input and writing it to that rank's output.
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
def persistent_scatter_gluon(
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
    src: gl.constexpr,
    BLOCK_SIZE_M: gl.constexpr,
    BLOCK_SIZE_N: gl.constexpr,
    GROUP_SIZE_M: gl.constexpr,
    COMM_SMS: gl.constexpr,
    THREADS_PER_WARP: gl.constexpr,
    WARPS_PER_CTA: gl.constexpr,
    TRACING: gl.constexpr = False,
):
    """
    Persistent scatter kernel using Gluon with flat-2D tiling.

    Only root rank (src) does work. For each destination rank i, loads
    input[i*M:(i+1)*M, :] and writes to rank i's output tensor.

    Uses a flat 1D index space of BLOCK_SIZE_M * BLOCK_SIZE_N elements,
    computing 2D row/col via integer div/mod.

    Args:
        IrisDeviceCtx: Gluon device context class for remote memory operations.
        context_tensor: Opaque tensor holding IrisDeviceCtx state.
        input_ptr: Pointer to input tensor on root (world_size * M, N), unused on non-root.
        output_ptr: Pointer to output tensor (M, N).
        M: Number of rows per rank in output.
        N: Number of columns.
        stride_in_m, stride_in_n: Row and column strides for input tensor.
        stride_out_m, stride_out_n: Row and column strides for output tensor.
        group_rank: This rank's index within the ProcessGroup (0..world_size-1).
        iris_rank: This rank's global index in the iris context (for RMA addressing).
        world_size: Total number of ranks in the group.
        rank_start: First iris rank in the group.
        rank_stride: Stride between consecutive iris ranks in the group.
        src: Source (root) rank within the group.
        BLOCK_SIZE_M: Number of rows per tile.
        BLOCK_SIZE_N: Number of columns per tile.
        GROUP_SIZE_M: Swizzle group size for M-dimension tiling.
        COMM_SMS: Number of CUs used for persistent scheduling.
        THREADS_PER_WARP: Threads per warp/wavefront (64 for AMD, 32 for NVIDIA).
        WARPS_PER_CTA: Number of warps per workgroup.
    """
    # Only root rank does work
    if group_rank != src:
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

        # Global row/col within the per-rank output (M x N)
        row = pid_m * BLOCK_SIZE_M + row_local
        col = pid_n * BLOCK_SIZE_N + col_local

        output_mask = (row < M) & (col < N)

        # Output offset (same for all destination ranks — each gets M x N)
        output_offsets = row * stride_out_m + col * stride_out_n
        output_ptrs = output_ptr + output_offsets

        # Traffic-shaped stores: stagger write order per src rank
        for rank_idx in range(world_size):
            dest_idx = (src + rank_idx) % world_size
            target_iris_rank = rank_start + dest_idx * rank_stride

            # Input rows for dest_idx: input[dest_idx*M:(dest_idx+1)*M, :]
            input_row = dest_idx * M + row
            input_mask = (input_row < (dest_idx + 1) * M) & (col < N)
            combined_mask = output_mask & input_mask

            # Load chunk from input
            input_offsets = input_row * stride_in_m + col * stride_in_n
            input_addr = input_ptr + input_offsets
            data = gl.load(input_addr, mask=combined_mask, other=0.0)

            if dest_idx == src:
                # Local destination: root writes to its own output
                gl.store(output_ptrs, data, mask=combined_mask, cache_modifier=".wt")
            else:
                # Remote destination: compute ptr_delta from hoisted local_base
                target_base = gl.load(ctx.heap_bases + target_iris_rank)
                ptr_delta = target_base - local_base
                output_ptrs_int = tl.cast(output_ptrs, gl.uint64)
                remote_ptrs_int = output_ptrs_int + ptr_delta
                remote_ptrs = tl.cast(remote_ptrs_int, output_ptrs.dtype)
                gl.store(remote_ptrs, data, mask=combined_mask)


def launch(
    input_tensor,
    output_tensor,
    ctx,
    rank_in_group,
    rank_global,
    world_size,
    rank_start,
    rank_stride,
    src,
    config,
):
    """Launch the Gluon scatter kernel."""
    M, N = output_tensor.shape[:2]
    stride_out_m, stride_out_n = output_tensor.stride(0), output_tensor.stride(1)

    # Input strides: only meaningful on root
    if rank_in_group == src:
        stride_in_m, stride_in_n = input_tensor.stride(0), input_tensor.stride(1)
    else:
        stride_in_m, stride_in_n = 0, 0

    # Gluon only supports the persistent variant
    if config.all_gather_variant != "persistent":
        raise ValueError(
            f"Gluon scatter only supports all_gather_variant='persistent', got '{config.all_gather_variant}'."
        )

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
            f"Gluon scatter requires block_size_m * block_size_n >= "
            f"threads_per_warp * num_warps ({threads_per_cta}), "
            f"got {block_size_m} * {block_size_n} = {total_elems}."
        )
    if total_elems % threads_per_cta != 0:
        raise ValueError(
            f"Gluon scatter requires block_size_m * block_size_n to be a "
            f"multiple of threads_per_warp * num_warps ({threads_per_cta}), "
            f"got {block_size_m} * {block_size_n} = {total_elems}. "
            f"Recommended: block_size_m=8, block_size_n=256."
        )

    context_tensor = ctx.get_device_context()
    tracing = getattr(ctx, "tracing", None)
    tracing_enabled = bool(tracing and getattr(tracing, "enabled", False))

    iris_launch(
        persistent_scatter_gluon,
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
        algorithm="scatter",
        rank=rank_global,
        dtype=output_tensor.dtype,
    )
