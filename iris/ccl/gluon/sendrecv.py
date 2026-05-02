# SPDX-License-Identifier: MIT
# Copyright (c) 2025 Advanced Micro Devices, Inc. All rights reserved.

"""
Gluon kernel for point-to-point send/recv communication.

Uses flat-2D tiling: a single 1D arange over BLOCK_SIZE_M * BLOCK_SIZE_N
elements with div/mod to compute 2D row/col indices. This gives one load
and one remote store per tile while staying within gluon's 1D BlockedLayout
framework.
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
def send_kernel_gluon(
    IrisDeviceCtx: gl.constexpr,
    context_tensor,
    input_ptr,
    output_ptr,
    flag_ptr,
    M,
    N,
    stride_in_m,
    stride_in_n,
    stride_out_m,
    stride_out_n,
    iris_rank: gl.constexpr,
    dst_iris_rank: gl.constexpr,
    tag: gl.constexpr,
    BLOCK_SIZE_M: gl.constexpr,
    BLOCK_SIZE_N: gl.constexpr,
    GROUP_SIZE_M: gl.constexpr,
    COMM_SMS: gl.constexpr,
    THREADS_PER_WARP: gl.constexpr,
    WARPS_PER_CTA: gl.constexpr,
    TRACING: gl.constexpr = False,
):
    """
    Gluon send kernel with flat-2D tiling.

    Loads local tiles and writes them to the receiver's output buffer via
    remote pointer translation. After all tiles are written, PID 0 signals
    completion by setting the flag on the receiver.

    Args:
        IrisDeviceCtx: Gluon device context class for remote memory operations.
        context_tensor: Opaque tensor holding IrisDeviceCtx state.
        input_ptr: Local input tensor to send, shape (M, N).
        output_ptr: Receiver's output tensor (in symmetric heap), shape (M, N).
        flag_ptr: Pointer to int32 flag on symmetric heap for signaling.
        M, N: Tensor dimensions.
        stride_in_m, stride_in_n: Strides for input tensor.
        stride_out_m, stride_out_n: Strides for output tensor.
        iris_rank: This rank's global iris rank.
        dst_iris_rank: Destination rank's global iris rank.
        tag: Communication tag.
        BLOCK_SIZE_M, BLOCK_SIZE_N: Block sizes for tiling.
        GROUP_SIZE_M: Swizzle group size.
        COMM_SMS: Number of CUs for persistent scheduling.
        THREADS_PER_WARP: Threads per warp/wavefront.
        WARPS_PER_CTA: Warps per workgroup.
    """
    ctx = IrisDeviceCtx.initialize(context_tensor, tracing=TRACING)

    pid = gl.program_id(0)

    num_pid_m = gl.cdiv(M, BLOCK_SIZE_M)
    num_pid_n = gl.cdiv(N, BLOCK_SIZE_N)
    total_tiles = num_pid_m * num_pid_n

    TOTAL_ELEMS: gl.constexpr = BLOCK_SIZE_M * BLOCK_SIZE_N
    ELEMS_PER_THREAD: gl.constexpr = TOTAL_ELEMS // (THREADS_PER_WARP * WARPS_PER_CTA)
    flat_layout: gl.constexpr = gl.BlockedLayout([ELEMS_PER_THREAD], [THREADS_PER_WARP], [WARPS_PER_CTA], [0])

    # Hoist heap base loads outside tile loop
    local_base = gl.load(ctx.heap_bases + iris_rank)
    target_base = gl.load(ctx.heap_bases + dst_iris_rank)
    ptr_delta = target_base - local_base

    for tile_id in range(pid, total_tiles, COMM_SMS):
        num_pid_in_group = GROUP_SIZE_M * num_pid_n
        group_id = tile_id // num_pid_in_group
        first_pid_m = group_id * GROUP_SIZE_M
        group_size_m = min(num_pid_m - first_pid_m, GROUP_SIZE_M)
        pid_m = first_pid_m + ((tile_id % num_pid_in_group) % group_size_m)
        pid_n = (tile_id % num_pid_in_group) // group_size_m

        flat_idx = gl.arange(0, TOTAL_ELEMS, layout=flat_layout)
        row_local = flat_idx // BLOCK_SIZE_N
        col_local = flat_idx % BLOCK_SIZE_N

        row = pid_m * BLOCK_SIZE_M + row_local
        col = pid_n * BLOCK_SIZE_N + col_local
        mask = (row < M) & (col < N)

        # Load from local input
        input_offsets = row * stride_in_m + col * stride_in_n
        input_addr = input_ptr + input_offsets
        data = gl.load(input_addr, mask=mask, other=0.0)

        # Write to receiver's output via remote pointer translation
        output_offsets = row * stride_out_m + col * stride_out_n
        output_ptrs = output_ptr + output_offsets
        output_ptrs_int = tl.cast(output_ptrs, gl.uint64)
        remote_ptrs_int = output_ptrs_int + ptr_delta
        remote_ptrs = tl.cast(remote_ptrs_int, output_ptrs.dtype)
        gl.store(remote_ptrs, data, mask=mask)

    # PID 0 signals receiver after all tiles written
    if pid == 0:
        flag_ptrs_int = tl.cast(flag_ptr, gl.uint64)
        remote_flag_int = flag_ptrs_int + ptr_delta
        remote_flag = tl.cast(remote_flag_int, tl.pointer_type(tl.int32))
        tl.atomic_xchg(remote_flag, 1, sem="release", scope="sys")


@gluon.jit
def recv_kernel_gluon(
    flag_ptr,
    MAX_SPINS: gl.constexpr = 1_000_000_000,
):
    """
    Gluon recv kernel: spin on flag until sender signals completion.

    Args:
        flag_ptr: Pointer to int32 flag on symmetric heap.
        MAX_SPINS: Maximum spin iterations before asserting timeout.
    """
    spin_count = 0
    while tl.atomic_cas(flag_ptr, 0, 0, sem="acquire", scope="sys") != 1:
        spin_count += 1
        tl.device_assert(spin_count < MAX_SPINS, "recv_kernel_gluon: timeout waiting for send")

    tl.atomic_xchg(flag_ptr, 0, sem="release", scope="sys")


def launch_send(
    input_tensor,
    output_tensor,
    flag_tensor,
    ctx,
    rank_global,
    dst_iris_rank,
    tag,
    config,
):
    """Launch the Gluon send kernel."""
    M, N = input_tensor.shape[:2]
    stride_in_m, stride_in_n = input_tensor.stride(0), input_tensor.stride(1)
    stride_out_m, stride_out_n = output_tensor.stride(0), output_tensor.stride(1)

    # Apply optimal defaults for gluon flat-2D kernel
    block_size_m = config.block_size_m
    block_size_n = config.block_size_n
    if block_size_m == 32 and block_size_n == 64:
        block_size_m = 8
        block_size_n = 256

    total_elems = block_size_m * block_size_n
    threads_per_cta = config.threads_per_warp * config.num_warps
    if total_elems < threads_per_cta:
        raise ValueError(
            f"Gluon send requires block_size_m * block_size_n >= "
            f"threads_per_warp * num_warps ({threads_per_cta}), "
            f"got {block_size_m} * {block_size_n} = {total_elems}."
        )
    if total_elems % threads_per_cta != 0:
        raise ValueError(
            f"Gluon send requires block_size_m * block_size_n to be a "
            f"multiple of threads_per_warp * num_warps ({threads_per_cta}), "
            f"got {block_size_m} * {block_size_n} = {total_elems}. "
            f"Recommended: block_size_m=8, block_size_n=256."
        )

    context_tensor = ctx.get_device_context()
    tracing = getattr(ctx, "tracing", None)
    tracing_enabled = bool(tracing and getattr(tracing, "enabled", False))

    iris_launch(
        send_kernel_gluon,
        (config.comm_sms,),
        IrisDeviceCtx,
        context_tensor,
        input_tensor,
        output_tensor,
        flag_tensor,
        M,
        N,
        stride_in_m,
        stride_in_n,
        stride_out_m,
        stride_out_n,
        rank_global,
        dst_iris_rank,
        tag,
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
        algorithm="send",
        rank=rank_global,
        dtype=input_tensor.dtype,
    )


def launch_recv(
    flag_tensor,
    rank_global,
    config,
):
    """Launch the Gluon recv kernel."""
    iris_launch(
        recv_kernel_gluon,
        (1,),
        flag_tensor,
        num_stages=config.num_stages,
        num_warps=config.num_warps,
        waves_per_eu=config.waves_per_eu,
        algorithm="recv",
        rank=rank_global,
    )
