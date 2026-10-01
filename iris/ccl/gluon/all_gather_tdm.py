# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.

"""All-gather through the gfx1250 tensor data movement (TDM) engine.

Why this exists. The gl.store path is latency-bound on this part: a remote
store does not retire early, so sustaining bandwidth means keeping bytes in
flight, and a store-based kernel can only buy in-flight bytes with more CUs.
Measured on gfx1250 (world=4, 128 MiB/rank, fp16), every CCL collective is
still near-linear in CU count at the 256-CU maximum -- they are CU-starved
rather than bandwidth-saturated.

TDM does not go through that path. One `tensor_load_to_lds` hands the hardware
a descriptor and the engine walks the whole tile, so issue cost is close to
independent of CU count.

NOT the same as async_copy. `global_load_async_to_lds` is still one memory op
per thread; it only skips the VGPR round trip, so it still scales with CU
count. The two lower to different instructions and different engines:

    async_copy  ->  global_load_async_to_lds / global_store_async_from_lds
    tdm         ->  tensor_load_to_lds       / tensor_store_from_lds

The descriptor carries `shape` and pads out-of-range reads with zero, so the
ragged tail needs no mask and no separate epilogue. That removes the class of
bug where a fully-masked async load issues no group, leaving a commit to
commit an empty one and a wait to return without waiting.

Caveat kept from the mini-iris work this is ported from: LDS and GL0 share one
SRAM, so depth that wins on an idle GPU may be unavailable when a GEMM is
co-resident. Measure BUFFERS rather than maxing it.
"""

import triton.language as tl
from triton.experimental import gluon
from triton.experimental.gluon import language as gl
from triton.experimental.gluon.language.amd.gfx1250 import tdm

from iris.host.tracing.kernel_artifacts import iris_launch
from iris.mem.gluon.context import Context as IrisDeviceCtx


@gluon.jit
def persistent_all_gather_tdm(
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
    BLOCK_SIZE_M: gl.constexpr,
    BLOCK_SIZE_N: gl.constexpr,
    GROUP_SIZE_M: gl.constexpr,
    COMM_SMS: gl.constexpr,
    THREADS_PER_WARP: gl.constexpr,
    WARPS_PER_CTA: gl.constexpr,
):
    ctx = IrisDeviceCtx.initialize(context_tensor, tracing=False)
    pid = gl.program_id(0)

    num_pid_m = gl.cdiv(M, BLOCK_SIZE_M)
    num_pid_n = gl.cdiv(N, BLOCK_SIZE_N)
    total_tiles = num_pid_m * num_pid_n

    # max_phase must be 1 for a descriptor's shared layout. No swizzle is
    # wanted anyway: every consumer of this LDS is the TDM engine itself, so
    # there is no bank-conflicting access pattern to break up.
    smem_layout: gl.constexpr = gl.SwizzledSharedLayout(vec=8, per_phase=1, max_phase=1, order=[1, 0])
    buf = gl.allocate_shared_memory(input_ptr.dtype.element_ty, [BLOCK_SIZE_M, BLOCK_SIZE_N], layout=smem_layout)

    # Hoist the local heap base: computing each peer's delta from it avoids a
    # second gl.load(heap_bases) per destination per tile.
    local_base = gl.load(ctx.heap_bases + iris_rank)

    src = tdm.make_tensor_descriptor(
        input_ptr, [M, N], [stride_in_m, stride_in_n], [BLOCK_SIZE_M, BLOCK_SIZE_N], smem_layout
    )

    for tile_id in range(pid, total_tiles, COMM_SMS):
        num_pid_in_group = GROUP_SIZE_M * num_pid_n
        group_id = tile_id // num_pid_in_group
        first_pid_m = group_id * GROUP_SIZE_M
        group_size_m = min(num_pid_m - first_pid_m, GROUP_SIZE_M)
        pid_m = first_pid_m + ((tile_id % num_pid_in_group) % group_size_m)
        pid_n = (tile_id % num_pid_in_group) // group_size_m

        tdm.async_load(src, [pid_m * BLOCK_SIZE_M, pid_n * BLOCK_SIZE_N], buf)
        tdm.async_wait(0)

        # This rank's contribution lands at output row group_rank*M + row.
        out_row = group_rank * M + pid_m * BLOCK_SIZE_M
        out_col = pid_n * BLOCK_SIZE_N

        # Traffic shaping: stagger the destination order by rank so that at any
        # instant each rank is writing to a different peer.
        for rank_idx in tl.static_range(world_size):
            dest_idx = (group_rank + rank_idx) % world_size
            target_iris_rank = rank_start + dest_idx * rank_stride
            target_base = gl.load(ctx.heap_bases + target_iris_rank)
            delta = target_base - local_base
            base = tl.cast(tl.cast(output_ptr, gl.uint64) + delta, output_ptr.dtype)
            dst = tdm.make_tensor_descriptor(
                base,
                [world_size * M, N],
                [stride_out_m, stride_out_n],
                [BLOCK_SIZE_M, BLOCK_SIZE_N],
                smem_layout,
            )
            tdm.async_store(dst, [out_row, out_col], buf)

        # The stores read from buf, so they must retire before it is reused.
        tdm.async_wait(0)


def launch(
    input_tensor,
    output_tensor,
    ctx,
    rank_in_group,
    rank_global,
    world_size,
    rank_start,
    rank_stride,
    config,
):
    """Launch the TDM all-gather kernel."""
    M, N = input_tensor.shape[:2]
    stride_in_m, stride_in_n = input_tensor.stride(0), input_tensor.stride(1)
    stride_out_m, stride_out_n = output_tensor.stride(0), output_tensor.stride(1)

    # The TDM block dimension is encoded in 16 bits.
    for name, v in (("block_size_m", config.block_size_m), ("block_size_n", config.block_size_n)):
        if v > 65535:
            raise ValueError(f"TDM {name} must be <= 65535, got {v}")

    iris_launch(
        persistent_all_gather_tdm,
        (config.comm_sms,),
        IrisDeviceCtx,
        ctx.get_device_context(),
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
        config.block_size_m,
        config.block_size_n,
        config.swizzle_size,
        config.comm_sms,
        config.threads_per_warp,
        config.num_warps,
        num_stages=config.num_stages,
        num_warps=config.num_warps,
        waves_per_eu=config.waves_per_eu,
        algorithm="all_gather",
        rank=rank_global,
        dtype=input_tensor.dtype,
    )
