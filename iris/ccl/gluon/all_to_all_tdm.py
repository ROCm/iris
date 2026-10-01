# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.

"""All-to-all through the gfx1250 tensor data movement (TDM) engine.

Same motivation as gluon/all_gather_tdm.py: the gl.store path is latency-bound
on this part, so a store-based kernel buys in-flight bytes with CUs and stays
CU-starved at the 256-CU maximum. TDM hands the hardware a descriptor and the
engine walks the tile, so issue cost is close to independent of CU count.

Layout. all_to_all is sharded by COLUMN, both sides (M, N*world):

    my input[:,  d*N : (d+1)*N]   ->   rank d's output[:, my_rank*N : ...]

so a tile is identified by a position within one (M, N) slice, and each peer
needs its own load (different source column band) as well as its own store.
That is unlike all_gather, where one load feeds every destination.

MEASURED OUTCOME: THIS LOSES. Keep it opt-in.

world=4, 128 MiB/rank, fp16, bn=256/nw=16, GB/s of per-GPU egress:

    CUs    TDM   triton   TDM/triton
     32  115.1    151.7        0.76x
     64  204.2    259.1        0.79x
    128  344.6    406.5        0.85x
    256  496.8    542.6        0.92x

and that is already the pipelined form. The first version issued
load/wait/store/wait per peer and measured 0.69-0.86x; staging every peer's
tile so all loads are in flight before the first store bought ~0.06x and no
more.

Why, and the rule it implies. TDM's win on all_gather comes from amortising one
descriptor setup and one load across N stores -- N times the egress per unit of
fixed overhead. all_to_all is 1 load : 1 store, so that fixed cost is paid per
tile of egress and never amortised, and the plain store path wins.

**TDM pays off where there is fan-out, not merely where there is bulk
movement.** Expect it to win on all_gather and broadcast-shaped traffic, and to
lose on any 1:1 exchange. The ratio does improve with CU count (0.76 -> 0.92),
so at a larger world size, where each tile fans out further, the crossover may
move; that is untested.
"""

import triton.language as tl
from triton.experimental import gluon
from triton.experimental.gluon import language as gl
from triton.experimental.gluon.language.amd.gfx1250 import tdm

from iris.host.tracing.kernel_artifacts import iris_launch
from iris.mem.gluon.context import Context as IrisDeviceCtx


@gluon.jit
def persistent_all_to_all_tdm(
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

    # N here is the per-peer column count; the tensors are (M, N*world_size).
    num_pid_m = gl.cdiv(M, BLOCK_SIZE_M)
    num_pid_n = gl.cdiv(N, BLOCK_SIZE_N)
    total_tiles = num_pid_m * num_pid_n
    full_n = N * world_size

    smem_layout: gl.constexpr = gl.SwizzledSharedLayout(vec=8, per_phase=1, max_phase=1, order=[1, 0])
    # One staging buffer PER PEER. all_to_all is 1 load : 1 store, so it cannot
    # amortize a load across destinations the way all_gather does -- issuing
    # load/wait/store/wait per peer serialises a full round trip each time and
    # measured 0.69-0.86x of the store path. Holding every peer's tile lets all
    # the loads be in flight before the first store.
    pool = gl.allocate_shared_memory(
        input_ptr.dtype.element_ty, [world_size, BLOCK_SIZE_M, BLOCK_SIZE_N], layout=smem_layout
    )

    local_base = gl.load(ctx.heap_bases + iris_rank)

    src = tdm.make_tensor_descriptor(
        input_ptr, [M, full_n], [stride_in_m, stride_in_n], [BLOCK_SIZE_M, BLOCK_SIZE_N], smem_layout
    )

    for tile_id in range(pid, total_tiles, COMM_SMS):
        num_pid_in_group = GROUP_SIZE_M * num_pid_n
        group_id = tile_id // num_pid_in_group
        first_pid_m = group_id * GROUP_SIZE_M
        group_size_m = min(num_pid_m - first_pid_m, GROUP_SIZE_M)
        pid_m = first_pid_m + ((tile_id % num_pid_in_group) % group_size_m)
        pid_n = (tile_id % num_pid_in_group) // group_size_m

        row = pid_m * BLOCK_SIZE_M
        col = pid_n * BLOCK_SIZE_N

        # Issue every peer's load first. Each reads a different source column
        # band, so unlike all_gather the loads cannot be hoisted out entirely --
        # but they can all be in flight at once.
        for rank_idx in tl.static_range(world_size):
            dest_idx = (group_rank + rank_idx) % world_size
            tdm.async_load(src, [row, dest_idx * N + col], pool.index(rank_idx))

        # Drain oldest-first. The wait count must include the stores already
        # issued, because TDM tracks loads and stores in ONE outstanding queue:
        # waiting on just the remaining loads would also force every earlier
        # store to retire and collapse the pipeline again.
        for rank_idx in tl.static_range(world_size):
            tdm.async_wait(world_size - 1 - rank_idx + rank_idx)
            dest_idx = (group_rank + rank_idx) % world_size
            target_iris_rank = rank_start + dest_idx * rank_stride
            target_base = gl.load(ctx.heap_bases + target_iris_rank)
            delta = target_base - local_base
            base = tl.cast(tl.cast(output_ptr, gl.uint64) + delta, output_ptr.dtype)
            dst = tdm.make_tensor_descriptor(
                base, [M, full_n], [stride_out_m, stride_out_n], [BLOCK_SIZE_M, BLOCK_SIZE_N], smem_layout
            )
            tdm.async_store(dst, [row, group_rank * N + col], pool.index(rank_idx))

        # The stores read from the pool, so they must retire before reuse.
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
    """Launch the TDM all-to-all kernel."""
    M, full_n = input_tensor.shape[:2]
    if full_n % world_size:
        raise ValueError(f"all_to_all needs N divisible by world_size, got {full_n} % {world_size}")
    N = full_n // world_size
    stride_in_m, stride_in_n = input_tensor.stride(0), input_tensor.stride(1)
    stride_out_m, stride_out_n = output_tensor.stride(0), output_tensor.stride(1)

    # The TDM block dimension is encoded in 16 bits.
    for name, v in (("block_size_m", config.block_size_m), ("block_size_n", config.block_size_n)):
        if v > 65535:
            raise ValueError(f"TDM {name} must be <= 65535, got {v}")

    iris_launch(
        persistent_all_to_all_tdm,
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
        algorithm="all_to_all",
        rank=rank_global,
        dtype=input_tensor.dtype,
    )
