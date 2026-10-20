# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.

"""All-gather through the gfx1250 tensor data movement (TDM) engine.

A store-based all-gather moves every byte through registers, one memory op per
thread per destination, and a remote store does not retire early: sustaining
bandwidth means keeping bytes in flight, and a store-based kernel can only buy
in-flight bytes with more CUs. TDM hands the engine a tensor descriptor instead.
One ``tensor_load_to_lds`` brings a whole tile into LDS and one
``tensor_store_from_lds`` per rank writes it out, so issue cost barely depends
on CU count. One load is amortized over ``world_size`` stores.

A remote store does not retire early, so one CU moving one tile at a time spends
most of its time waiting on the fabric. With ``Config.num_stages = S > 1`` each
program keeps an LDS ring of ``S`` tiles: tile loads are issued ``S - 1`` tiles
ahead of their stores, and stores of consecutive tiles overlap. ``S = 1`` is the
unpipelined load, store, wait loop.

This is not async copy: ``global_load_async_to_lds`` is still one memory op per
thread and still scales with CU count.

The descriptors carry the tensor shape. Loads past the edge of the input read
zeros and stores past the edge are dropped, so ragged tiles need no masks. Each
peer's store descriptor ends at the last row of this rank's shard, so a ragged
last tile is clipped there. A descriptor covering the whole output would only clip
at its end, and the tile's zero padding would land in the next rank's rows.
"""

import torch
import triton.language as tl
from triton.experimental import gluon
from triton.experimental.gluon import language as gl

try:
    from triton.experimental.gluon.language.amd.gfx1250 import tdm
except ImportError as e:
    raise ImportError(
        "all_gather_variant='tdm' needs Triton with Gluon gfx1250 TDM support: triton-lang/triton "
        "commit b5fea1e3f4 (#8392, which added TDM stores) or later."
    ) from e

from iris.host.tracing.kernel_artifacts import iris_launch
from iris.mem.gluon.context import Context as IrisDeviceCtx

# A TDM descriptor encodes each block dimension in 16 bits.
TDM_MAX_BLOCK_DIM = 65535


@gluon.jit
def _tile_offsets(
    k,
    pid,
    M,
    total_tiles,
    num_pid_m,
    num_pid_n,
    BLOCK_SIZE_M: gl.constexpr,
    BLOCK_SIZE_N: gl.constexpr,
    GROUP_SIZE_M: gl.constexpr,
    COMM_SMS: gl.constexpr,
):
    """(row, col) of this program's k-th tile. Past its last tile, row = M: every TDM op
    on such a tile is clipped to nothing but still issues, which keeps wait counts exact."""
    tile_id = pid + k * COMM_SMS
    t = gl.minimum(tile_id, total_tiles - 1)
    num_pid_in_group = GROUP_SIZE_M * num_pid_n
    group_id = t // num_pid_in_group
    first_pid_m = group_id * GROUP_SIZE_M
    group_size_m = gl.minimum(num_pid_m - first_pid_m, GROUP_SIZE_M)
    pid_m = first_pid_m + ((t % num_pid_in_group) % group_size_m)
    pid_n = (t % num_pid_in_group) // group_size_m
    row = gl.where(tile_id < total_tiles, pid_m * BLOCK_SIZE_M, M)
    return row, pid_n * BLOCK_SIZE_N


@gluon.jit
def _pipeline_step(
    k,
    WAIT_LOAD: gl.constexpr,
    WAIT_BUF: gl.constexpr,
    pid,
    M,
    total_tiles,
    num_pid_m,
    num_pid_n,
    group_rank: gl.constexpr,
    src,
    dsts,
    bufs,
    world_size: gl.constexpr,
    BLOCK_SIZE_M: gl.constexpr,
    BLOCK_SIZE_N: gl.constexpr,
    GROUP_SIZE_M: gl.constexpr,
    COMM_SMS: gl.constexpr,
    NUM_STAGES: gl.constexpr,
    LOOKAHEAD: gl.constexpr,
):
    """Store tile k to every rank, then load tile k + LOOKAHEAD."""
    row, col = _tile_offsets(
        k, pid, M, total_tiles, num_pid_m, num_pid_n, BLOCK_SIZE_M, BLOCK_SIZE_N, GROUP_SIZE_M, COMM_SMS
    )
    buf = bufs.index(k % NUM_STAGES)
    tdm.async_wait(WAIT_LOAD)
    # static_range unrolls this at compile time: the world_size stores issue back to back.
    for rank_idx in tl.static_range(world_size):
        tdm.async_store(dsts[rank_idx], [group_rank * M + row, col], buf)

    next_row, next_col = _tile_offsets(
        k + LOOKAHEAD, pid, M, total_tiles, num_pid_m, num_pid_n, BLOCK_SIZE_M, BLOCK_SIZE_N, GROUP_SIZE_M, COMM_SMS
    )
    tdm.async_wait(WAIT_BUF)
    tdm.async_load(src, [next_row, next_col], bufs.index((k + LOOKAHEAD) % NUM_STAGES))


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
    NUM_STAGES: gl.constexpr,
):
    ctx = IrisDeviceCtx.initialize(context_tensor, tracing=False)
    pid = gl.program_id(0)

    num_pid_m = gl.cdiv(M, BLOCK_SIZE_M)
    num_pid_n = gl.cdiv(N, BLOCK_SIZE_N)
    total_tiles = num_pid_m * num_pid_n
    num_local = gl.cdiv(gl.maximum(total_tiles - pid, 0), COMM_SMS)

    # max_phase must be 1 for a descriptor's shared layout. No swizzle is wanted
    # anyway: the only consumer of this LDS is the TDM engine itself.
    smem_layout: gl.constexpr = gl.SwizzledSharedLayout(vec=8, per_phase=1, max_phase=1, order=[1, 0])
    # A layout may have one rank fewer than its allocation: the leading stage dimension
    # is outermost, and bufs.index(i) is a 2D tile with smem_layout, as the descriptors expect.
    bufs = gl.allocate_shared_memory(
        input_ptr.dtype.element_ty, [NUM_STAGES, BLOCK_SIZE_M, BLOCK_SIZE_N], layout=smem_layout
    )

    src = tdm.make_tensor_descriptor(
        input_ptr, [M, N], [stride_in_m, stride_in_n], [BLOCK_SIZE_M, BLOCK_SIZE_N], smem_layout
    )

    # Build each peer's store descriptor once, before the tile loop: one heap-base
    # load and one descriptor per destination per kernel rather than per tile.
    # They end at this rank's last output row, (group_rank + 1) * M, so a ragged
    # last tile is clipped at the end of this rank's shard. Destinations are in
    # staggered order (group_rank + i) so that at any instant each rank is writing
    # to a different peer.
    local_base = gl.load(ctx.heap_bases + iris_rank)
    rows_through_mine = (group_rank + 1) * M
    dsts = ()
    for rank_idx in tl.static_range(world_size):
        dest_idx = (group_rank + rank_idx) % world_size
        target_iris_rank = rank_start + dest_idx * rank_stride
        delta = gl.load(ctx.heap_bases + target_iris_rank) - local_base
        base = tl.cast(tl.cast(output_ptr, gl.uint64) + delta, output_ptr.dtype)
        dsts += (
            tdm.make_tensor_descriptor(
                base, [rows_through_mine, N], [stride_out_m, stride_out_n], [BLOCK_SIZE_M, BLOCK_SIZE_N], smem_layout
            ),
        )

    # Software pipeline over this program's tiles k = 0, 1, ...: tile k lives in
    # buffer k % NUM_STAGES and is loaded LOOKAHEAD tiles before it is stored. Step k
    # stores tile k to every rank, then loads tile k + LOOKAHEAD into the buffer that
    # tile k + LOOKAHEAD - NUM_STAGES used.
    #
    # A wave's TDM ops retire in order and async_wait(n) waits until at most n are
    # outstanding, so each wait is the number of ops issued after the one it waits for:
    #   - tile k's load, before its stores: LOOKAHEAD - 1 whole steps since;
    #   - the stores of the tile whose buffer is reloaded: NUM_STAGES - LOOKAHEAD steps.
    # Waiting for an op that was never issued returns at once. A count is only unsafe
    # when fewer ops followed its target than it names, which happens in the first
    # LOOKAHEAD - 1 steps: those are peeled with their exact counts.
    LOOKAHEAD: gl.constexpr = NUM_STAGES - 1 if NUM_STAGES > 1 else 1
    OPS_PER_STEP: gl.constexpr = world_size + 1
    WAIT_LOAD: gl.constexpr = (LOOKAHEAD - 1) * OPS_PER_STEP
    WAIT_BUF: gl.constexpr = (NUM_STAGES - LOOKAHEAD) * OPS_PER_STEP

    for k in tl.static_range(LOOKAHEAD):
        row, col = _tile_offsets(
            k, pid, M, total_tiles, num_pid_m, num_pid_n, BLOCK_SIZE_M, BLOCK_SIZE_N, GROUP_SIZE_M, COMM_SMS
        )
        tdm.async_load(src, [row, col], bufs.index(k % NUM_STAGES))

    for k in tl.static_range(LOOKAHEAD - 1):
        # After tile k's load: the later prologue loads, then k whole steps.
        _pipeline_step(
            k,
            (LOOKAHEAD - 1 - k) + k * OPS_PER_STEP,
            WAIT_BUF,
            pid,
            M,
            total_tiles,
            num_pid_m,
            num_pid_n,
            group_rank,
            src,
            dsts,
            bufs,
            world_size,
            BLOCK_SIZE_M,
            BLOCK_SIZE_N,
            GROUP_SIZE_M,
            COMM_SMS,
            NUM_STAGES,
            LOOKAHEAD,
        )
    for k in range(LOOKAHEAD - 1, num_local):
        _pipeline_step(
            k,
            WAIT_LOAD,
            WAIT_BUF,
            pid,
            M,
            total_tiles,
            num_pid_m,
            num_pid_n,
            group_rank,
            src,
            dsts,
            bufs,
            world_size,
            BLOCK_SIZE_M,
            BLOCK_SIZE_N,
            GROUP_SIZE_M,
            COMM_SMS,
            NUM_STAGES,
            LOOKAHEAD,
        )

    # The stores read from LDS, so they must retire before the program exits.
    tdm.async_wait(0)


def _check_supported(input_tensor, config):
    arch = torch.cuda.get_device_properties(input_tensor.device).gcnArchName.split(":")[0]
    if arch != "gfx1250":
        raise ValueError(f"all_gather_variant='tdm' needs the gfx1250 tensor data movement engine, got {arch}")
    for name, value in (("block_size_m", config.block_size_m), ("block_size_n", config.block_size_n)):
        if value > TDM_MAX_BLOCK_DIM:
            raise ValueError(f"TDM {name} must be <= {TDM_MAX_BLOCK_DIM}, got {value}")
    ring_bytes = config.num_stages * config.block_size_m * config.block_size_n * input_tensor.element_size()
    max_lds = torch.cuda.get_device_properties(input_tensor.device).shared_memory_per_block
    if ring_bytes > max_lds:
        raise ValueError(
            f"TDM num_stages x block_size_m x block_size_n ({config.num_stages} x {config.block_size_m} x "
            f"{config.block_size_n}) is {ring_bytes} bytes, more than the {max_lds} bytes of LDS per workgroup"
        )


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
    _check_supported(input_tensor, config)
    M, N = input_tensor.shape[:2]
    if M == 0 or N == 0:
        # Nothing to move. The kernel's tile arithmetic assumes at least one tile.
        return
    stride_in_m, stride_in_n = input_tensor.stride(0), input_tensor.stride(1)
    stride_out_m, stride_out_n = output_tensor.stride(0), output_tensor.stride(1)

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
        config.num_stages,
        num_warps=config.num_warps,
        waves_per_eu=config.waves_per_eu,
        algorithm="all_gather",
        rank=rank_global,
        dtype=input_tensor.dtype,
    )
