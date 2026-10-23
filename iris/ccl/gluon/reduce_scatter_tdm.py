# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.

"""Reduce-scatter through the gfx1250 tensor data movement (TDM) engine.

Same contract as the Triton two-shot reduce-scatter: every rank holds an (M, N)
input, the tiles of the (M, N) output are split among the ranks by
``Config.all_reduce_distribution``, and each rank writes the sum over all ranks
of its own tiles to its own output. Tiles it does not own are left untouched.

The Triton kernel pulls each tile from every rank with one load per thread per
rank. Here one ``tensor_load_to_lds`` per rank pulls a whole tile into LDS, so
issue cost barely depends on CU count. The tile is summed in registers (fp32,
int32 for int8) and one ``tensor_store_from_lds`` writes the result.

Remote loads do not return early, so the tiles are pipelined through an LDS ring
of ``Config.num_stages`` slots. A slot holds world_size input tiles and one
output tile. Reading a tile out of LDS is synchronous, so a slot's input tiles
can be refilled as soon as it is reduced: loads run ``num_stages`` tiles ahead
of the reduction. ``num_stages = 1`` is the unpipelined load, wait, reduce,
store loop. Ranks are read in staggered order (group_rank + i), so at any instant
each rank reads from a different peer.

The descriptors carry the tensor shape. Loads past the edge read zeros and
stores past the edge are dropped, so ragged tiles need no masks.
"""

import torch
import triton.language as tl
from triton.experimental import gluon
from triton.experimental.gluon import language as gl
from triton.language.extra.hip import utils as hip_utils

try:
    from triton.experimental.gluon.language.amd.gfx1250 import tdm
except ImportError as e:
    raise ImportError(
        "reduce_scatter_variant='tdm' needs Triton with Gluon gfx1250 TDM support: triton-lang/triton "
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
    start_tile,
    tile_stride,
    num_owned,
    num_pid_m,
    num_pid_n,
    BLOCK_SIZE_M: gl.constexpr,
    BLOCK_SIZE_N: gl.constexpr,
    GROUP_SIZE_M: gl.constexpr,
    COMM_SMS: gl.constexpr,
):
    """(row, col) of this program's k-th owned tile. Past its last tile, row = M: every
    TDM op on such a tile is clipped to nothing but still issues, which keeps wait
    counts exact."""
    tile_offset = pid + k * COMM_SMS
    t = start_tile + gl.minimum(tile_offset, num_owned - 1) * tile_stride
    num_pid_in_group = GROUP_SIZE_M * num_pid_n
    group_id = t // num_pid_in_group
    first_pid_m = group_id * GROUP_SIZE_M
    group_size_m = gl.minimum(num_pid_m - first_pid_m, GROUP_SIZE_M)
    pid_m = first_pid_m + ((t % num_pid_in_group) % group_size_m)
    pid_n = (t % num_pid_in_group) // group_size_m
    row = gl.where(tile_offset < num_owned, pid_m * BLOCK_SIZE_M, M)
    return row, pid_n * BLOCK_SIZE_N


@gluon.jit
def _issue_loads(
    k,
    pid,
    M,
    start_tile,
    tile_stride,
    num_owned,
    num_pid_m,
    num_pid_n,
    srcs,
    bufs,
    world_size: gl.constexpr,
    BLOCK_SIZE_M: gl.constexpr,
    BLOCK_SIZE_N: gl.constexpr,
    GROUP_SIZE_M: gl.constexpr,
    COMM_SMS: gl.constexpr,
    NUM_STAGES: gl.constexpr,
):
    """Load tile k from every rank into the input tiles of slot k % NUM_STAGES."""
    row, col = _tile_offsets(
        k,
        pid,
        M,
        start_tile,
        tile_stride,
        num_owned,
        num_pid_m,
        num_pid_n,
        BLOCK_SIZE_M,
        BLOCK_SIZE_N,
        GROUP_SIZE_M,
        COMM_SMS,
    )
    slot = (k % NUM_STAGES) * (world_size + 1)
    for i in tl.static_range(world_size):
        tdm.async_load(srcs[i], [row, col], bufs.index(slot + i))


@gluon.jit
def _pipeline_step(
    k,
    WAIT_LOAD: gl.constexpr,
    pid,
    M,
    start_tile,
    tile_stride,
    num_owned,
    num_pid_m,
    num_pid_n,
    srcs,
    dst,
    bufs,
    world_size: gl.constexpr,
    BLOCK_SIZE_M: gl.constexpr,
    BLOCK_SIZE_N: gl.constexpr,
    GROUP_SIZE_M: gl.constexpr,
    COMM_SMS: gl.constexpr,
    NUM_STAGES: gl.constexpr,
    LOOKAHEAD: gl.constexpr,
    ACC_DTYPE: gl.constexpr,
):
    """Reduce tile k and store it, then load tile k + LOOKAHEAD from every rank."""
    WARPS: gl.constexpr = hip_utils.num_warps()
    LANES: gl.constexpr = hip_utils.num_threads() // WARPS
    # 8 contiguous elements per thread along a row; lanes fill a row first.
    TPW_N: gl.constexpr = min(LANES, max(BLOCK_SIZE_N // 8, 1))
    reg_layout: gl.constexpr = gl.BlockedLayout([1, 8], [LANES // TPW_N, TPW_N], [WARPS, 1], [1, 0])

    row, col = _tile_offsets(
        k,
        pid,
        M,
        start_tile,
        tile_stride,
        num_owned,
        num_pid_m,
        num_pid_n,
        BLOCK_SIZE_M,
        BLOCK_SIZE_N,
        GROUP_SIZE_M,
        COMM_SMS,
    )
    slot = (k % NUM_STAGES) * (world_size + 1)
    tdm.async_wait(WAIT_LOAD)
    acc = bufs.index(slot).load(reg_layout).to(ACC_DTYPE)
    for i in tl.static_range(1, world_size):
        acc += bufs.index(slot + i).load(reg_layout).to(ACC_DTYPE)
    out = bufs.index(slot + world_size)
    out.store(acc.to(bufs.dtype))
    tdm.async_store(dst, [row, col], out)

    # Tile k + LOOKAHEAD refills the input tiles of the slot tile k was just reduced
    # out of. Its output tile is written again only at step k + LOOKAHEAD, after the
    # wait for these loads, which were issued after this store: TDM ops retire in
    # order, so that wait covers the store too.
    _issue_loads(
        k + LOOKAHEAD,
        pid,
        M,
        start_tile,
        tile_stride,
        num_owned,
        num_pid_m,
        num_pid_n,
        srcs,
        bufs,
        world_size,
        BLOCK_SIZE_M,
        BLOCK_SIZE_N,
        GROUP_SIZE_M,
        COMM_SMS,
        NUM_STAGES,
    )


@gluon.jit
def persistent_reduce_scatter_tdm(
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
    DISTRIBUTION: gl.constexpr,
):
    ctx = IrisDeviceCtx.initialize(context_tensor, tracing=False)
    pid = gl.program_id(0)

    num_pid_m = gl.cdiv(M, BLOCK_SIZE_M)
    num_pid_n = gl.cdiv(N, BLOCK_SIZE_N)
    total_tiles = num_pid_m * num_pid_n

    # This rank's tiles, as in the Triton two-shot reduce-scatter: start_tile +
    # i * tile_stride for i < num_owned.
    if DISTRIBUTION == 0:
        start_tile = group_rank
        tile_stride = world_size
        num_owned = gl.cdiv(gl.maximum(total_tiles - start_tile, 0), world_size)
    else:
        tiles_per_rank = gl.cdiv(total_tiles, world_size)
        start_tile = group_rank * tiles_per_rank
        tile_stride = 1
        num_owned = gl.minimum(tiles_per_rank, gl.maximum(total_tiles - start_tile, 0))
    num_local = gl.cdiv(gl.maximum(num_owned - pid, 0), COMM_SMS)

    if output_ptr.dtype.element_ty == gl.int8:
        ACC_DTYPE: gl.constexpr = gl.int32
    else:
        ACC_DTYPE: gl.constexpr = gl.float32

    # max_phase must be 1 for a descriptor's shared layout. Rows are read back
    # 8 elements at a time, which an unswizzled row-major tile serves without conflicts.
    smem_layout: gl.constexpr = gl.SwizzledSharedLayout(vec=8, per_phase=1, max_phase=1, order=[1, 0])
    # NUM_STAGES slots of world_size input tiles and one output tile.
    bufs = gl.allocate_shared_memory(
        input_ptr.dtype.element_ty, [NUM_STAGES * (world_size + 1), BLOCK_SIZE_M, BLOCK_SIZE_N], layout=smem_layout
    )

    # One load descriptor per rank, built once, in staggered order.
    local_base = gl.load(ctx.heap_bases + iris_rank)
    srcs = ()
    for i in tl.static_range(world_size):
        src_idx = (group_rank + i) % world_size
        source_iris_rank = rank_start + src_idx * rank_stride
        delta = gl.load(ctx.heap_bases + source_iris_rank) - local_base
        base = tl.cast(tl.cast(input_ptr, gl.uint64) + delta, input_ptr.dtype)
        srcs += (
            tdm.make_tensor_descriptor(
                base, [M, N], [stride_in_m, stride_in_n], [BLOCK_SIZE_M, BLOCK_SIZE_N], smem_layout
            ),
        )
    dst = tdm.make_tensor_descriptor(
        output_ptr, [M, N], [stride_out_m, stride_out_n], [BLOCK_SIZE_M, BLOCK_SIZE_N], smem_layout
    )

    # Software pipeline over this program's tiles k = 0, 1, ...: tile k lives in slot
    # k % NUM_STAGES and is loaded LOOKAHEAD tiles before it is reduced. Step k
    # reduces tile k, stores it, then loads tile k + LOOKAHEAD: one store and
    # world_size loads.
    #
    # A wave's TDM ops retire in order and async_wait(n) waits until at most n are
    # outstanding, so the wait for tile k's loads is the number of ops issued after
    # them: LOOKAHEAD - 1 whole steps. In the first LOOKAHEAD - 1 steps tile k was
    # loaded in the prologue, with only (LOOKAHEAD - 1 - k) * world_size later
    # prologue loads and k steps behind it; those steps are peeled with their exact
    # counts, which equal (LOOKAHEAD - 1) * world_size + k.
    LOOKAHEAD: gl.constexpr = NUM_STAGES
    OPS_PER_STEP: gl.constexpr = world_size + 1
    WAIT_LOAD: gl.constexpr = (LOOKAHEAD - 1) * OPS_PER_STEP

    for k in tl.static_range(LOOKAHEAD):
        _issue_loads(
            k,
            pid,
            M,
            start_tile,
            tile_stride,
            num_owned,
            num_pid_m,
            num_pid_n,
            srcs,
            bufs,
            world_size,
            BLOCK_SIZE_M,
            BLOCK_SIZE_N,
            GROUP_SIZE_M,
            COMM_SMS,
            NUM_STAGES,
        )

    for k in tl.static_range(LOOKAHEAD - 1):
        _pipeline_step(
            k,
            (LOOKAHEAD - 1) * world_size + k,
            pid,
            M,
            start_tile,
            tile_stride,
            num_owned,
            num_pid_m,
            num_pid_n,
            srcs,
            dst,
            bufs,
            world_size,
            BLOCK_SIZE_M,
            BLOCK_SIZE_N,
            GROUP_SIZE_M,
            COMM_SMS,
            NUM_STAGES,
            LOOKAHEAD,
            ACC_DTYPE,
        )
    for k in range(LOOKAHEAD - 1, num_local):
        _pipeline_step(
            k,
            WAIT_LOAD,
            pid,
            M,
            start_tile,
            tile_stride,
            num_owned,
            num_pid_m,
            num_pid_n,
            srcs,
            dst,
            bufs,
            world_size,
            BLOCK_SIZE_M,
            BLOCK_SIZE_N,
            GROUP_SIZE_M,
            COMM_SMS,
            NUM_STAGES,
            LOOKAHEAD,
            ACC_DTYPE,
        )

    # The stores read from LDS and the lookahead loads write to it, so they must
    # retire before the program exits.
    tdm.async_wait(0)


def _check_supported(input_tensor, world_size, config):
    arch = torch.cuda.get_device_properties(input_tensor.device).gcnArchName.split(":")[0]
    if arch != "gfx1250":
        raise ValueError(f"reduce_scatter_variant='tdm' needs the gfx1250 tensor data movement engine, got {arch}")
    if config.num_stages < 1:
        raise ValueError(f"TDM num_stages must be >= 1, got {config.num_stages}")
    for name, value in (("block_size_m", config.block_size_m), ("block_size_n", config.block_size_n)):
        if value > TDM_MAX_BLOCK_DIM:
            raise ValueError(f"TDM {name} must be <= {TDM_MAX_BLOCK_DIM}, got {value}")
    tile_bytes = config.block_size_m * config.block_size_n * input_tensor.element_size()
    ring_bytes = config.num_stages * (world_size + 1) * tile_bytes
    max_lds = torch.cuda.get_device_properties(input_tensor.device).shared_memory_per_block
    if ring_bytes > max_lds:
        raise ValueError(
            f"TDM num_stages x (world_size + 1) x block_size_m x block_size_n ({config.num_stages} x "
            f"{world_size + 1} x {config.block_size_m} x {config.block_size_n}) is {ring_bytes} bytes, "
            f"more than the {max_lds} bytes of LDS per workgroup"
        )


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
):
    """Launch the TDM reduce-scatter kernel."""
    _check_supported(input_tensor, world_size, config)
    M, N = input_tensor.shape[:2]
    if M == 0 or N == 0:
        # Nothing to reduce. The kernel's tile arithmetic assumes at least one tile.
        return
    stride_in_m, stride_in_n = input_tensor.stride(0), input_tensor.stride(1)
    stride_out_m, stride_out_n = output_tensor.stride(0), output_tensor.stride(1)

    iris_launch(
        persistent_reduce_scatter_tdm,
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
        config.all_reduce_distribution,
        num_warps=config.num_warps,
        waves_per_eu=config.waves_per_eu,
        algorithm="reduce_scatter",
        rank=rank_global,
        dtype=input_tensor.dtype,
    )
