# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.

"""All-to-all through the gfx1250 tensor data movement (TDM) engine.

Input and output are (M, world_size * N). Column chunk d of this rank's input goes
to rank d, where it lands in column chunk group_rank of the output.

The store-based all-to-all moves every byte through registers, one memory op per
thread per row. Here one ``tensor_load_to_lds`` brings a whole tile of chunk d
into LDS and one ``tensor_store_from_lds`` writes it to rank d, so issue cost
barely depends on CU count. Unlike all-gather, each loaded tile is stored once:
there is no load to amortize, and a program moves one (tile, destination) pair
per load-store.

A remote store does not retire early, so the pairs are pipelined through an LDS
ring of ``Config.num_stages`` tiles: loads run ``ceil(num_stages / 2)`` pairs
ahead of their stores and the rest of the ring holds stores in flight.
``num_stages = 1`` is the unpipelined load, store, wait loop.

The pair order is tile-major: for each of a program's tiles, every destination
in staggered order (group_rank + i), so at any instant each rank writes to a
different peer. The destination loop is unrolled, so every descriptor and every
wait count is a compile-time constant.

The descriptors carry the tensor shape. Loads past the last row read zeros and
stores past the edge are dropped, so ragged tiles need no masks. Each peer's
store descriptor ends at the last column of this rank's chunk, so a ragged last
tile is clipped there rather than spilling into the next rank's chunk. A load
that runs past its chunk reads the next chunk's columns, which the store drops.
"""

import torch
import triton.language as tl
from triton.experimental import gluon
from triton.experimental.gluon import language as gl

try:
    from triton.experimental.gluon.language.amd.gfx1250 import tdm
except ImportError as e:
    raise ImportError(
        "all_to_all_variant='tdm' needs Triton with Gluon gfx1250 TDM support: triton-lang/triton "
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
    """(row, col) of this program's k-th tile, col within a chunk. Past its last tile,
    row = M: every TDM op on such a tile is clipped to nothing but still issues, which
    keeps wait counts exact."""
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
    I: gl.constexpr,
    WAIT_LOAD: gl.constexpr,
    WAIT_BUF: gl.constexpr,
    pid,
    M,
    N,
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
    """Store pair (k, I) to its destination, then load the pair LOOKAHEAD after it.

    Pair (k, I) is this program's k-th tile for the I-th destination in staggered
    order, and is pair number j = k * world_size + I."""
    j = k * world_size + I
    row, col = _tile_offsets(
        k, pid, M, total_tiles, num_pid_m, num_pid_n, BLOCK_SIZE_M, BLOCK_SIZE_N, GROUP_SIZE_M, COMM_SMS
    )
    tdm.async_wait(WAIT_LOAD)
    tdm.async_store(dsts[I], [row, group_rank * N + col], bufs.index(j % NUM_STAGES))

    # The pair LOOKAHEAD later: I and the tile step are both compile-time.
    NEXT_I: gl.constexpr = (I + LOOKAHEAD) % world_size
    NEXT_DK: gl.constexpr = (I + LOOKAHEAD) // world_size
    NEXT_D: gl.constexpr = (group_rank + NEXT_I) % world_size
    next_row, next_col = _tile_offsets(
        k + NEXT_DK, pid, M, total_tiles, num_pid_m, num_pid_n, BLOCK_SIZE_M, BLOCK_SIZE_N, GROUP_SIZE_M, COMM_SMS
    )
    tdm.async_wait(WAIT_BUF)
    tdm.async_load(src, [next_row, NEXT_D * N + next_col], bufs.index((j + LOOKAHEAD) % NUM_STAGES))


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
    bufs = gl.allocate_shared_memory(
        input_ptr.dtype.element_ty, [NUM_STAGES, BLOCK_SIZE_M, BLOCK_SIZE_N], layout=smem_layout
    )

    src = tdm.make_tensor_descriptor(
        input_ptr, [M, world_size * N], [stride_in_m, stride_in_n], [BLOCK_SIZE_M, BLOCK_SIZE_N], smem_layout
    )

    # One store descriptor per destination, built once, in staggered order. Each
    # ends at the last column of this rank's chunk, (group_rank + 1) * N, so a
    # ragged last tile is clipped there.
    local_base = gl.load(ctx.heap_bases + iris_rank)
    cols_through_mine = (group_rank + 1) * N
    dsts = ()
    for i in tl.static_range(world_size):
        dest_idx = (group_rank + i) % world_size
        target_iris_rank = rank_start + dest_idx * rank_stride
        delta = gl.load(ctx.heap_bases + target_iris_rank) - local_base
        base = tl.cast(tl.cast(output_ptr, gl.uint64) + delta, output_ptr.dtype)
        dsts += (
            tdm.make_tensor_descriptor(
                base, [M, cols_through_mine], [stride_out_m, stride_out_n], [BLOCK_SIZE_M, BLOCK_SIZE_N], smem_layout
            ),
        )

    # Software pipeline over this program's pairs j = k * world_size + i: pair j lives
    # in buffer j % NUM_STAGES and is loaded LOOKAHEAD pairs before it is stored. Each
    # pair is one load and one store, so the wait counts are those of the TDM
    # all-gather with world_size = 1:
    #   - pair j's load, before its store: LOOKAHEAD - 1 whole pairs since;
    #   - the store of the pair whose buffer is reloaded: NUM_STAGES - LOOKAHEAD pairs.
    # The first LOOKAHEAD - 1 pairs have fewer ops behind their loads and get exact
    # counts. They are peeled in whole tiles (PEEL_K of them) so the main loop starts
    # at i = 0; pairs past a program's last tile are clipped to nothing.
    #
    # TDM ops retire in order, so waiting for pair j's load also waits for every store
    # issued before it: at most LOOKAHEAD stores are in flight, and at most
    # NUM_STAGES - LOOKAHEAD by the buffer wait. Half the ring each maximizes that.
    # With LOOKAHEAD = NUM_STAGES - 1, as in the all-gather, the buffer wait drains all
    # but the newest store, and a deeper ring adds nothing (measured on gfx1250).
    LOOKAHEAD: gl.constexpr = (NUM_STAGES + 1) // 2
    OPS_PER_PAIR: gl.constexpr = 2
    WAIT_LOAD: gl.constexpr = (LOOKAHEAD - 1) * OPS_PER_PAIR
    WAIT_BUF: gl.constexpr = (NUM_STAGES - LOOKAHEAD) * OPS_PER_PAIR
    PEEL_K: gl.constexpr = (LOOKAHEAD - 1 + world_size - 1) // world_size

    for j in tl.static_range(LOOKAHEAD):
        row, col = _tile_offsets(
            j // world_size,
            pid,
            M,
            total_tiles,
            num_pid_m,
            num_pid_n,
            BLOCK_SIZE_M,
            BLOCK_SIZE_N,
            GROUP_SIZE_M,
            COMM_SMS,
        )
        d = (group_rank + j % world_size) % world_size
        tdm.async_load(src, [row, d * N + col], bufs.index(j % NUM_STAGES))

    for k in tl.static_range(PEEL_K):
        for i in tl.static_range(world_size):
            # After pair j's load: LOOKAHEAD - 1 - j later prologue loads, then j whole
            # pairs, i.e. LOOKAHEAD - 1 + j ops. From j = LOOKAHEAD - 1 on that is WAIT_LOAD.
            _pipeline_step(
                k,
                i,
                min(LOOKAHEAD - 1 + k * world_size + i, WAIT_LOAD),
                WAIT_BUF,
                pid,
                M,
                N,
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
    for k in range(PEEL_K, num_local):
        # static_range unrolls the destinations: every descriptor index is a constant.
        for i in tl.static_range(world_size):
            _pipeline_step(
                k,
                i,
                WAIT_LOAD,
                WAIT_BUF,
                pid,
                M,
                N,
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
        raise ValueError(f"all_to_all_variant='tdm' needs the gfx1250 tensor data movement engine, got {arch}")
    if config.num_stages < 1:
        raise ValueError(f"TDM num_stages must be >= 1, got {config.num_stages}")
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
    """Launch the TDM all-to-all kernel."""
    _check_supported(input_tensor, config)
    M, total_N = input_tensor.shape[:2]
    if total_N % world_size != 0:
        raise ValueError(f"all-to-all input has {total_N} columns, not a multiple of world_size {world_size}")
    if output_tensor.shape != input_tensor.shape:
        raise ValueError(
            f"all-to-all output shape {tuple(output_tensor.shape)} != input shape {tuple(input_tensor.shape)}"
        )
    N = total_N // world_size
    if M == 0 or N == 0:
        # Nothing to move. The kernel's tile arithmetic assumes at least one tile.
        return
    stride_in_m, stride_in_n = input_tensor.stride(0), input_tensor.stride(1)
    stride_out_m, stride_out_n = output_tensor.stride(0), output_tensor.stride(1)

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
        config.num_stages,
        num_warps=config.num_warps,
        waves_per_eu=config.waves_per_eu,
        algorithm="all_to_all",
        rank=rank_global,
        dtype=input_tensor.dtype,
    )
