# SPDX-License-Identifier: MIT
# Copyright (c) 2025-2026 Advanced Micro Devices, Inc. All rights reserved.

"""
Ring KV Transfer primitives for Iris.

Device functions for ring-topology KV data movement with flag-based synchronization.
Used as building blocks for fused ring prefill attention.

Phase 6: ring_kv_send, ring_kv_recv — push/pull KV tiles around the ring.
"""

import triton
import triton.language as tl
import iris
from iris.iris import DeviceContext


@triton.jit
def ring_kv_send(
    k_tile,
    v_tile,
    k_ring_buf_ptr,
    v_ring_buf_ptr,
    signal_flags_ptr,
    buf_slot,
    ctx: DeviceContext,
    BLOCK_K: tl.constexpr,
    HEAD_DIM: tl.constexpr,
):
    """
    Send a K/V tile to the next rank's ring buffer and signal readiness.

    Pushes k_tile and v_tile to next_rank = (ctx.rank + 1) % ctx.world_size
    via iris.store, then sets the corresponding signal flag via iris.atomic_xchg.

    Args:
        k_tile: [BLOCK_K, HEAD_DIM] key tile to send (in registers)
        v_tile: [BLOCK_K, HEAD_DIM] value tile to send (in registers)
        k_ring_buf_ptr: pointer to next rank's K ring buffer (sym heap)
        v_ring_buf_ptr: pointer to next rank's V ring buffer (sym heap)
        signal_flags_ptr: pointer to signal flags array (sym heap)
        buf_slot: which slot in the flags array (0 or 1 for double buffering)
        ctx: DeviceContext with rank/world_size/heap_bases
        BLOCK_K, HEAD_DIM: constexpr tile dimensions
    """
    next_rank = (ctx.rank + 1) % ctx.world_size

    # Compute flat offsets for the tile
    rows = tl.arange(0, BLOCK_K)
    cols = tl.arange(0, HEAD_DIM)
    offset = rows[:, None] * HEAD_DIM + cols[None, :]  # [BLOCK_K, HEAD_DIM]

    # Push K tile to next rank's ring buffer
    iris.store(
        k_ring_buf_ptr + offset,
        k_tile,
        ctx.rank,
        next_rank,
        ctx.heap_bases,
    )

    # Push V tile to next rank's ring buffer
    iris.store(
        v_ring_buf_ptr + offset,
        v_tile,
        ctx.rank,
        next_rank,
        ctx.heap_bases,
    )

    tl.debug_barrier()

    # Signal next rank: this slot is ready
    iris.atomic_xchg(
        signal_flags_ptr + buf_slot,
        1,
        ctx.rank,
        next_rank,
        ctx.heap_bases,
        sem="release",
        scope="sys",
    )


@triton.jit
def ring_kv_recv(
    k_ring_buf_ptr,
    v_ring_buf_ptr,
    signal_flags_ptr,
    buf_slot,
    ctx: DeviceContext,
    BLOCK_K: tl.constexpr,
    HEAD_DIM: tl.constexpr,
):
    """
    Receive a K/V tile from the previous rank's ring buffer (spin-wait).

    Spins on the local signal flag until the previous rank has pushed data,
    then loads the K and V tiles from the local ring buffer and resets the flag.

    Args:
        k_ring_buf_ptr: pointer to local K ring buffer (sym heap)
        v_ring_buf_ptr: pointer to local V ring buffer (sym heap)
        signal_flags_ptr: pointer to local signal flags array (sym heap, local)
        buf_slot: which slot to wait on (0 or 1 for double buffering)
        ctx: DeviceContext (rank/world_size/heap_bases)
        BLOCK_K, HEAD_DIM: constexpr tile dimensions

    Returns:
        (k_tile, v_tile): received key and value tiles [BLOCK_K, HEAD_DIM]
    """
    # Spin-wait: prev rank will atomic_xchg this flag to 1
    while tl.atomic_cas(signal_flags_ptr + buf_slot, 0, 0, sem="acquire", scope="sys") == 0:
        pass

    # Load received data from local ring buffer
    rows = tl.arange(0, BLOCK_K)
    cols = tl.arange(0, HEAD_DIM)
    offset = rows[:, None] * HEAD_DIM + cols[None, :]  # [BLOCK_K, HEAD_DIM]

    k_tile = tl.load(k_ring_buf_ptr + offset)
    v_tile = tl.load(v_ring_buf_ptr + offset)

    # Reset flag so sender can reuse the slot
    tl.atomic_xchg(signal_flags_ptr + buf_slot, 0, sem="release", scope="sys")

    return k_tile, v_tile
