# SPDX-License-Identifier: MIT
# Copyright (c) 2025-2026 Advanced Micro Devices, Inc. All rights reserved.

"""
SDMA (System DMA) device-side utilities for Triton kernels.

This module provides low-level Triton device functions for directly managing
SDMA queues from GPU kernels, including packet construction, queue reservation,
and submission operations.
"""

import triton
import triton.language as tl
from iris.mem.utils import wait_cnt

sdma_ep = None


def bind_sdma_ep():
    """Import rocm-xio and bind ``sdma_ep`` for the SDMA device functions."""
    global sdma_ep
    if sdma_ep is None:
        try:
            from xio import sdma_ep as imported
        except ModuleNotFoundError as exc:
            if exc.name != "xio":
                raise
            raise ModuleNotFoundError(
                "rocm-xio is required for the SDMA copy engine. "
                'Install it with `pip install iris[sdma]`, or drop "sdma" from engines '
                "to use shader load/store only.",
                name="xio",
            ) from exc
        sdma_ep = imported
    return sdma_ep


def queue_ctx_size():
    """Columns per destination rank in the copy-engine context tensor (``Iris.get_copy_engine_ctx()``).

    rocm-xio's device context (queue_buf, rptr, wptr, doorbell, cached_wptr, committed_wptr) followed by one
    Iris-owned slot caching the last observed HW read index (see ``can_write_up_to``).
    """
    return bind_sdma_ep().QUEUE_DEVICE_CTX_SIZE + 1


@triton.jit
def queue_ctx(copy_engine_ctx, to_rank):
    """Row of the copy-engine context tensor that describes this rank's queue to ``to_rank``."""
    return copy_engine_ctx + (sdma_ep.QUEUE_DEVICE_CTX_SIZE + 1) * to_rank


@triton.jit
def cached_read_index_slot(queue_ctx_row):
    """Pointer to the cached HW read index of the queue described by ``queue_ctx_row``."""
    return (queue_ctx_row + sdma_ep.QUEUE_DEVICE_CTX_SIZE).to(tl.pointer_type(tl.uint64))


@triton.jit
def wrap_into_ring(index: tl.uint64):
    queue_size_u32 = sdma_ep.SDMA_QUEUE_SIZE
    queue_size = queue_size_u32.to(tl.uint64)
    return index.to(tl.uint64) % queue_size


@triton.jit
def can_write_up_to(rptr, up_to_index: tl.uint64, cached_rptr: tl.pointer_type(tl.uint64)):
    """Check if there's space to write up to the given index in the ring buffer.

    Like SdmaQueueHandle::CanWriteUpto in rocm-xio: checks against the cached read index first and reads the HW
    read pointer, an uncached round trip of ~1.8 us, only when the cached value says the ring is full; the fresh
    value is then written back. The HW read index only grows, so any cached value (stale, or written by another
    workgroup) is a lower bound and the fast path never reports space that does not exist.
    """
    read_index = tl.load(cached_rptr)
    if (up_to_index - read_index) >= sdma_ep.SDMA_QUEUE_SIZE:
        read_index = tl.load(rptr, cache_modifier=".cv", volatile=True)
        tl.store(cached_rptr, read_index)
    return (up_to_index - read_index) < sdma_ep.SDMA_QUEUE_SIZE


@triton.jit
def acquire(
    read_ptr,
    cached_write_ptr: tl.pointer_type(tl.uint64),
    command_in_bytes: tl.uint64,
    cached_rptr: tl.pointer_type(tl.uint64),
):
    """
    Reserve ``command_in_bytes`` in the SDMA queue with a compare-and-swap on ``cached_write_ptr``.

    Returns the ring index (in bytes) where the command starts. The command may straddle the end of the
    ring; the packet writers wrap every dword (see ``_store_packet``), so no NOP padding is needed.
    ``cached_rptr`` is the queue's cached read index (``cached_read_index_slot``).

    Based on ReserveQueueSpace from anvil_device.hpp.
    """
    base = tl.zeros([], dtype=tl.uint64)
    stop_loop = False
    while not stop_loop:
        base = tl.load(cached_write_ptr, volatile=True)
        new_index = base + command_in_bytes
        if can_write_up_to(read_ptr, new_index, cached_rptr):
            if tl.atomic_cas(cached_write_ptr, base, new_index, sem="relaxed", scope="gpu") == base:
                stop_loop = True
    return base


@triton.jit
def acquire_fadd(
    read_ptr,
    cached_write_ptr: tl.pointer_type(tl.uint64),
    command_in_bytes: tl.uint64,
    cached_rptr: tl.pointer_type(tl.uint64),
):
    """
    Reserve ``command_in_bytes`` in the SDMA queue with an atomic add on ``cached_write_ptr``.

    Returns the ring index (in bytes) where the command starts, after waiting until the hardware has consumed
    enough of the ring to make room. The command may straddle the end of the ring; the packet writers wrap
    every dword (see ``_store_packet``), so no NOP padding is needed. ``cached_rptr`` is the queue's cached
    read index (``cached_read_index_slot``).
    """
    base = tl.atomic_add(cached_write_ptr, command_in_bytes, sem="relaxed", scope="gpu")
    while not can_write_up_to(read_ptr, base + command_in_bytes, cached_rptr):
        pass
    return base


@triton.jit
def submit(write_ptr, doorbell_ptr, committed_write_ptr, base, pending_wptr):
    """
    Submit SDMA commands to the hardware by updating write pointer and ringing the doorbell.

    Waits for previous threads to commit, then updates write pointer, rings doorbell,
    and updates committed pointer to allow subsequent threads to proceed.
    """
    while tl.load(committed_write_ptr, cache_modifier=".cv", volatile=True) != base:
        pass

    wait_cnt()
    tl.debug_barrier()

    tl.store(write_ptr, pending_wptr, cache_modifier=".wt")
    wait_cnt()
    tl.debug_barrier()

    # Ring doorbell
    tl.store(doorbell_ptr, pending_wptr, cache_modifier=".wt")
    wait_cnt()
    tl.debug_barrier()

    tl.store(committed_write_ptr, pending_wptr, cache_modifier=".wt")


@triton.jit
def quiet(read_ptr, cached_write_ptr: tl.pointer_type(tl.uint64)):
    """
    Wait for all submitted SDMA operations to complete.

    Polls the hardware read pointer until it catches up to the cached write pointer,
    ensuring all previously submitted SDMA packets have been processed.

    Args:
        read_ptr: Pointer to hardware read pointer
        cached_write_ptr: Pointer to cached write pointer
    """
    target_wptr = tl.load(cached_write_ptr, cache_modifier=".cv", volatile=True)
    while tl.load(read_ptr, cache_modifier=".cv", volatile=True) != target_wptr:
        pass


@triton.jit
def place_nop_packet(queue_ptr_u32, offset_bytes: tl.uint64, padding_bytes):
    """Place a NOP (no operation) packet for ring buffer padding."""
    num_padding_dwords = (padding_bytes // 4).to(tl.int32)
    for i in range(num_padding_dwords):
        if i == 0:
            value = ((num_padding_dwords - 1) & 0xFFFF) << 16
        else:
            value = 0
        tl.store(queue_ptr_u32 + _ring_dword(offset_bytes, i), value, cache_modifier=".wt")


# Packets are assembled in a register tensor (one dword per element) and written with a single vector store.
# A `.wt` store is lowered to a volatile store, and the AMDGPU backend waits for every volatile access to complete
# (s_waitcnt vmcnt(0)) before issuing the next one, so per-dword scalar stores serialize into one round trip to the
# uncached ring each (~6.7 us for a 112-byte copy + atomic). submit() still waits for the packet to land before it
# bumps the write pointer.
@triton.jit
def _packet_dword(pkt, idx, i: tl.constexpr, value):
    """Set dword ``i`` of the packet tensor ``pkt`` to ``value``."""
    return tl.where(idx == i, value, pkt).to(tl.uint32)


@triton.jit
def _ring_dword(offset_bytes: tl.uint64, i):
    """Ring position (in dwords) of dword ``i`` of a command placed at queue index ``offset_bytes``.

    Wraps every dword, so a command may straddle the end of the ring: the copy engine fetches the ring modulo
    its size, like it does for the amdgpu kernel driver, which never pads at the wrap either.
    """
    ring_dwords_u32 = sdma_ep.SDMA_QUEUE_SIZE // 4
    ring_dwords = ring_dwords_u32.to(tl.uint64)
    return (wrap_into_ring(offset_bytes) // 4 + i) % ring_dwords


@triton.jit
def _store_packet(queue_ptr_u32, offset_bytes: tl.uint64, pkt, idx, num_dwords: tl.constexpr):
    """Write the first ``num_dwords`` dwords of ``pkt`` to the ring at ``offset_bytes``, wrapping at its end."""
    tl.store(queue_ptr_u32 + _ring_dword(offset_bytes, idx), pkt, mask=idx < num_dwords, cache_modifier=".wt")


@triton.jit
def place_copy_packet(queue_ptr_u32, offset_bytes: tl.uint64, size_bytes: tl.uint32, src_ptr_val, dst_ptr_val):
    """Place a SDMA_PKT_COPY_LINEAR packet for 1D linear memory copy."""
    idx = tl.arange(0, 8)
    pkt = tl.zeros([8], dtype=tl.uint32)
    # offset 0: op + sub_op
    pkt = _packet_dword(pkt, idx, 0, 1)
    # offset 1: count
    pkt = _packet_dword(pkt, idx, 1, size_bytes - 1)
    # offset 2: parameters (0)
    # offset 3: src address 31:0
    pkt = _packet_dword(pkt, idx, 3, src_ptr_val.to(tl.uint32))
    # offset 4: src address 63:32
    pkt = _packet_dword(pkt, idx, 4, (src_ptr_val >> 32).to(tl.uint32))
    # offset 5: dst address 31:0
    pkt = _packet_dword(pkt, idx, 5, dst_ptr_val.to(tl.uint32))
    # offset 6: dst address 63:32
    pkt = _packet_dword(pkt, idx, 6, (dst_ptr_val >> 32).to(tl.uint32))
    _store_packet(queue_ptr_u32, offset_bytes, pkt, idx, 7)


@triton.jit
def copy_linear(copy_engine_ctx, to_rank, src, dst, nbytes):
    """
    Queue one COPY_LINEAR of ``nbytes`` from address ``src`` to address ``dst`` on this rank's
    device-initiated queue to ``to_rank``. ``nbytes`` is at most 1 GiB (a 30-bit count on SDMA 4.4).
    """
    handle = queue_ctx(copy_engine_ctx, to_rank)
    queue_ptr_u32 = tl.load(handle + 0).to(tl.pointer_type(tl.uint32))
    read_ptr = tl.load(handle + 1).to(tl.pointer_type(tl.uint64))
    write_ptr = tl.load(handle + 2).to(tl.pointer_type(tl.uint64))
    doorbell_ptr = tl.load(handle + 3).to(tl.pointer_type(tl.uint64))
    cached_write_ptr = tl.load(handle + 4).to(tl.pointer_type(tl.uint64))
    committed_write_ptr = tl.load(handle + 5).to(tl.pointer_type(tl.uint64))

    command_in_bytes = sdma_ep.COPY_LINEAR_COMMAND_BYTES
    base = acquire_fadd(read_ptr, cached_write_ptr, command_in_bytes, cached_read_index_slot(handle))
    place_copy_packet(queue_ptr_u32, base, nbytes.to(tl.uint32), src, dst)
    submit(write_ptr, doorbell_ptr, committed_write_ptr, base, base + command_in_bytes)


# atomic op codes and operation
# atomic add 32bit w/rtn: op 10, operation 15
# atomic add 64bit w/rtn: op 10, operation: 47 -> 32 + 15
# atomic add 32bit w/o rtn: op 10, operation: 79 -> 64 + 15
# atomic add 64bit w/o rtn: op 10, operation: 111 -> 96 + 15
# atomic cmp&swap 32bit w/rtn: op 10, operation: 8
# atomic cmp&swap 64bit w/rtn: op 10, operation: -> 32 + 8
# atomic cmp&swap 32bit w/o rtn: op 10, operation -> 64 + 8
# atomic cmp&swap 64bit w/o rtn: op 10, operation 104 -> 96 + 8
@triton.jit
def place_atomic_packet(
    queue_ptr_u32,
    offset_bytes: tl.uint64,
    dst_ptr_val,
    src_data,
    comp_data,
    OP: tl.constexpr,
    RETURN: tl.constexpr = False,
    IS_64_BIT: tl.constexpr = False,
):
    """
    Place a SDMA_PKT_ATOMIC packet for atomic memory operations.

    OP codes:
        15: atomic add (32/64-bit with/without return)
        8: atomic compare-and-swap (32/64-bit with/without return)
    Flags are encoded via IS_64_BIT (bit 5) and RETURN (bit 6).
    """
    if IS_64_BIT:
        OP = OP | (0x1 << 5)
    if not RETURN:
        OP = OP | (0x1 << 6)
    idx = tl.arange(0, 8)
    pkt = tl.zeros([8], dtype=tl.uint32)
    pkt = _packet_dword(pkt, idx, 0, ((OP & 0x7F) << 25) | (0xA & 0xFF))
    # offset 1: dst address 31:0
    pkt = _packet_dword(pkt, idx, 1, dst_ptr_val.to(tl.uint32))
    # offset 2: dst address 63:32
    pkt = _packet_dword(pkt, idx, 2, (dst_ptr_val >> 32).to(tl.uint32))
    # offset 3: src data 31:0
    pkt = _packet_dword(pkt, idx, 3, src_data)
    # offset 4: src data 63:32 (0 for 32-bit)
    if IS_64_BIT:
        pkt = _packet_dword(pkt, idx, 4, (tl.cast(src_data, tl.uint64) >> 32).to(tl.uint32))
    # offset 5: compare data 31:0
    pkt = _packet_dword(pkt, idx, 5, comp_data)
    # offset 6: compare data 63:32 (0 for 32-bit)
    if IS_64_BIT:
        pkt = _packet_dword(pkt, idx, 6, comp_data >> 32)
    # offset 7: loop timer + loop interval (0)
    _store_packet(queue_ptr_u32, offset_bytes, pkt, idx, 8)


@triton.jit
def place_atomic_add_packet(queue_ptr_u32, offset_bytes: tl.uint64, dst_ptr_val, val, IS_64_BIT: tl.constexpr = False):
    """Place an atomic add packet (OP=15, with return)."""
    place_atomic_packet(queue_ptr_u32, offset_bytes, dst_ptr_val, val, 0, 15, True, IS_64_BIT)


@triton.jit
def place_atomic_cas_packet(
    queue_ptr_u32,
    offset_bytes: tl.uint64,
    dst_ptr_val,
    compare_val,
    swap_val,
):
    """Place an atomic compare-and-swap packet (OP=8, with return)."""
    place_atomic_packet(queue_ptr_u32, offset_bytes, dst_ptr_val, swap_val, compare_val, 8, True)


# Size of one POLL_REGMEM packet (6 dwords); rocm-xio's sdma_ep does not export this one.
POLL_REGMEM_COMMAND_BYTES = tl.constexpr(24)


@triton.jit
def place_poll_regmem_packet(
    queue_ptr_u32,
    offset_bytes: tl.uint64,
    flag_ptr_val,
    expected_value,
    interval: tl.constexpr = 10,
    retry_count: tl.constexpr = 0xFFF,
):
    """
    Place a SDMA_PKT_POLL_REGMEM packet for memory polling.

    Polls memory location until (value >= expected_value).
    """
    header = ((1 & 0x1) << 31) | ((5 & 0x7) << 28) | (8 & 0xFF)
    dw5 = ((retry_count & 0xFFF) << 16) | (interval & 0xFFFF)

    idx = tl.arange(0, 8)
    pkt = tl.zeros([8], dtype=tl.uint32)
    pkt = _packet_dword(pkt, idx, 0, header)
    pkt = _packet_dword(pkt, idx, 1, flag_ptr_val.to(tl.uint32))
    pkt = _packet_dword(pkt, idx, 2, (flag_ptr_val.to(tl.uint64) >> 32).to(tl.uint32))
    pkt = _packet_dword(pkt, idx, 3, tl.cast(expected_value, tl.uint32))
    pkt = _packet_dword(pkt, idx, 4, 0xFFFFFFFF)
    pkt = _packet_dword(pkt, idx, 5, dw5)
    _store_packet(queue_ptr_u32, offset_bytes, pkt, idx, 6)


@triton.jit
def place_sub_window_copy_packet(
    queue_ptr_u32,
    offset_bytes: tl.uint64,
    src_ptr_val,
    dst_ptr_val,
    tile_width: tl.uint32,
    tile_height: tl.uint32,
    src_buffer_pitch: tl.uint32,
    dst_buffer_pitch: tl.uint32,
    src_x: tl.uint32,
    src_y: tl.uint32,
    dst_x: tl.uint32,
    dst_y: tl.uint32,
):
    """
    Place a SDMA_PKT_LINEAR_LARGE_SUB_WINDOW_COPY packet for 2D tile transfer.

    Copies a rectangular tile with arbitrary source/destination offsets.
    Note: pitch, slice_pitch and rect fields are 1-based (subtract 1 before writing).
    Args:
        queue_ptr_u32: Pointer to the SDMA queue buffer (as uint32 array)
        offset_bytes: Byte offset in the queue where to place the packet
        src_ptr_val: Source buffer base address
        dst_ptr_val: Destination buffer base address
        tile_width: Width of the tile to copy in bytes
        tile_height: Height of the tile to copy in rows
        src_buffer_pitch: Row stride of the source buffer in bytes
        dst_buffer_pitch: Row stride of the destination buffer in bytes
        src_x: Source X offset in bytes
        src_y: Source Y offset in rows
        dst_x: Destination X offset in bytes
        dst_y: Destination Y offset in rows
    """
    idx = tl.arange(0, 32)
    pkt = tl.zeros([32], dtype=tl.uint32)

    # DW 0: Header (op=1, sub_op=0x24)
    # op[7:0] = 1 (SDMA_OP_COPY), sub_op[15:8] = 0x24 (SDMA_SUBOP_COPY_LINEAR_SUB_WINDOW)
    pkt = _packet_dword(pkt, idx, 0, ((0x24 & 0xFF) << 8) | (0x1 & 0xFF))

    # DW 1-2: Source base address
    pkt = _packet_dword(pkt, idx, 1, src_ptr_val.to(tl.uint32))
    pkt = _packet_dword(pkt, idx, 2, (src_ptr_val >> 32).to(tl.uint32))

    # DW 3: Source X offset (bytes)
    pkt = _packet_dword(pkt, idx, 3, src_x)

    # DW 4: Source Y offset (rows)
    pkt = _packet_dword(pkt, idx, 4, src_y)

    # DW 5: Source Z offset (0 for 2D)

    # DW 6: Source pitch (1-based, so subtract 1)
    pkt = _packet_dword(pkt, idx, 6, src_buffer_pitch - 1)

    # DW 7-8: Source slice pitch (1-based, 0 means slice_pitch of 1, for 2D)

    # DW 9-10: Destination base address
    pkt = _packet_dword(pkt, idx, 9, dst_ptr_val.to(tl.uint32))
    pkt = _packet_dword(pkt, idx, 10, (dst_ptr_val >> 32).to(tl.uint32))

    # DW 11: Destination X offset (bytes)
    pkt = _packet_dword(pkt, idx, 11, dst_x)

    # DW 12: Destination Y offset (rows)
    pkt = _packet_dword(pkt, idx, 12, dst_y)

    # DW 13: Destination Z offset (0 for 2D)

    # DW 14: Destination pitch (1-based, so subtract 1)
    pkt = _packet_dword(pkt, idx, 14, dst_buffer_pitch - 1)

    # DW 15-16: Destination slice pitch (1-based, 0 means slice_pitch of 1, for 2D)

    # DW 17: Rectangle X (width in bytes, 1-based)
    pkt = _packet_dword(pkt, idx, 17, tile_width - 1)

    # DW 18: Rectangle Y (height in rows, 1-based)
    pkt = _packet_dword(pkt, idx, 18, tile_height - 1)

    # DW 19: Rectangle Z (depth, 1-based, 0 for 2D means depth of 1)
    _store_packet(queue_ptr_u32, offset_bytes, pkt, idx, 20)
