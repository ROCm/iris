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
from xio import sdma_ep


@triton.jit
def wait_cnt():
    tl.inline_asm_elementwise("s_waitcnt vmcnt(0)", "=r", [], dtype=tl.int32, is_pure=False, pack=1)


@triton.jit
def wrap_into_ring(index: tl.uint64):
    queue_size_u32 = sdma_ep.SDMA_QUEUE_SIZE
    queue_size = queue_size_u32.to(tl.uint64)
    return index.to(tl.uint64) % queue_size


@triton.jit
def can_write_up_to(rptr, up_to_index: tl.uint64):
    """Check if there's space to write up to the given index in the ring buffer."""
    hw_read_ptr = tl.load(rptr, cache_modifier=".cv", volatile=True)
    return (up_to_index - hw_read_ptr) < sdma_ep.SDMA_QUEUE_SIZE


@triton.jit
def acquire(
    queue_ptr_u32,
    read_ptr,
    write_ptr,
    doorbell_ptr,
    cached_write_ptr: tl.pointer_type(tl.uint64),
    committed_write_ptr,
    command_in_bytes: tl.uint64,
):
    """
    Reserve space in the SDMA queue.
    Returns (base_index, offset) where:
      - base_index: the index where the packet should be written (cur_index initially)
      - offset: padding bytes added for wraparound (0 if no wraparound)

    Based on ReserveQueueSpace from anvil_device.hpp.
    """
    queue_size_u32 = sdma_ep.SDMA_QUEUE_SIZE
    queue_size_in_bytes = queue_size_u32.to(tl.uint64)

    base_u32 = 0
    base = (base_u32).to(tl.uint64)
    offset_u32 = 0
    offset = (offset_u32).to(tl.uint64)

    stop_loop = False
    while not stop_loop:
        cur_index = tl.load(cached_write_ptr, volatile=True)
        offset = (offset_u32).to(tl.uint64)

        # Calculate current position in ring buffer
        cur_ring_pos = wrap_into_ring(cur_index)

        # Check if we need to wrap around
        if (cur_ring_pos + command_in_bytes) > queue_size_in_bytes:
            # Need to pad to end of ring before wrap around
            offset = queue_size_in_bytes - cur_ring_pos

        # Calculate new index including any wraparound padding
        new_index = cur_index + command_in_bytes + offset
        base = cur_index

        # Check if queue has space
        if can_write_up_to(read_ptr, new_index):
            # Try to atomically claim this space
            if tl.atomic_cas(cached_write_ptr, cur_index, new_index, sem="relaxed", scope="gpu") == cur_index:
                stop_loop = True
    return base, offset


# acquire function using atomic_add instead of atomic_cas
@triton.jit
def acquire_fadd(
    queue_ptr_u32,
    read_ptr,
    write_ptr,
    doorbell_ptr,
    cached_write_ptr: tl.pointer_type(tl.uint64),
    committed_write_ptr,
    command_in_bytes: tl.uint64,
):
    """
    Reserve space in the SDMA queue using atomic_add.
    Returns (base_index, 0) where base_index is where the packet should be written.

    Uses atomic_add instead of CAS. Immediately acquires space, and if wraparound
    is detected, places padding NOP packet, submits it, and tries again.
    Always returns a non-wrapping allocation.
    """
    queue_size_u32 = sdma_ep.SDMA_QUEUE_SIZE
    queue_size_in_bytes = queue_size_u32.to(tl.uint64)

    stop_loop = False
    base_u32 = 0
    base = (base_u32).to(tl.uint64)
    offset_u32 = 0
    offset = (offset_u32).to(tl.uint64)

    while not stop_loop:
        # Atomically acquire space for the command
        base = tl.atomic_add(cached_write_ptr, command_in_bytes, sem="relaxed", scope="gpu")
        end_index = base + command_in_bytes
        # Calculate current position in ring buffer
        cur_ring_pos = wrap_into_ring(base)

        # Block until there is space in the queue to write the command
        while not can_write_up_to(read_ptr, end_index):
            pass

        # Check if we need to wrap around
        if (cur_ring_pos + command_in_bytes) > queue_size_in_bytes:
            # Wrap detected - need to pad to end of ring
            padding_bytes = queue_size_in_bytes - cur_ring_pos

            # Place NOP packet at end of ring
            place_nop_packet(queue_ptr_u32, base, padding_bytes)

            # Place remaining NOP padding at the beginning of the ring.
            remaining_padding_bytes = command_in_bytes - padding_bytes
            place_nop_packet(queue_ptr_u32, base + padding_bytes, remaining_padding_bytes)

            # Submit the padding - update committed write pointer
            # This allows other threads to proceed past this padding
            submit(write_ptr, doorbell_ptr, committed_write_ptr, base, end_index)

            # Continue loop to acquire space for the actual command (will be at ring start)
        else:
            # No wrap - this allocation is good
            stop_loop = True

    return base, offset


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
    offset_ring_pos = wrap_into_ring(offset_bytes)
    offset_in_dwords = (offset_ring_pos // 4).to(tl.int32)
    for i in range(num_padding_dwords):
        if i == 0:
            tl.store(queue_ptr_u32 + offset_in_dwords, ((num_padding_dwords - 1) & 0xFFFF) << 16, cache_modifier=".wt")
        else:
            tl.store(queue_ptr_u32 + offset_in_dwords + i, 0, cache_modifier=".wt")


@triton.jit
def place_copy_packet(queue_ptr_u32, offset_bytes: tl.uint64, size_bytes: tl.uint32, src_ptr_val, dst_ptr_val):
    """Place a SDMA_PKT_COPY_LINEAR packet for 1D linear memory copy."""
    slot_ptr_u32 = queue_ptr_u32 + (wrap_into_ring(offset_bytes) // 4)
    # offset 0: op + sub_op
    tl.store(slot_ptr_u32 + 0, 1, cache_modifier=".wt")
    # offset 1: count
    tl.store(slot_ptr_u32 + 1, size_bytes - 1, cache_modifier=".wt")
    # offset 2: parameters
    tl.store(slot_ptr_u32 + 2, 0, cache_modifier=".wt")
    # offset 3: src address 31:0
    tl.store(slot_ptr_u32 + 3, src_ptr_val.to(tl.uint32), cache_modifier=".wt")
    # offset 4: src address 63:32
    tl.store(slot_ptr_u32 + 4, (src_ptr_val >> 32).to(tl.uint32), cache_modifier=".wt")
    # offset 5: dst address 31:0
    tl.store(slot_ptr_u32 + 5, dst_ptr_val.to(tl.uint32), cache_modifier=".wt")
    # offset 6: dst address 63:32
    tl.store(slot_ptr_u32 + 6, (dst_ptr_val >> 32).to(tl.uint32), cache_modifier=".wt")


# atomic op codes and operation
# atomic add 32bit w/rtn: op 10, operation 15
# atomic add 64bit w/rtn: op 10, operation: 47 -> 32 + 15
# atomic add 32bit w/o rtn: op 10, operation: 31 -> 64 + 15
# atomic add 64bit w/o rtn: op 10, operation: 63 -> 96 + 15
# atomic cmp&swap 32bit w/rtn: op 10, operation: 8
# atomic cmp&swap 64bit w/rtn: op 10, operation: -> 32 + 8
# atomic cmp&swap 32bit w/o rtn: op 10, operation -> 64 + 8
# atomic cmp&swap 64bit w/o rtn: op 10, operation 56 -> 06 + 8
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
    Flags are encoded via IS_64_BIT (bit 4) and RETURN (bit 5).
    """
    slot_ptr_u32 = queue_ptr_u32 + (wrap_into_ring(offset_bytes) // 4)
    if IS_64_BIT:
        OP = OP | (0x1 << 4)
    if not RETURN:
        OP = OP | (0x1 << 5)
    tl.store(slot_ptr_u32 + 0, ((OP & 0x7F) << 25) | (0xA & 0xFF), cache_modifier=".wt")
    # offset 1: dst address 31:0
    tl.store(slot_ptr_u32 + 1, dst_ptr_val.to(tl.uint32), cache_modifier=".wt")
    # offset 2: dst address 63:32
    tl.store(slot_ptr_u32 + 2, (dst_ptr_val >> 32).to(tl.uint32), cache_modifier=".wt")
    # offset 3: src data 31:0
    tl.store(slot_ptr_u32 + 3, src_data, cache_modifier=".wt")
    # offset 4: src data 63:32
    if IS_64_BIT:
        tl.store(slot_ptr_u32 + 4, (src_data >> 32).to(tl.uint32), cache_modifier=".wt")
    else:
        tl.store(slot_ptr_u32 + 4, 0, cache_modifier=".wt")
    # offset 5: compare data 31:0
    tl.store(slot_ptr_u32 + 5, comp_data, cache_modifier=".wt")
    # offset 6: compare data 63:32
    if IS_64_BIT:
        tl.store(slot_ptr_u32 + 6, (comp_data >> 32), cache_modifier=".wt")
    else:
        tl.store(slot_ptr_u32 + 6, 0, cache_modifier=".wt")
    # offset 7: loop timer + loop interval
    tl.store(slot_ptr_u32 + 7, 0, cache_modifier=".wt")


@triton.jit
def place_atomic_add_packet(queue_ptr_u32, offset_bytes: tl.uint64, dst_ptr_val, val):
    """Place an atomic add packet (OP=15, with return)."""
    place_atomic_packet(queue_ptr_u32, offset_bytes, dst_ptr_val, val, 0, 15, True)


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
    slot_ptr_u32 = queue_ptr_u32 + (wrap_into_ring(offset_bytes) // 4)
    header = ((1 & 0x1) << 31) | ((5 & 0x7) << 28) | (8 & 0xFF)
    dw5 = ((retry_count & 0xFFF) << 16) | (interval & 0xFFFF)

    tl.store(slot_ptr_u32 + 0, header, cache_modifier=".wt")
    tl.store(slot_ptr_u32 + 1, flag_ptr_val.to(tl.uint32), cache_modifier=".wt")
    tl.store(slot_ptr_u32 + 2, (flag_ptr_val.to(tl.uint64) >> 32).to(tl.uint32), cache_modifier=".wt")
    tl.store(slot_ptr_u32 + 3, expected_value.to(tl.uint32), cache_modifier=".wt")
    tl.store(slot_ptr_u32 + 4, 0xFFFFFFFF, cache_modifier=".wt")
    tl.store(slot_ptr_u32 + 5, dw5, cache_modifier=".wt")


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
    slot_ptr_u32 = queue_ptr_u32 + (wrap_into_ring(offset_bytes) // 4)

    # DW 0: Header (op=1, sub_op=0x24)
    # op[7:0] = 1 (SDMA_OP_COPY), sub_op[15:8] = 0x24 (SDMA_SUBOP_COPY_LINEAR_SUB_WINDOW)
    tl.store(slot_ptr_u32 + 0, ((0x24 & 0xFF) << 8) | (0x1 & 0xFF), cache_modifier=".wt")

    # DW 1-2: Source base address
    tl.store(slot_ptr_u32 + 1, src_ptr_val.to(tl.uint32), cache_modifier=".wt")
    tl.store(slot_ptr_u32 + 2, (src_ptr_val >> 32).to(tl.uint32), cache_modifier=".wt")

    # DW 3: Source X offset (bytes)
    tl.store(slot_ptr_u32 + 3, src_x, cache_modifier=".wt")

    # DW 4: Source Y offset (rows)
    tl.store(slot_ptr_u32 + 4, src_y, cache_modifier=".wt")

    # DW 5: Source Z offset (0 for 2D)
    tl.store(slot_ptr_u32 + 5, 0, cache_modifier=".wt")

    # DW 6: Source pitch (1-based, so subtract 1)
    tl.store(slot_ptr_u32 + 6, src_buffer_pitch - 1, cache_modifier=".wt")

    # DW 7-8: Source slice pitch (1-based, 0 means slice_pitch of 1, for 2D)
    tl.store(slot_ptr_u32 + 7, 0, cache_modifier=".wt")
    tl.store(slot_ptr_u32 + 8, 0, cache_modifier=".wt")

    # DW 9-10: Destination base address
    tl.store(slot_ptr_u32 + 9, dst_ptr_val.to(tl.uint32), cache_modifier=".wt")
    tl.store(slot_ptr_u32 + 10, (dst_ptr_val >> 32).to(tl.uint32), cache_modifier=".wt")

    # DW 11: Destination X offset (bytes)
    tl.store(slot_ptr_u32 + 11, dst_x, cache_modifier=".wt")

    # DW 12: Destination Y offset (rows)
    tl.store(slot_ptr_u32 + 12, dst_y, cache_modifier=".wt")

    # DW 13: Destination Z offset (0 for 2D)
    tl.store(slot_ptr_u32 + 13, 0, cache_modifier=".wt")

    # DW 14: Destination pitch (1-based, so subtract 1)
    tl.store(slot_ptr_u32 + 14, dst_buffer_pitch - 1, cache_modifier=".wt")

    # DW 15-16: Destination slice pitch (1-based, 0 means slice_pitch of 1, for 2D)
    tl.store(slot_ptr_u32 + 15, 0, cache_modifier=".wt")
    tl.store(slot_ptr_u32 + 16, 0, cache_modifier=".wt")

    # DW 17: Rectangle X (width in bytes, 1-based)
    tl.store(slot_ptr_u32 + 17, tile_width - 1, cache_modifier=".wt")

    # DW 18: Rectangle Y (height in rows, 1-based)
    tl.store(slot_ptr_u32 + 18, tile_height - 1, cache_modifier=".wt")

    # DW 19: Rectangle Z (depth, 1-based, 0 for 2D means depth of 1)
    tl.store(slot_ptr_u32 + 19, 0, cache_modifier=".wt")
