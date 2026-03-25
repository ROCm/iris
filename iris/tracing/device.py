# SPDX-License-Identifier: MIT
# Copyright (c) 2025-2026 Advanced Micro Devices, Inc. All rights reserved.

"""
Device-side tracing aggregate for Iris.

DeviceTracing is used inside Triton kernels to record events into trace buffers.
Bounds check uses Python `if event_idx.item() < max_events:` so we only store
when the event index is in range (avoids buffer overrun when event count exceeds capacity).
"""

import triton
import triton.language as tl
from triton.language.core import _aggregate as aggregate

from .. import device_utils


class _DeviceTracingCls:
    """
    Device-side tracing: records events into SoA buffers from inside Triton kernels.

    Created by DeviceContext.initialize() when tracing=True. Use record_event_start
    / record_event_end to bracket operations; events are exported via Tracing.export().

    Bounds check: we only store when event_idx.item() < max_events to avoid overrun.
    """

    enabled: tl.constexpr
    rank: tl.constexpr  # current rank (from ctx)
    max_events: tl.tensor  # scalar (0-dim)
    counter: tl.tensor  # pointer to int32 (event counter)
    op_index_counter: tl.tensor  # pointer to int32 (operation index counter)
    buf_event_id: tl.tensor
    buf_pid: tl.tensor
    buf_pid_m: tl.tensor
    buf_pid_n: tl.tensor
    buf_cur_rank: tl.tensor
    buf_target_rank: tl.tensor
    buf_xcc_id: tl.tensor
    buf_cu_id: tl.tensor
    buf_timestamp: tl.tensor
    buf_address: tl.tensor
    buf_duration_cycles: tl.tensor
    buf_op_index: tl.tensor
    buf_payload_size: tl.tensor

    def __init__(
        self,
        enabled,
        rank,
        max_events,
        counter,
        op_index_counter,
        buf_event_id,
        buf_pid,
        buf_pid_m,
        buf_pid_n,
        buf_cur_rank,
        buf_target_rank,
        buf_xcc_id,
        buf_cu_id,
        buf_timestamp,
        buf_address,
        buf_duration_cycles,
        buf_op_index,
        buf_payload_size,
    ):
        """Construct DeviceTracing (called from DeviceContext.initialize)."""
        self.enabled = enabled
        self.rank = rank
        self.max_events = max_events
        self.counter = counter
        self.op_index_counter = op_index_counter
        self.buf_event_id = buf_event_id
        self.buf_pid = buf_pid
        self.buf_pid_m = buf_pid_m
        self.buf_pid_n = buf_pid_n
        self.buf_cur_rank = buf_cur_rank
        self.buf_target_rank = buf_target_rank
        self.buf_xcc_id = buf_xcc_id
        self.buf_cu_id = buf_cu_id
        self.buf_timestamp = buf_timestamp
        self.buf_address = buf_address
        self.buf_duration_cycles = buf_duration_cycles
        self.buf_op_index = buf_op_index
        self.buf_payload_size = buf_payload_size

    @triton.jit
    def record_event_start(
        self,
        event_id: tl.constexpr,
        target_rank,
        address,
        pid_m,
        pid_n,
        mask=None,
    ):
        """
        Record start of a traced operation. Returns a handle for record_event_end.

        Only stores when event_idx.item() < max_events (bounds check).
        cur_rank is taken from the tracing context (ctx.rank).
        op_index is automatically tracked internally (0, 1, 2, ...).
        payload_size is automatically calculated from mask and datatype:
        - Counts True values in mask to get number of elements
        - Infers datatype size from address pointer type
        - Multiplies elements * bytes_per_element to get total bytes
        If mask is None, payload_size is set to 0 (unknown size).

        Args:
            event_id: Event type ID (constexpr)
            target_rank: Target rank for the operation
            address: Memory address(es) - can be 1D or 2D block of pointers.
                     The element type is inferred from address.type.element_ty
            pid_m: Program ID in M dimension
            pid_n: Program ID in N dimension
            mask: Optional mask tensor (1D or 2D) indicating valid elements.
                  If provided, payload_size is calculated as:
                  (count of True values) * (bytes per element from address dtype).
                  If None, payload_size is set to 0.
        """
        if not self.enabled:
            # Return dummy handle; record_event_end will no-op (0 < max_events is false when disabled)
            return tl.full((), 0, dtype=tl.int32)

        event_idx = tl.atomic_add(self.counter, 1)
        op_index = tl.atomic_add(self.op_index_counter, 1)

        # Calculate payload_size from mask and datatype
        if mask is not None:
            # Count True values in mask (True=1, False=0, so sum gives count of elements)
            mask_i32 = tl.cast(mask, tl.int32)
            num_elements = tl.sum(mask_i32)

            # Get element type from address pointer and calculate size in bytes
            # address can be 1D or 2D block of pointers, all with same element type
            # For blocks, use .dtype instead of .type (like in test_atomic_xchg_triton.py)
            # address.dtype is the pointer type, address.dtype.element_ty is the element dtype
            elem_type = address.dtype.element_ty
            # Get size in bytes using primitive_bitwidth (bits / 8 = bytes)
            bitwidth = elem_type.primitive_bitwidth
            elem_size_bytes = bitwidth // 8
            # Calculate total payload size in bytes
            payload_size = num_elements * elem_size_bytes
        else:
            # No mask provided, set to 0 to indicate unknown size
            payload_size = tl.full((), 0, dtype=tl.int32)

        if event_idx.item() < self.max_events.item():
            tl.store(self.buf_event_id + event_idx, event_id)
            tl.store(self.buf_pid + event_idx, tl.program_id(0))
            tl.store(self.buf_pid_m + event_idx, pid_m)
            tl.store(self.buf_pid_n + event_idx, pid_n)
            tl.store(self.buf_cur_rank + event_idx, self.rank)
            tl.store(self.buf_target_rank + event_idx, target_rank)
            tl.store(self.buf_xcc_id + event_idx, device_utils.get_xcc_id())
            tl.store(self.buf_cu_id + event_idx, device_utils.get_cu_id())
            tl.store(self.buf_timestamp + event_idx, device_utils.read_realtime())
            # Store one address per event: accept block of pointers (2D/1D) and take min as representative
            addr_i64 = tl.cast(address, tl.int64)
            tl.store(self.buf_address + event_idx, tl.min(addr_i64))
            tl.store(self.buf_duration_cycles + event_idx, tl.full((), 0, dtype=tl.int64))
            tl.store(self.buf_op_index + event_idx, op_index)
            tl.store(self.buf_payload_size + event_idx, payload_size)
        return event_idx

    @triton.jit
    def record_event_end(self, handle):
        """
        Record end timestamp for the event started with record_event_start(handle).

        Only stores when handle.item() < max_events (bounds check).
        """
        if not self.enabled:
            return

        end_ts = device_utils.read_realtime()
        if handle.item() < self.max_events.item():
            tl.store(self.buf_duration_cycles + handle, end_ts)


# Mark __init__ as Triton builtin so dependency finder accepts it when hashing kernels.
_DeviceTracingCls.__init__.__triton_builtin__ = True
DeviceTracing = aggregate(_DeviceTracingCls)
