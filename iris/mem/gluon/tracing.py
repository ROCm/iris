# SPDX-License-Identifier: MIT
# Copyright (c) 2025-2026 Advanced Micro Devices, Inc. All rights reserved.

"""
Gluon device-side tracing: records events into SoA buffers from inside Gluon kernels.
"""

from triton.language.core import _aggregate as aggregate
from triton.experimental import gluon
from triton.experimental.gluon import language as gl
import triton.language as tl
from iris.mem import utils as device_utils


class _GluonDeviceTracingCls:
    """
    Gluon-native device-side tracing: records events into SoA buffers from inside Gluon kernels.

    Created by IrisDeviceCtx.initialize() when tracing=True. Use record_event_start
    / record_event_end to bracket operations; events are exported via Tracing.export().
    """

    enabled: tl.constexpr
    rank: gl.tensor
    max_events: gl.tensor
    counter: gl.tensor
    op_index_counter: gl.tensor
    buf_event_id: gl.tensor
    buf_pid: gl.tensor
    buf_pid_m: gl.tensor
    buf_pid_n: gl.tensor
    buf_cur_rank: gl.tensor
    buf_target_rank: gl.tensor
    buf_xcc_id: gl.tensor
    buf_cu_id: gl.tensor
    buf_timestamp: gl.tensor
    buf_address: gl.tensor
    buf_duration_cycles: gl.tensor
    buf_op_index: gl.tensor
    buf_payload_size: gl.tensor

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
        """Construct GluonDeviceTracing (called from IrisDeviceCtx.initialize)."""
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

    @gluon.jit
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

        Only stores when event_idx < max_events (bounds check).
        cur_rank is taken from the tracing context (self.rank).

        Args:
            event_id: Event type ID (constexpr)
            target_rank: Target rank for the operation
            address: Memory address(es) - 1D or 2D block of pointers.
            pid_m: Program ID in M dimension
            pid_n: Program ID in N dimension
            mask: Optional mask tensor indicating valid elements (1D or 2D).
        """
        if not self.enabled:
            return tl.cast(0, tl.int32)

        # Guard against runtime-disabled tracing: when the kernel is compiled
        # with tracing=True but the host context has tracing disabled, the
        # counter pointers are null and max_events is 0. Skip all work.
        if self.max_events <= 0:
            return tl.cast(0, tl.int32)

        event_idx = tl.atomic_add(self.counter, 1)
        op_index = tl.atomic_add(self.op_index_counter, 1)

        # Calculate payload_size from mask and datatype
        if mask is not None:
            mask_i32 = tl.cast(mask, tl.int32)
            num_elements = gl.sum(mask_i32)
            elem_type = address.dtype.element_ty
            bitwidth = elem_type.primitive_bitwidth
            elem_size_bytes = bitwidth // 8
            payload_size = num_elements * tl.cast(elem_size_bytes, tl.int32)
        else:
            payload_size = tl.cast(0, tl.int32)

        if event_idx < self.max_events:
            tl.store(self.buf_event_id + event_idx, tl.cast(event_id, tl.int32))
            tl.store(self.buf_pid + event_idx, tl.cast(gl.program_id(0), tl.int32))
            tl.store(self.buf_pid_m + event_idx, tl.cast(pid_m, tl.int32))
            tl.store(self.buf_pid_n + event_idx, tl.cast(pid_n, tl.int32))
            tl.store(self.buf_cur_rank + event_idx, tl.cast(self.rank, tl.int32))
            tl.store(self.buf_target_rank + event_idx, tl.cast(target_rank, tl.int32))
            tl.store(self.buf_xcc_id + event_idx, device_utils.get_xcc_id())
            tl.store(self.buf_cu_id + event_idx, device_utils.get_cu_id())
            tl.store(self.buf_timestamp + event_idx, device_utils.read_realtime())
            addr_i64 = tl.cast(address, tl.int64)
            tl.store(self.buf_address + event_idx, gl.min(addr_i64))
            tl.store(self.buf_duration_cycles + event_idx, tl.cast(0, tl.int64))
            tl.store(self.buf_op_index + event_idx, op_index)
            tl.store(self.buf_payload_size + event_idx, tl.cast(payload_size, tl.int32))
        return event_idx

    @gluon.jit
    def record_event_end(self, handle):
        """
        Record end timestamp for the event started with record_event_start(handle).

        Only stores when handle < max_events (bounds check).
        """
        if not self.enabled:
            return

        end_ts = device_utils.read_realtime()
        if handle < self.max_events:
            tl.store(self.buf_duration_cycles + handle, end_ts)


_GluonDeviceTracingCls.__init__.__triton_builtin__ = True
Tracing = aggregate(_GluonDeviceTracingCls)
GluonDeviceTracing = Tracing  # backward compat
