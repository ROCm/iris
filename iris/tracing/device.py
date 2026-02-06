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
    counter: tl.tensor  # pointer to int32
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

    def __init__(
        self,
        enabled,
        rank,
        max_events,
        counter,
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
    ):
        """Construct DeviceTracing (called from DeviceContext.initialize)."""
        self.enabled = enabled
        self.rank = rank
        self.max_events = max_events
        self.counter = counter
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

    @triton.jit
    def record_event_start(
        self,
        event_id: tl.constexpr,
        target_rank,
        address,
        pid_m,
        pid_n,
    ):
        """
        Record start of a traced operation. Returns a handle for record_event_end.

        Only stores when event_idx.item() < max_events (bounds check).
        cur_rank is taken from the tracing context (ctx.rank).
        """
        if not self.enabled:
            # Return dummy handle; record_event_end will no-op (0 < max_events is false when disabled)
            return tl.full((), 0, dtype=tl.int32)

        event_idx = tl.atomic_add(self.counter, 1)
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
