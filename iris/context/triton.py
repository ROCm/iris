# SPDX-License-Identifier: MIT
# Copyright (c) 2025-2026 Advanced Micro Devices, Inc. All rights reserved.

"""
Triton device-side context and tracing for Iris.

Provides ``TritonContext`` — a device-side aggregate that decodes the context
tensor from ``Iris.get_device_context()`` and exposes RMA operations (load,
store, copy, get, put, atomics) inside ``@triton.jit`` kernels.

Also provides ``TritonDeviceTracing`` — a device-side aggregate for recording
trace events into SoA buffers from inside Triton kernels.

Example::

    import iris
    from iris.context import TritonContext

    ctx = iris.iris(heap_size=2**30)
    context_tensor = ctx.get_device_context()

    @triton.jit
    def kernel(context_tensor, rank: tl.constexpr, world_size: tl.constexpr):
        ctx = TritonContext.initialize(context_tensor, rank, world_size)
        data = ctx.load(buffer + offsets, from_rank=1, mask=mask)
"""

import triton
import triton.language as tl
from triton.language.core import _aggregate as aggregate

from iris import device_utils


# ---------------------------------------------------------------------------
# Pointer translation (shared with free functions in iris.py)
# ---------------------------------------------------------------------------


@triton.jit
def _translate_ptr(ptr, from_rank, to_rank, heap_bases, hint: tl.constexpr = None):
    """Translate a pointer from one rank's address space to another."""
    from_base = tl.load(heap_bases + from_rank)
    to_base = tl.load(heap_bases + to_rank)
    ptr_int = tl.cast(ptr, tl.uint64)
    offset = ptr_int - from_base
    to_base_byte = tl.cast(to_base, tl.pointer_type(tl.int8))
    translated_ptr_byte = to_base_byte + offset
    translated_ptr = tl.cast(translated_ptr_byte, ptr.dtype)
    if hint is not None:
        translated_ptr = tl.max_contiguous(tl.multiple_of(translated_ptr, hint), hint)
    return translated_ptr


# ---------------------------------------------------------------------------
# Device-side tracing
# ---------------------------------------------------------------------------


class _TritonDeviceTracingCls:
    """
    Device-side tracing: records events into SoA buffers from inside Triton kernels.

    Created by TritonContext.initialize() when tracing=True. Use record_event_start
    / record_event_end to bracket operations; events are exported via Tracing.export().
    """

    enabled: tl.constexpr
    rank: tl.constexpr
    max_events: tl.tensor
    counter: tl.tensor
    op_index_counter: tl.tensor
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
        """Construct TritonDeviceTracing (called from TritonContext.initialize)."""
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
            return tl.full((), 0, dtype=tl.int32)

        event_idx = tl.atomic_add(self.counter, 1)
        op_index = tl.atomic_add(self.op_index_counter, 1)

        # Calculate payload_size from mask and datatype
        if mask is not None:
            mask_i32 = tl.cast(mask, tl.int32)
            num_elements = tl.sum(mask_i32)
            elem_type = address.dtype.element_ty
            bitwidth = elem_type.primitive_bitwidth
            elem_size_bytes = bitwidth // 8
            payload_size = num_elements * elem_size_bytes
        else:
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

        Only stores when handle < max_events (bounds check).
        """
        if not self.enabled:
            return

        end_ts = device_utils.read_realtime()
        if handle.item() < self.max_events.item():
            tl.store(self.buf_duration_cycles + handle, end_ts)


_TritonDeviceTracingCls.__init__.__triton_builtin__ = True
TritonDeviceTracing = aggregate(_TritonDeviceTracingCls)


# ---------------------------------------------------------------------------
# Device-side context
# ---------------------------------------------------------------------------


@aggregate
class TritonContext:
    """
    Triton device-side context that decodes the tensor from Iris.get_device_context().

    This aggregate provides an object-oriented interface for Iris device operations,
    eliminating the need to pass heap_bases to every function call.

    Usage::

        import iris
        from iris.context import TritonContext

        shmem = iris.iris()
        context_tensor = shmem.get_device_context()

        @triton.jit
        def my_kernel(context_tensor, rank: tl.constexpr, world_size: tl.constexpr, ...):
            ctx = TritonContext.initialize(context_tensor, rank, world_size)
            data = ctx.load(buffer + offsets, from_rank=1, mask=mask)
            ctx.store(buffer + offsets, data, to_rank=1, mask=mask)

    Attributes:
        rank: Current rank (constexpr)
        world_size: Total number of ranks (constexpr)
        heap_bases: Heap base pointers for all ranks (tensor)
        tracing: TritonDeviceTracing instance (active when tracing=True)
    """

    rank: tl.constexpr
    world_size: tl.constexpr
    heap_bases: tl.tensor
    tracing: TritonDeviceTracing

    @triton.constexpr_function
    def __init__(self, rank, world_size, heap_bases, tracing):
        """
        Internal constructor - use TritonContext.initialize() instead.

        Args:
            rank: Current rank (constexpr)
            world_size: Total number of ranks (constexpr)
            heap_bases: Heap base pointers for all ranks (tensor)
            tracing: TritonDeviceTracing instance
        """
        self.rank = tl.constexpr(rank)
        self.world_size = tl.constexpr(world_size)
        self.heap_bases = heap_bases
        self.tracing = tracing

    @staticmethod
    @triton.jit
    def initialize(context_tensor, rank, world_size, tracing: tl.constexpr = False):
        """
        Initialize TritonContext from the encoded context tensor.

        The context tensor has the format:
        ``[cur_rank, num_ranks, heap_base_0, ..., heap_base_N, trace_info...]``

        Args:
            context_tensor: Pointer to encoded context data (from Iris.get_device_context())
            rank: Current rank (must be constexpr in kernel signature)
            world_size: Total number of ranks (must be constexpr in kernel signature)
            tracing: Enable event tracing (constexpr, default: False)

        Returns:
            TritonContext: Initialized device context
        """
        # Extract heap bases (from index 2 onwards)
        heap_bases = context_tensor + 2

        if tracing:
            # Extract tracing info (starts after heap_bases)
            trace_info_idx = 2 + world_size + 1  # Skip: cur_rank, num_ranks, heap_bases, trace_enabled flag
            max_events = tl.load(context_tensor + trace_info_idx + 0)
            trace_counter_ptr = tl.load(context_tensor + trace_info_idx + 1)
            op_index_counter_ptr = tl.load(context_tensor + trace_info_idx + 2)

            # Cast counter pointers
            trace_counter = tl.cast(trace_counter_ptr, tl.pointer_type(tl.int32))
            op_index_counter = tl.cast(op_index_counter_ptr, tl.pointer_type(tl.int32))

            # Extract trace buffer pointers (13 buffers)
            base_idx = trace_info_idx + 3
            trace_buf_event_id = tl.cast(tl.load(context_tensor + base_idx + 0), tl.pointer_type(tl.int32))
            trace_buf_pid = tl.cast(tl.load(context_tensor + base_idx + 1), tl.pointer_type(tl.int32))
            trace_buf_pid_m = tl.cast(tl.load(context_tensor + base_idx + 2), tl.pointer_type(tl.int32))
            trace_buf_pid_n = tl.cast(tl.load(context_tensor + base_idx + 3), tl.pointer_type(tl.int32))
            trace_buf_cur_rank = tl.cast(tl.load(context_tensor + base_idx + 4), tl.pointer_type(tl.int32))
            trace_buf_target_rank = tl.cast(tl.load(context_tensor + base_idx + 5), tl.pointer_type(tl.int32))
            trace_buf_xcc_id = tl.cast(tl.load(context_tensor + base_idx + 6), tl.pointer_type(tl.int32))
            trace_buf_cu_id = tl.cast(tl.load(context_tensor + base_idx + 7), tl.pointer_type(tl.int32))
            trace_buf_timestamp = tl.cast(tl.load(context_tensor + base_idx + 8), tl.pointer_type(tl.int64))
            trace_buf_address = tl.cast(tl.load(context_tensor + base_idx + 9), tl.pointer_type(tl.int64))
            trace_buf_duration_cycles = tl.cast(tl.load(context_tensor + base_idx + 10), tl.pointer_type(tl.int64))
            trace_buf_op_index = tl.cast(tl.load(context_tensor + base_idx + 11), tl.pointer_type(tl.int32))
            trace_buf_payload_size = tl.cast(tl.load(context_tensor + base_idx + 12), tl.pointer_type(tl.int32))

            device_tracing = TritonDeviceTracing(
                enabled=tracing,
                rank=rank,
                max_events=max_events,
                counter=trace_counter,
                op_index_counter=op_index_counter,
                buf_event_id=trace_buf_event_id,
                buf_pid=trace_buf_pid,
                buf_pid_m=trace_buf_pid_m,
                buf_pid_n=trace_buf_pid_n,
                buf_cur_rank=trace_buf_cur_rank,
                buf_target_rank=trace_buf_target_rank,
                buf_xcc_id=trace_buf_xcc_id,
                buf_cu_id=trace_buf_cu_id,
                buf_timestamp=trace_buf_timestamp,
                buf_address=trace_buf_address,
                buf_duration_cycles=trace_buf_duration_cycles,
                buf_op_index=trace_buf_op_index,
                buf_payload_size=trace_buf_payload_size,
            )

            return TritonContext(rank, world_size, heap_bases, device_tracing)
        else:
            # When tracing disabled, use dummy pointers (never dereferenced)
            dummy_ptr_i32 = tl.cast(context_tensor, tl.pointer_type(tl.int32))
            dummy_ptr_i64 = tl.cast(context_tensor, tl.pointer_type(tl.int64))
            max_events_zero = tl.full((), 0, dtype=tl.int32)
            device_tracing = TritonDeviceTracing(
                enabled=False,
                rank=rank,
                max_events=max_events_zero,
                counter=dummy_ptr_i32,
                op_index_counter=dummy_ptr_i32,
                buf_event_id=dummy_ptr_i32,
                buf_pid=dummy_ptr_i32,
                buf_pid_m=dummy_ptr_i32,
                buf_pid_n=dummy_ptr_i32,
                buf_cur_rank=dummy_ptr_i32,
                buf_target_rank=dummy_ptr_i32,
                buf_xcc_id=dummy_ptr_i32,
                buf_cu_id=dummy_ptr_i32,
                buf_timestamp=dummy_ptr_i64,
                buf_address=dummy_ptr_i64,
                buf_duration_cycles=dummy_ptr_i64,
                buf_op_index=dummy_ptr_i32,
                buf_payload_size=dummy_ptr_i32,
            )

            return TritonContext(rank, world_size, heap_bases, device_tracing)

    @triton.jit
    def _translate(self, ptr, from_rank, to_rank, hint: tl.constexpr = None):
        """Internal pointer translation between rank address spaces."""
        return _translate_ptr(ptr, from_rank, to_rank, self.heap_bases, hint)

    @triton.jit
    def load(
        self,
        pointer,
        from_rank,
        mask=None,
        other=None,
        cache_modifier=None,
        volatile=False,
        hint: tl.constexpr = None,
    ):
        """
        Loads a value from the specified rank's memory location.

        Args:
            pointer: Pointer in the current rank's address space.
            from_rank: The rank ID from which to read the data.
            mask: Optional mask for conditional loading.
            other: Value to return for masked-out elements.
            cache_modifier: Controls cache behavior (".ca", ".cg", ".cv").
            volatile: If True, disables compiler reordering optimizations.
            hint: Vectorization hint for the translated pointer.

        Returns:
            The loaded value from the target memory location.
        """
        translated_ptr = self._translate(pointer, self.rank, from_rank, hint)
        result = tl.load(translated_ptr, mask=mask, other=other, cache_modifier=cache_modifier, volatile=volatile)
        return result

    @triton.jit
    def store(self, pointer, value, to_rank, mask=None, cache_modifier=None, hint: tl.constexpr = None):
        """
        Writes data to the specified rank's memory location.

        Args:
            pointer: Pointer in the current rank's address space.
            value: The tensor of elements to be stored.
            to_rank: The rank ID to which the data will be written.
            mask: Optional mask for conditional storing.
            cache_modifier: Controls cache behavior (".wb", ".cg", ".cs", ".wt").
            hint: Vectorization hint for the translated pointer.
        """
        translated_ptr = self._translate(pointer, self.rank, to_rank, hint)
        tl.store(translated_ptr, value, mask=mask, cache_modifier=cache_modifier)

    @triton.jit
    def get(
        self,
        from_ptr,
        to_ptr,
        from_rank,
        mask=None,
        other=None,
        load_cache_modifier=None,
        store_cache_modifier=None,
        hint: tl.constexpr = None,
    ):
        """
        Copies data from the specified rank's memory into current rank's local memory.

        Args:
            from_ptr: Pointer to remote memory in current rank's address space.
            to_ptr: Pointer to local memory in current rank.
            from_rank: The rank ID from which to read the data.
            mask: Optional mask for conditional operations.
            other: Value for masked-out elements during load.
            load_cache_modifier: Cache behavior for the load.
            store_cache_modifier: Cache behavior for the store.
            hint: Vectorization hint for the translated pointer.
        """
        translated_from_ptr = self._translate(from_ptr, self.rank, from_rank, hint)
        data = tl.load(translated_from_ptr, mask=mask, other=other, cache_modifier=load_cache_modifier)
        tl.store(to_ptr, data, mask=mask, cache_modifier=store_cache_modifier)

    @triton.jit
    def put(
        self,
        from_ptr,
        to_ptr,
        to_rank,
        mask=None,
        other=None,
        load_cache_modifier=None,
        store_cache_modifier=None,
        hint: tl.constexpr = None,
    ):
        """
        Copies data from current rank's local memory to the specified rank's memory.

        Args:
            from_ptr: Pointer to local memory in current rank.
            to_ptr: Pointer to remote memory in current rank's address space.
            to_rank: The rank ID to which the data will be written.
            mask: Optional mask for conditional operations.
            other: Value for masked-out elements during load.
            load_cache_modifier: Cache behavior for the load.
            store_cache_modifier: Cache behavior for the store.
            hint: Vectorization hint for the translated pointer.
        """
        translated_to_ptr = self._translate(to_ptr, self.rank, to_rank, hint)
        data = tl.load(from_ptr, mask=mask, other=other, cache_modifier=load_cache_modifier)
        tl.store(translated_to_ptr, data, mask=mask, cache_modifier=store_cache_modifier)

    @triton.jit
    def copy(
        self,
        src_ptr,
        dst_ptr,
        from_rank,
        to_rank,
        mask=None,
        other=None,
        load_cache_modifier=None,
        store_cache_modifier=None,
        hint: tl.constexpr = None,
    ):
        """
        Copies data from one rank's memory to another rank's memory.

        Args:
            src_ptr: Pointer referencing from_rank's local memory.
            dst_ptr: Pointer referencing to_rank's local memory.
            from_rank: The rank ID that owns src_ptr (source rank).
            to_rank: The rank ID that will receive the data (destination rank).
            mask: Optional mask for conditional operations.
            other: Value for masked-out elements during load.
            load_cache_modifier: Cache behavior for the load.
            store_cache_modifier: Cache behavior for the store.
            hint: Vectorization hint for the translated pointers.
        """
        cur_base = tl.load(self.heap_bases + self.rank)
        from_base = tl.load(self.heap_bases + from_rank)
        to_base = tl.load(self.heap_bases + to_rank)

        src_ptr_int = tl.cast(src_ptr, tl.uint64)
        src_offset = src_ptr_int - cur_base

        dst_ptr_int = tl.cast(dst_ptr, tl.uint64)
        dst_offset = dst_ptr_int - cur_base

        from_base_byte = tl.cast(from_base, tl.pointer_type(tl.int8))
        to_base_byte = tl.cast(to_base, tl.pointer_type(tl.int8))

        translated_src = tl.cast(from_base_byte + src_offset, src_ptr.dtype)
        translated_dst = tl.cast(to_base_byte + dst_offset, src_ptr.dtype)

        if hint is not None:
            translated_src = tl.max_contiguous(tl.multiple_of(translated_src, hint), hint)
            translated_dst = tl.max_contiguous(tl.multiple_of(translated_dst, hint), hint)

        data = tl.load(translated_src, mask=mask, other=other, cache_modifier=load_cache_modifier)
        tl.store(translated_dst, data, mask=mask, cache_modifier=store_cache_modifier)

    @triton.jit
    def atomic_add(self, pointer, val, to_rank, mask=None, sem=None, scope=None, hint: tl.constexpr = None):
        """Performs an atomic add at the specified rank's memory location."""
        translated_ptr = self._translate(pointer, self.rank, to_rank, hint)
        return tl.atomic_add(translated_ptr, val, mask=mask, sem=sem, scope=scope)

    @triton.jit
    def atomic_sub(self, pointer, val, to_rank, mask=None, sem=None, scope=None, hint: tl.constexpr = None):
        """Atomically subtracts data from the specified rank's memory location."""
        translated_ptr = self._translate(pointer, self.rank, to_rank, hint)
        return tl.atomic_sub(translated_ptr, val, mask=mask, sem=sem, scope=scope)

    @triton.jit
    def atomic_cas(self, pointer, cmp, val, to_rank, sem=None, scope=None, hint: tl.constexpr = None):
        """Performs an atomic compare-and-swap at the specified rank's memory location."""
        translated_ptr = self._translate(pointer, self.rank, to_rank, hint)
        return tl.atomic_cas(translated_ptr, cmp, val, sem=sem, scope=scope)

    @triton.jit
    def atomic_xchg(self, pointer, val, to_rank, mask=None, sem=None, scope=None, hint: tl.constexpr = None):
        """Performs an atomic exchange at the specified rank's memory location."""
        translated_ptr = self._translate(pointer, self.rank, to_rank, hint)
        return tl.atomic_xchg(translated_ptr, val, mask=mask, sem=sem, scope=scope)

    @triton.jit
    def atomic_xor(self, pointer, val, to_rank, mask=None, sem=None, scope=None, hint: tl.constexpr = None):
        """Performs an atomic XOR at the specified rank's memory location."""
        translated_ptr = self._translate(pointer, self.rank, to_rank, hint)
        return tl.atomic_xor(translated_ptr, val, mask=mask, sem=sem, scope=scope)

    @triton.jit
    def atomic_and(self, pointer, val, to_rank, mask=None, sem=None, scope=None, hint: tl.constexpr = None):
        """Performs an atomic AND at the specified rank's memory location."""
        translated_ptr = self._translate(pointer, self.rank, to_rank, hint)
        return tl.atomic_and(translated_ptr, val, mask=mask, sem=sem, scope=scope)

    @triton.jit
    def atomic_or(self, pointer, val, to_rank, mask=None, sem=None, scope=None, hint: tl.constexpr = None):
        """Performs an atomic OR at the specified rank's memory location."""
        translated_ptr = self._translate(pointer, self.rank, to_rank, hint)
        return tl.atomic_or(translated_ptr, val, mask=mask, sem=sem, scope=scope)

    @triton.jit
    def atomic_min(self, pointer, val, to_rank, mask=None, sem=None, scope=None, hint: tl.constexpr = None):
        """Performs an atomic minimum at the specified rank's memory location."""
        translated_ptr = self._translate(pointer, self.rank, to_rank, hint)
        return tl.atomic_min(translated_ptr, val, mask=mask, sem=sem, scope=scope)

    @triton.jit
    def atomic_max(self, pointer, val, to_rank, mask=None, sem=None, scope=None, hint: tl.constexpr = None):
        """Performs an atomic maximum at the specified rank's memory location."""
        translated_ptr = self._translate(pointer, self.rank, to_rank, hint)
        return tl.atomic_max(translated_ptr, val, mask=mask, sem=sem, scope=scope)
