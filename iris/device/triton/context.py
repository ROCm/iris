# SPDX-License-Identifier: MIT
# Copyright (c) 2025-2026 Advanced Micro Devices, Inc. All rights reserved.

"""
Triton device-side context for Iris RMA operations.
"""

import triton
import triton.language as tl
from triton.language.core import _aggregate as aggregate
from iris.device.utils import get_xcc_id, get_cu_id, read_realtime  # noqa: F401 — used by DeviceTracing
from iris.device.triton.tracing import DeviceTracing


@triton.jit
def __translate(ptr, from_rank, to_rank, heap_bases, hint: tl.constexpr = None):
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


@aggregate
class DeviceContext:
    """
    Device-side context that encapsulates rank and heap_bases for ergonomic Iris operations.

    This aggregate provides an object-oriented interface for Iris device operations,
    eliminating the need to pass heap_bases to every function call.

    Usage:
        import iris
        from iris import DeviceContext

        # Host-side: Get encoded context tensor
        shmem = iris.iris()
        context_tensor = shmem.get_device_context()

        @triton.jit
        def my_kernel(context_tensor, rank: tl.constexpr, world_size: tl.constexpr, ...):
            # Initialize device context from encoded tensor
            ctx = DeviceContext.initialize(context_tensor, rank, world_size)

            # Use object-oriented API
            data = ctx.load(buffer + offsets, from_rank=1, mask=mask)
            ctx.store(buffer + offsets, data, to_rank=1, mask=mask)
            old_val = ctx.atomic_add(counter, 1, to_rank=1)

    Attributes:
        rank: Current rank (constexpr)
        world_size: Total number of ranks (constexpr)
        heap_bases: Heap base pointers for all ranks (tensor)
        trace_enabled: Whether tracing is enabled (constexpr)
        max_trace_events: Maximum number of trace events (constexpr)
        trace_counter: Pointer to atomic event counter (tensor)
        trace_buf_pid: Pointer to pid buffer (tensor)
        trace_buf_pid_m: Pointer to pid_m buffer (tensor)
        trace_buf_pid_n: Pointer to pid_n buffer (tensor)
        trace_buf_cur_rank: Pointer to cur_rank buffer (tensor)
        trace_buf_target_rank: Pointer to target_rank buffer (tensor)
        trace_buf_xcc_id: Pointer to xcc_id buffer (tensor)
        trace_buf_cu_id: Pointer to cu_id buffer (tensor)
        trace_buf_timestamp: Pointer to timestamp buffer (tensor)
        trace_buf_address: Pointer to address buffer (tensor)
    """

    rank: tl.constexpr
    world_size: tl.constexpr
    heap_bases: tl.tensor
    tracing: DeviceTracing

    @triton.constexpr_function
    def __init__(self, rank, world_size, heap_bases, tracing):
        """
        Internal constructor - use DeviceContext.initialize() instead.

        Args:
            rank: Current rank (constexpr)
            world_size: Total number of ranks (constexpr)
            heap_bases: Heap base pointers for all ranks (tensor)
            tracing: DeviceTracing instance
        """
        self.rank = tl.constexpr(rank)
        self.world_size = tl.constexpr(world_size)
        self.heap_bases = heap_bases
        self.tracing = tracing

    @staticmethod
    @triton.jit
    def initialize(context_tensor, rank, world_size, tracing: tl.constexpr = False):
        """
        Initialize DeviceContext from the encoded context tensor.

        The context tensor has the format:
        - [cur_rank, num_ranks, heap_base_0, ..., heap_base_N, trace_info...]
        - If tracing=True: extracts trace buffer pointers from context_tensor

        Args:
            context_tensor: Pointer to encoded context data (from Iris.get_device_context())
            rank: Current rank (must be constexpr in kernel signature)
            world_size: Total number of ranks (must be constexpr in kernel signature)
            tracing: Enable event tracing (constexpr, default: False)

        Returns:
            DeviceContext: Initialized device context

        Example:
            >>> import iris
            >>> from iris import DeviceContext
            >>>
            >>> ctx = iris.iris()
            >>> ctx.tracing.enable(max_events=1_000_000)
            >>> context_tensor = ctx.get_device_context()
            >>>
            >>> @triton.jit
            >>> def kernel(context_tensor, rank: tl.constexpr, world_size: tl.constexpr, ...):
            >>>     # Without tracing
            >>>     ctx = DeviceContext.initialize(context_tensor, rank, world_size)
            >>>
            >>>     # With tracing
            >>>     ctx = DeviceContext.initialize(context_tensor, rank, world_size, tracing=True)
            >>>     mask = tl.full([64], True, dtype=tl.int1)  # Example mask
            >>>     ctx.tracing.record_event_start(event_id=TraceEvent().put, target_rank=1, address=ptr, pid_m=0, pid_n=0, mask=mask)
        """
        # Extract heap bases (from index 2 onwards)
        heap_bases = context_tensor + 2  # Offset pointer to start at heap bases

        if tracing:
            # Extract tracing info (starts after heap_bases)
            trace_info_idx = 2 + world_size + 1  # Skip: cur_rank, num_ranks, heap_bases, trace_enabled flag
            max_events = tl.load(context_tensor + trace_info_idx + 0)
            trace_counter_ptr = tl.load(context_tensor + trace_info_idx + 1)
            op_index_counter_ptr = tl.load(context_tensor + trace_info_idx + 2)

            # Cast counter pointers to pointer type
            trace_counter = tl.cast(trace_counter_ptr, tl.pointer_type(tl.int32))
            op_index_counter = tl.cast(op_index_counter_ptr, tl.pointer_type(tl.int32))

            # Extract trace buffer pointers (13 buffers)
            base_idx = trace_info_idx + 3  # Updated: +3 because we now have op_index_counter
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

            # Create DeviceTracing instance
            device_tracing = DeviceTracing(
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

            return DeviceContext(rank, world_size, heap_bases, device_tracing)
        else:
            # When tracing disabled, use dummy pointers (never dereferenced; we return early in record_*)
            dummy_ptr_i32 = tl.cast(context_tensor, tl.pointer_type(tl.int32))
            dummy_ptr_i64 = tl.cast(context_tensor, tl.pointer_type(tl.int64))
            max_events_zero = tl.full((), 0, dtype=tl.int32)
            device_tracing = DeviceTracing(
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

            return DeviceContext(rank, world_size, heap_bases, device_tracing)

    @triton.jit
    def _translate(self, ptr, from_rank, to_rank, hint: tl.constexpr = None):
        """Internal pointer translation between rank address spaces."""
        return __translate(ptr, from_rank, to_rank, self.heap_bases, hint)

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

        This method performs a memory read operation by translating the pointer
        from the current rank's address space to the `from_rank`'s address space and loading
        data from the target memory location. If the current rank and `from_rank` are the same,
        this performs a local load operation.

        Args:
            pointer (triton.PointerType, or block of dtype=triton.PointerType): Pointer in the current rank's address space that will be translated to the `from_rank`'s address space.
            from_rank (int): The rank ID from which to read the data.
            mask (Block of triton.int1, optional): If mask[idx] is false, do not load the data at address pointer[idx]. Defaults to None.
            other (Block, optional): Value to return for masked-out elements. If not provided, the result for masked-out elements is undefined. Defaults to None.
            cache_modifier (str, optional): Controls cache behavior of the load.

                Supported values:
                    - None: *(default)* — Same as ".ca". Uses cache at all levels (CU, L2, LLC) with LRU policy.
                    - ".ca": Cache at all levels (CU, L2, LLC) with LRU policy
                    - ".cg": Bypasses the CU (L1) cache, streams through L2, and may hit in LLC but the line is not retained or inserted.
                    - ".cv": Bypasses all GPU caches (CU and L2) and fetches directly from system memory. If data exists in the LLC, it may hit, but is not retained or inserted.
                            Ensures global coherence by invalidating stale GPU cache lines.

            volatile (bool, optional): If True, disables compiler optimizations that
                could reorder or eliminate the load. Defaults to False.
            hint (int or tuple, optional): Vectorization hint for the translated pointer. Defaults to None.

        Returns:
            Block: The loaded value from the target memory location.

        Example:
            >>> data = ctx.load(buffer + offsets, from_rank=1, mask=mask)
        """
        translated_ptr = self._translate(pointer, self.rank, from_rank, hint)
        result = tl.load(translated_ptr, mask=mask, other=other, cache_modifier=cache_modifier, volatile=volatile)
        return result

    @triton.jit
    def store(self, pointer, value, to_rank, mask=None, cache_modifier=None, hint: tl.constexpr = None):
        """
        Writes data to the specified rank's memory location.

        This method performs a memory write operation by translating the pointer
        from the current rank's address space to the `to_rank`'s address space and storing
        the provided data to the target memory location. If the current rank and `to_rank` are the same,
        this performs a local store operation.

        Args:
            pointer (triton.PointerType, or block of dtype=triton.PointerType): Pointer in the current rank's address space that will be translated to the `to_rank`'s address space.
            value (Block): The tensor of elements to be stored.
            to_rank (int): The rank ID to which the data will be written.
            mask (Block of triton.int1, optional): If mask[idx] is false, do not store the data at address pointer[idx]. Defaults to None.
            cache_modifier (str, optional): Controls cache behavior of the store. Supported values are:

                - None: *(default)* — Same as ".wb". Uses write-back caching at all levels (CU, L2, LLC) with LRU policy.
                - ".wb": Write-back. Write-allocate on L1 miss, inserted into caches and written back later.
                - ".cg": Cache Global. Equivalent to ".wb" — stored through L1 → L2 → LLC under LRU.
                - ".cs": Cache Streaming. Bypasses L1, streamed through L2, not retained in LLC.
                - ".wt": Write-Through. Bypasses L1 and L2 (coherent cache bypass), may hit in LLC with LRU.

        Returns:
            None

        Example:
            >>> ctx.store(buffer + offsets, values, to_rank=1, mask=mask)
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

        This method performs a remote load operation by translating `from_ptr` from the current
        rank's address space to the `from_rank`'s address space, loading the data, and storing
        it to `to_ptr` in the current rank's local memory. If the current rank and `from_rank`
        are the same, this performs a local copy operation.

        Args:
            from_ptr (triton.PointerType, or block of dtype=triton.PointerType): Pointer in the current rank's address space that references memory in `from_rank`.
            to_ptr (triton.PointerType, or block of dtype=triton.PointerType): Pointer to local memory in current rank where the data will be written.
            from_rank (int): The rank ID from which to read the data.
            mask (Block of triton.int1, optional): If mask[idx] is false, do not load from from_ptr[idx] and do not store to to_ptr[idx]. Defaults to None.
            other (Block, optional): Value to return for masked-out elements during the load operation. If not provided, the result for masked-out elements is undefined. Defaults to None.
            load_cache_modifier (str, optional): Controls cache behavior of the load. Supported values are:
                - None: *(default)* — Same as ".ca". Uses cache at all levels (CU, L2, LLC) with LRU policy.
                - ".ca": Cache at all levels (CU, L2, LLC) with LRU policy.
                - ".cg": Bypasses the CU (L1) cache, streams through L2, and may hit in LLC but the line is not retained or inserted.
                - ".cv": Bypasses all GPU caches (CU and L2) and fetches directly from system memory. If data exists in the LLC, it may hit, but is not retained or inserted.

            store_cache_modifier (str, optional): Controls cache behavior of the store. Supported values are:
                - None: *(default)* — Same as ".wb". Uses write-back caching at all levels (CU, L2, LLC) with LRU policy.
                - ".wb": Write-back. Write-allocate on L1 miss, inserted into caches and written back later.
                - ".cg": Cache Global. Equivalent to ".wb" — stored through L1 → L2 → LLC under LRU.
                - ".cs": Cache Streaming. Bypasses L1, streamed through L2, not retained in LLC.
                - ".wt": Write-Through. Bypasses L1 and L2 (coherent cache bypass), may hit in LLC with LRU.

        Returns:
            None

        Example:
            >>> ctx.get(remote_ptr + offsets, local_ptr + offsets, from_rank=1, mask=mask)
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

        This method performs a remote store operation by loading data from `from_ptr` in the
        current rank's local memory, translating `to_ptr` from the current rank's address space
        to the `to_rank`'s address space, and storing the data to the target memory location.
        If the current rank and `to_rank` are the same, this performs a local copy operation.

        Args:
            from_ptr (triton.PointerType, or block of dtype=triton.PointerType): Pointer to local memory in current rank from which to read data.
            to_ptr (triton.PointerType, or block of dtype=triton.PointerType): Pointer in the current rank's address space that references memory in `to_rank`.
            to_rank (int): The rank ID to which the data will be written.
            mask (Block of triton.int1, optional): If mask[idx] is false, do not load from from_ptr[idx] and do not store to to_ptr[idx]. Defaults to None.
            other (Block, optional): Value to return for masked-out elements during the load operation. If not provided, the result for masked-out elements is undefined. Defaults to None.
            load_cache_modifier (str, optional): Controls cache behavior of the load. Supported values are:
                - None: *(default)* — Same as ".ca". Uses cache at all levels (CU, L2, LLC) with LRU policy.
                - ".ca": Cache at all levels (CU, L2, LLC) with LRU policy.
                - ".cg": Bypasses the CU (L1) cache, streams through L2, and may hit in LLC but the line is not retained or inserted.
                - ".cv": Bypasses all GPU caches (CU and L2) and fetches directly from system memory. If data exists in the LLC, it may hit, but is not retained or inserted.

            store_cache_modifier (str, optional): Controls cache behavior of the store. Supported values are:
                - None: *(default)* — Same as ".wb". Uses write-back caching at all levels (CU, L2, LLC) with LRU policy.
                - ".wb": Write-back. Write-allocate on L1 miss, inserted into caches and written back later.
                - ".cg": Cache Global. Equivalent to ".wb" — stored through L1 → L2 → LLC under LRU.
                - ".cs": Cache Streaming. Bypasses L1, streamed through L2, not retained in LLC.
                - ".wt": Write-Through. Bypasses L1 and L2 (coherent cache bypass), may hit in LLC with LRU.

        Returns:
            None

        Example:
            >>> ctx.put(local_ptr + offsets, remote_ptr + offsets, to_rank=1, mask=mask)
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

        This method performs a data transfer by translating `src_ptr` from the current rank's
        address space to the `from_rank`'s address space, performing a masked load from the
        translated source, translating `dst_ptr` to the `to_rank`'s address space, and storing
        the loaded data to the target memory location. If `from_rank` and `to_rank` are the same,
        this performs a local copy operation. It is undefined behaviour if the current rank is
        neither `from_rank` nor `to_rank`.

        Args:
            src_ptr (triton.PointerType, or block of dtype=triton.PointerType): Pointer in the current rank's address space that references `from_rank`'s local memory.
            dst_ptr (triton.PointerType, or block of dtype=triton.PointerType): Pointer in the current rank's address space that references `to_rank`'s local memory.
            from_rank (int): The rank ID that owns `src_ptr` (source rank).
            to_rank (int): The rank ID that will receive the data (destination rank).
            mask (Block of triton.int1, optional): If mask[idx] is false, do not load from src_ptr[idx] and do not store to dst_ptr[idx]. Defaults to None.
            other (Block, optional): Value to return for masked-out elements during the load operation. If not provided, the result for masked-out elements is undefined. Defaults to None.
            load_cache_modifier (str, optional): Controls cache behavior of the load. Supported values are:
                - None: *(default)* — Same as ".ca". Uses cache at all levels (CU, L2, LLC) with LRU policy.
                - ".ca": Cache at all levels (CU, L2, LLC) with LRU policy.
                - ".cg": Bypasses the CU (L1) cache, streams through L2, and may hit in LLC but the line is not retained or inserted.
                - ".cv": Bypasses all GPU caches (CU and L2) and fetches directly from system memory. If data exists in the LLC, it may hit, but is not retained or inserted.

            store_cache_modifier (str, optional): Controls cache behavior of the store. Supported values are:
                - None: *(default)* — Same as ".wb". Uses write-back caching at all levels (CU, L2, LLC) with LRU policy.
                - ".wb": Write-back. Write-allocate on L1 miss, inserted into caches and written back later.
                - ".cg": Cache Global. Equivalent to ".wb" — stored through L1 → L2 → LLC under LRU.
                - ".cs": Cache Streaming. Bypasses L1, streamed through L2, not retained in LLC.
                - ".wt": Write-Through. Bypasses L1 and L2 (coherent cache bypass), may hit in LLC with LRU.

        Returns:
            None

        Example:
            >>> ctx.copy(src_ptr + offsets, dst_ptr + offsets, from_rank=1, to_rank=0, mask=mask)
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
        translated_dst = tl.cast(to_base_byte + dst_offset, dst_ptr.dtype)

        if hint is not None:
            translated_src = tl.max_contiguous(tl.multiple_of(translated_src, hint), hint)
            translated_dst = tl.max_contiguous(tl.multiple_of(translated_dst, hint), hint)

        data = tl.load(translated_src, mask=mask, other=other, cache_modifier=load_cache_modifier)
        tl.store(translated_dst, data, mask=mask, cache_modifier=store_cache_modifier)

    @triton.jit
    def atomic_add(self, pointer, val, to_rank, mask=None, sem=None, scope=None, hint: tl.constexpr = None):
        """
        Performs an atomic add at the specified rank's memory location.

        This method performs an atomic addition operation by translating the pointer
        from the current rank's address space to the `to_rank`'s address space and atomically
        adding the provided data to the `to_rank` memory location. If the current rank and
        `to_rank` are the same, this performs a local atomic addition operation.

        Args:
            pointer (triton.PointerType, or block of dtype=triton.PointerType): The memory locations in the current rank's address space that will be translated to the `to_rank`'s address space.
            val (Block of dtype=pointer.dtype.element_ty): The values with which to perform the atomic operation.
            to_rank (int): The rank ID to which the atomic operation will be performed.
            mask (Block of triton.int1, optional): If mask[idx] is false, do not perform the atomic operation at address pointer[idx]. Defaults to None.
            sem (str, optional): Specifies the memory semantics for the operation. Acceptable values are "acquire", "release", "acq_rel" (stands for "ACQUIRE_RELEASE"), and "relaxed". If not provided, the function defaults to using "acq_rel" semantics.
            scope (str, optional): Defines the scope of threads that observe the synchronizing effect of the atomic operation. Acceptable values are "gpu" (default), "cta" (cooperative thread array, thread block), or "sys" (stands for "SYSTEM"). The default value is "gpu".

        Returns:
            Block: The data stored at pointer before the atomic operation.

        Example:
            >>> old_val = ctx.atomic_add(counter, 1, to_rank=1)
        """
        translated_ptr = self._translate(pointer, self.rank, to_rank, hint)
        return tl.atomic_add(translated_ptr, val, mask=mask, sem=sem, scope=scope)

    @triton.jit
    def atomic_sub(self, pointer, val, to_rank, mask=None, sem=None, scope=None, hint: tl.constexpr = None):
        """
        Atomically subtracts data from the specified rank's memory location.

        This method performs an atomic subtraction operation by translating the pointer
        from the current rank's address space to the `to_rank`'s address space and atomically
        subtracting the provided data from the `to_rank` memory location. If the current rank
        and `to_rank` are the same, this performs a local atomic subtraction operation.

        Args:
            pointer (triton.PointerType, or block of dtype=triton.PointerType): Pointer in the current rank's address space that will be translated to the `to_rank`'s address space.
            val (Block): The tensor of elements to be subtracted atomically.
            to_rank (int): The rank ID to which the atomic operation will be performed.
            mask (Block of triton.int1, optional): If mask[idx] is false, do not perform the atomic operation at address pointer[idx]. Defaults to None.
            sem (str, optional): Specifies the memory semantics for the operation. Acceptable values are "acquire", "release", "acq_rel", and "relaxed". Defaults to "acq_rel".
            scope (str, optional): Defines the scope of threads that observe the synchronizing effect. Acceptable values are "gpu" (default), "cta", or "sys". The default value is "gpu".

        Returns:
            Block: The data stored at pointer before the atomic operation.
        """
        translated_ptr = self._translate(pointer, self.rank, to_rank, hint)
        return tl.atomic_sub(translated_ptr, val, mask=mask, sem=sem, scope=scope)

    @triton.jit
    def atomic_cas(self, pointer, cmp, val, to_rank, sem=None, scope=None, hint: tl.constexpr = None):
        """
        Performs an atomic compare-and-swap at the specified rank's memory location.

        This method performs an atomic compare-and-swap operation by translating the pointer
        from the current rank's address space to the `to_rank`'s address space and atomically
        comparing the value at the memory location with `cmp`. If they match, it replaces the
        value with `val`. If the current rank and `to_rank` are the same, this performs a local
        atomic CAS operation.

        Args:
            pointer (triton.PointerType, or block of dtype=triton.PointerType): The memory location in the current rank's address space that will be translated to the `to_rank`'s address space.
            cmp (Block): The expected value to compare against.
            val (Block): The new value to store if comparison succeeds.
            to_rank (int): The rank ID to which the atomic operation will be performed.
            sem (str, optional): Specifies the memory semantics for the operation. Acceptable values are "acquire", "release", "acq_rel", and "relaxed". Defaults to "acq_rel".
            scope (str, optional): Defines the scope of threads that observe the synchronizing effect. Acceptable values are "gpu" (default), "cta", or "sys". The default value is "gpu".

        Returns:
            Block: The data stored at pointer before the atomic operation.
        """
        translated_ptr = self._translate(pointer, self.rank, to_rank, hint)
        return tl.atomic_cas(translated_ptr, cmp, val, sem=sem, scope=scope)

    @triton.jit
    def atomic_xchg(self, pointer, val, to_rank, mask=None, sem=None, scope=None, hint: tl.constexpr = None):
        """
        Performs an atomic exchange at the specified rank's memory location.

        This method performs an atomic exchange operation by translating the pointer
        from the current rank's address space to the `to_rank`'s address space and atomically
        swapping the value at the memory location with `val`. If the current rank and `to_rank`
        are the same, this performs a local atomic exchange operation.

        Args:
            pointer (triton.PointerType, or block of dtype=triton.PointerType): The memory locations in the current rank's address space that will be translated to the `to_rank`'s address space.
            val (Block): The new values to store.
            to_rank (int): The rank ID to which the atomic operation will be performed.
            mask (Block of triton.int1, optional): If mask[idx] is false, do not perform the atomic operation at address pointer[idx]. Defaults to None.
            sem (str, optional): Specifies the memory semantics for the operation. Acceptable values are "acquire", "release", "acq_rel", and "relaxed". Defaults to "acq_rel".
            scope (str, optional): Defines the scope of threads that observe the synchronizing effect. Acceptable values are "gpu" (default), "cta", or "sys". The default value is "gpu".

        Returns:
            Block: The data stored at pointer before the atomic operation.
        """
        translated_ptr = self._translate(pointer, self.rank, to_rank, hint)
        return tl.atomic_xchg(translated_ptr, val, mask=mask, sem=sem, scope=scope)

    @triton.jit
    def atomic_xor(self, pointer, val, to_rank, mask=None, sem=None, scope=None, hint: tl.constexpr = None):
        """
        Performs an atomic XOR at the specified rank's memory location.

        This method performs an atomic bitwise XOR operation by translating the pointer
        from the current rank's address space to the `to_rank`'s address space and atomically
        XOR'ing the value at the memory location with `val`. If the current rank and `to_rank`
        are the same, this performs a local atomic XOR operation.

        Args:
            pointer (triton.PointerType, or block of dtype=triton.PointerType): The memory locations in the current rank's address space that will be translated to the `to_rank`'s address space.
            val (Block): The values to XOR with.
            to_rank (int): The rank ID to which the atomic operation will be performed.
            mask (Block of triton.int1, optional): If mask[idx] is false, do not perform the atomic operation at address pointer[idx]. Defaults to None.
            sem (str, optional): Specifies the memory semantics for the operation. Acceptable values are "acquire", "release", "acq_rel", and "relaxed". Defaults to "acq_rel".
            scope (str, optional): Defines the scope of threads that observe the synchronizing effect. Acceptable values are "gpu" (default), "cta", or "sys". The default value is "gpu".

        Returns:
            Block: The data stored at pointer before the atomic operation.
        """
        translated_ptr = self._translate(pointer, self.rank, to_rank, hint)
        return tl.atomic_xor(translated_ptr, val, mask=mask, sem=sem, scope=scope)

    @triton.jit
    def atomic_and(self, pointer, val, to_rank, mask=None, sem=None, scope=None, hint: tl.constexpr = None):
        """
        Performs an atomic AND at the specified rank's memory location.

        This method performs an atomic bitwise AND operation by translating the pointer
        from the current rank's address space to the `to_rank`'s address space and atomically
        AND'ing the value at the memory location with `val`. If the current rank and `to_rank`
        are the same, this performs a local atomic AND operation.

        Args:
            pointer (triton.PointerType, or block of dtype=triton.PointerType): The memory locations in the current rank's address space that will be translated to the `to_rank`'s address space.
            val (Block): The values to AND with.
            to_rank (int): The rank ID to which the atomic operation will be performed.
            mask (Block of triton.int1, optional): If mask[idx] is false, do not perform the atomic operation at address pointer[idx]. Defaults to None.
            sem (str, optional): Specifies the memory semantics for the operation. Acceptable values are "acquire", "release", "acq_rel", and "relaxed". Defaults to "acq_rel".
            scope (str, optional): Defines the scope of threads that observe the synchronizing effect. Acceptable values are "gpu" (default), "cta", or "sys". The default value is "gpu".

        Returns:
            Block: The data stored at pointer before the atomic operation.
        """
        translated_ptr = self._translate(pointer, self.rank, to_rank, hint)
        return tl.atomic_and(translated_ptr, val, mask=mask, sem=sem, scope=scope)

    @triton.jit
    def atomic_or(self, pointer, val, to_rank, mask=None, sem=None, scope=None, hint: tl.constexpr = None):
        """
        Performs an atomic OR at the specified rank's memory location.

        This method performs an atomic bitwise OR operation by translating the pointer
        from the current rank's address space to the `to_rank`'s address space and atomically
        OR'ing the value at the memory location with `val`. If the current rank and `to_rank`
        are the same, this performs a local atomic OR operation.

        Args:
            pointer (triton.PointerType, or block of dtype=triton.PointerType): The memory locations in the current rank's address space that will be translated to the `to_rank`'s address space.
            val (Block): The values to OR with.
            to_rank (int): The rank ID to which the atomic operation will be performed.
            mask (Block of triton.int1, optional): If mask[idx] is false, do not perform the atomic operation at address pointer[idx]. Defaults to None.
            sem (str, optional): Specifies the memory semantics for the operation. Acceptable values are "acquire", "release", "acq_rel", and "relaxed". Defaults to "acq_rel".
            scope (str, optional): Defines the scope of threads that observe the synchronizing effect. Acceptable values are "gpu" (default), "cta", or "sys". The default value is "gpu".

        Returns:
            Block: The data stored at pointer before the atomic operation.
        """
        translated_ptr = self._translate(pointer, self.rank, to_rank, hint)
        return tl.atomic_or(translated_ptr, val, mask=mask, sem=sem, scope=scope)

    @triton.jit
    def atomic_min(self, pointer, val, to_rank, mask=None, sem=None, scope=None, hint: tl.constexpr = None):
        """
        Performs an atomic minimum at the specified rank's memory location.

        This method performs an atomic minimum operation by translating the pointer
        from the current rank's address space to the `to_rank`'s address space and atomically
        updating the memory location to the minimum of its current value and `val`. If the
        current rank and `to_rank` are the same, this performs a local atomic min operation.

        Args:
            pointer (triton.PointerType, or block of dtype=triton.PointerType): The memory locations in the current rank's address space that will be translated to the `to_rank`'s address space.
            val (Block): The values to compare with.
            to_rank (int): The rank ID to which the atomic operation will be performed.
            mask (Block of triton.int1, optional): If mask[idx] is false, do not perform the atomic operation at address pointer[idx]. Defaults to None.
            sem (str, optional): Specifies the memory semantics for the operation. Acceptable values are "acquire", "release", "acq_rel", and "relaxed". Defaults to "acq_rel".
            scope (str, optional): Defines the scope of threads that observe the synchronizing effect. Acceptable values are "gpu" (default), "cta", or "sys". The default value is "gpu".

        Returns:
            Block: The data stored at pointer before the atomic operation.
        """
        translated_ptr = self._translate(pointer, self.rank, to_rank, hint)
        return tl.atomic_min(translated_ptr, val, mask=mask, sem=sem, scope=scope)

    @triton.jit
    def atomic_max(self, pointer, val, to_rank, mask=None, sem=None, scope=None, hint: tl.constexpr = None):
        """
        Performs an atomic maximum at the specified rank's memory location.

        This method performs an atomic maximum operation by translating the pointer
        from the current rank's address space to the `to_rank`'s address space and atomically
        updating the memory location to the maximum of its current value and `val`. If the
        current rank and `to_rank` are the same, this performs a local atomic max operation.

        Args:
            pointer (triton.PointerType, or block of dtype=triton.PointerType): The memory locations in the current rank's address space that will be translated to the `to_rank`'s address space.
            val (Block): The values to compare with.
            to_rank (int): The rank ID to which the atomic operation will be performed.
            mask (Block of triton.int1, optional): If mask[idx] is false, do not perform the atomic operation at address pointer[idx]. Defaults to None.
            sem (str, optional): Specifies the memory semantics for the operation. Acceptable values are "acquire", "release", "acq_rel", and "relaxed". Defaults to "acq_rel".
            scope (str, optional): Defines the scope of threads that observe the synchronizing effect. Acceptable values are "gpu" (default), "cta", or "sys". The default value is "gpu".

        Returns:
            Block: The data stored at pointer before the atomic operation.
        """
        translated_ptr = self._translate(pointer, self.rank, to_rank, hint)
        return tl.atomic_max(translated_ptr, val, mask=mask, sem=sem, scope=scope)
