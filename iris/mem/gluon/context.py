# SPDX-License-Identifier: MIT
# Copyright (c) 2025-2026 Advanced Micro Devices, Inc. All rights reserved.

"""
Gluon device-side context for Iris RMA operations.
"""

from triton.language.core import _aggregate as aggregate
from triton.experimental import gluon
from triton.experimental.gluon import language as gl
import triton.language as tl
from iris.mem.gluon.tracing import Tracing


@aggregate
class Context:
    """
    Gluon device-side context that decodes the tensor from Iris.get_device_context().

    This aggregate encapsulates the `heap_bases` pointer and provides
    device-side methods for memory operations and atomics using Gluon.

    Attributes:
        cur_rank: Current rank ID
        num_ranks: Total number of ranks
        heap_bases: Pointer to array of heap base addresses for all ranks
        tracing: Tracing instance (active when tracing=True)
    """

    cur_rank: gl.tensor
    num_ranks: gl.tensor
    heap_bases: gl.tensor
    tracing: Tracing

    @gluon.constexpr_function
    def __init__(self, cur_rank, num_ranks, heap_bases, tracing):
        self.cur_rank = cur_rank
        self.num_ranks = num_ranks
        self.heap_bases = heap_bases
        self.tracing = tracing

    @staticmethod
    @gluon.jit
    def initialize(context_tensor, tracing: gl.constexpr = False):
        """
        Initialize `Context` from the encoded tensor.

        The context tensor has the format:
        ``[cur_rank, num_ranks, heap_base_0, heap_base_1, ..., trace_info...]``

        If tracing is enabled on the host (via ``shmem.tracing.enable()``), the
        context tensor also contains tracing buffer pointers after the heap bases.

        Args:
            context_tensor: Pointer to encoded context data
            tracing: Enable event tracing (constexpr, default: False)

        Returns:
            `Context`: Initialized device context
        """
        # Decode the tensor: [cur_rank, num_ranks, heap_base_0, heap_base_1, ...]
        cur_rank = gl.load(context_tensor + 0)
        num_ranks = gl.load(context_tensor + 1)

        # Extract heap bases (from index 2 onwards)
        heap_bases = context_tensor + 2  # Offset pointer to start at heap bases

        if tracing:
            # Extract tracing info: starts after heap_bases, then skip trace_enabled flag.
            # Layout: [cur_rank, num_ranks, heap_base_0..N-1, trace_enabled, max_events,
            #          trace_counter_ptr, op_index_counter_ptr, buf_event_id, ...(13 buffers)]
            #
            # When tracing is disabled at the host, the context tensor is padded with
            # zeros in the same positions (max_events=0, null pointers). On device,
            # the tracing helpers (e.g., record_event_start) first early-return when
            # max_events <= 0 and then guard all writes with a bounds check
            # (event_idx < max_events), so decoding potentially null pointers here is
            # safe as long as those invariants are preserved.
            trace_info_base = 2 + num_ranks + 1  # skip cur_rank, num_ranks, heap_bases, trace_enabled
            max_events = tl.cast(gl.load(context_tensor + trace_info_base + 0), tl.int32)
            trace_counter_ptr = gl.load(context_tensor + trace_info_base + 1)
            op_index_counter_ptr = gl.load(context_tensor + trace_info_base + 2)

            # Cast counter pointers
            trace_counter = tl.cast(trace_counter_ptr, tl.pointer_type(tl.int32))
            op_index_counter = tl.cast(op_index_counter_ptr, tl.pointer_type(tl.int32))

            # Extract trace buffer pointers (13 buffers, same order as Iris._build_device_context)
            buf_base = trace_info_base + 3
            buf_event_id = tl.cast(gl.load(context_tensor + buf_base + 0), tl.pointer_type(tl.int32))
            buf_pid = tl.cast(gl.load(context_tensor + buf_base + 1), tl.pointer_type(tl.int32))
            buf_pid_m = tl.cast(gl.load(context_tensor + buf_base + 2), tl.pointer_type(tl.int32))
            buf_pid_n = tl.cast(gl.load(context_tensor + buf_base + 3), tl.pointer_type(tl.int32))
            buf_cur_rank = tl.cast(gl.load(context_tensor + buf_base + 4), tl.pointer_type(tl.int32))
            buf_target_rank = tl.cast(gl.load(context_tensor + buf_base + 5), tl.pointer_type(tl.int32))
            buf_xcc_id = tl.cast(gl.load(context_tensor + buf_base + 6), tl.pointer_type(tl.int32))
            buf_cu_id = tl.cast(gl.load(context_tensor + buf_base + 7), tl.pointer_type(tl.int32))
            buf_timestamp = tl.cast(gl.load(context_tensor + buf_base + 8), tl.pointer_type(tl.int64))
            buf_address = tl.cast(gl.load(context_tensor + buf_base + 9), tl.pointer_type(tl.int64))
            buf_duration_cycles = tl.cast(gl.load(context_tensor + buf_base + 10), tl.pointer_type(tl.int64))
            buf_op_index = tl.cast(gl.load(context_tensor + buf_base + 11), tl.pointer_type(tl.int32))
            buf_payload_size = tl.cast(gl.load(context_tensor + buf_base + 12), tl.pointer_type(tl.int32))

            device_tracing = Tracing(
                enabled=tracing,
                rank=cur_rank,
                max_events=max_events,
                counter=trace_counter,
                op_index_counter=op_index_counter,
                buf_event_id=buf_event_id,
                buf_pid=buf_pid,
                buf_pid_m=buf_pid_m,
                buf_pid_n=buf_pid_n,
                buf_cur_rank=buf_cur_rank,
                buf_target_rank=buf_target_rank,
                buf_xcc_id=buf_xcc_id,
                buf_cu_id=buf_cu_id,
                buf_timestamp=buf_timestamp,
                buf_address=buf_address,
                buf_duration_cycles=buf_duration_cycles,
                buf_op_index=buf_op_index,
                buf_payload_size=buf_payload_size,
            )
        else:
            # When tracing disabled, use dummy pointers (never dereferenced)
            dummy_ptr_i32 = tl.cast(context_tensor, tl.pointer_type(tl.int32))
            dummy_ptr_i64 = tl.cast(context_tensor, tl.pointer_type(tl.int64))
            max_events_zero = tl.cast(0, tl.int32)
            device_tracing = Tracing(
                enabled=tracing,
                rank=cur_rank,
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

        return Context(cur_rank, num_ranks, heap_bases, device_tracing)

    @gluon.jit
    def _translate(self, ptr, from_rank, to_rank):
        """
        Internal function to translate a pointer from one rank's address space to another.

        Args:
            ptr: Pointer in the `from_rank`'s address space
            from_rank: Source rank ID
            to_rank: Target rank ID

        Returns:
            Translated pointer in the `to_rank`'s address space
        """
        from_base = gl.load(self.heap_bases + from_rank)
        to_base = gl.load(self.heap_bases + to_rank)
        # convert to int to compute difference
        ptr_int = tl.cast(ptr, gl.uint64)
        # Find the offset from from_rank heap
        offset = ptr_int - from_base
        # Byte cast for byte offset addition
        to_base_byte = tl.cast(to_base, gl.pointer_type(gl.int8))
        # Find the offset into the to_rank heap
        translated_ptr_byte = to_base_byte + offset
        # Cast to_base back to pointer type
        translated_ptr = tl.cast(translated_ptr_byte, ptr.dtype)

        # Optimization to vectorize the load/store - similar to iris.py
        # This enables the compiler to generate dwordx4 or wider loads
        # Note: Gluon uses scalar multiples, not 2D tuples like Triton
        # ptr = gl.max_contiguous(gl.multiple_of(ptr, 64), 64)
        # translated_ptr = gl.max_contiguous(gl.multiple_of(translated_ptr, 64), 64)

        return translated_ptr

    @gluon.jit
    def load(self, pointer, from_rank, mask=None, other=None, cache_modifier=None, volatile=False):
        """
        Loads a value from the specified rank's memory location to the current rank.

        Args:
            pointer: Pointer in the `from_rank`'s address space
            from_rank: The rank ID from which to read the data
            mask: Optional mask for conditional loading
            other: Value to return for masked-out elements. If not provided, the result for masked-out elements is undefined.
            cache_modifier (str, optional): Controls cache behavior of the load.

                Supported values:
                    - None: *(default)* — Same as ".ca". Uses cache at all levels (CU, L2, LLC) with LRU policy.
                    - ".ca": Cache at all levels (CU, L2, LLC) with LRU policy.
                    - ".cg": Bypasses the CU (L1) cache, streams through L2, and may hit in LLC but the line is not retained or inserted.
                    - ".cv": Bypasses all GPU caches (CU and L2) and fetches directly from system memory. If data exists in the LLC, it may hit, but is not retained or inserted.
                            Ensures global coherence by invalidating stale GPU cache lines.

            volatile (bool, optional): If True, disables compiler optimizations that
                could reorder or eliminate the load. Defaults to False.

        Returns:
            The loaded value from the target memory location

        Example:
            >>> # Load from rank 1 to current rank
            >>> data = ctx.load(buffer + offsets, 1, mask=mask)
        """
        translated_ptr = self._translate(pointer, self.cur_rank, from_rank)
        result = gl.load(translated_ptr, mask=mask, other=other, cache_modifier=cache_modifier, volatile=volatile)
        return result

    @gluon.jit
    def store(self, pointer, value, to_rank, mask=None, cache_modifier=None):
        """
        Writes data from the current rank to the specified rank's memory location.

        Args:
            pointer: Pointer in the current rank's address space
            value: The value to store
            to_rank: The rank ID to which the data will be written
            mask: Optional mask for conditional storing
            cache_modifier (str, optional): Controls cache behavior of the store. Supported values are:

                - None: *(default)* — Same as ".wb". Uses write-back caching at all levels (CU, L2, LLC) with LRU policy.
                - ".wb": Write-back. Write-allocate on L1 miss, inserted into caches and written back later.
                - ".cg": Cache Global. Equivalent to ".wb" — stored through L1 → L2 → LLC under LRU.
                - ".cs": Cache Streaming. Bypasses L1, streamed through L2, not retained in LLC.
                - ".wt": Write-Through. Bypasses L1 and L2 (coherent cache bypass), may hit in LLC with LRU.

        Example:
            >>> # Store from current rank to rank 1
            >>> ctx.store(buffer + offsets, values, 1, mask=mask)
        """
        translated_ptr = self._translate(pointer, self.cur_rank, to_rank)
        gl.store(translated_ptr, value, mask=mask, cache_modifier=cache_modifier)

    @gluon.jit
    def get(
        self, from_ptr, to_ptr, from_rank, mask=None, other=None, load_cache_modifier=None, store_cache_modifier=None
    ):
        """
        Copies data from the specified rank's memory to the current rank's local memory.

        Args:
            from_ptr: Pointer to remote memory in `from_rank`'s address space
            to_ptr: Pointer to local memory in current rank
            from_rank: The rank ID from which to read the data
            mask: Optional mask for conditional operations
            other: Value to return for masked-out elements during the load operation. If not provided, the result for masked-out elements is undefined.
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

        Example:
            >>> # Copy from rank 1 to current rank's local memory
            >>> ctx.get(remote_ptr + offsets, local_ptr + offsets, 1, mask=mask)
        """
        translated_from_ptr = self._translate(from_ptr, self.cur_rank, from_rank)
        data = gl.load(translated_from_ptr, mask=mask, other=other, cache_modifier=load_cache_modifier)
        gl.store(to_ptr, data, mask=mask, cache_modifier=store_cache_modifier)

    @gluon.jit
    def put(
        self, from_ptr, to_ptr, to_rank, mask=None, other=None, load_cache_modifier=None, store_cache_modifier=None
    ):
        """
        Copies data from the current rank's local memory to the specified rank's memory.

        Args:
            from_ptr: Pointer to local memory in current rank
            to_ptr: Pointer to remote memory in `to_rank`'s address space
            to_rank: The rank ID to which the data will be written
            mask: Optional mask for conditional operations
            other: Value to return for masked-out elements during the load operation. If not provided, the result for masked-out elements is undefined.
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

        Example:
            >>> # Copy from current rank's local memory to rank 1
            >>> ctx.put(local_ptr + offsets, remote_ptr + offsets, 1, mask=mask)
        """
        translated_to_ptr = self._translate(to_ptr, self.cur_rank, to_rank)
        data = gl.load(from_ptr, mask=mask, other=other, cache_modifier=load_cache_modifier)
        gl.store(translated_to_ptr, data, mask=mask, cache_modifier=store_cache_modifier)

    @gluon.jit
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
    ):
        """
        Copies data from the specified rank's memory into the destination rank's memory.

        This function performs the transfer by translating `src_ptr` from the `from_rank`'s address
        space to the `to_rank`'s address space, performing a masked load from the translated
        source, and storing the loaded data to `dst_ptr` in the `to_rank` memory location.
        If `from_rank` and `to_rank` are the same, this function performs a local copy operation.
        It is undefined behaviour if neither `from_rank` nor `to_rank` is the `cur_rank`.

        Args:
            src_ptr: Pointer in the `from_rank`'s local memory from which to read data
            dst_ptr: Pointer in the `to_rank`'s local memory where the data will be written
            from_rank: The rank ID that owns `src_ptr` (source rank)
            to_rank: The rank ID that will receive the data (destination rank)
            mask: Optional mask for conditional operations
            other: Value to return for masked-out elements during the load operation. If not provided, the result for masked-out elements is undefined.
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

        Example:
            >>> # Copy from rank 1 to rank 0 (current rank must be either 1 or 0)
            >>> ctx.copy(remote_ptr + offsets, local_ptr + offsets, 1, 0, mask=mask)
        """
        cur_base = gl.load(self.heap_bases + self.cur_rank)
        from_base = gl.load(self.heap_bases + from_rank)
        to_base = gl.load(self.heap_bases + to_rank)

        src_ptr_int = tl.cast(src_ptr, gl.uint64)
        src_offset = src_ptr_int - cur_base

        dst_ptr_int = tl.cast(dst_ptr, gl.uint64)
        dst_offset = dst_ptr_int - cur_base

        from_base_byte = tl.cast(from_base, gl.pointer_type(gl.int8))
        to_base_byte = tl.cast(to_base, gl.pointer_type(gl.int8))

        translated_src = tl.cast(from_base_byte + src_offset, src_ptr.dtype)
        translated_dst = tl.cast(to_base_byte + dst_offset, dst_ptr.dtype)

        data = gl.load(translated_src, mask=mask, other=other, cache_modifier=load_cache_modifier)
        gl.store(translated_dst, data, mask=mask, cache_modifier=store_cache_modifier)

    @gluon.jit
    def atomic_add(self, pointer, val, to_rank, mask=None, sem=None, scope=None):
        """
        Performs an atomic add at the specified rank's memory location.

        Args:
            pointer: The memory location in the current rank's address space
            val: The value to add
            to_rank: The rank ID to which the atomic operation will be performed
            mask: Optional mask for conditional operations
            sem: Memory semantics (acquire, release, acq_rel, relaxed)
            scope: Scope of synchronization (gpu, cta, sys)

        Returns:
            The value at the memory location before the atomic operation

        Example:
            >>> # Atomically add to rank 1's memory
            >>> old_val = ctx.atomic_add(buffer, 5, 1)
        """
        translated_ptr = self._translate(pointer, self.cur_rank, to_rank)
        return gl.atomic_add(translated_ptr, val, mask=mask, sem=sem, scope=scope)

    @gluon.jit
    def atomic_sub(self, pointer, val, to_rank, mask=None, sem=None, scope=None):
        """
        Atomically subtracts data from the specified rank's memory location.

        Args:
            pointer: Pointer in the current rank's address space
            val: The value to subtract
            to_rank: The rank ID to which the atomic operation will be performed
            mask: Optional mask for conditional operations
            sem: Memory semantics (acquire, release, acq_rel, relaxed)
            scope: Scope of synchronization (gpu, cta, sys)

        Returns:
            The value at the memory location before the atomic operation

        Example:
            >>> # Atomically subtract from rank 1's memory
            >>> old_val = ctx.atomic_sub(buffer, 3, 1)
        """
        translated_ptr = self._translate(pointer, self.cur_rank, to_rank)
        return gl.atomic_sub(translated_ptr, val, mask=mask, sem=sem, scope=scope)

    @gluon.jit
    def atomic_cas(self, pointer, cmp, val, to_rank, sem=None, scope=None):
        """
        Atomically compares and exchanges the specified rank's memory location.

        Args:
            pointer: Pointer in the current rank's address space
            cmp: The expected value to compare
            val: The new value to write if comparison succeeds
            to_rank: The rank ID to which the atomic operation will be performed
            sem: Memory semantics (acquire, release, acq_rel, relaxed)
            scope: Scope of synchronization (gpu, cta, sys)

        Returns:
            The value at the memory location before the atomic operation

        Example:
            >>> # Compare-and-swap on rank 1's memory
            >>> old_val = ctx.atomic_cas(flag + pid, 0, 1, 1, sem="release", scope="sys")
        """
        translated_ptr = self._translate(pointer, self.cur_rank, to_rank)
        return gl.atomic_cas(translated_ptr, cmp, val, sem=sem, scope=scope)

    @gluon.jit
    def atomic_xchg(self, pointer, val, to_rank, mask=None, sem=None, scope=None):
        """
        Performs an atomic exchange at the specified rank's memory location.

        Args:
            pointer: The memory location in the current rank's address space
            val: The value to exchange
            to_rank: The rank ID to which the atomic operation will be performed
            mask: Optional mask for conditional operations
            sem: Memory semantics (acquire, release, acq_rel, relaxed)
            scope: Scope of synchronization (gpu, cta, sys)

        Returns:
            The value at the memory location before the atomic operation

        Example:
            >>> # Exchange value with rank 1's memory
            >>> old_val = ctx.atomic_xchg(buffer, 99, 1)
        """
        translated_ptr = self._translate(pointer, self.cur_rank, to_rank)
        return gl.atomic_xchg(translated_ptr, val, mask=mask, sem=sem, scope=scope)

    @gluon.jit
    def atomic_xor(self, pointer, val, to_rank, mask=None, sem=None, scope=None):
        """
        Performs an atomic xor at the specified rank's memory location.

        Args:
            pointer: The memory location in the current rank's address space
            val: The value to xor
            to_rank: The rank ID to which the atomic operation will be performed
            mask: Optional mask for conditional operations
            sem: Memory semantics (acquire, release, acq_rel, relaxed)
            scope: Scope of synchronization (gpu, cta, sys)

        Returns:
            The value at the memory location before the atomic operation

        Example:
            >>> # Atomically XOR with rank 1's memory
            >>> old_val = ctx.atomic_xor(buffer, 0xFF, 1)
        """
        translated_ptr = self._translate(pointer, self.cur_rank, to_rank)
        return gl.atomic_xor(translated_ptr, val, mask=mask, sem=sem, scope=scope)

    @gluon.jit
    def atomic_and(self, pointer, val, to_rank, mask=None, sem=None, scope=None):
        """
        Performs an atomic and at the specified rank's memory location.

        Args:
            pointer: The memory location in the current rank's address space
            val: The value to and
            to_rank: The rank ID to which the atomic operation will be performed
            mask: Optional mask for conditional operations
            sem: Memory semantics (acquire, release, acq_rel, relaxed)
            scope: Scope of synchronization (gpu, cta, sys)

        Returns:
            The value at the memory location before the atomic operation

        Example:
            >>> # Atomically AND with rank 1's memory
            >>> old_val = ctx.atomic_and(buffer, 0x0F, 1)
        """
        translated_ptr = self._translate(pointer, self.cur_rank, to_rank)
        return gl.atomic_and(translated_ptr, val, mask=mask, sem=sem, scope=scope)

    @gluon.jit
    def atomic_or(self, pointer, val, to_rank, mask=None, sem=None, scope=None):
        """
        Performs an atomic or at the specified rank's memory location.

        Args:
            pointer: The memory location in the current rank's address space
            val: The value to or
            to_rank: The rank ID to which the atomic operation will be performed
            mask: Optional mask for conditional operations
            sem: Memory semantics (acquire, release, acq_rel, relaxed)
            scope: Scope of synchronization (gpu, cta, sys)

        Returns:
            The value at the memory location before the atomic operation

        Example:
            >>> # Atomically OR with rank 1's memory
            >>> old_val = ctx.atomic_or(buffer, 0xF0, 1)
        """
        translated_ptr = self._translate(pointer, self.cur_rank, to_rank)
        return gl.atomic_or(translated_ptr, val, mask=mask, sem=sem, scope=scope)

    @gluon.jit
    def atomic_min(self, pointer, val, to_rank, mask=None, sem=None, scope=None):
        """
        Performs an atomic min at the specified rank's memory location.

        Args:
            pointer: The memory location in the current rank's address space
            val: The value to compare and potentially store
            to_rank: The rank ID to which the atomic operation will be performed
            mask: Optional mask for conditional operations
            sem: Memory semantics (acquire, release, acq_rel, relaxed)
            scope: Scope of synchronization (gpu, cta, sys)

        Returns:
            The value at the memory location before the atomic operation

        Example:
            >>> # Atomically compute minimum with rank 1's memory
            >>> old_val = ctx.atomic_min(buffer, 10, 1)
        """
        translated_ptr = self._translate(pointer, self.cur_rank, to_rank)
        return gl.atomic_min(translated_ptr, val, mask=mask, sem=sem, scope=scope)

    @gluon.jit
    def atomic_max(self, pointer, val, to_rank, mask=None, sem=None, scope=None):
        """
        Performs an atomic max at the specified rank's memory location.

        Args:
            pointer: The memory location in the current rank's address space
            val: The value to compare and potentially store
            to_rank: The rank ID to which the atomic operation will be performed
            mask: Optional mask for conditional operations
            sem: Memory semantics (acquire, release, acq_rel, relaxed)
            scope: Scope of synchronization (gpu, cta, sys)

        Returns:
            The value at the memory location before the atomic operation

        Example:
            >>> # Atomically compute maximum with rank 1's memory
            >>> old_val = ctx.atomic_max(buffer, 100, 1)
        """
        translated_ptr = self._translate(pointer, self.cur_rank, to_rank)
        return gl.atomic_max(translated_ptr, val, mask=mask, sem=sem, scope=scope)


IrisDeviceCtx = Context  # backward compat
