# SPDX-License-Identifier: MIT
# Copyright (c) 2025-2026 Advanced Micro Devices, Inc. All rights reserved.

"""
Triton device-side RMA operations (load, store, copy, get, put, atomics).

These are the module-level functional API. For the OO API, see DeviceContext.
"""

import triton
import triton.language as tl
from iris.mem.triton.context import __translate


@triton.jit
def load(
    pointer,
    to_rank,
    from_rank,
    heap_bases,
    mask=None,
    other=None,
    cache_modifier=None,
    volatile=False,
    hint: tl.constexpr = None,
):
    """
    Loads a value from the specified rank's memory location.

    This function performs a memory read operation by translating the pointer
    from the `from_rank`'s address space to the `to_rank`'s address space and loading
    data from the target memory location. The load is **local** when
    ``to_rank == from_rank``, and **remote** (cross-GPU) otherwise.

    The ``cache_modifier`` is passed through to the underlying ``tl.load()``
    call. Cache modifiers control instruction-level cache behavior by setting
    the appropriate scope (``SC0``, ``SC1``) and non-temporal (``NT``) bits
    in the load instruction, following the CDNA ISA.

    Args:
        pointer (triton.PointerType, or block of dtype=triton.PointerType): Pointer in the `from_rank`'s address space that will be translated to the `to_rank`'s address space. Must be the current rank where the pointer is local.
        to_rank (int): The rank ID to which the pointer will be translated. Must be the current rank where the pointer is local.
        from_rank (int): The rank ID from which to read the data.
        heap_bases (triton.PointerType): Array containing the heap base addresses for all ranks.
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
            could reorder or eliminate the load.
        hint (int or tuple, optional): Vectorization hint passed to tl.multiple_of / tl.max_contiguous on the translated pointer. Use a scalar for 1-D (e.g. 16) or a tuple for N-D (e.g. (1, 16)). Defaults to None (no hint).

    Returns:
        Block: The loaded value from the target memory location.

    Example:
        >>> @triton.jit
        >>> def kernel(ptr, heap_bases):
        >>>     # Load data from rank 1's memory into the current rank
        >>>     cur_rank = 0      # Current rank
        >>>     remote_rank = 1   # Remote rank to load from
        >>>     data = iris.load(ptr, cur_rank, remote_rank, heap_bases)
        >>>     return data
    """
    translated_ptr = __translate(pointer, to_rank, from_rank, heap_bases, hint)
    result = tl.load(translated_ptr, mask=mask, other=other, cache_modifier=cache_modifier, volatile=volatile)
    return result


@triton.jit
def store(
    pointer,
    value,
    from_rank,
    to_rank,
    heap_bases,
    mask=None,
    hint: tl.constexpr = None,
    cache_modifier=None,
):
    """
    Writes data to the specified rank's memory location.

    This function performs a memory write operation by translating the pointer
    from the `from_rank`'s address space to the `to_rank`'s address space and storing
    the provided data to the target memory location. The store is **local** when
    ``from_rank == to_rank``, and **remote** (cross-GPU) otherwise.

    The ``cache_modifier`` is passed through to the underlying ``tl.store()``
    call. Cache modifiers control instruction-level cache behavior by setting
    the appropriate scope (``SC0``, ``SC1``) and non-temporal (``NT``) bits
    in the store instruction, following the CDNA ISA.

    Args:
        pointer (triton.PointerType, or block of dtype=triton.PointerType): Pointer in the `from_rank`'s address space that will be translated to the `to_rank`'s address space. Must be the current rank where the pointer is local.
        value (Block): The tensor of elements to be stored.
        from_rank (int): The rank ID from which the pointer originates. Must be the current rank where the pointer is local.
        to_rank (int): The rank ID to which the data will be written.
        heap_bases (triton.PointerType): Array containing the heap base addresses for all ranks.
        mask (Block of triton.int1, optional): If mask[idx] is false, do not store the data at address pointer[idx]. Defaults to None.
        hint (int or tuple, optional): Vectorization hint passed to tl.multiple_of / tl.max_contiguous on the translated pointer. Use a scalar for 1-D (e.g. 16) or a tuple for N-D (e.g. (1, 16)). Defaults to None (no hint).
        cache_modifier (str, optional): Controls cache behavior of the store. Supported values are:

            - None: *(default)* — Same as ".wb". Uses write-back caching at all levels (CU, L2, LLC) with LRU policy.
            - ".wb": Write-back. Write-allocate on L1 miss, inserted into caches and written back later.
            - ".cg": Cache Global. Equivalent to ".wb" — stored through L1 → L2 → LLC under LRU.
            - ".cs": Cache Streaming. Bypasses L1, streamed through L2, not retained in LLC.
            - ".wt": Write-Through. Bypasses L1 and L2 (coherent cache bypass), may hit in LLC with LRU.

    Returns:
        None

    Example:
        >>> @triton.jit
        >>> def kernel(ptr, heap_bases):
        >>>     # Store value 42 into rank 1's heap from rank 0
        >>>     cur_rank = 0      # Current rank (source)
        >>>     remote_rank = 1   # Remote rank (destination)
        >>>     value = 42
        >>>     iris.store(ptr, value, cur_rank, remote_rank, heap_bases)
    """
    translated_ptr = __translate(pointer, from_rank, to_rank, heap_bases, hint)
    tl.store(translated_ptr, value, mask=mask, cache_modifier=cache_modifier)


@triton.jit
def copy(
    src_ptr,
    dst_ptr,
    from_rank,
    to_rank,
    cur_rank,
    heap_bases,
    mask=None,
    other=None,
    load_cache_modifier=None,
    store_cache_modifier=None,
    hint: tl.constexpr = None,
):
    """
    Copies data from the specified rank's memory into the destination rank's memory.
    This function performs the transfer by translating `src_ptr` from the `from_rank`'s address
    space to the `to_rank`'s address space, performing a masked load from the translated
    source, and storing the loaded data to `dst_ptr` in the `to_rank` memory location.
    It is undefined behaviour if neither `from_rank` nor `to_rank` is the `cur_rank`.

    The load is from ``from_rank`` (remote if ``from_rank != cur_rank``) and the store is to
    ``to_rank`` (remote if ``to_rank != cur_rank``).

    Args:
        src_ptr (triton.PointerType, or block of dtype=triton.PointerType): Pointer in the `from_rank`'s local memory from which to read data.
        dst_ptr (triton.PointerType, or block of dtype=triton.PointerType): Pointer in the `to_rank`'s local memory where the data will be written.
        from_rank (int): The rank ID that owns `src_ptr` (source rank).
        to_rank (int): The rank ID that will receive the data (destination rank).
        cur_rank (int): The rank ID issuing the copy operation. Must be either `from_rank` or `to_rank`.
        heap_bases (triton.PointerType): Array containing the heap base addresses for all ranks.
        mask (Block of triton.int1, optional): If mask[idx] is false, do not load from the translated src_ptr[idx] and do not store to dst_ptr[idx]. Defaults to None.
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
        hint (int or tuple, optional): Vectorization hint passed to tl.multiple_of / tl.max_contiguous on the translated pointers. Use a scalar for 1-D (e.g. 16) or a tuple for N-D (e.g. (1, 16)). Defaults to None (no hint).

    Returns:
        None

    Example:
        >>> @triton.jit
        >>> def kernel(remote_ptr, local_ptr, heap_bases):
        >>>     from_rank = 1
        >>>     to_rank = 0
        >>>     iris.copy(remote_ptr, local_ptr, from_rank, to_rank, to_rank, heap_bases)
    """

    cur_base = tl.load(heap_bases + cur_rank)

    from_base = tl.load(heap_bases + from_rank)
    to_base = tl.load(heap_bases + to_rank)

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
def get(
    from_ptr,
    to_ptr,
    from_rank,
    to_rank,
    heap_bases,
    mask=None,
    other=None,
    load_cache_modifier=None,
    store_cache_modifier=None,
    hint: tl.constexpr = None,
):
    """
    Copies data from the specified rank's memory to the current rank's local memory.

    This function performs a memory read operation by translating the `from_ptr`
    from the current rank's address space to the `from_rank`'s address space, loading data
    from the `from_rank`'s memory location, and storing it to the local `to_ptr`.

    The load is **remote** when ``from_rank != to_rank`` (reading from a peer GPU), while the
    store is **always local** (writing to `to_ptr` in the current rank's own memory).

    Args:
        from_ptr (triton.PointerType, or block of dtype=triton.PointerType): Pointer in the current rank's address space that will be translated to the `from_rank`'s address space. Must be the current rank where the pointer is local.
        to_ptr (triton.PointerType, or block of dtype=triton.PointerType): Pointer in the current rank's local memory where the data will be stored.
        from_rank (int): The `from_rank` ID from which to read the data.
        to_rank (int): The current rank ID where the data will be stored.
        heap_bases (triton.PointerType): Array containing the heap base addresses for all ranks.
        mask (Block of triton.int1, optional): If mask[idx] is false, do not load the data at address from_ptr[idx] and do not store to to_ptr[idx]. Defaults to None.
        other (Block, optional): Value to return for masked-out elements during the load operation. If not provided, the result for masked-out elements is undefined. Defaults to None.
        load_cache_modifier (str, optional): Controls cache behavior of the load (remote when ``from_rank != to_rank``). Supported values are:
            - None: *(default)* — Same as ".ca". Uses cache at all levels (CU, L2, LLC) with LRU policy.
            - ".ca": Cache at all levels (CU, L2, LLC) with LRU policy.
            - ".cg": Bypasses the CU (L1) cache, streams through L2, and may hit in LLC but the line is not retained or inserted.
            - ".cv": Bypasses all GPU caches (CU and L2) and fetches directly from system memory. If data exists in the LLC, it may hit, but is not retained or inserted.

        store_cache_modifier (str, optional): Controls cache behavior of the store. The store is always to local memory (``to_ptr``). Supported values are:
            - None: *(default)* — Same as ".wb". Uses write-back caching at all levels (CU, L2, LLC) with LRU policy.
            - ".wb": Write-back. Write-allocate on L1 miss, inserted into caches and written back later.
            - ".cg": Cache Global. Equivalent to ".wb" — stored through L1 → L2 → LLC under LRU.
            - ".cs": Cache Streaming. Bypasses L1, streamed through L2, not retained in LLC.
            - ".wt": Write-Through. Bypasses L1 and L2 (coherent cache bypass), may hit in LLC with LRU.
        hint (int or tuple, optional): Vectorization hint passed to tl.multiple_of / tl.max_contiguous on the translated pointer. Use a scalar for 1-D (e.g. 16) or a tuple for N-D (e.g. (1, 16)). Defaults to None (no hint).

    Returns:
        None

    Example:
        >>> @triton.jit
        >>> def kernel(remote_ptr, local_ptr, heap_bases):
        >>>     from_rank = 1
        >>>     to_rank = 0
        >>>     iris.get(remote_ptr, local_ptr, from_rank, to_rank, heap_bases)
    """
    translated_from_ptr = __translate(from_ptr, from_rank, to_rank, heap_bases, hint)

    data = tl.load(translated_from_ptr, mask=mask, other=other, cache_modifier=load_cache_modifier)

    tl.store(to_ptr, data, mask=mask, cache_modifier=store_cache_modifier)


@triton.jit
def put(
    from_ptr,
    to_ptr,
    from_rank,
    to_rank,
    heap_bases,
    mask=None,
    other=None,
    load_cache_modifier=None,
    store_cache_modifier=None,
    hint: tl.constexpr = None,
):
    """
    Copies data from the current rank's local memory to the specified rank's memory.
    This function performs a memory write operation by loading data from the current
    rank's `from_ptr`, translating the `to_ptr` from the current rank's address
    space to the `to_rank`'s address space, and storing the data to the `to_rank` memory location.

    The load is **always local** (reading from the current rank's own ``from_ptr``), while the
    store is **remote** when ``from_rank != to_rank`` (writing to a peer GPU).

    Args:
        from_ptr (triton.PointerType, or block of dtype=triton.PointerType): Pointer in the current rank's local memory from which to read data.
        to_ptr (triton.PointerType, or block of dtype=triton.PointerType): Pointer in the current rank's address space that will be translated to the `to_rank`'s address space. Must be the current rank where the pointer is local.
        from_rank (int): The current rank ID from which to read the data.
        to_rank (int): The `to_rank` ID to which the data will be written.
        heap_bases (triton.PointerType): Array containing the heap base addresses for all ranks.
        mask (Block of triton.int1, optional): If mask[idx] is false, do not load the data at address from_ptr[idx] and do not store to to_ptr[idx]. Defaults to None.
        other (Block, optional): Value to return for masked-out elements during the load operation. If not provided, the result for masked-out elements is undefined. Defaults to None.

        load_cache_modifier (str, optional): Controls cache behavior of the load (always local). Supported values are:
            - None: *(default)* — Same as ".ca". Uses cache at all levels (CU, L2, LLC) with LRU policy.
            - ".ca": Cache at all levels (CU, L2, LLC) with LRU policy.
            - ".cg": Bypasses the CU (L1) cache, streams through L2, and may hit in LLC but the line is not retained or inserted.
            - ".cv": Bypasses all GPU caches (CU and L2) and fetches directly from system memory. If data exists in the LLC, it may hit, but is not retained or inserted.

        store_cache_modifier (str, optional): Controls cache behavior of the store (remote when ``from_rank != to_rank``). Supported values are:
            - None: *(default)* — Same as ".wb". Uses write-back caching at all levels (CU, L2, LLC) with LRU policy.
            - ".wb": Write-back. Write-allocate on L1 miss, inserted into caches and written back later.
            - ".cg": Cache Global. Equivalent to ".wb" — stored through L1 → L2 → LLC under LRU.
            - ".cs": Cache Streaming. Bypasses L1, streamed through L2, not retained in LLC.
            - ".wt": Write-Through. Bypasses L1 and L2 (coherent cache bypass), may hit in LLC with LRU.
        hint (int or tuple, optional): Vectorization hint passed to tl.multiple_of / tl.max_contiguous on the translated pointer. Use a scalar for 1-D (e.g. 16) or a tuple for N-D (e.g. (1, 16)). Defaults to None (no hint).

    Returns:
        None

    Example:
        >>> @triton.jit
        >>> def kernel(local_ptr, remote_ptr, heap_bases):
        >>>     from_rank = 0
        >>>     to_rank = 1
        >>>     iris.put(local_ptr, remote_ptr, from_rank, to_rank, heap_bases)
    """
    translated_to_ptr = __translate(to_ptr, from_rank, to_rank, heap_bases, hint)

    data = tl.load(from_ptr, mask=mask, other=other, cache_modifier=load_cache_modifier)

    tl.store(translated_to_ptr, data, mask=mask, cache_modifier=store_cache_modifier)


@triton.jit
def atomic_add(
    pointer, val, from_rank, to_rank, heap_bases, mask=None, sem=None, scope=None, hint: tl.constexpr = None
):
    """
    Performs an atomic add at the specified rank's memory location.

    This function performs an atomic addition operation by translating the pointer
    from the `from_rank`'s address space to the `to_rank`'s address space and atomically
    adding the provided data to the `to_rank` memory location. If the `from_rank` and `to_rank` are the same,
    this function performs a local atomic addition operation.

    Args:
        pointer (triton.PointerType, or block of dtype=triton.PointerType): The memory locations in the `from_rank`'s address space that will be translated to the `to_rank`'s address space. Must be the current rank where the pointer is local.
        val (Block of dtype=pointer.dtype.element_ty): The values with which to perform the atomic operation.
        from_rank (int): The rank ID from which the pointer originates. Must be the current rank where the pointer is local.
        to_rank (int): The rank ID to which the atomic operation will be performed.
        heap_bases (triton.PointerType): Array containing the heap base addresses for all ranks.
        mask (Block of triton.int1, optional): If mask[idx] is false, do not perform the atomic operation at address pointer[idx]. Defaults to None.
        sem (str, optional): Specifies the memory semantics for the operation. Acceptable values are "acquire", "release", "acq_rel" (stands for "ACQUIRE_RELEASE"), and "relaxed". If not provided, the function defaults to using "acq_rel" semantics.
        scope (str, optional): Defines the scope of threads that observe the synchronizing effect of the atomic operation. Acceptable values are "gpu" (default), "cta" (cooperative thread array, thread block), or "sys" (stands for "SYSTEM"). The default value is "gpu".
        hint (int or tuple, optional): Vectorization hint passed to tl.multiple_of / tl.max_contiguous on the translated pointer. Defaults to None (no hint).

    Returns:
        Block: The data stored at pointer before the atomic operation.

    Example:
        >>> @triton.jit
        >>> def kernel(ptr, heap_bases):
        >>>     # Atomically add 5 to rank 1's memory from rank 0
        >>>     cur_rank = 0      # Current rank (source)
        >>>     remote_rank = 1   # Remote rank (destination)
        >>>     increment = 5
        >>>     old_val = iris.atomic_add(ptr, increment, cur_rank, remote_rank, heap_bases)
    """
    translated_ptr = __translate(pointer, from_rank, to_rank, heap_bases, hint)
    return tl.atomic_add(translated_ptr, val, mask=mask, sem=sem, scope=scope)


@triton.jit
def atomic_sub(
    pointer, val, from_rank, to_rank, heap_bases, mask=None, sem=None, scope=None, hint: tl.constexpr = None
):
    """
    Atomically subtracts data from the specified rank's memory location.

    This function performs an atomic subtraction operation by translating the pointer
    from the `from_rank`'s address space to the `to_rank`'s address space and atomically
        subtracting the provided data from the `to_rank` memory location. If the `from_rank` and `to_rank` are the same,
    this function performs a local atomic subtraction operation.

    Args:
        pointer (triton.PointerType, or block of dtype=triton.PointerType): Pointer in the `from_rank`'s address space that will be translated to the `to_rank`'s address space. Must be the current rank where the pointer is local.
        val (Block): The tensor of elements to be subtracted atomically.
        from_rank (int): The rank ID from which the pointer originates. Must be the current rank where the pointer is local.
        to_rank (int): The rank ID to which the atomic operation will be performed.
        heap_bases (triton.PointerType): Array containing the heap base addresses for all ranks.
        mask (Block of triton.int1, optional): If mask[idx] is false, do not perform the atomic operation at address pointer[idx]. Defaults to None.
        sem (str, optional): Specifies the memory semantics for the operation. Acceptable values are "acquire", "release", "acq_rel" (stands for "ACQUIRE_RELEASE"), and "relaxed". Defaults to "acq_rel".
        scope (str, optional): Defines the scope of threads that observe the synchronizing effect of the atomic operation. Acceptable values are "gpu" (default), "cta" (cooperative thread array, thread block), or "sys" (stands for "SYSTEM"). Defaults to "gpu".
        hint (int or tuple, optional): Vectorization hint passed to tl.multiple_of / tl.max_contiguous on the translated pointer. Defaults to None (no hint).

    Returns:
        Block: The value at the memory location before the atomic subtraction.

    Example:
        >>> @triton.jit
        >>> def kernel(ptr, heap_bases):
        >>>     # Atomically subtract 3 from rank 2's memory from rank 0
        >>>     cur_rank = 0      # Current rank (source)
        >>>     remote_rank = 2   # Remote rank (destination)
        >>>     decrement = 3
        >>>     old_val = iris.atomic_sub(ptr, decrement, cur_rank, remote_rank, heap_bases)
    """
    translated_ptr = __translate(pointer, from_rank, to_rank, heap_bases, hint)
    return tl.atomic_sub(translated_ptr, val, mask=mask, sem=sem, scope=scope)


@triton.jit
def atomic_cas(pointer, cmp, val, from_rank, to_rank, heap_bases, sem=None, scope=None, hint: tl.constexpr = None):
    """
    Atomically compares and exchanges the specified rank's memory location.

    This function performs an atomic compare-and-swap operation by translating the pointer
    from the `from_rank`'s address space to the `to_rank`'s address space and atomically
    comparing the current value with the expected value, then writing the new value if they match.
    If the `from_rank` and `to_rank` are the same, this function performs a local atomic compare-and-swap operation.

    Args:
        pointer (triton.PointerType, or block of dtype=triton.PointerType): Pointer in the `from_rank`'s address space that will be translated to the `to_rank`'s address space. Must be the current rank where the pointer is local.
        cmp (Block): The expected value to be compared with the current value at the memory location.
        val (Block): The new value to be written if the compare succeeds.
        from_rank (int): The rank ID from which the pointer originates. Must be the current rank where the pointer is local.
        to_rank (int): The rank ID to which the atomic operation will be performed.
        heap_bases (triton.PointerType): Array containing the heap base addresses for all ranks.
        sem (str, optional): Specifies the memory semantics for the operation. Acceptable values are "acquire", "release", "acq_rel" (stands for "ACQUIRE_RELEASE"), and "relaxed". Defaults to "acq_rel".
        scope (str, optional): Defines the scope of threads that observe the synchronizing effect of the atomic operation. Acceptable values are "gpu" (default), "cta" (cooperative thread array, thread block), or "sys" (stands for "SYSTEM"). Defaults to "gpu".
        hint (int or tuple, optional): Vectorization hint passed to tl.multiple_of / tl.max_contiguous on the translated pointer. Defaults to None (no hint).

    Returns:
        Block: The value contained at the memory location before the atomic operation attempt.

    Example:
        >>> @triton.jit
        >>> def kernel(ptr, heap_bases):
        >>>     # Compare-and-swap on rank 1's memory from rank 0
        >>>     cur_rank = 0      # Current rank (source)
        >>>     remote_rank = 1   # Remote rank (destination)
        >>>     expected = 0
        >>>     new_val = 42
        >>>     old_val = iris.atomic_cas(ptr, expected, new_val, cur_rank, remote_rank, heap_bases)
    """
    translated_ptr = __translate(pointer, from_rank, to_rank, heap_bases, hint)
    return tl.atomic_cas(translated_ptr, cmp, val, sem=sem, scope=scope)


@triton.jit
def atomic_xchg(
    pointer, val, from_rank, to_rank, heap_bases, mask=None, sem=None, scope=None, hint: tl.constexpr = None
):
    """
    Performs an atomic exchange at the specified rank's memory location.

    This function performs an atomic exchange operation by translating the pointer
    from the `from_rank`'s address space to the `to_rank`'s address space and atomically
    exchanging the current value with the provided new value. If the `from_rank` and `to_rank` are the same,
    this function performs a local atomic exchange operation.

    Args:
        pointer (triton.PointerType, or block of dtype=triton.PointerType): The memory locations in the `from_rank`'s address space that will be translated to the `to_rank`'s address space. Must be the current rank where the pointer is local.
        val (Block of dtype=pointer.dtype.element_ty): The values with which to perform the atomic operation.
        from_rank (int): The rank ID from which the pointer originates. Must be the current rank where the pointer is local.
        to_rank (int): The rank ID to which the atomic operation will be performed.
        heap_bases (triton.PointerType): Array containing the heap base addresses for all ranks.
        mask (Block of triton.int1, optional): If mask[idx] is false, do not perform the atomic operation at address pointer[idx]. Defaults to None.
        sem (str, optional): Specifies the memory semantics for the operation. Acceptable values are "acquire", "release", "acq_rel" (stands for "ACQUIRE_RELEASE"), and "relaxed". If not provided, the function defaults to using "acq_rel" semantics.
        scope (str, optional): Defines the scope of threads that observe the synchronizing effect of the atomic operation. Acceptable values are "gpu" (default), "cta" (cooperative thread array, thread block), or "sys" (stands for "SYSTEM"). The default value is "gpu".
        hint (int or tuple, optional): Vectorization hint passed to tl.multiple_of / tl.max_contiguous on the translated pointer. Defaults to None (no hint).

    Returns:
        Block: The data stored at pointer before the atomic operation.

    Example:
        >>> @triton.jit
        >>> def kernel(ptr, heap_bases):
        >>>     # Exchange value with rank 1's memory from rank 0
        >>>     cur_rank = 0      # Current rank (source)
        >>>     remote_rank = 1   # Remote rank (destination)
        >>>     new_value = 99
        >>>     old_val = iris.atomic_xchg(ptr, new_value, cur_rank, remote_rank, heap_bases)
    """
    translated_ptr = __translate(pointer, from_rank, to_rank, heap_bases, hint)
    return tl.atomic_xchg(translated_ptr, val, mask=mask, sem=sem, scope=scope)


@triton.jit
def atomic_xor(
    pointer, val, from_rank, to_rank, heap_bases, mask=None, sem=None, scope=None, hint: tl.constexpr = None
):
    """
    Performs an atomic xor at the specified rank's memory location.

    This function performs an atomic xor operation by translating the pointer
    from the `from_rank`'s address space to the `to_rank`'s address space and atomically
    xoring the provided data to the `to_rank` memory location. If the `from_rank` and `to_rank` are the same,
    this function performs a local atomic xor operation.

    Args:
        pointer (triton.PointerType, or block of dtype=triton.PointerType): The memory locations in the `from_rank`'s address space that will be translated to the `to_rank`'s address space. Must be the current rank where the pointer is local.
        val (Block of dtype=pointer.dtype.element_ty): The values with which to perform the atomic operation.
        from_rank (int): The rank ID from which the pointer originates. Must be the current rank where the pointer is local.
        to_rank (int): The rank ID to which the atomic operation will be performed.
        heap_bases (triton.PointerType): Array containing the heap base addresses for all ranks.
        mask (Block of triton.int1, optional): If mask[idx] is false, do not perform the atomic operation at address pointer[idx]. Defaults to None.
        sem (str, optional): Specifies the memory semantics for the operation. Acceptable values are "acquire", "release", "acq_rel" (stands for "ACQUIRE_RELEASE"), and "relaxed". If not provided, the function defaults to using "acq_rel" semantics.
        scope (str, optional): Defines the scope of threads that observe the synchronizing effect of the atomic operation. Acceptable values are "gpu" (default), "cta" (cooperative thread array, thread block), or "sys" (stands for "SYSTEM"). The default value is "gpu".
        hint (int or tuple, optional): Vectorization hint passed to tl.multiple_of / tl.max_contiguous on the translated pointer. Defaults to None (no hint).

    Returns:
        Block: The data stored at pointer before the atomic operation.

    Example:
        >>> @triton.jit
        >>> def kernel(ptr, heap_bases):
        >>>     # Atomically XOR with rank 1's memory from rank 0
        >>>     cur_rank = 0      # Current rank (source)
        >>>     remote_rank = 1   # Remote rank (destination)
        >>>     mask_val = 0xFF
        >>>     old_val = iris.atomic_xor(ptr, mask_val, cur_rank, remote_rank, heap_bases)
    """
    translated_ptr = __translate(pointer, from_rank, to_rank, heap_bases, hint)
    return tl.atomic_xor(translated_ptr, val, mask=mask, sem=sem, scope=scope)


@triton.jit
def atomic_and(
    pointer, val, from_rank, to_rank, heap_bases, mask=None, sem=None, scope=None, hint: tl.constexpr = None
):
    """
    Performs an atomic and at the specified rank's memory location.

    This function performs an atomic and operation by translating the pointer
    from the `from_rank`'s address space to the `to_rank`'s address space and atomically
    anding the provided data to the `to_rank` memory location. If the `from_rank` and `to_rank` are the same,
    this function performs a local atomic and operation.

    Args:
        pointer (triton.PointerType, or block of dtype=triton.PointerType): The memory locations in the `from_rank`'s address space that will be translated to the `to_rank`'s address space. Must be the current rank where the pointer is local.
        val (Block of dtype=pointer.dtype.element_ty): The values with which to perform the atomic operation.
        from_rank (int): The rank ID from which the pointer originates. Must be the current rank where the pointer is local.
        to_rank (int): The rank ID to which the atomic operation will be performed.
        heap_bases (triton.PointerType): Array containing the heap base addresses for all ranks.
        mask (Block of triton.int1, optional): If mask[idx] is false, do not perform the atomic operation at address pointer[idx]. Defaults to None.
        sem (str, optional): Specifies the memory semantics for the operation. Acceptable values are "acquire", "release", "acq_rel" (stands for "ACQUIRE_RELEASE"), and "relaxed". If not provided, the function defaults to using "acq_rel" semantics.
        scope (str, optional): Defines the scope of threads that observe the synchronizing effect of the atomic operation. Acceptable values are "gpu" (default), "cta" (cooperative thread array, thread block), or "sys" (stands for "SYSTEM"). The default value is "gpu".
        hint (int or tuple, optional): Vectorization hint passed to tl.multiple_of / tl.max_contiguous on the translated pointer. Defaults to None (no hint).

    Returns:
        Block: The data stored at pointer before the atomic operation.

    Example:
        >>> @triton.jit
        >>> def kernel(ptr, heap_bases):
        >>>     # Atomically AND with rank 1's memory from rank 0
        >>>     cur_rank = 0      # Current rank (source)
        >>>     remote_rank = 1   # Remote rank (destination)
        >>>     mask_val = 0x0F
        >>>     old_val = iris.atomic_and(ptr, mask_val, cur_rank, remote_rank, heap_bases)
    """
    translated_ptr = __translate(pointer, from_rank, to_rank, heap_bases, hint)
    return tl.atomic_and(translated_ptr, val, mask=mask, sem=sem, scope=scope)


@triton.jit
def atomic_or(pointer, val, from_rank, to_rank, heap_bases, mask=None, sem=None, scope=None, hint: tl.constexpr = None):
    """
    Performs an atomic or at the specified rank's memory location.

    This function performs an atomic or operation by translating the pointer
    from the `from_rank`'s address space to the `to_rank`'s address space and atomically
    oring the provided data to the `to_rank` memory location. If the `from_rank` and `to_rank` are the same,
    this function performs a local atomic or operation.

    Args:
        pointer (triton.PointerType, or block of dtype=triton.PointerType): The memory locations in the `from_rank`'s address space that will be translated to the `to_rank`'s address space. Must be the current rank where the pointer is local.
        val (Block of dtype=pointer.dtype.element_ty): The values with which to perform the atomic operation.
        from_rank (int): The rank ID from which the pointer originates. Must be the current rank where the pointer is local.
        to_rank (int): The rank ID to which the atomic operation will be performed.
        heap_bases (triton.PointerType): Array containing the heap base addresses for all ranks.
        mask (Block of triton.int1, optional): If mask[idx] is false, do not perform the atomic operation at address pointer[idx]. Defaults to None.
        sem (str, optional): Specifies the memory semantics for the operation. Acceptable values are "acquire", "release", "acq_rel" (stands for "ACQUIRE_RELEASE"), and "relaxed". If not provided, the function defaults to using "acq_rel" semantics.
        scope (str, optional): Defines the scope of threads that observe the synchronizing effect of the atomic operation. Acceptable values are "gpu" (default), "cta" (cooperative thread array, thread block), or "sys" (stands for "SYSTEM"). The default value is "gpu".
        hint (int or tuple, optional): Vectorization hint passed to tl.multiple_of / tl.max_contiguous on the translated pointer. Defaults to None (no hint).

    Returns:
        Block: The data stored at pointer before the atomic operation.

    Example:
        >>> @triton.jit
        >>> def kernel(ptr, heap_bases):
        >>>     # Atomically OR with rank 1's memory from rank 0
        >>>     cur_rank = 0      # Current rank (source)
        >>>     remote_rank = 1   # Remote rank (destination)
        >>>     mask_val = 0xF0
        >>>     old_val = iris.atomic_or(ptr, mask_val, cur_rank, remote_rank, heap_bases)
    """
    translated_ptr = __translate(pointer, from_rank, to_rank, heap_bases, hint)
    return tl.atomic_or(translated_ptr, val, mask=mask, sem=sem, scope=scope)


@triton.jit
def atomic_min(
    pointer, val, from_rank, to_rank, heap_bases, mask=None, sem=None, scope=None, hint: tl.constexpr = None
):
    """
    Performs an atomic min at the specified rank's memory location.

    This function performs an atomic min operation by translating the pointer
    from the `from_rank`'s address space to the `to_rank`'s address space and atomically
    performing the min on the provided data to the `to_rank` memory location. If the `from_rank` and `to_rank` are the same,
    this function performs a local atomic min operation.

    Args:
        pointer (triton.PointerType, or block of dtype=triton.PointerType): The memory locations in the `from_rank`'s address space that will be translated to the `to_rank`'s address space. Must be the current rank where the pointer is local.
        val (Block of dtype=pointer.dtype.element_ty): The values with which to perform the atomic operation.
        from_rank (int): The rank ID from which the pointer originates. Must be the current rank where the pointer is local.
        to_rank (int): The rank ID to which the atomic operation will be performed.
        heap_bases (triton.PointerType): Array containing the heap base addresses for all ranks.
        mask (Block of triton.int1, optional): If mask[idx] is false, do not perform the atomic operation at address pointer[idx]. Defaults to None.
        sem (str, optional): Specifies the memory semantics for the operation. Acceptable values are "acquire", "release", "acq_rel" (stands for "ACQUIRE_RELEASE"), and "relaxed". If not provided, the function defaults to using "acq_rel" semantics.
        scope (str, optional): Defines the scope of threads that observe the synchronizing effect of the atomic operation. Acceptable values are "gpu" (default), "cta" (cooperative thread array, thread block), or "sys" (stands for "SYSTEM"). The default value is "gpu".
        hint (int or tuple, optional): Vectorization hint passed to tl.multiple_of / tl.max_contiguous on the translated pointer. Defaults to None (no hint).

    Returns:
        Block: The data stored at pointer before the atomic operation.

    Example:
        >>> @triton.jit
        >>> def kernel(ptr, heap_bases):
        >>>     # Atomically find minimum with rank 1's memory from rank 0
        >>>     cur_rank = 0      # Current rank (source)
        >>>     remote_rank = 1   # Remote rank (destination)
        >>>     new_val = 10
        >>>     old_val = iris.atomic_min(ptr, new_val, cur_rank, remote_rank, heap_bases)
    """
    translated_ptr = __translate(pointer, from_rank, to_rank, heap_bases, hint)
    return tl.atomic_min(translated_ptr, val, mask=mask, sem=sem, scope=scope)


@triton.jit
def atomic_max(
    pointer, val, from_rank, to_rank, heap_bases, mask=None, sem=None, scope=None, hint: tl.constexpr = None
):
    """
    Performs an atomic max at the specified rank's memory location.

    This function performs an atomic max operation by translating the pointer
    from the `from_rank`'s address space to the `to_rank`'s address space and atomically
    performing the max on the provided data to the `to_rank` memory location. If the `from_rank` and `to_rank` are the same,
    this function performs a local atomic max operation.

    Args:
        pointer (triton.PointerType, or block of dtype=triton.PointerType): The memory locations in the `from_rank`'s address space that will be translated to the `to_rank`'s address space. Must be the current rank where the pointer is local.
        val (Block of dtype=pointer.dtype.element_ty): The values with which to perform the atomic operation.
        from_rank (int): The rank ID from which the pointer originates. Must be the current rank where the pointer is local.
        to_rank (int): The rank ID to which the atomic operation will be performed.
        heap_bases (triton.PointerType): Array containing the heap base addresses for all ranks.
        mask (Block of triton.int1, optional): If mask[idx] is false, do not perform the atomic operation at address pointer[idx]. Defaults to None.
        sem (str, optional): Specifies the memory semantics for the operation. Acceptable values are "acquire", "release", "acq_rel" (stands for "ACQUIRE_RELEASE"), and "relaxed". If not provided, the function defaults to using "acq_rel" semantics.
        scope (str, optional): Defines the scope of threads that observe the synchronizing effect of the atomic operation. Acceptable values are "gpu" (default), "cta" (cooperative thread array, thread block), or "sys" (stands for "SYSTEM"). The default value is "gpu".
        hint (int or tuple, optional): Vectorization hint passed to tl.multiple_of / tl.max_contiguous on the translated pointer. Defaults to None (no hint).

    Returns:
        Block: The data stored at pointer before the atomic operation.

    Example:
        >>> @triton.jit
        >>> def kernel(ptr, heap_bases):
        >>>     # Atomically find maximum with rank 1's memory from rank 0
        >>>     cur_rank = 0      # Current rank (source)
        >>>     remote_rank = 1   # Remote rank (destination)
        >>>     new_val = 100
        >>>     old_val = iris.atomic_max(ptr, new_val, cur_rank, remote_rank, heap_bases)
    """
    translated_ptr = __translate(pointer, from_rank, to_rank, heap_bases, hint)
    return tl.atomic_max(translated_ptr, val, mask=mask, sem=sem, scope=scope)
