# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.

"""
Power-of-two VMem allocator using HIP's virtual memory management APIs.

This allocator provides efficient reuse of virtual memory allocations by
rounding all requests up to the next power-of-two size class and maintaining
per-class free lists for O(1) allocation and deallocation.
"""

import os
import weakref
from collections import deque
from typing import Dict, List, Tuple
from threading import Lock

import torch

from .base import BaseAllocator
from ..hip import (
    get_allocation_granularity,
    get_address_range,
    export_dmabuf_handle,
    mem_import_from_shareable_handle,
    mem_create,
    mem_address_reserve,
    mem_map,
    mem_unmap,
    mem_address_free,
    mem_release,
    mem_set_access,
    hipMemAccessDesc,
    hipMemLocationTypeDevice,
    hipMemAccessFlagsProtReadWrite,
)


def _next_pow2(n: int) -> int:
    """Round n up to the next power of two (>= 1)."""
    if n <= 1:
        return 1
    return 1 << (n - 1).bit_length()


# Module-level element-size cache: avoids creating a temporary tensor on every allocation.
_DTYPE_ELEMENT_SIZE: Dict[torch.dtype, int] = {}


def _element_size(dtype: torch.dtype) -> int:
    """Return the element size in bytes for *dtype*, using a module-level cache."""
    if dtype not in _DTYPE_ELEMENT_SIZE:
        _DTYPE_ELEMENT_SIZE[dtype] = torch.empty((), dtype=dtype).element_size()
    return _DTYPE_ELEMENT_SIZE[dtype]


class _CUDAArrayInterface:
    """
    Minimal ``__cuda_array_interface__`` wrapper.

    Lets ``torch.as_tensor`` create a tensor view over a raw device-memory
    pointer without going through PyTorch's caching allocator.
    """

    __slots__ = ("ptr", "nbytes", "device")

    def __init__(self, ptr: int, nbytes: int, device: torch.device) -> None:
        self.ptr = ptr
        self.nbytes = nbytes
        self.device = device

    @property
    def __cuda_array_interface__(self) -> dict:
        return {
            "shape": (self.nbytes,),
            "typestr": "|u1",
            "data": (self.ptr, False),
            "version": 3,
        }


class VMemPow2Allocator(BaseAllocator):
    """
    Power-of-two virtual memory allocator using HIP's VMem APIs.

    All allocation requests are rounded up to the nearest power-of-two size
    class (minimum: HIP allocation granularity).  Freed blocks are returned to
    per-class free lists.  When a free-listed VA is reused, the old physical
    handle is released (``mem_unmap`` + ``mem_release``) and fresh physical
    memory is mapped in its place (``mem_create`` + ``mem_map``).

    Physical memory is therefore renewed at **reuse time** rather than at
    free time.  This design respects the ROCm constraint that
    ``hipMemSetAccess`` must be called cumulatively from ``base_va``
    (see rocm-systems#2667): the VA range always remains contiguous, so the
    cumulative access call never encounters unmapped gaps.

    A ``weakref`` finalizer is registered on every returned tensor's storage
    so that blocks are automatically returned to the free list when the last
    view of a tensor is garbage collected, without requiring explicit calls to
    :meth:`free`.

    .. note::
        Callers that release a tensor whose memory may still be in use by an
        in-flight CUDA kernel should call ``torch.cuda.synchronize()`` before
        dropping the last reference (or before calling :meth:`free`) to avoid
        races during the physical remap that happens on next reuse.

    Args:
        heap_size:   Total virtual address space to reserve, in bytes.
        device_id:   HIP/CUDA device index.
        rank:        Current process rank.
        world_size:  Total number of ranks.
    """

    def __init__(
        self,
        heap_size: int,
        device_id: int,
        rank: int,
        world_size: int,
    ):
        super().__init__(heap_size, device_id, rank, world_size)
        self.device = torch.device(f"cuda:{device_id}")
        self.lock = Lock()

        # HIP allocation granularity (always a power of two, e.g. 2 MiB on MI300X).
        self.granularity = get_allocation_granularity(self.device_id)
        self.min_size_class: int = self.granularity

        # Align the heap to the granularity.
        self.aligned_heap_size = (heap_size + self.granularity - 1) & ~(self.granularity - 1)
        self.va_size = self.aligned_heap_size
        self.base_va: int = mem_address_reserve(self.va_size, self.granularity, 0)

        # Bootstrap: map one minimal segment so we always have something mapped.
        self.minimal_size: int = self.min_size_class
        self.minimal_handle = mem_create(self.minimal_size, self.device_id)
        mem_map(self.base_va, self.minimal_size, 0, self.minimal_handle)

        # Access descriptors: allow read/write from every peer device.
        self.access_descs: List[hipMemAccessDesc] = []
        for peer_device_id in range(world_size):
            desc = hipMemAccessDesc()
            desc.location.type = hipMemLocationTypeDevice
            desc.location.id = peer_device_id
            desc.flags = hipMemAccessFlagsProtReadWrite
            self.access_descs.append(desc)

        # ROCm: mem_set_access must be called cumulatively from base_va
        # (see rocm-systems#2667).  We therefore always call it as
        # mem_set_access(base_va, cumulative_size, ...) so the range is
        # always contiguous from the reservation start.
        self.cumulative_mapped_size: int = self.minimal_size
        mem_set_access(self.base_va, self.cumulative_mapped_size, self.access_descs)

        # Physical-segment tracking.
        # Maps VA-offset -> (size, is_imported, handle, va).
        # Segments remain physically mapped at all times (even when on the free list).
        self.allocations: Dict[int, Tuple] = {}
        # Ordered list of (offset, size) for get_allocation_segments().
        self.allocation_order: List[Tuple[int, int]] = []
        self._track_allocation(0, self.minimal_size, False, self.minimal_handle, self.base_va)

        # Per-offset generation counter.  Incremented every time a VA block is
        # remapped with fresh physical memory (i.e., each reuse from the free
        # list).  The symmetric heap uses (offset, size, generation) as the
        # deduplication key so peers re-import when physical backing changes.
        self._segment_generation: Dict[int, int] = {0: 0}

        # Next available VA offset for a brand-new physical segment.
        self.current_offset: int = self.minimal_size

        # Free lists: size_class (power-of-two bytes) -> [(offset, va), ...]
        # Physical memory at these VAs is STILL MAPPED; it will be replaced
        # when the VA is next popped from the list (_remap_free_block).
        self.free_lists: Dict[int, List[Tuple[int, int]]] = {}

        # Logical-allocation tracking: va -> size_class (needed by free()).
        self.logical_allocations: Dict[int, int] = {}

        # Weak-reference finalizers: va -> weakref.finalize object.
        # Stored so we can detach them on explicit free() to prevent double-free.
        # NOTE: finalizers are attached to tensor.untyped_storage(), not to the
        # Python tensor wrapper, so they survive tensor.reshape() / other view ops.
        self._finalizers: Dict[int, weakref.finalize] = {}

        # Pending GC-triggered frees.  The GC callback appends here instead of
        # acquiring the lock directly, which would deadlock if GC fires while
        # the lock is already held on the same thread.  Entries are processed at
        # the start of allocate() and free().
        # Thread-safety note: ``deque.append`` and ``deque.popleft`` are atomic
        # in CPython due to the GIL.  This is a documented property of
        # ``collections.deque`` for single-element operations and is relied on
        # here to allow GC finalizers (running without the allocator lock) to
        # safely enqueue work for the owning thread.
        self._pending_free: deque = deque()

        self.world_size = world_size

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _track_allocation(self, offset: int, size: int, is_imported: bool, handle, va: int) -> None:
        """Record a physical segment for cleanup and segmented DMA-BUF export."""
        self.allocations[offset] = (size, is_imported, handle, va)
        self.allocation_order.append((offset, size))

    def _size_class(self, size_bytes: int) -> int:
        """Return the smallest power-of-two size class >= size_bytes and >= granularity."""
        return max(_next_pow2(max(size_bytes, 1)), self.min_size_class)

    def _map_new_segment(self, size_class: int) -> Tuple[int, int]:
        """
        Map a fresh physical segment of *size_class* bytes at the next
        available VA offset and extend cumulative access.

        VA offsets are aligned to the HIP granularity so that consecutive
        segments remain contiguous, which is required by the cumulative
        ``hipMemSetAccess`` call.

        Returns:
            ``(offset, va)`` of the newly mapped segment.

        Raises:
            RuntimeError: If the heap VA space is exhausted.
        """
        aligned_offset = (self.current_offset + self.granularity - 1) & ~(self.granularity - 1)

        if aligned_offset + size_class > self.aligned_heap_size:
            raise RuntimeError(
                f"VMemPow2Allocator: out of VA space. "
                f"Need {size_class} bytes at offset {aligned_offset}, "
                f"heap size is {self.aligned_heap_size}, "
                f"current offset is {self.current_offset}."
            )

        va = self.base_va + aligned_offset
        handle = mem_create(size_class, self.device_id)
        mem_map(va, size_class, 0, handle)

        # Extend cumulative access to include the new segment.
        new_cumulative = aligned_offset + size_class
        if new_cumulative > self.cumulative_mapped_size:
            self.cumulative_mapped_size = new_cumulative
            mem_set_access(self.base_va, self.cumulative_mapped_size, self.access_descs)

        self._track_allocation(aligned_offset, size_class, False, handle, va)
        self._segment_generation[aligned_offset] = 0
        self.current_offset = aligned_offset + size_class
        return aligned_offset, va

    def _remap_free_block(self, offset: int, va: int, size_class: int) -> None:
        """
        Refresh the physical backing of a free-listed VA block.

        The *old* physical handle is released (``mem_unmap`` + ``mem_release``)
        and fresh physical memory is created (``mem_create`` + ``mem_map``) at
        the same VA.  Cumulative access is re-set for the full range after the
        physical swap so the remapped segment is accessible.

        The generation counter for *offset* is incremented so the symmetric
        heap detects the physical change and re-imports the segment on peers.
        """
        alloc_info = self.allocations.get(offset)
        if alloc_info is not None:
            _old_size, _is_imported, old_handle, _old_va = alloc_info
            if old_handle is not None:
                mem_unmap(_old_va, _old_size)
                mem_release(old_handle)

        new_handle = mem_create(size_class, self.device_id)
        mem_map(va, size_class, 0, new_handle)

        # Re-set cumulative access after the physical swap (access is tied to
        # the physical mapping and must be restored after unmap/remap).
        mem_set_access(self.base_va, self.cumulative_mapped_size, self.access_descs)

        self.allocations[offset] = (size_class, False, new_handle, va)
        self._segment_generation[offset] = self._segment_generation.get(offset, 0) + 1

    def _process_pending_frees(self) -> None:
        """
        Process GC-triggered frees that were queued to avoid lock re-entry.

        Must be called while holding ``self.lock``.
        """
        while self._pending_free:
            va, size_class = self._pending_free.popleft()
            if va not in self.logical_allocations:
                continue  # already freed manually
            del self.logical_allocations[va]
            self._finalizers.pop(va, None)
            offset = va - self.base_va
            self.free_lists.setdefault(size_class, []).append((offset, va))

    def _register_finalizer(self, tensor: torch.Tensor, va: int, size_class: int) -> None:
        """Register a weak-reference GC finalizer on the tensor's storage.

        The finalizer is attached to ``tensor.untyped_storage()`` (the shared
        C++ storage object) rather than to the Python tensor wrapper.  This is
        essential because callers routinely create new wrappers over the same
        storage (e.g. via ``.reshape()``): if the finalizer were on the wrapper
        it would fire as soon as the first wrapper was discarded, even while
        other wrappers still hold the storage alive.  Attaching to the storage
        ensures the block is only freed when *every* view has been released.
        """
        allocator_ref = weakref.ref(self)

        def _gc_free() -> None:
            alloc = allocator_ref()
            if alloc is None:
                return
            # Enqueue rather than locking directly to avoid deadlock when GC
            # fires inside a locked section on the same thread.
            alloc._pending_free.append((va, size_class))

        self._finalizers[va] = weakref.finalize(tensor.untyped_storage(), _gc_free)

    def _make_tensor_view(self, va: int, size_class: int, num_elements: int, dtype: torch.dtype) -> torch.Tensor:
        """Create a PyTorch tensor view over the given VA-backed device memory."""
        elem_sz = _element_size(dtype)
        interface_size = (size_class // elem_sz) * elem_sz
        cuda_array = _CUDAArrayInterface(va, interface_size, self.device)
        tensor_bytes = torch.as_tensor(cuda_array, device=self.device)
        full = tensor_bytes.view(dtype)
        if num_elements == 0:
            return full.narrow(0, 1, 0)
        return full.narrow(0, 0, num_elements)

    # ------------------------------------------------------------------
    # BaseAllocator interface
    # ------------------------------------------------------------------

    def get_base_address(self) -> int:
        """Return the base virtual address of this allocator's VA range."""
        return self.base_va

    def get_minimum_allocation_size(self) -> int:
        """Minimum allocation size in bytes (one size-class / granule)."""
        return self.granularity

    def allocate(self, num_elements: int, dtype: torch.dtype, alignment: int = 1024) -> torch.Tensor:
        """
        Allocate a tensor on the power-of-two symmetric heap.

        If a block of the required size class is on the free list it is reused
        (old physical memory released, fresh physical memory mapped at the same
        VA); otherwise a new segment is mapped at the next available offset.

        Args:
            num_elements: Number of tensor elements.
            dtype:        PyTorch data type.
            alignment:    Ignored; alignment is guaranteed by the size class.

        Returns:
            A PyTorch tensor of shape ``(num_elements,)``.

        Raises:
            RuntimeError: If the heap VA space is exhausted.
        """
        with self.lock:
            self._process_pending_frees()

            elem_sz = _element_size(dtype)
            size_bytes = num_elements * elem_sz
            size_class = self._size_class(size_bytes)

            # Try the free list first; otherwise map a new segment.
            if size_class in self.free_lists and self.free_lists[size_class]:
                offset, va = self.free_lists[size_class].pop()
                self._remap_free_block(offset, va, size_class)
            else:
                offset, va = self._map_new_segment(size_class)

            self.logical_allocations[va] = size_class
            tensor = self._make_tensor_view(va, size_class, num_elements, dtype)
            self._register_finalizer(tensor, va, size_class)
            return tensor

    def free(self, tensor: torch.Tensor) -> None:
        """
        Return a tensor's VA block to the free list.

        The block remains physically mapped so that the VA range stays
        contiguous (required for the cumulative ``hipMemSetAccess`` call).
        Physical memory is released and renewed when the block is next
        reused from the free list.

        Zero-element tensors are silently ignored.

        Args:
            tensor: A tensor previously returned by :meth:`allocate`.

        Raises:
            ValueError: If the tensor was not allocated by this allocator or
                        was already freed.
        """
        if tensor.numel() == 0:
            return

        with self.lock:
            self._process_pending_frees()

            va = tensor.data_ptr()
            if va not in self.logical_allocations:
                raise ValueError(
                    f"VMemPow2Allocator.free(): tensor at VA 0x{va:x} was not "
                    "allocated by this allocator (or was already freed)."
                )
            size_class = self.logical_allocations.pop(va)

            # Detach the GC finalizer to prevent double-free.
            fin = self._finalizers.pop(va, None)
            if fin is not None:
                fin.detach()

            offset = va - self.base_va
            self.free_lists.setdefault(size_class, []).append((offset, va))

    def get_device(self) -> torch.device:
        """Return the PyTorch device for this allocator."""
        return self.device

    def owns_tensor(self, tensor: torch.Tensor) -> bool:
        """
        Return True if *tensor*'s data pointer lies within this heap's VA range.

        The check is purely address-based; zero-element tensors are checked by
        pointer rather than being unconditionally claimed as owned, which would
        incorrectly claim externally-created zero-element tensors.

        Args:
            tensor: PyTorch tensor to check.
        """
        if not tensor.is_cuda:
            return False
        ptr = tensor.data_ptr()
        # data_ptr() returns 0 for tensors that have no storage (e.g. meta tensors)
        # or for certain zero-element tensors on some backends.  Such tensors are
        # never part of this allocator's heap.
        if ptr == 0:
            return False
        return self.base_va <= ptr < self.base_va + self.aligned_heap_size

    # ------------------------------------------------------------------
    # Symmetric-heap segment API (used by SymmetricHeap.refresh_peer_access)
    # ------------------------------------------------------------------

    def get_allocation_segments(self) -> List[Tuple[int, int, int, int]]:
        """
        Return the ordered list of physical segments for DMA-BUF export.

        All tracked segments are included (both live and free-listed), since
        free-listed blocks remain physically mapped.

        Each element is ``(offset, size, va, generation)`` where *generation*
        is a monotonically increasing counter bumped each time the block is
        remapped with fresh physical memory.  The symmetric heap uses
        ``(offset, size, generation)`` as the de-duplication key so that
        remapped segments are recognised as new and peer ranks re-import them.

        Returns:
            List of ``(offset, size, va, generation)`` tuples in allocation order.
        """
        segments = []
        for offset, size in self.allocation_order:
            va = self.base_va + offset
            generation = self._segment_generation.get(offset, 0)
            segments.append((offset, size, va, generation))
        return segments

    # ------------------------------------------------------------------
    # as_symmetric() support
    # ------------------------------------------------------------------

    def import_external_tensor(self, external_tensor: torch.Tensor) -> torch.Tensor:
        """
        Import an external PyTorch tensor into the symmetric heap.

        The returned tensor **shares physical memory** with the original;
        changes to one are immediately visible in the other.

        Args:
            external_tensor: A contiguous CUDA tensor allocated by PyTorch.

        Returns:
            A tensor view in the symmetric heap that shares memory with
            *external_tensor*.

        Raises:
            RuntimeError: If the tensor is not a contiguous CUDA tensor or
                          the heap VA space is exhausted.
        """
        with self.lock:
            if not external_tensor.is_cuda:
                raise RuntimeError("VMemPow2Allocator: can only import CUDA tensors.")
            if not external_tensor.is_contiguous():
                raise RuntimeError(
                    "VMemPow2Allocator: only contiguous tensors can be imported; "
                    "call .contiguous() before as_symmetric()."
                )

            external_ptr = external_tensor.data_ptr()
            alloc_base, alloc_size = get_address_range(external_ptr)
            offset_in_alloc = external_ptr - alloc_base

            # Export first so we know the actual export_size before the OOM check.
            dmabuf_fd, _export_base, export_size = export_dmabuf_handle(alloc_base, alloc_size)
            aligned_export_size = (export_size + self.granularity - 1) & ~(self.granularity - 1)
            aligned_offset = (self.current_offset + self.granularity - 1) & ~(self.granularity - 1)

            if aligned_offset + aligned_export_size > self.aligned_heap_size:
                os.close(dmabuf_fd)
                raise RuntimeError(
                    f"VMemPow2Allocator: out of VA space for import. "
                    f"Need {aligned_export_size} bytes at offset {aligned_offset}, "
                    f"heap size is {self.aligned_heap_size}."
                )

            try:
                imported_handle = mem_import_from_shareable_handle(dmabuf_fd)
            finally:
                os.close(dmabuf_fd)

            target_va = self.base_va + aligned_offset
            mem_map(target_va, aligned_export_size, 0, imported_handle)

            new_cumulative = aligned_offset + aligned_export_size
            if new_cumulative > self.cumulative_mapped_size:
                self.cumulative_mapped_size = new_cumulative
                mem_set_access(self.base_va, self.cumulative_mapped_size, self.access_descs)

            tensor_va = target_va + offset_in_alloc
            self._track_allocation(aligned_offset, aligned_export_size, True, imported_handle, target_va)
            self._segment_generation[aligned_offset] = 0
            self.current_offset = aligned_offset + aligned_export_size

            tensor_size = external_tensor.numel() * external_tensor.element_size()
            cuda_array = _CUDAArrayInterface(tensor_va, tensor_size, self.device)
            tensor_bytes = torch.as_tensor(cuda_array, device=self.device)
            return tensor_bytes.view(external_tensor.dtype).reshape(external_tensor.shape)

    # ------------------------------------------------------------------
    # Resource management
    # ------------------------------------------------------------------

    def close(self) -> None:
        """Release all VMem resources (unmap, release handles, free VA range)."""
        if getattr(self, "_closed", False):
            return

        with self.lock:
            # Detach all GC finalizers so they cannot fire after close().
            for fin in self._finalizers.values():
                fin.detach()
            self._finalizers.clear()
            self._pending_free.clear()

            # Release all physical segments (both live and free-listed).
            for offset, alloc_info in self.allocations.items():
                _size, _is_imported, handle, va = alloc_info
                if handle is not None:
                    mem_unmap(va, _size)
                    mem_release(handle)

            self.allocations.clear()
            self.free_lists.clear()
            self.logical_allocations.clear()

            if getattr(self, "base_va", 0):
                mem_address_free(self.base_va, self.va_size)
                self.base_va = 0

            self._closed = True

    def __del__(self) -> None:
        """Cleanup VMem resources on garbage collection."""
        self.close()

    # ------------------------------------------------------------------
    # Diagnostics
    # ------------------------------------------------------------------

    def stats(self) -> dict:
        """
        Return a snapshot of allocator statistics.

        Returns:
            A dict with keys:

            * ``heap_size``            – requested heap size in bytes
            * ``aligned_heap_size``    – actual VA reservation in bytes
            * ``granularity``          – HIP allocation granularity in bytes
            * ``current_offset``       – bytes consumed from VA space
            * ``num_segments``         – number of physical segments ever mapped
            * ``num_live_allocations`` – logical allocations currently in use
            * ``free_list_counts``     – dict of {size_class: count} for free lists
        """
        with self.lock:
            return {
                "heap_size": self.heap_size,
                "aligned_heap_size": self.aligned_heap_size,
                "granularity": self.granularity,
                "current_offset": self.current_offset,
                "num_segments": len(self.allocation_order),
                "num_live_allocations": len(self.logical_allocations),
                "free_list_counts": {sc: len(bl) for sc, bl in self.free_lists.items() if bl},
            }
