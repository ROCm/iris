# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.

"""
Chunked VMem allocator with power-of-two free lists and GC-based deallocation.

Design:
- Reserve large VA range up front (cheap, just address space)
- Map physical memory in large chunks (e.g. 256 MiB)
- hipMemSetAccess called once per chunk (not per allocation)
- Sub-allocate from chunks with bump pointer
- Power-of-two free lists for O(1) alloc/free reuse
- GC via weakref finalizers on tensor.untyped_storage()
- Free/reuse is pure bookkeeping (no HIP calls, no physical remap)

Cost model:
- Init: ~170us per chunk (create + map + set_access for 1 device)
- Per-allocation: 0us (bump or free-list pop, no HIP calls)
- Per-free: 0us (push to free list, no HIP calls)
- Chunk growth: ~170us (rare, every chunk_size bytes)
"""

import math
import os
import weakref
from collections import defaultdict, deque
from threading import Lock

import torch

from .base import BaseAllocator
from ..hip import (
    get_allocation_granularity,
    get_address_range,
    export_dmabuf_handle,
    mem_create,
    mem_address_reserve,
    mem_map,
    mem_unmap,
    mem_address_free,
    mem_release,
    mem_set_access,
    mem_import_from_shareable_handle,
    hipMemAccessDesc,
    hipMemLocationTypeDevice,
    hipMemAccessFlagsProtReadWrite,
)


# Module-level CUDAArrayInterface to avoid repeated class creation
class _CUDAArrayInterface:
    __slots__ = ("ptr", "nbytes")

    def __init__(self, ptr, nbytes):
        self.ptr = ptr
        self.nbytes = nbytes

    @property
    def __cuda_array_interface__(self):
        return {
            "shape": (self.nbytes,),
            "typestr": "|u1",
            "data": (self.ptr, False),
            "version": 3,
        }


# Cached element sizes to avoid torch.tensor([], dtype=...).element_size() overhead
_DTYPE_ELEMENT_SIZE = {}


def _element_size(dtype):
    if dtype not in _DTYPE_ELEMENT_SIZE:
        _DTYPE_ELEMENT_SIZE[dtype] = torch.tensor([], dtype=dtype).element_size()
    return _DTYPE_ELEMENT_SIZE[dtype]


def _next_power_of_two(n):
    """Return the smallest power of two >= n."""
    if n <= 0:
        return 1
    return 1 << (n - 1).bit_length()


class VMemChunkedAllocator(BaseAllocator):
    """
    Chunked VMem allocator with power-of-two free lists.

    Physical memory is allocated in large chunks that are mapped once and
    never remapped. Free/reuse is pure bookkeeping -- no HIP calls, no
    physical remap, no peer coordination.

    Args:
        heap_size: Initial heap size in bytes (best effort; will grow if exceeded)
        device_id: GPU device ID
        cur_rank: Current process rank
        num_ranks: Total number of ranks
        chunk_size: Size of each physical chunk in bytes (default 256 MiB)
        va_size: Total VA reservation size (default 64 GiB)
    """

    # Default chunk size: 256 MiB
    DEFAULT_CHUNK_SIZE = 256 * 1024 * 1024
    # Default VA size: 0 means auto-size (8x heap_size, min 256 MiB)
    DEFAULT_VA_SIZE = 0

    def __init__(
        self,
        heap_size: int,
        device_id: int,
        cur_rank: int,
        num_ranks: int,
        chunk_size: int = DEFAULT_CHUNK_SIZE,
        va_size: int = DEFAULT_VA_SIZE,
    ):
        super().__init__(heap_size, device_id, cur_rank, num_ranks)
        self.device = torch.device(f"cuda:{device_id}")
        self.lock = Lock()
        self.granularity = get_allocation_granularity(device_id)

        # Chunk configuration -- cap at heap_size to avoid overshooting VA
        effective_chunk = min(chunk_size, max(heap_size, self.granularity))
        self.chunk_size = max(effective_chunk, self.granularity)
        # Align chunk_size to granularity
        self.chunk_size = (self.chunk_size + self.granularity - 1) & ~(self.granularity - 1)

        # VA reservation -- larger than heap_size for growth + imports headroom
        # Default: 16x heap_size (min 32 MiB) for tests with small heaps
        if va_size == 0:
            va_size = max(32 * 1024 * 1024, heap_size * 16)
        self.va_size = max(va_size, heap_size * 4)
        # Align VA size to chunk_size
        self.va_size = (self.va_size + self.chunk_size - 1) & ~(self.chunk_size - 1)
        self.base_va = mem_address_reserve(self.va_size, self.granularity, 0)

        # Access descriptor for local device only (in torchrun, each process owns 1 GPU)
        self.local_access_desc = hipMemAccessDesc()
        self.local_access_desc.location.type = hipMemLocationTypeDevice
        self.local_access_desc.location.id = device_id
        self.local_access_desc.flags = hipMemAccessFlagsProtReadWrite

        # Chunk tracking: list of (handle, va, size)
        self.chunks = []
        # Import chunks are tracked separately -- they cannot be re-exported
        # for peer sharing (they were imported from DMA-BUF, not VMem-created).
        self._import_chunks = []
        self.mapped_extent = 0  # total bytes of physical mapped into VA

        # Pre-allocate initial chunks to cover heap_size
        n_initial_chunks = max(1, math.ceil(heap_size / self.chunk_size))
        for _ in range(n_initial_chunks):
            self._grow_chunk()

        # Bump allocator state
        self.bump = 0
        # Minimum alignment for sub-allocations (at least granularity, at least 1024)
        self.min_alignment = max(self.granularity, 1024)

        # Power-of-two free lists: size_class -> [va_offset, ...]
        self.free_lists = defaultdict(list)
        # Track allocation sizes for free: va_offset -> size_class
        self.alloc_sizes = {}

        # Pending frees from GC finalizers (thread-safe deque)
        self._pending_free = deque()

        # Track weakrefs to prevent premature GC
        self._weak_refs = set()

        # Closed flag
        self._closed = False

    def _grow_chunk(self):
        """
        Map a new physical chunk into the VA range.

        This is the only operation that makes HIP calls during normal operation.
        It happens rarely (every chunk_size bytes of net-new memory).
        """
        if self.mapped_extent + self.chunk_size > self.va_size:
            raise RuntimeError(
                f"VMemChunkedAllocator: VA space exhausted. "
                f"mapped_extent={self.mapped_extent}, chunk_size={self.chunk_size}, "
                f"va_size={self.va_size}"
            )

        handle = mem_create(self.chunk_size, self.device_id)
        va = self.base_va + self.mapped_extent
        mem_map(va, self.chunk_size, 0, handle)
        mem_set_access(va, self.chunk_size, self.local_access_desc)

        self.chunks.append((handle, va, self.chunk_size))
        self.mapped_extent += self.chunk_size

    def _process_pending_frees(self):
        """Process pending frees from GC finalizers. Call with lock held."""
        while self._pending_free:
            offset, size_class = self._pending_free.popleft()
            self.free_lists[size_class].append(offset)

    def _free_callback(self, offset, size_class, ref_id):
        """Called by weakref finalizer when a tensor's storage is GC'd."""
        if self._closed:
            return
        self._pending_free.append((offset, size_class))
        # Remove the weak ref tracker
        self._weak_refs.discard(ref_id)

    def get_base_address(self) -> int:
        return self.base_va

    def get_minimum_allocation_size(self) -> int:
        return self.granularity

    def get_device(self) -> torch.device:
        return self.device

    def allocate(self, num_elements: int, dtype: torch.dtype, alignment: int = 1024) -> torch.Tensor:
        with self.lock:
            self._process_pending_frees()

            elem_size = _element_size(dtype)
            size_bytes = num_elements * elem_size
            # Minimum allocation is one granule
            size_bytes = max(size_bytes, self.granularity)
            # Round to next power of two for free-list bucketing
            size_class = _next_power_of_two(size_bytes)
            # Ensure alignment to granularity
            aligned_size = max(size_class, self.min_alignment)
            aligned_size = (aligned_size + self.granularity - 1) & ~(self.granularity - 1)

            # Try free list first
            if self.free_lists[aligned_size]:
                offset = self.free_lists[aligned_size].pop()
            else:
                # Bump allocate
                # Align the bump pointer
                aligned_bump = (self.bump + self.min_alignment - 1) & ~(self.min_alignment - 1)

                # Grow if needed
                while aligned_bump + aligned_size > self.mapped_extent:
                    self._grow_chunk()

                offset = aligned_bump
                self.bump = aligned_bump + aligned_size

            # Track for free
            self.alloc_sizes[offset] = aligned_size

            # Create tensor via CUDAArrayInterface
            va = self.base_va + offset
            interface_size = (aligned_size // elem_size) * elem_size
            iface = _CUDAArrayInterface(va, interface_size)
            tensor_bytes = torch.as_tensor(iface, device=self.device)
            full = tensor_bytes.view(dtype)
            if num_elements == 0:
                tensor = full.narrow(0, 1, 0)
            else:
                tensor = full.narrow(0, 0, num_elements)

            # Attach GC weak ref for automatic free
            ref_id = id(tensor.untyped_storage())
            weakref.finalize(
                tensor.untyped_storage(),
                self._free_callback,
                offset,
                aligned_size,
                ref_id,
            )
            self._weak_refs.add(ref_id)

            return tensor

    def owns_tensor(self, tensor: torch.Tensor) -> bool:
        if not tensor.is_cuda:
            return False
        if tensor.numel() == 0:
            return True
        ptr = tensor.data_ptr()
        return self.base_va <= ptr < self.base_va + self.va_size

    def get_allocation_chunks(self):
        """
        Get list of physical chunks for peer sharing.

        Returns:
            List of (chunk_index, offset, size, handle) tuples.
        """
        result = []
        for i, (handle, va, size) in enumerate(self.chunks):
            offset = va - self.base_va
            result.append((i, offset, size, handle))
        return result

    def get_num_chunks(self):
        """Return the number of mapped chunks."""
        return len(self.chunks)

    def import_external_tensor(self, external_tensor: torch.Tensor) -> torch.Tensor:
        """
        Import an external tensor into the symmetric heap via DMA-BUF.

        The imported tensor shares physical memory with the original --
        writes to one are visible in the other (zero-copy).

        The import is placed beyond the current bump pointer in the VA range.
        """
        with self.lock:
            if not external_tensor.is_cuda:
                raise RuntimeError("Can only import CUDA tensors")
            if not external_tensor.is_contiguous():
                raise RuntimeError("Only contiguous tensors can be imported; call .contiguous() before as_symmetric()")

            external_ptr = external_tensor.data_ptr()
            alloc_base, alloc_size = get_address_range(external_ptr)
            offset_in_alloc = external_ptr - alloc_base
            aligned_alloc_size = (alloc_size + self.granularity - 1) & ~(self.granularity - 1)

            # Place import beyond current bump, aligned to granularity
            import_offset = (self.bump + self.granularity - 1) & ~(self.granularity - 1)

            # Grow chunks if needed to cover the import region
            while import_offset + aligned_alloc_size > self.mapped_extent:
                self._grow_chunk()

            # We need to unmap the existing chunk region at import_offset
            # and remap with the imported DMA-BUF handle instead.
            # Actually, the import region overlaps with a pre-mapped chunk.
            # We need to unmap that portion and remap with the imported handle.
            #
            # Simpler approach: use a separate VA range for imports that
            # doesn't overlap with the chunk region. But that breaks the
            # single-VA-range requirement.
            #
            # Correct approach: reserve import space in the VA range BEYOND
            # the chunk region. The import region uses unmapped VA (no chunk
            # was mapped there). We just map the imported handle directly.

            # Recalculate: place import at the END of mapped_extent
            # (or beyond, in unmapped VA space)
            import_offset = self.mapped_extent
            # Grow VA if needed (but not physical chunks)
            if import_offset + aligned_alloc_size > self.va_size:
                raise RuntimeError(
                    f"VMemChunkedAllocator: VA space exhausted for import. "
                    f"import_offset={import_offset}, size={aligned_alloc_size}, "
                    f"va_size={self.va_size}"
                )

            # Export external tensor as DMA-BUF
            dmabuf_fd, export_base, export_size = export_dmabuf_handle(alloc_base, alloc_size)
            try:
                aligned_export_size = (export_size + self.granularity - 1) & ~(self.granularity - 1)

                # Import the DMA-BUF as a VMem handle
                imported_handle = mem_import_from_shareable_handle(dmabuf_fd)
            finally:
                os.close(dmabuf_fd)

            # Map at import offset in our VA range
            target_va = self.base_va + import_offset
            mem_map(target_va, aligned_export_size, 0, imported_handle)
            # Imported DMA-BUF handles from PyTorch's allocator may already
            # have device access set.  hipMemSetAccess can fail with
            # "invalid argument" on such handles, so treat the error as
            # non-fatal — the mapping itself is sufficient.
            try:
                mem_set_access(target_va, aligned_export_size, self.local_access_desc)
            except RuntimeError:
                pass

            # Track as an import chunk (cleanup only, NOT peer-shared)
            self._import_chunks.append((imported_handle, target_va, aligned_export_size))
            # Advance mapped_extent past the import
            self.mapped_extent = import_offset + aligned_export_size
            # Also advance bump past the import so future allocs don't collide
            self.bump = self.mapped_extent

            # Create tensor view at the correct offset within the import
            tensor_va = target_va + offset_in_alloc
            tensor_size = external_tensor.numel() * external_tensor.element_size()
            iface = _CUDAArrayInterface(tensor_va, tensor_size)
            tensor_bytes = torch.as_tensor(iface, device=self.device)
            imported_tensor = tensor_bytes.view(external_tensor.dtype).reshape(external_tensor.shape)

            return imported_tensor

    def get_stats(self):
        """Return allocator statistics."""
        total_free = sum(len(v) for v in self.free_lists.values())
        free_bytes = sum(size_class * len(offsets) for size_class, offsets in self.free_lists.items())
        return {
            "num_chunks": len(self.chunks),
            "mapped_bytes": self.mapped_extent,
            "bump": self.bump,
            "num_active_allocs": len(self.alloc_sizes),
            "num_free_blocks": total_free,
            "free_bytes": free_bytes,
            "va_size": self.va_size,
            "chunk_size": self.chunk_size,
            "granularity": self.granularity,
        }

    def close(self):
        """Release all VMem resources."""
        if self._closed:
            return
        self._closed = True

        with self.lock:
            # Disable finalizers
            self._weak_refs.clear()
            self._pending_free.clear()
            self.free_lists.clear()
            self.alloc_sizes.clear()

            # Synchronize GPU before unmapping -- async kernels (.zero_(), .fill_())
            # may still be accessing mapped memory. Unmapping while kernels are
            # in-flight causes hipErrorUnknown on subsequent GPU operations.
            try:
                torch.cuda.synchronize(self.device)
            except Exception:
                pass

            # Unmap and release all chunks (regular + imported)
            for handle, va, size in self.chunks + self._import_chunks:
                try:
                    mem_unmap(va, size)
                except Exception:
                    pass
                try:
                    mem_release(handle)
                except Exception:
                    pass
            self.chunks.clear()
            self._import_chunks.clear()

            # Free VA range
            if self.base_va:
                try:
                    mem_address_free(self.base_va, self.va_size)
                except Exception:
                    pass
                self.base_va = 0

    def __del__(self):
        self.close()
