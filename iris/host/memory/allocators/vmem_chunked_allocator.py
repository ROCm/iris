# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.

"""
Chunked VMem allocator with power-of-two free lists and GC-based deallocation.

Design:
- Reserve large VA range up front (cheap, just address space)
- Map physical memory in large chunks (e.g. 256 MiB)
- Driver applies access once per chunk (not per allocation)
- Sub-allocate from chunks with bump pointer
- Power-of-two free lists for O(1) alloc/free reuse
- GC via weakref finalizers on tensor.untyped_storage()
- Free/reuse is pure bookkeeping (no driver calls, no physical remap)

Cost model:
- Init: ~170us per chunk (create + map + set_access for 1 device)
- Per-allocation: 0us (bump or free-list pop, no driver calls)
- Per-free: 0us (push to free list, no driver calls)
- Chunk growth: ~170us (rare, every chunk_size bytes)
"""

import math
import logging
import weakref
from collections import defaultdict, deque
from dataclasses import dataclass
from threading import Lock
from typing import List, Optional

import torch

from .base import BaseAllocator
from iris.drivers.base import LocalAllocation, PeerMapping
from iris.drivers.factory import DriverFactory
from iris.host.distributed.topology import (
    InterconnectLevel,
    TopologyMap,
    _detect_vendor,
)

logger = logging.getLogger("iris.host.memory.allocators.vmem_chunked_allocator")


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


def _is_power_of_two(n: int) -> bool:
    return n > 0 and (n & (n - 1)) == 0


@dataclass
class _SharedRegion:
    """Exported heap region tracked for peer refresh."""

    va: int
    size: int
    allocation: Optional[LocalAllocation] = None


class VMemChunkedAllocator(BaseAllocator):
    """
    Chunked VMem allocator with power-of-two free lists.

    Physical memory is allocated in large chunks that are mapped once and
    never remapped. Free/reuse is pure bookkeeping -- no driver calls, no
    physical remap, no peer coordination.

    Driver tier selection: this allocator picks ONE driver per rank based on
    the topology. If any peer is in the same fabric domain on a different
    host, the fabric driver is used for ALL peers including local ones.
    Mixed-tier jobs work correctly but local peers may incur a small overhead
    vs a pure intra-node driver. A future per-peer driver selection would
    require a higher-level coordinator. The chosen tier is exposed via the
    `transport_tier` property and the underlying driver via the `driver`
    attribute; both are part of the public contract for orchestration layers.

    Lifetime: this allocator's GC finalizers, attached to every allocated
    tensor's storage, hold a strong reference to the allocator itself. As a
    consequence, del allocator does NOT destroy the allocator while any tensor
    allocated by it is still live. To release resources deterministically, call
    allocator.close() explicitly, or use the allocator as a context manager:

        with VMemChunkedAllocator(...) as alloc:
            tensor = alloc.allocate(...)
            ...
        # alloc.close() called automatically here, even if tensor is still alive.

    The tensor must not be used after close(). The __del__ method is a
    best-effort backup; do not rely on it for release ordering.

    Args:
        heap_size: Initial heap size in bytes (best effort; will grow if exceeded)
        device_id: GPU device ID
        cur_rank: Current process rank
        num_ranks: Total number of ranks
        chunk_size: Size of each physical chunk in bytes (default 256 MiB)
        va_size: Total VA reservation size (default 64 GiB)
        topology: Optional cluster topology used to select local vs fabric driver
    """

    # Default chunk size: 256 MiB
    DEFAULT_CHUNK_SIZE = 256 * 1024 * 1024
    # Default VA size: 0 means auto-size (128 GiB)
    DEFAULT_VA_SIZE = 0

    def __init__(
        self,
        heap_size: int,
        device_id: int,
        cur_rank: int,
        num_ranks: int,
        chunk_size: int = DEFAULT_CHUNK_SIZE,
        va_size: int = DEFAULT_VA_SIZE,
        *,
        topology: Optional[TopologyMap] = None,
    ) -> None:
        super().__init__(heap_size, device_id, cur_rank, num_ranks)
        self.device = torch.device(f"cuda:{device_id}")
        self.lock = Lock()
        self._closed = False
        self.driver = None

        # Collections initialized first so close() is always safe to call,
        # even if a later __init__ step raises.
        self.bump = 0
        self.free_lists = defaultdict(list)
        self.alloc_sizes = {}
        self._pending_free = deque()
        self.chunks: List[LocalAllocation] = []
        self._shared_regions: List[_SharedRegion] = []
        self._peer_mappings: List[PeerMapping] = []
        self._imported_heap_mappings: List[PeerMapping] = []
        self.mapped_extent = 0
        self.base_va = 0
        self.granularity = 0

        vendor = _detect_vendor()
        if vendor == "unknown":
            raise RuntimeError("VMemChunkedAllocator: could not detect GPU vendor; no compatible driver available")

        interconnect = InterconnectLevel.INTRA_NODE
        if topology is not None:
            own_info = topology.gpu_info.get(cur_rank)
            if own_info is None:
                logger.warning(
                    "Rank %d not found in topology.gpu_info; defaulting to INTRA_NODE driver. "
                    "This may indicate a topology/rank-assignment mismatch.",
                    cur_rank,
                )
            else:
                own_domain = own_info.fabric_info.domain_key
                if own_domain:
                    for peer_rank, peer_info in topology.gpu_info.items():
                        if peer_rank == cur_rank:
                            continue
                        if peer_info.hostname != own_info.hostname and peer_info.fabric_info.domain_key == own_domain:
                            interconnect = InterconnectLevel.INTRA_RACK_FABRIC
                            logger.info(
                                "Rank %d using fabric driver: peer %d on host %s shares fabric domain %s",
                                cur_rank,
                                peer_rank,
                                peer_info.hostname,
                                own_domain,
                            )
                            break

        self.driver = DriverFactory.create_driver(vendor, interconnect)
        self.driver.initialize(device_id)
        self._interconnect = interconnect
        logger.info(
            "VMemChunkedAllocator initialized: vendor=%s, interconnect=%s, device=%d, rank=%d/%d",
            vendor,
            interconnect.name,
            device_id,
            cur_rank,
            num_ranks,
        )
        self.granularity = self.driver.get_minimum_granularity()
        if not _is_power_of_two(self.granularity):
            raise RuntimeError(
                f"VMemChunkedAllocator: driver granularity {self.granularity} "
                f"is not a power of two; bitmask alignment math will not work. "
                f"This indicates a driver bug."
            )

        # Chunk configuration -- cap at heap_size to avoid overshooting VA
        effective_chunk = min(chunk_size, max(heap_size, self.granularity))
        self.chunk_size = max(effective_chunk, self.granularity)
        # granularity is a power of two (asserted above), so the bitmask trick
        # (x + a - 1) & ~(a - 1) is safe for this alignment.
        self.chunk_size = (self.chunk_size + self.granularity - 1) & ~(self.granularity - 1)
        if not _is_power_of_two(self.chunk_size):
            raise RuntimeError(
                f"VMemChunkedAllocator: computed chunk_size {self.chunk_size} "
                f"is not a power of two. Pass a power-of-two chunk_size to the "
                f"constructor (default {self.DEFAULT_CHUNK_SIZE} is safe)."
            )

        # VA reservation -- just address space, no physical memory cost.
        # Default: 128 GiB (plenty of room for growth + imports).
        if va_size == 0:
            va_size = 128 * 1024 * 1024 * 1024  # 128 GiB
        # va_size = max(va_size, heap_size * 4) -- gives 4x headroom for growth
        # and imports. For very large heaps (>100 GiB) this can exceed sensible
        # VA budgets; if reserve_va fails, the caller should pass an explicit
        # va_size. Not capped here because the right cap depends on workload.
        self.va_size = max(va_size, heap_size * 4)
        # chunk_size is a power of two (asserted above), so bitmask alignment is safe.
        self.va_size = (self.va_size + self.chunk_size - 1) & ~(self.chunk_size - 1)
        self.base_va = self.driver.reserve_va(self.va_size, self.granularity)

        self.min_alignment = max(self.granularity, 1024)

        # Pre-allocate initial chunks to cover heap_size
        n_initial_chunks = max(1, math.ceil(heap_size / self.chunk_size))
        for _ in range(n_initial_chunks):
            self._grow_chunk()

    def _grow_chunk(self):
        """Map a new physical chunk into the VA range."""
        if self.mapped_extent + self.chunk_size > self.va_size:
            raise RuntimeError(
                f"VMemChunkedAllocator: VA space exhausted. "
                f"mapped_extent={self.mapped_extent}, "
                f"chunk_size={self.chunk_size}, va_size={self.va_size}"
            )

        target_va = self.base_va + self.mapped_extent
        alloc_kwargs = {}
        if self.driver.__class__.__name__ == "LocalHipDriver":
            alloc_kwargs = {
                "access_va": self.base_va,
                "access_size": self.mapped_extent + self.chunk_size,
            }
        allocation = self.driver.allocate_exportable(self.chunk_size, va=target_va, **alloc_kwargs)
        self.chunks.append(allocation)
        self._shared_regions.append(_SharedRegion(va=allocation.va, size=allocation.size, allocation=allocation))
        self.mapped_extent += self.chunk_size

    def _process_pending_frees(self):
        """Process pending frees from GC finalizers. Call with lock held."""
        while self._pending_free:
            offset, size_class = self._pending_free.popleft()
            self.free_lists[size_class].append(offset)

    def _free_callback(self, offset, size_class):
        """Called by weakref finalizer when a tensor's storage is GC'd.

        NOTE: this runs without holding self.lock. That is intentional -- GC
        finalizers can fire from any thread, and acquiring a lock from a
        finalizer risks deadlock with the thread the GC interrupted. Safety
        relies on:
        - deque.append being atomic under the GIL
        - stale appends after close() being benign because the deque is cleared
          and never read again.
        """
        if self._closed:
            return
        self._pending_free.append((offset, size_class))

    def get_base_address(self) -> int:
        return self.base_va

    def get_minimum_allocation_size(self) -> int:
        return self.granularity

    def get_device(self) -> torch.device:
        return self.device

    @property
    def transport_tier(self) -> InterconnectLevel:
        """Return the interconnect tier this allocator's driver operates over.

        Used by orchestration layers to choose between local FD-based and
        fabric-handle peer setup. Stable for the allocator lifetime.
        """
        return self._interconnect

    def allocate(self, num_elements: int, dtype: torch.dtype, alignment: int = 1024) -> torch.Tensor:
        with self.lock:
            if num_elements == 0:
                return torch.empty(0, dtype=dtype, device=self.device)

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

                needed = aligned_bump + aligned_size - self.mapped_extent
                if needed > 0:
                    n_new_chunks = math.ceil(needed / self.chunk_size)
                    if self.mapped_extent + n_new_chunks * self.chunk_size > self.va_size:
                        raise RuntimeError(
                            f"VMemChunkedAllocator: requested allocation needs "
                            f"{n_new_chunks} chunks ({n_new_chunks * self.chunk_size} bytes), "
                            f"but only {self.va_size - self.mapped_extent} bytes of VA remain."
                        )
                    for _ in range(n_new_chunks):
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
            tensor = full.narrow(0, 0, num_elements)

            # Attach GC weak ref for automatic free
            weakref.finalize(
                tensor.untyped_storage(),
                self._free_callback,
                offset,
                aligned_size,
            )

            return tensor

    def owns_tensor(self, tensor: torch.Tensor) -> bool:
        if not tensor.is_cuda:
            return False
        if tensor.numel() == 0:
            return True
        ptr = tensor.data_ptr()
        if self.base_va <= ptr < self.base_va + self.va_size:
            return True
        for mapping in self._peer_mappings:
            if mapping.remote_va <= ptr < mapping.remote_va + mapping.size:
                return True
        return False

    def get_allocation_chunks(self):
        """
        Get list of exported heap regions for peer sharing.

        This includes both ordinary chunk-backed heap regions and imported
        external tensors that have been permanently mapped into the heap for
        RMA-safe symmetric addressing.

        Each call invokes driver export once per returned region, which on AMD
        allocates a fresh DMA-BUF file descriptor per chunk. The caller OWNS
        those FDs and must close them after delivering them to peers (e.g. via
        SCM_RIGHTS or pidfd_getfd). Calling this method multiple times will
        leak FDs unless the caller closes the previous batch.

        This method is intended to be called once per allocator lifetime during
        peer setup. If you need to re-export after a chunk grow, cache the
        previous handle_bytes externally and only call this for new chunks.

        Returns:
            List of (chunk_index, offset, size, handle_bytes) tuples,
            where handle_bytes is the serialized peer handle for the chunk.
        """
        return self.get_allocation_chunks_since(0)

    def get_allocation_chunks_since(self, start_index: int):
        """
        Get exported heap regions at index >= start_index, with their handles.

        Used by orchestration layers that have already imported the first
        `start_index` regions and only need handles for newly-added ones.
        Each call invokes driver.export_handle once per returned chunk; for
        drivers that allocate kernel resources (FDs) at export time, this
        avoids re-exporting chunks the caller has already processed.

        Args:
            start_index: chunks at indices [0, start_index) are skipped.

        Returns:
            List of (chunk_index, offset, size, handle_bytes) tuples for
            chunks at index >= start_index. Indices are absolute (matching
            get_allocation_chunks's output), not relative to start_index.

        Raises:
            ValueError: if start_index is negative or > current chunk count.
        """
        if start_index < 0:
            raise ValueError(f"start_index must be non-negative, got {start_index}")
        if start_index > len(self._shared_regions):
            raise ValueError(f"start_index {start_index} exceeds exported region count {len(self._shared_regions)}")

        result = []
        for i in range(start_index, len(self._shared_regions)):
            region = self._shared_regions[i]
            offset = region.va - self.base_va
            if region.allocation is not None:
                handle_bytes = self.driver.export_handle(region.allocation)
            else:
                handle_bytes = self.driver.export_pointer_handle(region.va, region.size)
            result.append((i, offset, region.size, handle_bytes))
        return result

    def get_num_chunks(self):
        """Return the number of exported heap regions."""
        return len(self._shared_regions)

    def import_external_tensor(self, external_tensor: torch.Tensor) -> torch.Tensor:
        """Import an external tensor into the symmetric heap (zero-copy).

        The imported tensor shares physical memory with the original -- writes
        to one are visible in the other. On the chunked allocator path, the
        imported allocation is mapped into the heap's VA layout and retained
        until allocator.close() so peer translation remains valid for RMA.

        Raises:
            DriverNotSupported: This operation requires DMA-BUF support and is
                currently AMD-only. On NVIDIA, the local driver does not
                implement export_pointer_handle for arbitrary device pointers,
                and this method will raise.
            RuntimeError: If the input tensor is not on a CUDA/HIP device or
                is not contiguous.
        """
        with self.lock:
            if not external_tensor.is_cuda:
                raise RuntimeError("Can only import CUDA/HIP tensors")
            if not external_tensor.is_contiguous():
                raise RuntimeError("Only contiguous tensors can be imported; call .contiguous() before as_symmetric()")

            external_ptr = external_tensor.data_ptr()
            tensor_size = external_tensor.numel() * external_tensor.element_size()
            alloc_base, alloc_size = self.driver.get_address_range(external_ptr)
            offset_in_alloc = external_ptr - alloc_base
            aligned_alloc_size = (alloc_size + self.granularity - 1) & ~(self.granularity - 1)

            target_offset = (self.mapped_extent + self.granularity - 1) & ~(self.granularity - 1)
            if target_offset + aligned_alloc_size > self.va_size:
                raise RuntimeError(
                    f"VMemChunkedAllocator: imported tensor needs {aligned_alloc_size} bytes "
                    f"at offset {target_offset}, but only "
                    f"{self.va_size - target_offset} bytes of VA remain."
                )

            target_base_va = self.base_va + target_offset
            handle_bytes = self.driver.export_pointer_handle(alloc_base, alloc_size)
            import_kwargs = {}
            if self.driver.__class__.__name__ == "LocalHipDriver":
                import_kwargs = {
                    "access_va": self.base_va,
                    "access_size": target_offset + aligned_alloc_size,
                }
            mapping = self.driver.import_and_map(
                self.cur_rank,
                handle_bytes,
                aligned_alloc_size,
                va=target_base_va,
                **import_kwargs,
            )
            self._imported_heap_mappings.append(mapping)
            self._shared_regions.append(_SharedRegion(va=target_base_va, size=aligned_alloc_size, allocation=None))
            self.mapped_extent = target_offset + aligned_alloc_size
            self.bump = max(self.bump, self.mapped_extent)

            tensor_va = target_base_va + offset_in_alloc
            iface = _CUDAArrayInterface(tensor_va, tensor_size)
            tensor_bytes = torch.as_tensor(iface, device=self.device)
            return tensor_bytes.view(external_tensor.dtype).reshape(external_tensor.shape)

    def _import_release_callback(self, mapping: PeerMapping) -> None:
        """Called when an imported tensor's storage is GC'd.

        Unlike _free_callback, this finalizer DOES take self.lock. The lock,
        combined with using self._peer_mappings.remove(mapping) as the gate,
        is what makes cleanup race-free against close() and release_peer_chunk:
        only one code path can successfully remove a given mapping from the
        list, and that code path owns the cleanup_import call.

        The self._closed check before the lock is a fast-path optimization only
        -- it is NOT a correctness gate. On weakly-ordered architectures the
        finalizer thread may not see _closed=True until the lock acquire forces
        a memory barrier. That's fine: when it does acquire the lock, it will
        find the mapping already removed by close() and return via the
        ValueError path.

        Acquiring a lock from a finalizer normally risks deadlock, but here:
        - imported tensors are not allocated frequently (a few per process)
        - the lock is held only for one driver call + one list mutation
        - the hot allocate path uses a different finalizer (_free_callback)
          which does NOT take the lock, so allocate can never block this
          finalizer indirectly.
        """
        if self._closed:
            return
        with self.lock:
            try:
                self._peer_mappings.remove(mapping)
            except ValueError:
                return
            try:
                self.driver.cleanup_import(mapping)
            except Exception as exc:
                logger.warning("cleanup_import failed in finalizer: %s", exc)

    def import_peer_chunk(self, peer_rank: int, handle_bytes: bytes, size: int) -> int:
        """
        Import a serialized chunk handle from a peer rank.

        Returns the local virtual address where the chunk was mapped.
        The caller is responsible for calling release_peer_chunk when done.
        """
        with self.lock:
            mapping = self.driver.import_and_map(peer_rank, handle_bytes, size, va=None)
            self._peer_mappings.append(mapping)
            return mapping.remote_va

    def release_peer_chunk(self, remote_va: int) -> None:
        """Release a peer chunk previously imported via import_peer_chunk.

        Idempotent -- calling this on a remote_va that's already been released
        (or that was never imported via import_peer_chunk) is a no-op and logs
        at DEBUG. Do NOT use this for tensors returned by import_external_tensor;
        those remain mapped until allocator.close() so their heap offsets stay
        valid for peer RMA.
        """
        with self.lock:
            for i, mapping in enumerate(self._peer_mappings):
                if mapping.remote_va == remote_va:
                    try:
                        self.driver.cleanup_import(mapping)
                    finally:
                        self._peer_mappings.pop(i)
                    return
            logger.debug(
                "release_peer_chunk: no mapping at va=0x%x (already released or never imported via import_peer_chunk)",
                remote_va,
            )

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

        if self.driver is None:
            return

        with self.lock:
            # Clear allocator bookkeeping before releasing mappings/chunks.
            self._pending_free.clear()
            self.free_lists.clear()
            self.alloc_sizes.clear()

            try:
                torch.cuda.synchronize(self.device)
            except Exception:
                pass

            # Release imported peer mappings
            for mapping in self._peer_mappings:
                try:
                    self.driver.cleanup_import(mapping)
                except Exception:
                    pass
            self._peer_mappings.clear()

            # Release imported symmetric mappings that were inserted into the
            # heap VA layout via import_external_tensor.
            for mapping in self._imported_heap_mappings:
                try:
                    self.driver.cleanup_import(mapping)
                except Exception:
                    pass
            self._imported_heap_mappings.clear()
            self._shared_regions.clear()

            # Release locally-mapped chunks. Each chunk has _va_owned=False,
            # so cleanup_local will unmap and release the physical handle
            # but will NOT free VA
            for alloc in self.chunks:
                try:
                    self.driver.cleanup_local(alloc)
                except Exception:
                    pass
            self.chunks.clear()

            # Free the master VA reservation last.
            if self.base_va:
                try:
                    self.driver.free_va(self.base_va, self.va_size)
                except Exception:
                    pass
                self.base_va = 0

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        self.close()
        return False

    def __del__(self):
        try:
            self.close()
        except Exception:
            pass
