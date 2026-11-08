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
- A block becomes reusable only once every rank has freed it (see reuse()),
  so allocation offsets stay identical across ranks whenever each rank's GC runs

Cost model:
- Init: ~170us per chunk (create + map + set_access for 1 device)
- Per-allocation: 0us (bump or free-list pop, no driver calls)
- Per-free: 0us (push to a local released list, no driver calls)
- Chunk growth: ~170us (rare, every chunk_size bytes)
"""

import math
import logging
import weakref
from collections import defaultdict, deque
from threading import Lock
from typing import List, NamedTuple, Optional, Tuple

import torch

from .base import BaseAllocator
from iris.drivers.base import DriverNotSupported, ExportableMemory, LocalAllocation, MappingPlacement, PeerMapping
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


class Bucket(NamedTuple):
    """A free-list key: a block's size and alignment, both in bytes.

    A freed block is reused for a request with the same ``size`` and a compatible
    alignment (see allocate() / _take_reusable_block for the matching rule, and
    _free_callback / reuse() for how freed blocks are filed back under their bucket).
    """

    size: int
    alignment: int


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
    # Largest per-allocation alignment the heap guarantees on the *absolute address*.
    # base_va is reserved aligned to (at least) this, so a request's aligned heap offset
    # and its aligned data_ptr() are the same thing for any alignment up to this bound
    # (see allocate()). Requests above it are rejected rather than silently returning a
    # misaligned pointer. 256 MiB covers TMA/page/DMA alignment needs; VA alignment is free.
    MAX_ALIGNMENT = 256 * 1024 * 1024
    # Reconcile frees (SymmetricHeap.collect()) after handing out 1/COLLECT_FRACTION of the mapped heap
    COLLECT_FRACTION = 8

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
        self.free_lists: dict[Bucket, list] = defaultdict(list)
        self.alloc_sizes = {}
        self._pending_free = deque()
        # Freed on this rank but not yet reusable: offset -> Bucket
        self._released: dict[int, Bucket] = {}
        # offset -> the bump value an over-aligned block was allocated at (i.e. where its
        # alignment padding begins). reuse() rolls the bump pointer back to this point so the
        # skipped padding is reclaimed, not stranded. Only set when a block has padding.
        self._pad_start: dict[int, int] = {}
        # Bytes handed out since the last reuse(): identical on every rank, like the allocation sequence
        self._bytes_since_collect = 0
        self.chunks: List[LocalAllocation] = []
        self._shared_regions: List[ExportableMemory] = []
        self._peer_mappings: List[PeerMapping] = []
        self._imported_heap_mappings: List[PeerMapping] = []
        self.mapped_extent = 0
        self.base_va = 0
        self.granularity = 0
        self.min_granularity = 0

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
        # Back memory in units of the driver's recommended granule (the efficient unit),
        # and remember the true minimum for reporting. On stacks where the two differ,
        # the recommended value is what the allocator aligns and maps with.
        self.granularity = self.driver.get_recommended_granularity()
        # min_granularity is reporting-only. Its HIP query is less tested for some handle
        # types (e.g. the fabric handle type), so don't let a failed query break allocator
        # construction; fall back to the recommended granule with a debug log.
        try:
            self.min_granularity = self.driver.get_minimum_granularity()
        except Exception as exc:
            logger.debug("get_minimum_granularity() failed (%s); reporting recommended granule instead", exc)
            self.min_granularity = self.granularity
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
        # (x + a - 1) & ~(a - 1) is safe for granularity alignment. The chunk size
        # itself need not be a power of two; VA rounding below uses arithmetic so
        # arbitrary (granularity-aligned) chunk sizes work for any heap_size.
        self.chunk_size = (self.chunk_size + self.granularity - 1) & ~(self.granularity - 1)

        # VA reservation -- just address space, no physical memory cost.
        # Default: 128 GiB (plenty of room for growth + imports).
        if va_size == 0:
            va_size = 128 * 1024 * 1024 * 1024  # 128 GiB
        # va_size = max(va_size, heap_size * 4) -- gives 4x headroom for growth
        # and imports. For very large heaps (>100 GiB) this can exceed sensible
        # VA budgets; if reserve_va fails, the caller should pass an explicit
        # va_size. Not capped here because the right cap depends on workload.
        self.va_size = max(va_size, heap_size * 4)
        # chunk_size may not be a power of two, so round the VA reservation up to a whole
        # number of chunks with integer arithmetic (a VA size shouldn't depend on float
        # division staying exact), rather than with a bitmask.
        self.va_size = -(-self.va_size // self.chunk_size) * self.chunk_size
        # Reserve the VA arena aligned to the largest alignment we promise on the absolute
        # address, not merely the granule. Because base_va is then a multiple of every
        # permitted alignment, aligning a block's heap offset also aligns its data_ptr().
        # Both granularity and MAX_ALIGNMENT are powers of two, so the max is too.
        self._max_alignment = max(self.granularity, self.MAX_ALIGNMENT)
        self.base_va = self.driver.reserve_va(self.va_size, self._max_alignment)

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
        placement = MappingPlacement(target_va, arena_base=self.base_va)
        allocation = self.driver.allocate_exportable(self.chunk_size, placement)
        self.chunks.append(allocation)
        self._shared_regions.append(ExportableMemory(va=allocation.va, size=allocation.size, allocation=allocation))
        self.mapped_extent += self.chunk_size

    def _process_pending_frees(self) -> None:
        """Move frees from GC finalizers to the released list. Call with lock held."""
        while self._pending_free:
            offset, bucket = self._pending_free.popleft()
            self.alloc_sizes.pop(offset, None)
            self._released[offset] = bucket

    def released_offsets(self) -> List[int]:
        """Offsets freed on this rank that are not yet reusable, sorted."""
        with self.lock:
            self._process_pending_frees()
            return sorted(self._released)

    def collect_due(self) -> bool:
        """True once 1/COLLECT_FRACTION of the mapped heap has been handed out since the last reuse()."""
        with self.lock:
            return self._bytes_since_collect >= self.mapped_extent // self.COLLECT_FRACTION

    def reuse(self, offsets) -> None:
        """
        Make released blocks reusable, and return every free block at the top of the heap (released
        this round or earlier) to the bump pointer. Call only via SymmetricHeap.collect(), which
        passes the same offsets on every rank: those every rank has released. That keeps the free
        lists, and every later offset, identical across ranks.
        """
        with self.lock:
            self._bytes_since_collect = 0
            candidates: dict[int, Bucket] = {offset: self._released.pop(offset) for offset in offsets}
            # Blocks freed in earlier rounds can reach the top once the blocks above them are freed
            for bucket, free in self.free_lists.items():
                candidates.update((offset, bucket) for offset in free)
                free.clear()
            # Blocks at the top of the bump region go back to the bump pointer, not a free list.
            # by_end maps a block's end offset to its start; rolling back lands on the start of
            # the block's alignment padding (_pad_start) rather than the block's own offset, so
            # the padding is reclaimed too and the block directly below (whose end == this
            # padding start) chains next.
            by_end = {offset + bucket.size: offset for offset, bucket in candidates.items()}
            while self.bump in by_end:
                offset = by_end.pop(self.bump)
                del candidates[offset]
                self.bump = self._pad_start.pop(offset, offset)
            # Highest offset first, so pop() hands out the lowest blocks and keeps those near the top
            # free for later rollbacks
            for offset in sorted(candidates, reverse=True):
                self.free_lists[candidates[offset]].append(offset)

    def _free_callback(self, offset: int, bucket: Bucket) -> None:
        """Called by weakref finalizer when a tensor's storage is GC'd.

        ``bucket`` is the free-list key the block must be filed back under.

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
        self._pending_free.append((offset, bucket))

    def get_base_address(self) -> int:
        return self.base_va

    @property
    def base_alignment(self) -> int:
        """Alignment the heap base VA is reserved to, and the largest per-allocation alignment
        honored on the absolute address (see allocate()).

        Peer VA arenas (SymmetricHeap._refresh_peer_access_*) must be reserved to this same
        alignment: a symmetric object sits at an identical offset on every rank, so an importer's
        translated address heap_bases[peer] + offset is only aligned to the requested alignment if
        the peer arena base is aligned exactly as the owner's base_va is.
        """
        return self._max_alignment

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

    def _check_open(self) -> None:
        """Reject operations once the allocator has been closed.

        This is a single read of a plain bool, so it is safe to call with or
        without self.lock held. Most call sites take the lock anyway because
        they go on to mutate allocator state; get_allocation_chunks_since()
        does not take the lock (see its docstring) and calls this unlocked.

        reuse(), released_offsets() and collect_due() are intentionally *not*
        guarded: they are reached only via SymmetricHeap.collect(), never as
        public post-close entry points, so an explicit guard would be dead code.
        """
        if self._closed:
            raise RuntimeError("VMemChunkedAllocator: operation attempted after close()")

    def _take_reusable_block(self, bucket: Bucket) -> Optional[Tuple[int, Bucket]]:
        """Pop a free (offset, source_bucket) that satisfies ``bucket``, or return None.

        Prefers an exact (size, alignment) match. Failing that, accepts a same-size block
        filed under a *larger* alignment: its offset is already a multiple of the smaller
        requested alignment, so it satisfies the request. The smallest sufficient alignment
        is chosen first to avoid wasting the most-aligned blocks. The choice depends only on
        free_lists and the request, both identical on every rank, so offsets stay symmetric.

        The source bucket is returned so a caller that must roll back (a failed allocation)
        can file the block back under the exact key it came from, not the request's key.
        """
        exact = self.free_lists.get(bucket)
        if exact:
            return exact.pop(), bucket
        for candidate in sorted(
            b
            for b, offs in self.free_lists.items()
            if b.size == bucket.size and b.alignment > bucket.alignment and offs
        ):
            return self.free_lists[candidate].pop(), candidate
        return None

    def allocate(
        self, num_elements: int, dtype: torch.dtype, alignment: int = 1024, grow: bool = True
    ) -> Optional[torch.Tensor]:
        """
        Allocate from a free list or the bump pointer. With ``grow=False``, return None instead of
        mapping new chunks.

        A requested ``alignment`` up to ``MAX_ALIGNMENT`` is honored on the absolute address: the
        returned tensor's ``data_ptr()`` (and its offset from the heap base) is a multiple of
        ``alignment``. This holds because the heap base is reserved aligned to ``MAX_ALIGNMENT``,
        so aligning the heap offset aligns the address identically on every rank. An ``alignment``
        above ``MAX_ALIGNMENT`` is rejected rather than silently returning a misaligned pointer.
        """
        with self.lock:
            self._check_open()
            if num_elements == 0:
                return torch.empty(0, dtype=dtype, device=self.device)

            self._process_pending_frees()

            elem_size = _element_size(dtype)
            size_bytes = num_elements * elem_size
            # Minimum allocation is one granule
            size_bytes = max(size_bytes, self.granularity)
            # Round to next power of two for free-list bucketing
            size_class = _next_power_of_two(size_bytes)
            # Effective alignment: at least min_alignment, raised to honor an over-aligned
            # request. Rounded to a power of two so the bitmask alignment math holds.
            effective_align = max(self.min_alignment, _next_power_of_two(max(1, alignment)))
            if effective_align > self._max_alignment:
                raise ValueError(
                    f"VMemChunkedAllocator: requested alignment {alignment} (effective "
                    f"{effective_align}) exceeds the heap's maximum alignment {self._max_alignment}. "
                    f"The heap base is reserved aligned to {self._max_alignment}; a larger alignment "
                    f"cannot be guaranteed on the absolute address. Raise MAX_ALIGNMENT or request less."
                )
            # Block size is independent of alignment: the bucket key already records the alignment,
            # so an over-aligned request no longer inflates the block (a 4 KiB request with 64 MiB
            # alignment stays a granule, not 64 MiB). Alignment is handled by padding the bump below.
            aligned_size = (size_class + self.granularity - 1) & ~(self.granularity - 1)
            bucket = Bucket(aligned_size, effective_align)

            pad_start: Optional[int] = None
            reused = self._take_reusable_block(bucket)
            if reused is not None:
                offset, reused_bucket = reused  # reused_bucket is the exact key to restore on rollback
                from_free_list = True
                new_bump = self.bump
                grow_chunks = 0
            else:
                from_free_list = False
                reused_bucket = None
                # Bump allocate -- align the bump pointer to the effective alignment. The skipped
                # range [self.bump, aligned_bump) is this block's alignment padding; remember its
                # start so reuse() can roll the bump back past it (see _pad_start).
                aligned_bump = (self.bump + effective_align - 1) & ~(effective_align - 1)
                pad_start = self.bump

                needed = aligned_bump + aligned_size - self.mapped_extent
                if needed > 0 and not grow:
                    return None
                grow_chunks = 0
                if needed > 0:
                    grow_chunks = math.ceil(needed / self.chunk_size)
                    if self.mapped_extent + grow_chunks * self.chunk_size > self.va_size:
                        raise RuntimeError(
                            f"VMemChunkedAllocator: requested allocation needs "
                            f"{grow_chunks} chunks ({grow_chunks * self.chunk_size} bytes), "
                            f"but only {self.va_size - self.mapped_extent} bytes of VA remain."
                        )

                offset = aligned_bump
                new_bump = aligned_bump + aligned_size

            # Build the tensor BEFORE growing chunks or mutating bookkeeping, so a construction
            # failure leaves the allocator untouched: no bump advance, no orphaned reservation,
            # and -- critically -- no chunk grown-but-unexported (a grown chunk that peers never
            # import, which a later non-growing allocation would read over as unmapped peer VA).
            # torch.as_tensor over a CUDA array interface only wraps the pointer; it does not
            # touch the (possibly still unmapped) memory, so building before the grow is safe.
            #
            # try/finally with a `committed` flag makes the rollback cover *every* exception
            # type (including KeyboardInterrupt), not just Exception. The finalizer is armed
            # before the bookkeeping commit: once the block is owned by GC it is guaranteed to
            # be freed, and the finally never also returns it to a free list (no double-manage).
            committed = False
            try:
                va = self.base_va + offset
                interface_size = (aligned_size // elem_size) * elem_size
                iface = _CUDAArrayInterface(va, interface_size)
                tensor_bytes = torch.as_tensor(iface, device=self.device)
                full = tensor_bytes.view(dtype)
                tensor = full.narrow(0, 0, num_elements)

                # Tensor built successfully -- now it is safe to grow chunks.
                for _ in range(grow_chunks):
                    self._grow_chunk()

                # Arm the GC weak ref before committing bookkeeping, so the block can never be
                # left live-but-unfinalized if a later step raised.
                weakref.finalize(
                    tensor.untyped_storage(),
                    self._free_callback,
                    offset,
                    bucket,
                )

                self.bump = new_bump
                self.alloc_sizes[offset] = aligned_size
                self._bytes_since_collect += aligned_size
                if pad_start is not None and pad_start != offset:
                    # This block owns alignment padding [pad_start, offset) below it; reuse() rolls
                    # the bump back to pad_start, not offset, so the padding is not stranded.
                    self._pad_start[offset] = pad_start

                committed = True
                return tensor
            finally:
                if not committed and from_free_list:
                    # Build/grow/finalizer-registration failed: return the popped free-list block
                    # to the exact bucket it came from. Nothing was grown into use, the bump
                    # pointer was never advanced, and no finalizer owns this offset.
                    self.free_lists[reused_bucket].append(offset)

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

        Does not take self.lock: per the "once per allocator lifetime during
        peer setup" contract above, this is not expected to race with allocate().

        Args:
            start_index: chunks at indices [0, start_index) are skipped.

        Returns:
            List of (chunk_index, offset, size, handle_bytes) tuples for
            chunks at index >= start_index. Indices are absolute (matching
            get_allocation_chunks's output), not relative to start_index.

        Raises:
            ValueError: if start_index is negative or > current chunk count.
            RuntimeError: if the allocator has been closed.
        """
        self._check_open()
        if start_index < 0:
            raise ValueError(f"start_index must be non-negative, got {start_index}")
        if start_index > len(self._shared_regions):
            raise ValueError(f"start_index {start_index} exceeds exported region count {len(self._shared_regions)}")

        result = []
        for i in range(start_index, len(self._shared_regions)):
            region = self._shared_regions[i]
            offset = region.va - self.base_va
            handle_bytes = self.driver.export_handle(region)
            result.append((i, offset, region.size, handle_bytes))
        return result

    def get_num_chunks(self):
        """Return the number of exported heap regions."""
        return len(self._shared_regions)

    def _can_return_external_tensor_alias(self) -> bool:
        return self.num_ranks == 1 and self.driver.__class__.__name__ == "LocalCudaDriver"

    def _external_tensor_alias(self, external_tensor: torch.Tensor) -> torch.Tensor:
        logger.info(
            "Returning a local CUDA tensor alias because this external allocation cannot be mapped into the Iris heap"
        )
        return external_tensor.view(external_tensor.shape)

    def import_external_tensor(self, external_tensor: torch.Tensor) -> torch.Tensor:
        """Import an external tensor into the symmetric heap (zero-copy).

        The imported tensor shares physical memory with the original -- writes
        to one are visible in the other. On the chunked allocator path, the
        imported allocation is mapped into the heap's VA layout and retained
        until allocator.close() so peer translation remains valid for RMA.

        Raises:
            DriverNotSupported: If the active driver cannot export and remap
                the external allocation. Single-rank local CUDA jobs may return
                a direct tensor alias instead, because no peer VA translation is
                needed.
            RuntimeError: If the input tensor is not on a CUDA/HIP device or
                is not contiguous.
        """
        with self.lock:
            self._check_open()
            if not external_tensor.is_cuda:
                raise RuntimeError("Can only import CUDA/HIP tensors")
            if not external_tensor.is_contiguous():
                raise RuntimeError("Only contiguous tensors can be imported; call .contiguous() before as_symmetric()")
            if self._can_return_external_tensor_alias():
                return self._external_tensor_alias(external_tensor)

            external_ptr = external_tensor.data_ptr()
            tensor_size = external_tensor.numel() * external_tensor.element_size()
            try:
                alloc_base, alloc_size = self.driver.get_address_range(external_ptr)
            except DriverNotSupported:
                if self._can_return_external_tensor_alias():
                    return self._external_tensor_alias(external_tensor)
                raise

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
            try:
                handle_bytes = self.driver.export_handle(ExportableMemory(alloc_base, alloc_size))
            except DriverNotSupported:
                if self._can_return_external_tensor_alias():
                    return self._external_tensor_alias(external_tensor)
                raise

            placement = MappingPlacement(target_base_va, arena_base=self.base_va)
            mapping = self.driver.import_and_map(
                self.cur_rank,
                handle_bytes,
                aligned_alloc_size,
                placement,
            )
            self._imported_heap_mappings.append(mapping)
            self._shared_regions.append(ExportableMemory(va=target_base_va, size=aligned_alloc_size))
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
        list, and that code path owns the cleanup call.

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
                self.driver.cleanup(mapping)
            except Exception as exc:
                logger.warning("cleanup failed in finalizer: %s", exc)

    def import_peer_chunk(self, peer_rank: int, handle_bytes: bytes, size: int) -> int:
        """
        Import a serialized chunk handle from a peer rank.

        Returns the local virtual address where the chunk was mapped.
        The caller is responsible for calling release_peer_chunk when done.
        """
        with self.lock:
            self._check_open()
            mapping = self.driver.import_and_map(peer_rank, handle_bytes, size)
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
                        self.driver.cleanup(mapping)
                    finally:
                        self._peer_mappings.pop(i)
                    return
            logger.debug(
                "release_peer_chunk: no mapping at va=0x%x (already released or never imported via import_peer_chunk)",
                remote_va,
            )

    def get_stats(self):
        """Return allocator statistics."""
        with self.lock:
            self._process_pending_frees()
            total_free = sum(len(v) for v in self.free_lists.values())
            free_bytes = sum(bucket.size * len(offsets) for bucket, offsets in self.free_lists.items())
            return {
                "num_chunks": len(self.chunks),
                "mapped_bytes": self.mapped_extent,
                "bump": self.bump,
                "num_active_allocs": len(self.alloc_sizes),
                "num_free_blocks": total_free,
                "num_released_blocks": len(self._released),
                "free_bytes": free_bytes,
                "va_size": self.va_size,
                "chunk_size": self.chunk_size,
                # "granularity" is kept as a backward-compat alias of "recommended_granularity".
                "granularity": self.granularity,
                "recommended_granularity": self.granularity,
                "min_granularity": self.min_granularity,
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
            self._released.clear()
            self._pad_start.clear()
            self.free_lists.clear()
            self.alloc_sizes.clear()

            try:
                torch.cuda.synchronize(self.device)
            except Exception:
                pass

            # Release imported peer mappings
            for mapping in self._peer_mappings:
                try:
                    self.driver.cleanup(mapping)
                except Exception:
                    pass
            self._peer_mappings.clear()

            # Release imported symmetric mappings that were inserted into the
            # heap VA layout via import_external_tensor.
            for mapping in self._imported_heap_mappings:
                try:
                    self.driver.cleanup(mapping)
                except Exception:
                    pass
            self._imported_heap_mappings.clear()
            self._shared_regions.clear()

            # Release locally-mapped chunks. Each chunk has _va_owned=False,
            # so cleanup will unmap and release the physical handle
            # but will NOT free VA
            for alloc in self.chunks:
                try:
                    self.driver.cleanup(alloc)
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
