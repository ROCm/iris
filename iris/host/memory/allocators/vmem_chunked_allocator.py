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
from typing import List, NamedTuple, Optional, Sequence, Tuple, Union

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


def _prev_power_of_two(n: int) -> int:
    """Return the largest power of two <= n (0 if n <= 0)."""
    if n <= 0:
        return 0
    return 1 << (n.bit_length() - 1)


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


def _aligned_block_size(size_bytes: int, granularity: int) -> int:
    """Bytes an allocation of ``size_bytes`` actually consumes: at least one granule, bucketed to a
    power of two for the free lists, then granule-aligned. Shared by the live allocator (allocate(),
    _reservation_bytes) and the pure plan_capacity() estimator so both agree on per-allocation
    sizing -- otherwise a reservation or capacity plan can disagree with what allocate() really uses.

    The block size is independent of a request's ``alignment``: the bucket key records the alignment
    and the bump pointer absorbs any alignment padding, so this models every allocation's block size
    regardless of alignment. What it cannot model is the bump-pointer padding an over-aligned request
    adds on top (that depends on the live bump position), so a capacity estimate built from this is
    exact for default-aligned allocations and a lower bound once over-alignment is in play.
    """
    size_bytes = max(size_bytes, granularity)
    size_class = _next_power_of_two(size_bytes)
    return (size_class + granularity - 1) & ~(granularity - 1)


def _block_size(bucket: Bucket) -> int:
    """Block size (bytes) a free-list key represents. Keys are ``Bucket(size, alignment)``; the
    block size is the bucket's ``size`` field."""
    return bucket.size


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
        reserved_bytes: Bytes pre-mapped on top of heap_size for Iris's own control state, so
            heap_size stays usable to the caller. Either a single byte count (rounded up to a
            granule) or a list of per-allocation byte counts (each rounded independently; see
            _reservation_bytes). Keyword-only.
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
        reserved_bytes: Union[int, Sequence[int]] = 0,
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
        # Imported regions by heap VA: (source range, source storage). Peers get these
        # regions exported from the source allocation, because HIP 7.16 and later refuse
        # to export memory that was itself imported. Holding the storage keeps torch from
        # freeing or reusing that allocation while it is in the heap.
        self._import_sources = {}
        self.mapped_extent = 0
        self.base_va = 0
        self.granularity = 0
        self.min_granularity = 0
        # Bytes set aside for Iris's own control state so heap_size stays usable to the caller.
        # Sized once the granularity is known (below), then pre-mapped on top of heap_size. May be a
        # single byte count or a list of per-allocation byte counts (see _reservation_bytes).
        self._requested_reserved_bytes = reserved_bytes
        self.reserved_bytes = 0
        # Bytes currently allocated for internal (Iris) control state, and each internal offset mapped
        # to the block size backing it. Storing the size here (rather than reading it back from the
        # free-list key on free) keeps internal accounting independent of the free-list key type.
        self.internal_bytes = 0
        self._internal_offsets = {}

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

        self.min_alignment = max(self.granularity, 1024)

        # Internal reservation, pre-mapped on top of heap_size so Iris's control state does not come
        # out of the caller's usable budget. Each internal allocation consumes a whole number of
        # granules (bucketed to a power of two like any allocation), so a reservation given as a list
        # of per-allocation sizes is rounded per allocation -- summing first and rounding once would
        # under-reserve by up to a granule per allocation.
        self.reserved_bytes = self._reservation_bytes(self._requested_reserved_bytes)

        # Chunk configuration -- cap at heap_size to avoid overshooting VA. chunk_size derives from
        # heap_size alone, not heap_size + reserved_bytes: the internal reservation is mapped as its
        # own exact-sized region at the end of __init__, so folding it into chunk_size would only
        # round a few MiB of control state up to a whole extra chunk. Chunk sizes need not be a power
        # of two (VA rounding below is arithmetic), so this split is a sizing choice, not a
        # correctness one.
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
        # Include the internal reservation in the VA budget so growth + imports still get headroom
        # on top of heap_size + the reservation.
        self.va_size = max(va_size, (heap_size + self.reserved_bytes) * 4)
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

        # Pre-map chunk_size chunks to cover heap_size, so the caller gets ~heap_size of usable
        # space before any growth is triggered.
        n_initial_chunks = max(1, math.ceil(heap_size / self.chunk_size))
        for _ in range(n_initial_chunks):
            self._grow_chunk()
        # Back the internal reservation with its own region of exactly reserved_bytes (already a
        # granule multiple) rather than rounding heap_size + reserved up to whole chunks -- a few
        # MiB of control state would otherwise map a whole extra chunk. The region is granule- (not
        # chunk-) sized; peers import it like any other region since each carries its own size.
        if self.reserved_bytes > 0:
            self._map_physical_region(self.reserved_bytes)

    def _map_physical_region(self, size: int) -> None:
        """Map one physical region of ``size`` bytes (a granule multiple) at the top of the VA arena.

        Regions stay contiguous and granule-aligned. Growth uses ``chunk_size`` via :meth:`_grow_chunk`;
        the one-time internal reservation uses its exact (sub-chunk) size so it never maps a whole
        extra chunk.
        """
        if self.mapped_extent + size > self.va_size:
            raise RuntimeError(
                "VMemChunkedAllocator: VA space exhausted while mapping a region "
                f"(size={size}). {self._capacity_message()}. "
                "Increase va_size."
            )

        target_va = self.base_va + self.mapped_extent
        placement = MappingPlacement(target_va, arena_base=self.base_va)
        allocation = self.driver.allocate_exportable(size, placement)
        self.chunks.append(allocation)
        self._shared_regions.append(ExportableMemory(va=allocation.va, size=allocation.size, allocation=allocation))
        self.mapped_extent += size

    def _grow_chunk(self):
        """Map a new physical chunk (``chunk_size``) into the VA range."""
        self._map_physical_region(self.chunk_size)

    def _process_pending_frees(self) -> None:
        """Move frees from GC finalizers to the released list. Call with lock held."""
        while self._pending_free:
            offset, bucket = self._pending_free.popleft()
            self.alloc_sizes.pop(offset, None)
            internal_size = self._internal_offsets.pop(offset, None)
            if internal_size is not None:
                self.internal_bytes -= internal_size
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

    def _aligned_alloc_size(self, size_bytes: int) -> int:
        """Bytes an allocation of ``size_bytes`` actually consumes (see module-level
        :func:`_aligned_block_size`)."""
        return _aligned_block_size(size_bytes, self.granularity)

    def _reservation_bytes(self, requested) -> int:
        """Size the internal reservation from ``requested`` (a byte count or a list of per-allocation
        byte counts). Both forms are rounded per allocation via :meth:`_aligned_alloc_size`: a list
        rounds each entry, and a scalar is treated as one allocation of that size. A scalar is *not*
        rounded to a bare granule -- a real internal allocation of ``requested`` bytes consumes the
        power-of-two-bucketed block size, so bare-granule rounding would under-reserve a scalar like
        5 MiB (reserve 6 MiB, allocation uses 8 MiB). :meth:`plan_capacity` uses the identical rule
        statically. (``SymmetricHeap`` always forwards a list; the scalar form is for direct users.)"""
        if isinstance(requested, (list, tuple)):
            return sum(self._aligned_alloc_size(int(b)) for b in requested if b and int(b) > 0)
        req = max(0, int(requested))
        return self._aligned_alloc_size(req) if req > 0 else 0

    def allocate(
        self,
        num_elements: int,
        dtype: torch.dtype,
        alignment: int = 1024,
        grow: bool = True,
        internal: bool = False,
    ) -> Optional[torch.Tensor]:
        """
        Allocate from a free list or the bump pointer. With ``grow=False``, return None instead of
        mapping new chunks.

        A requested ``alignment`` up to ``MAX_ALIGNMENT`` is honored on the absolute address: the
        returned tensor's ``data_ptr()`` (and its offset from the heap base) is a multiple of
        ``alignment``. This holds because the heap base is reserved aligned to ``MAX_ALIGNMENT``,
        so aligning the heap offset aligns the address identically on every rank. An ``alignment``
        above ``MAX_ALIGNMENT`` is rejected rather than silently returning a misaligned pointer.

        ``internal=True`` marks the allocation as Iris control state (e.g. device-barrier flags),
        so it is counted against ``reserved_bytes`` rather than the user's usable budget in
        ``get_stats()``.
        """
        with self.lock:
            self._check_open()
            if num_elements == 0:
                return torch.empty(0, dtype=dtype, device=self.device)

            self._process_pending_frees()

            elem_size = _element_size(dtype)
            size_bytes = num_elements * elem_size
            # Size the allocation: min one granule, bucketed to a power of two, granule-aligned.
            # Block size is independent of alignment (the bucket key records the alignment and the
            # bump pointer absorbs any padding), so an over-aligned request does not inflate the
            # block -- a 4 KiB request with 64 MiB alignment stays a granule, not 64 MiB. Shared with
            # the internal reservation and plan_capacity() via _aligned_block_size so they agree.
            aligned_size = self._aligned_alloc_size(size_bytes)
            # Effective alignment: at least min_alignment, raised to honor an over-aligned request.
            # Rounded to a power of two so the bitmask alignment math holds.
            effective_align = max(self.min_alignment, _next_power_of_two(max(1, alignment)))
            if effective_align > self._max_alignment:
                raise ValueError(
                    f"VMemChunkedAllocator: requested alignment {alignment} (effective "
                    f"{effective_align}) exceeds the heap's maximum alignment {self._max_alignment}. "
                    f"The heap base is reserved aligned to {self._max_alignment}; a larger alignment "
                    f"cannot be guaranteed on the absolute address. Raise MAX_ALIGNMENT or request less."
                )
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
                            "VMemChunkedAllocator: out of capacity -- cannot satisfy allocation "
                            f"(needs {grow_chunks} new chunk(s), {grow_chunks * self.chunk_size} bytes). "
                            f"{self._capacity_message(requested_bytes=aligned_size)}. "
                            "Increase heap_size or va_size."
                        )

                offset = aligned_bump
                new_bump = aligned_bump + aligned_size

            # Build the tensor BEFORE growing chunks or mutating bookkeeping, so a failed
            # construction rolls back cleanly (nothing grown, bump untouched). Just a local-rollback
            # nicety -- peer correctness comes from SymmetricHeap refreshing on chunk-count change.
            # Safe because we pass an explicit device index, so as_tensor only wraps the pointer
            # (torch <2.9 instead probes the unmapped VA via getDeviceFromPtr -- then grow first).
            # The try/finally below returns a popped free-list block on any failure (committed flag).
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
                # GC now owns this offset, so mark committed before the bookkeeping below: the
                # finally must not also re-file it on a late failure (double-manage).
                committed = True

                self.bump = new_bump
                self.alloc_sizes[offset] = aligned_size
                self._bytes_since_collect += aligned_size
                if pad_start is not None and pad_start != offset:
                    # This block owns alignment padding [pad_start, offset) below it; reuse() rolls
                    # the bump back to pad_start, not offset, so the padding is not stranded.
                    self._pad_start[offset] = pad_start
                # Count the internal reservation inside the committed block, after the finalizer is
                # armed. The finalizer's offset -> _internal_offsets decrement in
                # _process_pending_frees is then guaranteed a matching entry, and a failed build or
                # grow (which never reaches here) can never leave internal_bytes over-counted or a
                # phantom offset in _internal_offsets.
                if internal:
                    self.internal_bytes += aligned_size
                    self._internal_offsets[offset] = aligned_size

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
            source = self._import_sources.get(region.va)
            handle_bytes = self.driver.export_handle(source[0] if source is not None else region)
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
            self._import_sources[target_base_va] = (
                ExportableMemory(alloc_base, alloc_size),
                external_tensor.untyped_storage(),
            )
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

    def _capacity_fields(self):
        """Capacity accounting derived from current state. Caller holds the lock (or is in init).

        All figures are in bytes.

        ``usable_bytes`` is the user budget still available: ``heap_size`` minus the bytes user
        allocations have taken. Internal (Iris control-state) allocations are backed by the separate
        reservation, so they are excluded and never reduce it. Clamped at 0; tracks real memory, not
        the VA reservation.

        ``va_headroom_bytes`` and ``largest_allocatable_bytes`` are bounded by the VA reservation and
        growth, not by physical HBM (which the allocator cannot observe), so they can exceed both
        ``usable_bytes`` and the currently mapped size; reaching them triggers chunk growth and may
        fail on a real HBM limit first. ``largest_allocatable_bytes`` is the largest single request
        that still succeeds: a request of ``S`` bytes consumes ``_aligned_block_size(S)`` (rounded up
        to a power of two), so it is the largest power of two that fits in the growable VA, or the
        largest ready free block, whichever is larger.

        ``free_in_mapped_bytes`` is the already-mapped, currently-free space. It still counts the
        mapped reservation headroom that internal allocations have not consumed; ``usable_bytes`` and
        ``va_headroom_bytes`` exclude that unconsumed reservation.
        """
        used_bytes = sum(self.alloc_sizes.values())
        free_list_bytes = sum(_block_size(k) * len(offsets) for k, offsets in self.free_lists.items())
        largest_free = max((_block_size(k) for k, offsets in self.free_lists.items() if offsets), default=0)
        free_in_mapped = (self.mapped_extent - self.bump) + free_list_bytes
        pending_internal = max(0, self.reserved_bytes - self.internal_bytes)
        growable = max(0, self.va_size - self.bump - pending_internal)
        # User budget left: heap_size minus what user allocations have taken. Internal control state
        # is covered by the reservation, so subtract only the non-internal portion of used_bytes.
        user_used = max(0, used_bytes - self.internal_bytes)
        usable_bytes = max(0, self.heap_size - user_used)
        return {
            "heap_size": self.heap_size,
            "reserved_bytes": self.reserved_bytes,
            "internal_bytes": self.internal_bytes,
            "used_bytes": used_bytes,
            "free_bytes": free_list_bytes,
            "free_in_mapped_bytes": free_in_mapped,
            "usable_bytes": usable_bytes,
            "va_headroom_bytes": growable + free_list_bytes,
            "largest_allocatable_bytes": max(_prev_power_of_two(growable), largest_free),
        }

    def _capacity_message(self, requested_bytes=None):
        """One-line capacity breakdown for out-of-capacity errors. Caller holds the lock."""
        c = self._capacity_fields()
        parts = []
        if requested_bytes is not None:
            parts.append(f"requested={requested_bytes}")
        parts += [
            f"usable_bytes={c['usable_bytes']}",
            f"va_headroom_bytes={c['va_headroom_bytes']}",
            f"largest_allocatable_bytes={c['largest_allocatable_bytes']}",
            f"used_bytes={c['used_bytes']}",
            f"internal_bytes={c['internal_bytes']}",
            f"reserved_bytes={c['reserved_bytes']}",
            f"free_in_mapped_bytes={c['free_in_mapped_bytes']}",
            f"mapped_bytes={self.mapped_extent}",
            f"va_size={self.va_size}",
        ]
        return ", ".join(parts)

    @staticmethod
    def plan_capacity(
        target_user_bytes: int,
        *,
        granularity: int,
        reserved_bytes: Union[int, Sequence[int]] = 0,
        chunk_size: int = DEFAULT_CHUNK_SIZE,
        va_size: int = DEFAULT_VA_SIZE,
    ) -> dict:
        """Predict the backing an allocator needs for ``target_user_bytes`` of usable space.

        Pure (no device calls) so callers can size ``heap_size`` deterministically. ``granularity``
        is a required keyword: pass the device's granularity (``allocator.granularity``, or the
        driver's recommended granule -- 2 MiB on AMD). It has no default because a wrong granule
        silently mis-sizes the reservation (a smaller granule rounds each reserved allocation down
        toward its raw size, so a real, larger device granule reserves more than predicted). Mirrors
        ``__init__`` sizing: chunk size derives from the user target, the reservation is mapped as its
        own region on top, and VA gets 4x headroom.

        ``reserved_bytes`` matches the constructor: a scalar is sized as one internal allocation and a
        list is rounded per entry, both via ``_aligned_block_size`` (power-of-two bucketed, then
        granule-aligned), using the same rule as the live allocator so the prediction matches a real
        allocator built with the same reservation.

        A non-power-of-two ``target_user_bytes`` is fully supported: chunk sizes need not be powers
        of two, so the built allocator accepts it and this plan matches what it would map (the
        estimate is granule-accurate for any target, not only power-of-two ones).
        """
        g = max(1, granularity)
        if isinstance(reserved_bytes, (list, tuple)):
            reserved = sum(_aligned_block_size(int(b), g) for b in reserved_bytes if b and int(b) > 0)
        else:
            b = max(0, int(reserved_bytes))
            reserved = _aligned_block_size(b, g) if b > 0 else 0
        # Mirror __init__'s chunk sizing: derive from the user target, cap at chunk_size, round up to
        # a whole granule. Chunk sizes need not be a power of two, so this holds for any target.
        effective_chunk = min(chunk_size, max(target_user_bytes, g))
        cs = max(effective_chunk, g)
        cs = -(-cs // g) * g
        total = target_user_bytes + reserved
        # Exact integer ceil-division (-(-a // b)), not math.ceil(a / b), so large byte counts never
        # lose precision to float; cs above uses the same idiom.
        heap_chunks = max(1, -(-target_user_bytes // cs))
        # The reservation is mapped as its own region of exactly `reserved` (see __init__), not as
        # whole chunks, so it contributes one region and `reserved` bytes -- not a chunk multiple.
        n_chunks = heap_chunks + (1 if reserved > 0 else 0)
        mapped = heap_chunks * cs + reserved
        if va_size == 0:
            va_size = 128 * 1024 * 1024 * 1024
        va = max(va_size, total * 4)
        va = -(-va // cs) * cs
        return {
            "chunks": n_chunks,
            "chunk_size": cs,
            "mapped_bytes": mapped,
            "reserved_bytes": reserved,
            "heap_size": target_user_bytes,
            # Budget the caller asked to be usable (matches get_stats()'s usable_bytes on a fresh,
            # empty allocator). The allocator maps >= this (mapped_bytes) but the user budget is the
            # target; internal control state is paid from the reservation, not this budget.
            "usable_bytes": target_user_bytes,
            # VA/growth-bounded headroom on a fresh allocator (bump=0, whole reservation pending).
            "va_headroom_bytes": max(0, va - reserved),
            "va_size": va,
        }

    def get_stats(self):
        """Return allocator statistics, including capacity/accounting fields."""
        with self.lock:
            self._process_pending_frees()
            total_free = sum(len(v) for v in self.free_lists.values())
            capacity = self._capacity_fields()
            return {
                "num_chunks": len(self.chunks),
                "mapped_bytes": self.mapped_extent,
                "bump": self.bump,
                "num_active_allocs": len(self.alloc_sizes),
                "num_free_blocks": total_free,
                "num_released_blocks": len(self._released),
                "va_size": self.va_size,
                "chunk_size": self.chunk_size,
                # "granularity" is kept as a backward-compat alias of "recommended_granularity".
                "granularity": self.granularity,
                "recommended_granularity": self.granularity,
                "min_granularity": self.min_granularity,
                **capacity,
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
            self._internal_offsets.clear()
            self.internal_bytes = 0

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
            self._import_sources.clear()

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
