# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.

"""
Tests for VMemChunkedAllocator.

Tests cover:
- Basic allocation and data integrity
- Multiple allocations and non-overlap
- Power-of-two free list reuse via GC, once every rank has freed a block
- Chunk growth on overflow
- as_symmetric (import external tensor)
- owns_tensor detection
- heap_bases stability
- Cross-rank RMA (peer memory access)
- OOM handling
- Allocator stats
- Thread safety
"""

import gc
import threading

import pytest
import torch

import iris
from iris.host.memory.allocators import vmem_chunked_allocator as vmem_chunked_module


ALLOC_TYPE = "vmem_chunked"


class _ConstructionBoom:
    """Drop-in for _CUDAArrayInterface whose construction raises, to simulate a tensor-build
    failure narrowly (only the allocator's construction path, not the whole process's
    torch.as_tensor). See the rollback tests below."""

    def __init__(self, *args, **kwargs):
        raise RuntimeError("injected tensor construction failure")


def test_chunked_creation():
    """Test that chunked VMem allocator can be created."""
    ctx = iris.iris(1 << 20, allocator_type=ALLOC_TYPE)
    assert ctx.cur_rank >= 0
    assert ctx.num_ranks >= 1
    assert ctx.heap_size == 1 << 20


def test_chunked_basic_allocation():
    """Test basic allocation and data integrity."""
    ctx = iris.iris(1 << 20, allocator_type=ALLOC_TYPE)
    tensor = ctx.zeros(1024, dtype=torch.float32)

    assert tensor.shape == (1024,)
    assert tensor.device.type == "cuda"
    assert torch.all(tensor == 0)

    tensor.fill_(42.0)
    assert torch.all(tensor == 42.0)


def test_chunked_multiple_allocations():
    """Test multiple allocations don't overlap."""
    ctx = iris.iris(4 << 20, allocator_type=ALLOC_TYPE)

    tensors = []
    for i in range(20):
        t = ctx.zeros(256, dtype=torch.float32)
        t.fill_(float(i))
        tensors.append(t)

    # Verify each tensor retains its value (no overlap)
    for i, t in enumerate(tensors):
        assert torch.all(t == float(i)), f"tensor {i} corrupted: expected {float(i)}, got {t[0].item()}"


def test_chunked_dtypes():
    """Test allocation with various dtypes."""
    ctx = iris.iris(4 << 20, allocator_type=ALLOC_TYPE)

    for dtype in [torch.float32, torch.float16, torch.bfloat16, torch.int32, torch.int64]:
        t = ctx.zeros(100, dtype=dtype)
        assert t.dtype == dtype
        assert t.shape == (100,)


def test_chunked_zero_elements():
    """Test zero-element allocation."""
    ctx = iris.iris(1 << 20, allocator_type=ALLOC_TYPE)
    t = ctx.zeros(0, dtype=torch.float32)
    assert t.numel() == 0
    assert t.shape == (0,)


def test_chunked_gc_free_reuse():
    """Test that freed memory is reused via GC."""
    ctx = iris.iris(4 << 20, allocator_type=ALLOC_TYPE)
    alloc = ctx.heap.allocator

    # Allocate a tensor
    t = ctx.zeros(1024, dtype=torch.float32)
    ptr1 = t.data_ptr()

    # Drop the tensor and make freed memory reusable
    del t
    ctx.reclaim_freed()

    # Allocate again -- should reuse the freed block
    t2 = ctx.zeros(1024, dtype=torch.float32)
    ptr2 = t2.data_ptr()

    # The reused block should be at the same offset (same free list bucket)
    assert ptr2 == ptr1, f"Expected reuse at 0x{ptr1:x}, got 0x{ptr2:x}"


def test_chunked_gc_multiple_reuse():
    """Test multiple rounds of alloc-free-reuse."""
    ctx = iris.iris(4 << 20, allocator_type=ALLOC_TYPE)

    for _ in range(10):
        t = ctx.zeros(512, dtype=torch.float32)
        t.fill_(99.0)
        torch.cuda.synchronize()
        assert torch.all(t == 99.0)
        del t
        gc.collect()


def test_chunked_free_list_size_classes():
    """Test that different sizes use different free list buckets.

    SymmetricHeap.allocate() bumps element counts to at least
    granularity / element_size, so we must pick sizes that remain in
    distinct power-of-two buckets after that rounding.
    """
    ctx = iris.iris(256 << 20, allocator_type=ALLOC_TYPE)
    alloc = ctx.heap.allocator
    elem_size = 4  # float32

    # Compute the minimum element count (the floor imposed by SymmetricHeap)
    min_elems = max(1, (alloc.granularity + elem_size - 1) // elem_size)

    # Pick three sizes that land in clearly different power-of-two buckets:
    #   small  = 1x granularity  (min_elems elements)
    #   medium = 4x granularity  (min_elems * 4 elements)
    #   large  = 16x granularity (min_elems * 16 elements)
    size_small = min_elems
    size_medium = min_elems * 4
    size_large = min_elems * 16

    small = ctx.zeros(size_small, dtype=torch.float32)
    medium = ctx.zeros(size_medium, dtype=torch.float32)
    large = ctx.zeros(size_large, dtype=torch.float32)
    # Keeps the freed blocks below the top of the heap, so they go to the free lists
    sentinel = ctx.zeros(size_small, dtype=torch.float32)  # noqa: F841

    small_ptr = small.data_ptr()
    medium_ptr = medium.data_ptr()
    large_ptr = large.data_ptr()

    # Free all
    del small, medium, large
    ctx.reclaim_freed()

    # Re-allocate -- each should reuse from its size class
    small2 = ctx.zeros(size_small, dtype=torch.float32)
    medium2 = ctx.zeros(size_medium, dtype=torch.float32)
    large2 = ctx.zeros(size_large, dtype=torch.float32)

    assert small2.data_ptr() == small_ptr
    assert medium2.data_ptr() == medium_ptr
    assert large2.data_ptr() == large_ptr


@pytest.mark.parametrize("keep_every", [0, 16])
def test_chunked_churn_then_large_allocation_does_not_grow(keep_every):
    """
    A churned scratch buffer must not fragment the heap so that a later large allocation grows it:
    freed blocks at the top return to the bump pointer, and periodic collect() calls bound the rest
    (also when buffers kept every ``keep_every`` iterations stop the rollback).
    """
    ctx = iris.iris(64 << 20, allocator_type=ALLOC_TYPE)
    alloc = ctx.heap.allocator
    scratch_bytes, keep_bytes = 2 * alloc.granularity, alloc.granularity  # different size classes
    chunks = alloc.get_num_chunks()
    bump = alloc.bump

    kept = []
    for i in range(3 * alloc.mapped_extent // scratch_bytes):
        t = ctx.zeros(scratch_bytes // 4, dtype=torch.float32)
        t.fill_(1.0)
        del t
        if keep_every and i % keep_every == keep_every - 1:
            kept.append(ctx.zeros(keep_bytes // 4, dtype=torch.float32))

    big = ctx.zeros(alloc.chunk_size // 2 // 4, dtype=torch.float32)
    assert alloc.get_num_chunks() == chunks

    # Once everything is freed, the whole heap returns to the bump pointer
    del big
    kept.clear()
    ctx.reclaim_freed()
    assert alloc.bump == bump


def test_chunked_rollback_includes_earlier_free_blocks():
    """A block freed in an earlier collect() returns to the bump pointer once the blocks above it are freed."""
    ctx = iris.iris(64 << 20, allocator_type=ALLOC_TYPE)
    alloc = ctx.heap.allocator
    bump = alloc.bump

    a = ctx.zeros(1024, dtype=torch.float32)
    b = ctx.zeros(1024, dtype=torch.float32)
    del a
    ctx.reclaim_freed()  # a is below b, so it goes to a free list
    del b
    ctx.reclaim_freed()  # b is at the top, and a is once b rolls back

    assert alloc.bump == bump
    assert alloc.get_stats()["num_free_blocks"] == 0


def test_chunked_free_list_hands_out_lowest_block_first():
    """Reusable blocks go out lowest offset first, keeping those near the top free for later rollbacks."""
    ctx = iris.iris(64 << 20, allocator_type=ALLOC_TYPE)

    a = ctx.zeros(1024, dtype=torch.float32)
    b = ctx.zeros(1024, dtype=torch.float32)
    c = ctx.zeros(1024, dtype=torch.float32)  # noqa: F841 -- keeps a and b below the top
    a_ptr = a.data_ptr()
    del a, b
    ctx.reclaim_freed()

    d = ctx.zeros(1024, dtype=torch.float32)
    assert d.data_ptr() == a_ptr


def test_chunked_freed_block_waits_for_collect():
    """A freed block is only reused after collect(), so every rank's offsets change at the same point."""
    ctx = iris.iris(64 << 20, allocator_type=ALLOC_TYPE)

    t = ctx.zeros(1024, dtype=torch.float32)
    ptr = t.data_ptr()
    del t
    gc.collect()

    t2 = ctx.zeros(1024, dtype=torch.float32)
    assert t2.data_ptr() != ptr
    assert ctx.heap.allocator.get_stats()["num_released_blocks"] >= 1

    ctx.reclaim_freed()
    t3 = ctx.zeros(1024, dtype=torch.float32)
    assert t3.data_ptr() == ptr


def test_chunked_reuses_before_growing():
    """An allocation that would grow the heap first reuses blocks freed on every rank."""
    ctx = iris.iris(1 << 20, allocator_type=ALLOC_TYPE)
    alloc = ctx.heap.allocator
    size = alloc.granularity // 4  # float32 elements in one granule

    # Fill the mapped chunks, so the next allocation of this size would grow the heap
    held = [ctx.zeros(size, dtype=torch.float32)]
    while alloc.bump + alloc.granularity <= alloc.mapped_extent:
        held.append(ctx.zeros(size, dtype=torch.float32))
    chunks = alloc.get_num_chunks()
    ptr = held[0].data_ptr()
    del held[0]

    t = ctx.zeros(size, dtype=torch.float32)
    assert t.data_ptr() == ptr
    assert alloc.get_num_chunks() == chunks


def _offsets_per_rank(ctx, t):
    offset = torch.tensor([t.data_ptr() - int(ctx.heap.heap_bases_cpu[ctx.cur_rank])], device=t.device)
    every = [torch.empty_like(offset) for _ in range(ctx.num_ranks)]
    torch.distributed.all_gather(every, offset)
    return [int(o.item()) for o in every]


def test_chunked_offsets_symmetric_across_gc_timing():
    """GC running on one rank before another must not change where the next allocation lands."""
    ctx = iris.iris(64 << 20, allocator_type=ALLOC_TYPE)
    if ctx.num_ranks < 2:
        pytest.skip("Requires at least 2 ranks")

    gc.disable()
    try:
        # A tensor reachable only through a reference cycle: freed when the cyclic GC runs
        a = ctx.zeros(4096, dtype=torch.float32)
        holder = {"t": a}
        holder["self"] = holder
        del a, holder
        if ctx.cur_rank == 0:
            gc.collect()
        b = ctx.zeros(4096, dtype=torch.float32)
        offsets = _offsets_per_rank(ctx, b)
        assert len(set(offsets)) == 1, f"offsets differ across ranks: {offsets}"

        # A block one rank still holds is not handed out, even after collect()
        c = ctx.zeros(4096, dtype=torch.float32)
        c_ptr = c.data_ptr()
        keep = c if ctx.cur_rank == 0 else None
        del c
        ctx.reclaim_freed()
        d = ctx.zeros(4096, dtype=torch.float32)
        offsets = _offsets_per_rank(ctx, d)
        assert len(set(offsets)) == 1, f"offsets differ across ranks: {offsets}"
        assert d.data_ptr() != c_ptr
        # d reuses the block freed everywhere; had c been reused, the bump pointer would sit at c
        e = ctx.zeros(4096, dtype=torch.float32)
        offsets = _offsets_per_rank(ctx, e)
        assert len(set(offsets)) == 1, f"offsets differ across ranks: {offsets}"
        assert e.data_ptr() != c_ptr
        del keep
    finally:
        gc.enable()
    ctx.barrier()


def test_chunked_chunk_growth():
    """Test that allocator grows chunks when needed."""
    # Small chunk size to force growth
    chunk_size = 1 << 20  # 1 MiB chunks
    ctx = iris.iris(
        1 << 20,
        allocator_type=ALLOC_TYPE,
    )
    alloc = ctx.heap.allocator

    initial_chunks = alloc.get_num_chunks()

    # Allocate more than one chunk's worth
    tensors = []
    total = 0
    while alloc.get_num_chunks() <= initial_chunks:
        t = ctx.zeros(32768, dtype=torch.float32)  # 128 KiB each
        tensors.append(t)
        total += 32768 * 4

    assert alloc.get_num_chunks() > initial_chunks


def test_chunked_owns_tensor():
    """Test owns_tensor detection."""
    ctx = iris.iris(1 << 20, allocator_type=ALLOC_TYPE)

    heap_tensor = ctx.zeros(100, dtype=torch.float32)
    assert ctx.heap.allocator.owns_tensor(heap_tensor)

    external_tensor = torch.zeros(100, dtype=torch.float32, device=ctx.device)
    assert not ctx.heap.allocator.owns_tensor(external_tensor)

    del heap_tensor, external_tensor
    torch.cuda.synchronize()
    torch.cuda.empty_cache()


def test_chunked_heap_bases():
    """Test that heap bases are stable and properly set."""
    ctx = iris.iris(1 << 20, allocator_type=ALLOC_TYPE)

    assert ctx.heap_bases.shape == (ctx.num_ranks,)
    base = int(ctx.heap_bases[ctx.cur_rank].item())
    assert base > 0

    # Allocate several tensors -- base should not change
    for _ in range(10):
        ctx.zeros(100, dtype=torch.float32)

    assert int(ctx.heap_bases[ctx.cur_rank].item()) == base


def test_chunked_heap_bases_multirank():
    """Test heap bases across ranks."""
    ctx = iris.iris(1 << 20, allocator_type=ALLOC_TYPE)

    if ctx.num_ranks > 1:
        for peer in range(ctx.num_ranks):
            if peer != ctx.cur_rank:
                assert int(ctx.heap_bases[peer].item()) > 0
                assert int(ctx.heap_bases[peer].item()) != int(ctx.heap_bases[ctx.cur_rank].item())


def test_chunked_import_external_tensor():
    """Test as_symmetric (import external tensor)."""
    ctx = iris.iris(4 << 20, allocator_type=ALLOC_TYPE)

    original = torch.randn(100, dtype=torch.float32, device=ctx.device)
    original_data = original.clone()

    imported = ctx.as_symmetric(original)

    # Should have same data
    assert torch.allclose(imported, original_data)

    # Shared memory -- writes visible both ways
    imported.fill_(42.0)
    assert torch.all(original == 42.0)

    original.fill_(99.0)
    assert torch.all(imported == 99.0)


def test_chunked_import_tensor_survives_ctx():
    """Test that original tensor survives ctx destruction."""
    original = torch.randn(100, dtype=torch.float32, device="cuda")
    original_data = original.clone()

    ctx = iris.iris(4 << 20, allocator_type=ALLOC_TYPE)
    imported = ctx.as_symmetric(original)
    assert torch.allclose(imported, original_data)

    del ctx, imported
    gc.collect()
    torch.cuda.synchronize()

    # Original should still be valid
    assert torch.all(original == original_data)
    original.fill_(123.0)
    assert torch.all(original == 123.0)


def test_chunked_stats():
    """Test allocator statistics."""
    ctx = iris.iris(4 << 20, allocator_type=ALLOC_TYPE)
    alloc = ctx.heap.allocator

    stats = alloc.get_stats()
    assert stats["num_chunks"] >= 1
    assert stats["mapped_bytes"] > 0
    assert stats["va_size"] > 0
    assert stats["granularity"] > 0

    # Allocate some tensors
    t1 = ctx.zeros(100, dtype=torch.float32)
    t2 = ctx.zeros(200, dtype=torch.float32)

    stats = alloc.get_stats()
    assert stats["num_active_allocs"] >= 2
    assert stats["bump"] > 0


def test_chunked_single_rank_does_not_refresh_per_allocation(monkeypatch):
    """Single-rank allocations must not call refresh_peer_access().

    Regression: tying the post-allocation refresh to "chunk count != shared count" without
    gating on num_ranks made 1-rank jobs refresh on every allocation, because
    _shared_chunk_counts is never created for a 1-rank job (so my_shared stayed 0 while the
    real count was >= 1). Single-rank has no peers and a base VA that never moves, so a
    post-allocation refresh is pure overhead.
    """
    ctx = iris.iris(4 << 20, allocator_type=ALLOC_TYPE)
    if ctx.num_ranks != 1:
        pytest.skip("single-rank behavior only")

    calls = []
    monkeypatch.setattr(ctx.heap, "refresh_peer_access", lambda *a, **k: calls.append(1))
    for _ in range(5):
        ctx.zeros(256, dtype=torch.float32)
    assert calls == [], "single-rank allocate must not refresh peer access"


@pytest.mark.skipif(torch.cuda.device_count() < 2, reason="Requires at least 2 GPUs")
def test_chunked_peer_bases_aligned_like_owner():
    """Imported peer arenas must be aligned exactly like the owner's base_va.

    A symmetric object sits at the same offset on every rank, so an importer's translated
    address (heap_bases[peer] + offset) is only aligned to the requested alignment if the peer
    arena base is aligned as the owner's base is. Before the peer-arena fix the arenas were only
    granule-aligned, so an over-aligned allocation looked aligned on the owner but misaligned to
    every importer.
    """
    ctx = iris.iris(1 << 20, allocator_type=ALLOC_TYPE)
    if ctx.num_ranks < 2:
        pytest.skip("Requires at least 2 ranks")
    alloc = ctx.heap.allocator

    # Touch the heap and exchange, so peer arenas are actually reserved.
    t = ctx.zeros(1024, dtype=torch.float32)  # noqa: F841
    ctx.barrier()

    align = alloc.base_alignment
    assert alloc.get_base_address() % align == 0, "owner base must be aligned to base_alignment"
    for peer in range(ctx.num_ranks):
        base = int(ctx.heap_bases[peer].item())
        assert base % align == 0, f"peer {peer} base 0x{base:x} not aligned to 0x{align:x}"


@pytest.mark.skipif(torch.cuda.device_count() < 2, reason="Requires at least 2 GPUs")
def test_chunked_multirank_exchange():
    """Test FD exchange and peer access across ranks."""
    ctx = iris.iris(1 << 20, allocator_type=ALLOC_TYPE)

    if ctx.num_ranks < 2:
        pytest.skip("Requires at least 2 ranks")

    tensor = ctx.zeros(1024, dtype=torch.float32)
    tensor.fill_(float(ctx.cur_rank * 100))

    ctx.barrier()

    # Verify peer heap bases are set
    for peer in range(ctx.num_ranks):
        if peer != ctx.cur_rank:
            assert int(ctx.heap_bases[peer].item()) > 0

    # Verify local data still intact after exchange
    assert torch.all(tensor == float(ctx.cur_rank * 100))


def test_chunked_thread_safety():
    """Test concurrent allocations from multiple threads."""
    ctx = iris.iris(16 << 20, allocator_type=ALLOC_TYPE)
    alloc = ctx.heap.allocator
    results = []
    errors = []

    def alloc_free_loop(thread_id, n):
        try:
            for i in range(n):
                t = alloc.allocate(100, torch.float32)
                t.fill_(float(thread_id * 1000 + i))
                torch.cuda.synchronize()
                val = t[0].item()
                assert val == float(thread_id * 1000 + i)
                results.append(val)
                del t
                gc.collect()
        except Exception as e:
            errors.append((thread_id, e))

    threads = []
    for tid in range(4):
        t = threading.Thread(target=alloc_free_loop, args=(tid, 10))
        threads.append(t)
        t.start()

    for t in threads:
        t.join()

    assert len(errors) == 0, f"Thread errors: {errors}"
    assert len(results) == 40


def test_chunked_close():
    """Test explicit close releases resources."""
    ctx = iris.iris(1 << 20, allocator_type=ALLOC_TYPE)
    alloc = ctx.heap.allocator

    t = ctx.zeros(100, dtype=torch.float32)
    del t
    gc.collect()

    alloc.close()
    assert alloc._closed
    assert len(alloc.chunks) == 0


def test_chunked_no_refresh_on_reuse():
    """Test that reusing freed memory does NOT trigger refresh_peer_access."""
    ctx = iris.iris(4 << 20, allocator_type=ALLOC_TYPE)
    alloc = ctx.heap.allocator

    # Track initial chunk count
    initial_chunks = alloc.get_num_chunks()

    # Alloc-free-reuse cycle should not grow chunks
    for _ in range(20):
        t = ctx.zeros(256, dtype=torch.float32)
        t.fill_(1.0)
        torch.cuda.synchronize()
        del t
        gc.collect()

    assert alloc.get_num_chunks() == initial_chunks


def test_chunked_large_allocation():
    """Test allocation larger than default alignment."""
    ctx = iris.iris(64 << 20, allocator_type=ALLOC_TYPE)

    # 4 MiB tensor
    t = ctx.zeros(1024 * 1024, dtype=torch.float32)
    assert t.shape == (1024 * 1024,)
    t.fill_(7.0)
    torch.cuda.synchronize()
    assert torch.all(t == 7.0)


def test_chunked_mixed_alloc_free_pattern():
    """Test interleaved alloc and free with varying sizes."""
    ctx = iris.iris(32 << 20, allocator_type=ALLOC_TYPE)

    active = []
    for i in range(50):
        size = (i % 5 + 1) * 100
        t = ctx.zeros(size, dtype=torch.float32)
        t.fill_(float(i))
        active.append(t)

        # Free every 3rd tensor
        if i % 3 == 0 and active:
            del active[0]
            gc.collect()

    # Verify remaining tensors
    for t in active:
        torch.cuda.synchronize()
        assert t.numel() > 0


def _make_chunked(heap_size, **kwargs):
    """Construct a VMemChunkedAllocator directly for single-rank unit tests."""
    from iris.host.memory.allocators import VMemChunkedAllocator

    dev = torch.cuda.current_device()
    return VMemChunkedAllocator(heap_size, dev, 0, 1, **kwargs)


def test_chunked_internal_state_not_counted_against_heap():
    """reserved_bytes is pre-mapped on top of heap_size so the user budget stays usable."""
    heap_size = 8 << 20
    reserved = 2 << 20

    baseline = _make_chunked(heap_size)
    try:
        base_mapped = baseline.get_stats()["mapped_bytes"]
    finally:
        baseline.close()

    alloc = _make_chunked(heap_size, reserved_bytes=reserved)
    try:
        gran = alloc.granularity
        rounded_reserved = -(-reserved // gran) * gran
        stats = alloc.get_stats()
        assert stats["reserved_bytes"] == rounded_reserved
        # The reserved budget is mapped on top of the user heap, not carved out of it.
        assert stats["mapped_bytes"] >= heap_size + rounded_reserved
        assert stats["mapped_bytes"] > base_mapped
        assert stats["usable_bytes"] >= heap_size

        # Consuming the whole internal budget as internal allocations still leaves the user able to
        # allocate out of the reserved headroom.
        internal_t = alloc.allocate(rounded_reserved // 4, torch.float32, internal=True)
        assert alloc.get_stats()["internal_bytes"] >= rounded_reserved
        user_t = alloc.allocate(heap_size // 4, torch.float32)
        assert user_t is not None
        del internal_t, user_t
    finally:
        alloc.close()


def test_chunked_reserved_rounds_per_allocation():
    """A list reservation rounds each entry to a granule, so N tiny internal allocations reserve
    N granules -- not one granule for the summed bytes (which would under-reserve)."""
    heap_size = 8 << 20
    # Two separate tiny control allocations (e.g. barrier flags + SDMA counters). Summed they are
    # far under one granule, but each consumes a whole granule when allocated.
    alloc = _make_chunked(heap_size, reserved_bytes=[8, 8])
    try:
        g = alloc.granularity
        stats = alloc.get_stats()
        assert stats["reserved_bytes"] == 2 * g

        # Consuming both as internal allocations stays within the reservation: internal_bytes never
        # exceeds reserved_bytes, so the user's usable budget is untouched.
        a = alloc.allocate(1, torch.int32, internal=True)
        b = alloc.allocate(1, torch.int64, internal=True)
        internal = alloc.get_stats()
        assert internal["internal_bytes"] == 2 * g
        assert internal["internal_bytes"] <= internal["reserved_bytes"]
        assert internal["usable_bytes"] >= heap_size
        del a, b
    finally:
        alloc.close()


def test_chunked_reserved_scalar_rounds_to_granule():
    """A single scalar reservation keeps the simple lump behavior: rounded up to one granule."""
    alloc = _make_chunked(8 << 20, reserved_bytes=8)
    try:
        assert alloc.get_stats()["reserved_bytes"] == alloc.granularity
    finally:
        alloc.close()


def test_chunked_reserved_empty_and_zero_are_noops():
    """An empty list, a list of zeros, or a zero scalar reserve nothing (no accidental granule)."""
    for reserved in ([], [0, 0], 0):
        alloc = _make_chunked(8 << 20, reserved_bytes=reserved)
        try:
            assert alloc.get_stats()["reserved_bytes"] == 0
        finally:
            alloc.close()


def test_chunked_internal_bytes_round_trip_on_free():
    """internal_bytes rises as control state is allocated and falls back as each block is freed --
    the free path must not strand internal bytes (which would permanently shrink usable space)."""
    alloc = _make_chunked(8 << 20, reserved_bytes=[8, 8])
    try:
        g = alloc.granularity
        assert alloc.get_stats()["internal_bytes"] == 0
        a = alloc.allocate(1, torch.int32, internal=True)
        b = alloc.allocate(1, torch.int64, internal=True)
        assert alloc.get_stats()["internal_bytes"] == 2 * g
        # get_stats() processes pending frees, so internal_bytes reflects each GC'd block.
        del a
        gc.collect()
        assert alloc.get_stats()["internal_bytes"] == g
        del b
        gc.collect()
        assert alloc.get_stats()["internal_bytes"] == 0
    finally:
        alloc.close()


def test_chunked_free_internal_leaves_user_intact():
    """Freeing internal control state drops internal_bytes to zero but leaves the user block used --
    the two kinds of accounting must move independently, not clobber each other."""
    alloc = _make_chunked(16 << 20, reserved_bytes=[8])
    try:
        g = alloc.granularity
        internal_t = alloc.allocate(1, torch.int32, internal=True)
        user_t = alloc.allocate(4096, torch.float32)
        used_with_both = alloc.get_stats()["used_bytes"]
        del internal_t
        gc.collect()
        s = alloc.get_stats()
        assert s["internal_bytes"] == 0
        # Exactly the internal granule left used_bytes; the user block is untouched.
        assert s["used_bytes"] == used_with_both - g
        assert s["num_active_allocs"] == 1
        assert user_t.numel() == 4096
        del user_t
    finally:
        alloc.close()


def test_chunked_internal_over_reservation_is_safe():
    """If internal allocations exceed the reservation, accounting stays non-negative and consistent:
    pending_internal clamps at zero, so no capacity figure underflows or goes negative."""
    alloc = _make_chunked(8 << 20, reserved_bytes=[8])  # reserve one granule
    try:
        g = alloc.granularity
        assert alloc.get_stats()["reserved_bytes"] == g
        held = [alloc.allocate(1, torch.int32, internal=True) for _ in range(3)]
        stats = alloc.get_stats()
        assert stats["internal_bytes"] == 3 * g
        assert stats["internal_bytes"] > stats["reserved_bytes"]
        assert stats["usable_bytes"] >= 0
        assert 0 <= stats["largest_allocatable_bytes"] <= stats["usable_bytes"]
        assert stats["used_bytes"] >= stats["internal_bytes"]
        del held
    finally:
        alloc.close()


def test_chunked_internal_accounting_not_committed_on_build_failure(monkeypatch):
    """If tensor construction raises, the internal reservation must NOT have been counted -- otherwise
    internal_bytes over-counts and a phantom offset lingers in _internal_offsets for an allocation
    that never produced a tensor (and so has no finalizer to undo it). Reverting the fix (counting
    internal bytes before the build) leaves internal_bytes advanced here and fails this test."""
    alloc = _make_chunked(8 << 20, reserved_bytes=[8])
    try:
        before = alloc.get_stats()["internal_bytes"]
        monkeypatch.setattr(torch, "as_tensor", lambda *a, **k: (_ for _ in ()).throw(RuntimeError("boom")))
        with pytest.raises(RuntimeError):
            alloc.allocate(1, torch.int32, internal=True)
        # Patch is still active here, but get_stats()/_process_pending_frees() do not build tensors.
        assert alloc.get_stats()["internal_bytes"] == before
    finally:
        alloc.close()


def test_chunked_capacity_report_fields():
    """get_stats() exposes usable/used/free/largest/internal/reserved with sane invariants."""
    ctx = iris.iris(4 << 20, allocator_type=ALLOC_TYPE)
    alloc = ctx.heap.allocator
    stats = alloc.get_stats()
    for key in (
        "reserved_bytes",
        "internal_bytes",
        "used_bytes",
        "free_in_mapped_bytes",
        "usable_bytes",
        "largest_allocatable_bytes",
    ):
        assert key in stats, f"missing capacity field {key!r}"

    # Iris allocates its device-barrier flags (and SDMA counters, if enabled) as internal control
    # state. The reservation must cover all of it, so heap_size stays fully usable to the caller.
    assert stats["internal_bytes"] > 0
    assert stats["internal_bytes"] <= stats["reserved_bytes"]
    assert stats["used_bytes"] >= stats["internal_bytes"]
    assert stats["largest_allocatable_bytes"] <= stats["usable_bytes"]

    before = alloc.get_stats()["used_bytes"]
    t = ctx.zeros(4096, dtype=torch.float32)
    after = alloc.get_stats()["used_bytes"]
    assert after > before
    del t


def test_chunked_sizing_helper_roundtrip():
    """plan_capacity() predicts a size whose built allocator reports >= the target usable bytes."""
    from iris.host.memory.allocators import VMemChunkedAllocator

    target = 16 << 20
    reserved = 1 << 20
    plan = VMemChunkedAllocator.plan_capacity(target, reserved_bytes=reserved)
    assert plan["usable_bytes"] >= target
    assert plan["mapped_bytes"] >= target + plan["reserved_bytes"]

    alloc = _make_chunked(target, reserved_bytes=reserved)
    try:
        assert alloc.get_stats()["usable_bytes"] >= target
    finally:
        alloc.close()


def test_plan_capacity_list_reservation_rounds_per_allocation():
    """plan_capacity() rounds a *list* reservation per allocation, matching a real list-built
    allocator. Summing the bytes and rounding once (the old behavior) would under-reserve: here two
    tiny control allocations each consume a whole granule, so the reservation must be two granules,
    not one. A one-line revert to scalar rounding of the summed bytes turns 2*g into g and fails."""
    from iris.host.memory.allocators import VMemChunkedAllocator

    target = 16 << 20
    g = 2 << 20  # exaggerated granule so each tiny allocation clearly rounds up to a full granule
    plan_list = VMemChunkedAllocator.plan_capacity(target, reserved_bytes=[8, 8], granularity=g)
    assert plan_list["reserved_bytes"] == 2 * g

    # The same summed bytes as a single lump reserve only one granule -- the under-reservation the
    # list form fixes. This asserts plan_capacity actually distinguishes the two.
    plan_lump = VMemChunkedAllocator.plan_capacity(target, reserved_bytes=16, granularity=g)
    assert plan_lump["reserved_bytes"] == g

    # Zero/empty entries are dropped, not rounded up to a spurious granule.
    assert VMemChunkedAllocator.plan_capacity(target, reserved_bytes=[0, 0], granularity=g)["reserved_bytes"] == 0
    assert VMemChunkedAllocator.plan_capacity(target, reserved_bytes=[], granularity=g)["reserved_bytes"] == 0


def test_internal_reservation_sizes_cover_iris_control_state():
    """The reservation list Iris hands the heap must include *every* control-state allocation it
    makes: device-barrier flags always, and the SDMA all-gather counters when SDMA is enabled. The
    bug this guards: the SDMA counters were omitted from the reservation, so with engines=("sdma",)
    the last granule of heap_size was silently consumed by Iris. Pure wiring check (no SDMA hardware
    needed) -- a real engines=("sdma",) run would require rocm-xio, which is unavailable here."""
    from iris.host.iris import _internal_reservation_sizes

    num_ranks = 8
    without = _internal_reservation_sizes(num_ranks, sdma_enabled=False)
    assert without == [num_ranks * num_ranks * torch.int32.itemsize]  # matches _device_barrier_flags int32[R, R]

    with_sdma = _internal_reservation_sizes(num_ranks, sdma_enabled=True)
    # Enabling SDMA adds exactly one more reserved allocation (int64[R] counters); it never drops the
    # flags or changes their size. A revert that forgets the SDMA entry makes with_sdma == without.
    assert with_sdma == without + [num_ranks * torch.int64.itemsize]


def test_chunked_oom_message_has_breakdown():
    """An out-of-capacity error prints the capacity breakdown, not an opaque message."""
    alloc = _make_chunked(1 << 20, chunk_size=1 << 20, va_size=4 << 20)
    try:
        tensors = []
        with pytest.raises(RuntimeError) as excinfo:
            for _ in range(64):
                tensors.append(alloc.allocate(256 * 1024, torch.float32))  # 1 MiB each
        msg = str(excinfo.value).lower()
        for token in ("usable", "largest", "va_size"):
            assert token in msg, f"OOM message missing {token!r}: {excinfo.value}"
        del tensors
    finally:
        alloc.close()


if __name__ == "__main__":
    test_chunked_creation()
    test_chunked_basic_allocation()
    test_chunked_multiple_allocations()
    test_chunked_gc_free_reuse()
