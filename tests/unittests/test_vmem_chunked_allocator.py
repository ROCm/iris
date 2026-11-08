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


def test_chunked_granularity_reports_min_and_recommended():
    """Driver/allocator must report a true minimum and a recommended granule, not conflate them.

    This is an end-to-end sanity check only. The authoritative per-flag routing coverage is the
    GPU-free driver mock tests in test_drivers.py; on stacks where minimum == recommended this
    test cannot by itself distinguish a correctly fixed driver from a broken one, so it asserts
    only the invariants (min <= recommended, recommended a multiple of min, stats expose both).
    """
    ctx = iris.iris(1 << 20, allocator_type=ALLOC_TYPE)
    alloc = ctx.heap.allocator
    driver = alloc.driver

    min_g = driver.get_minimum_granularity()
    rec_g = driver.get_recommended_granularity()
    assert min_g > 0 and rec_g > 0
    assert min_g <= rec_g, f"minimum {min_g} must not exceed recommended {rec_g}"
    assert rec_g % min_g == 0, f"recommended {rec_g} must be a multiple of minimum {min_g}"

    stats = alloc.get_stats()
    assert stats["min_granularity"] == min_g
    assert stats["recommended_granularity"] == rec_g
    # The allocator backs memory with the recommended granule.
    assert alloc.granularity == rec_g


def test_chunked_honors_requested_alignment_above_granule():
    """allocate(alignment=...) above the granule must align the absolute address.

    Repro: before the fix the alignment argument was ignored beyond the fixed min_alignment,
    so an over-aligned request's data_ptr() was not a multiple of the requested alignment.
    """
    ctx = iris.iris(64 << 20, allocator_type=ALLOC_TYPE)
    alloc = ctx.heap.allocator
    base = alloc.get_base_address()
    alignment = 4 * alloc.granularity

    # Push the bump pointer to a non-(4*granule) multiple so alignment is actually exercised.
    sentinel = alloc.allocate(alloc.granularity // 4, torch.float32)  # one granule
    assert sentinel is not None

    t = alloc.allocate(1024, torch.float32, alignment=alignment)
    assert t is not None
    # The *absolute address*, not merely the heap offset, must be aligned: callers pass
    # alignment= for TMA descriptors / tl.multiple_of / page-aligned DMA, which look at data_ptr().
    assert t.data_ptr() % alignment == 0, f"address 0x{t.data_ptr():x} not aligned to 0x{alignment:x}"
    offset = t.data_ptr() - base
    assert offset % alignment == 0

    # Keep a live allocation ABOVE the over-aligned block so that, when freed, it lands in a
    # free-list bucket instead of merging straight back into the bump pointer. Without this the
    # block would be at the top of the heap and the "reuse" below would really be a fresh bump
    # that only coincidentally re-aligns -- not a test of the (size, alignment) free-list key.
    above = alloc.allocate(alloc.granularity // 4, torch.float32)
    assert above is not None

    del t
    ctx.reclaim_freed()
    bucket = next(b for b, offs in alloc.free_lists.items() if offset in offs)
    assert bucket.alignment == max(alloc.min_alignment, alignment), "block must be filed by its alignment"
    bump_before_reuse = alloc.bump

    # The next request of the same size/alignment pops that exact block from the free list --
    # the bump pointer does not move, proving a real free-list reuse keyed by (size, alignment).
    t2 = alloc.allocate(1024, torch.float32, alignment=alignment)
    assert alloc.bump == bump_before_reuse, "expected a free-list reuse, not a bump allocation"
    assert t2.data_ptr() % alignment == 0
    assert (t2.data_ptr() - base) == offset, "expected the freed over-aligned block to be reused"


def test_chunked_over_aligned_block_returns_bump_to_start():
    """Freeing an over-aligned block must reclaim its alignment padding, not strand it.

    Repro (MAJOR): allocate one granule, then a block aligned to 4*granule (which skips padding
    to reach the aligned offset), free both and reclaim. Before the fix the bump pointer rolled
    back only to the aligned offset -- the skipped padding, and every block below it, was lost to
    free lists and the heap could never shrink back below it.
    """
    ctx = iris.iris(64 << 20, allocator_type=ALLOC_TYPE)
    alloc = ctx.heap.allocator
    bump_before = alloc.bump
    alignment = 4 * alloc.granularity

    a = alloc.allocate(alloc.granularity // 4, torch.float32)  # one granule, no padding
    b = alloc.allocate(1024, torch.float32, alignment=alignment)  # forces padding before b
    assert a is not None and b is not None
    assert alloc.bump > bump_before

    del a, b
    ctx.reclaim_freed()

    assert alloc.bump == bump_before, "bump did not return to its starting value -- padding stranded"
    assert alloc.get_stats()["num_free_blocks"] == 0, "blocks stranded in free lists instead of rolled back"


def test_chunked_alignment_above_max_is_rejected():
    """An alignment larger than the heap's guaranteed maximum must raise, not silently misalign.

    The heap base is reserved aligned to MAX_ALIGNMENT; beyond that the absolute address can no
    longer be guaranteed aligned, so the allocator rejects the request instead of returning a
    pointer that only looks aligned in heap-offset space.
    """
    ctx = iris.iris(4 << 20, allocator_type=ALLOC_TYPE)
    alloc = ctx.heap.allocator
    with pytest.raises(ValueError, match="exceeds the heap's maximum alignment"):
        alloc.allocate(1024, torch.float32, alignment=alloc._max_alignment * 2)


def test_chunked_rejects_operations_after_close():
    """allocate() after close() must raise instead of silently succeeding."""
    ctx = iris.iris(1 << 20, allocator_type=ALLOC_TYPE)
    alloc = ctx.heap.allocator
    alloc.close()
    with pytest.raises(RuntimeError):
        alloc.allocate(128, torch.float32)


def test_chunked_import_external_tensor_rejects_after_close():
    """import_external_tensor() after close() must raise, matching allocate()'s guard.

    Repro: before the fix, only allocate() was guarded against use-after-close; this entry
    point would run past a closed allocator and touch driver/bookkeeping state that close()
    already tore down.
    """
    ctx = iris.iris(4 << 20, allocator_type=ALLOC_TYPE)
    alloc = ctx.heap.allocator
    original = torch.randn(16, dtype=torch.float32, device=ctx.device)
    alloc.close()
    with pytest.raises(RuntimeError):
        alloc.import_external_tensor(original)


def test_chunked_import_peer_chunk_rejects_after_close():
    """import_peer_chunk() after close() must raise before touching the torn-down driver.

    The handle bytes below are not a real peer handle: _check_open() must reject the call
    before the allocator ever tries to parse or import it, so garbage input is harmless here.
    """
    ctx = iris.iris(1 << 20, allocator_type=ALLOC_TYPE)
    alloc = ctx.heap.allocator
    alloc.close()
    with pytest.raises(RuntimeError):
        alloc.import_peer_chunk(0, b"\x00" * 64, alloc.granularity)


def test_chunked_get_allocation_chunks_since_rejects_after_close():
    """get_allocation_chunks_since() after close() must raise instead of exporting stale chunks.

    Repro: before the fix, this could run after close() released self.chunks/_shared_regions
    and either return an empty/garbage result or fail deep inside driver.export_handle instead
    of a clear, immediate error.
    """
    ctx = iris.iris(1 << 20, allocator_type=ALLOC_TYPE)
    alloc = ctx.heap.allocator
    alloc.close()
    with pytest.raises(RuntimeError):
        alloc.get_allocation_chunks_since(0)


def test_chunked_non_power_of_two_heap_size():
    """A non-power-of-two heap size below the default chunk must construct, allocate, and grow.

    Repro: 6 MiB with a 2 MiB granule produced a non-power-of-two chunk size and raised
    "computed chunk_size ... is not a power of two" during construction.
    """
    ctx = iris.iris(6 << 20, allocator_type=ALLOC_TYPE)
    alloc = ctx.heap.allocator
    assert not _is_pow2(alloc.chunk_size), "test wants a non-power-of-two chunk_size to exercise growth"

    t = ctx.zeros(1024, dtype=torch.float32)
    t.fill_(3.0)
    torch.cuda.synchronize()
    assert torch.all(t == 3.0)

    # Allocate past the first chunk so chunk growth runs with a non-power-of-two chunk_size,
    # not just initial construction. The VA rounding there uses math.ceil, not a bitmask.
    chunks0 = alloc.get_num_chunks()
    hold = [t]
    while alloc.get_num_chunks() == chunks0:
        hold.append(ctx.zeros(alloc.chunk_size // 4, dtype=torch.float32))
    assert alloc.get_num_chunks() > chunks0
    hold[-1].fill_(5.0)
    torch.cuda.synchronize()
    assert torch.all(hold[-1] == 5.0)


def _is_pow2(n):
    return n > 0 and (n & (n - 1)) == 0


def test_chunked_failed_allocation_rolls_back(monkeypatch):
    """A tensor-construction failure must not advance bump/alloc bookkeeping.

    Repro: before the fix, bump and alloc_sizes were mutated before the tensor was
    built, so a failure left an orphaned reservation (leak) with no finalizer.
    """
    ctx = iris.iris(4 << 20, allocator_type=ALLOC_TYPE)
    alloc = ctx.heap.allocator

    bump_before = alloc.bump
    active_before = alloc.get_stats()["num_active_allocs"]

    # Narrowly fail only the allocator's own tensor construction (patch the interface class it
    # uses), not the whole process's torch.as_tensor.
    monkeypatch.setattr(vmem_chunked_module, "_CUDAArrayInterface", _ConstructionBoom)
    with pytest.raises(RuntimeError):
        alloc.allocate(1024, torch.float32)
    monkeypatch.undo()

    assert alloc.bump == bump_before
    assert alloc.get_stats()["num_active_allocs"] == active_before

    # The allocator is still usable after the failed allocation.
    t = alloc.allocate(1024, torch.float32)
    assert t is not None


def test_chunked_failed_allocation_rolls_back_free_list_block(monkeypatch):
    """A tensor-construction failure on the free-list reuse path must return the block.

    This exercises the other half of the rollback fix: when allocate() serves a request from
    free_lists[bucket] (not the bump pointer), it pops the block *before* building the tensor.
    Repro: before the fix, a construction failure here would drop that offset on the floor --
    no bump/alloc_sizes damage (that part never ran), but the block would be leaked forever
    since no finalizer was ever attached to recover it.
    """
    ctx = iris.iris(4 << 20, allocator_type=ALLOC_TYPE)
    alloc = ctx.heap.allocator
    base = alloc.get_base_address()

    # Allocate a block and bury it under a second, live allocation so reclaim_freed() files it
    # into a free-list bucket instead of merging it straight back into the bump pointer.
    t = alloc.allocate(1024, torch.float32)
    assert t is not None
    freed_offset = t.data_ptr() - base
    keep_alive = alloc.allocate(1024, torch.float32)
    assert keep_alive is not None
    del t
    ctx.reclaim_freed()

    bucket = next(bucket for bucket, offsets in alloc.free_lists.items() if freed_offset in offsets)
    free_list_before = list(alloc.free_lists[bucket])
    bump_before = alloc.bump

    monkeypatch.setattr(vmem_chunked_module, "_CUDAArrayInterface", _ConstructionBoom)
    with pytest.raises(RuntimeError):
        alloc.allocate(1024, torch.float32)
    monkeypatch.undo()

    # The bump pointer is untouched (the free-list path never advances it), and the popped
    # block is back in the same bucket it came from.
    assert alloc.bump == bump_before
    assert sorted(alloc.free_lists[bucket]) == sorted(free_list_before)
    assert freed_offset in alloc.free_lists[bucket]

    # It's reusable afterward, landing at the same offset.
    t2 = alloc.allocate(1024, torch.float32)
    assert (t2.data_ptr() - base) == freed_offset


def test_chunked_failed_allocation_during_growth_grows_nothing(monkeypatch):
    """A construction failure on an allocation that must grow a chunk must grow nothing.

    This is the third rollback case (beyond the bump and free-list paths): a request larger than
    the remaining mapped extent. Because the tensor is built BEFORE _grow_chunk() runs, a failure
    leaves no chunk mapped-but-unexported -- which would otherwise be invisible locally but let a
    peer RMA into unmapped VA. Verifies no chunk was added and the bump pointer did not move.
    """
    ctx = iris.iris(1 << 20, allocator_type=ALLOC_TYPE)
    alloc = ctx.heap.allocator
    chunks_before = alloc.get_num_chunks()
    bump_before = alloc.bump

    # Sized to overflow the currently mapped extent, forcing at least one _grow_chunk().
    grow_elems = (alloc.mapped_extent - alloc.bump + alloc.chunk_size) // 4

    monkeypatch.setattr(vmem_chunked_module, "_CUDAArrayInterface", _ConstructionBoom)
    with pytest.raises(RuntimeError):
        alloc.allocate(grow_elems, torch.float32)
    monkeypatch.undo()

    assert alloc.get_num_chunks() == chunks_before, "a chunk was grown despite the failed allocation"
    assert alloc.bump == bump_before

    # The same request succeeds once construction works, and this time it does grow.
    t = alloc.allocate(grow_elems, torch.float32)
    assert t is not None
    assert alloc.get_num_chunks() > chunks_before


def test_chunked_reuses_block_with_larger_alignment(monkeypatch):
    """A freed block may serve a same-size request needing a smaller alignment.

    A block filed under a larger alignment already sits at an offset that satisfies any smaller
    (power-of-two) alignment, so reuse is safe and avoids fragmenting the heap. The choice is
    deterministic (smallest sufficient alignment first) and depends only on free-list contents,
    which are identical across ranks, so offsets stay symmetric.
    """
    ctx = iris.iris(64 << 20, allocator_type=ALLOC_TYPE)
    alloc = ctx.heap.allocator
    big_align = 4 * alloc.granularity

    # Free a highly-aligned block into a free list (keep one above it so it is not rolled back).
    t = alloc.allocate(1024, torch.float32, alignment=big_align)
    offset = t.data_ptr() - alloc.get_base_address()
    above = alloc.allocate(alloc.granularity // 4, torch.float32)  # noqa: F841
    assert above is not None
    del t
    ctx.reclaim_freed()
    assert any(offset in offs for b, offs in alloc.free_lists.items() if b.alignment == big_align)

    # A same-size request with the default (smaller) alignment reuses that exact block.
    bump_before = alloc.bump
    t2 = alloc.allocate(1024, torch.float32)  # default alignment < big_align
    assert alloc.bump == bump_before, "expected cross-alignment free-list reuse, not a bump allocation"
    assert (t2.data_ptr() - alloc.get_base_address()) == offset


if __name__ == "__main__":
    test_chunked_creation()
    test_chunked_basic_allocation()
    test_chunked_multiple_allocations()
    test_chunked_gc_free_reuse()
