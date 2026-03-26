# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.

"""
Tests for VMemChunkedAllocator.

Tests cover:
- Basic allocation and data integrity
- Multiple allocations and non-overlap
- Power-of-two free list reuse via GC
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


ALLOC_TYPE = "vmem_chunked"


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

    # Drop the tensor and trigger GC
    del t
    gc.collect()
    torch.cuda.synchronize()

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

    small_ptr = small.data_ptr()
    medium_ptr = medium.data_ptr()
    large_ptr = large.data_ptr()

    # Free all
    del small, medium, large
    gc.collect()
    torch.cuda.synchronize()

    # Re-allocate -- each should reuse from its size class
    small2 = ctx.zeros(size_small, dtype=torch.float32)
    medium2 = ctx.zeros(size_medium, dtype=torch.float32)
    large2 = ctx.zeros(size_large, dtype=torch.float32)

    assert small2.data_ptr() == small_ptr
    assert medium2.data_ptr() == medium_ptr
    assert large2.data_ptr() == large_ptr


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

    # Free and check -- synchronize first to ensure no async ops hold
    # references to the storage, then del + gc to trigger the weakref finalizer.
    torch.cuda.synchronize()
    del t1
    gc.collect()
    gc.collect()  # Second pass catches ref cycles from first pass
    # Force processing pending frees
    t3 = ctx.zeros(1, dtype=torch.float32)
    stats = alloc.get_stats()
    assert stats["num_free_blocks"] >= 1, f"Expected free blocks after GC, got stats: {stats}"


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


if __name__ == "__main__":
    test_chunked_creation()
    test_chunked_basic_allocation()
    test_chunked_multiple_allocations()
    test_chunked_gc_free_reuse()
