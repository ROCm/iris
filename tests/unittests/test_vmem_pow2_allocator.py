# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.

"""
Tests for the power-of-two VMem allocator (VMemPow2Allocator).
"""

import gc
import threading

import pytest
import torch

import iris
from iris.allocators.vmem_pow2_allocator import _next_pow2


# ---------------------------------------------------------------------------
# Unit tests for _next_pow2 helper
# ---------------------------------------------------------------------------


def test_next_pow2_small():
    assert _next_pow2(1) == 1
    assert _next_pow2(2) == 2
    assert _next_pow2(3) == 4
    assert _next_pow2(4) == 4
    assert _next_pow2(5) == 8
    assert _next_pow2(7) == 8
    assert _next_pow2(8) == 8
    assert _next_pow2(9) == 16


def test_next_pow2_large():
    assert _next_pow2(1 << 20) == 1 << 20
    assert _next_pow2((1 << 20) + 1) == 1 << 21


def test_next_pow2_one():
    assert _next_pow2(0) == 1


# ---------------------------------------------------------------------------
# Allocator creation
# ---------------------------------------------------------------------------


def test_vmem_pow2_allocator_creation():
    """VMemPow2Allocator can be created via the iris context."""
    ctx = iris.iris(4 << 20, allocator_type="vmem_pow2")

    assert ctx.cur_rank >= 0
    assert ctx.num_ranks >= 1
    assert ctx.heap_size == 4 << 20

    from iris.allocators.vmem_pow2_allocator import VMemPow2Allocator

    assert isinstance(ctx.heap.allocator, VMemPow2Allocator)


# ---------------------------------------------------------------------------
# Basic allocation
# ---------------------------------------------------------------------------


def test_vmem_pow2_basic_allocation():
    """Basic tensor allocation and write."""
    ctx = iris.iris(8 << 20, allocator_type="vmem_pow2")

    tensor = ctx.zeros(1024, dtype=torch.float32)

    assert tensor.shape == (1024,)
    assert tensor.device.type == "cuda"
    assert torch.all(tensor == 0)

    tensor.fill_(42.0)
    assert torch.all(tensor == 42.0)


def test_vmem_pow2_multiple_allocations():
    """Multiple allocations from the same heap."""
    ctx = iris.iris(16 << 20, allocator_type="vmem_pow2")

    tensors = []
    for i in range(8):
        t = ctx.zeros(256, dtype=torch.float32)
        t.fill_(float(i))
        tensors.append(t)

    for i, t in enumerate(tensors):
        assert torch.all(t == float(i)), f"Tensor {i} has wrong value."


def test_vmem_pow2_different_dtypes():
    """Allocations with different dtypes."""
    ctx = iris.iris(16 << 20, allocator_type="vmem_pow2")

    t_f32 = ctx.zeros(128, dtype=torch.float32)
    t_f16 = ctx.zeros(128, dtype=torch.float16)
    t_i32 = ctx.zeros(128, dtype=torch.int32)

    t_f32.fill_(1.0)
    t_f16.fill_(2.0)
    t_i32.fill_(3)

    assert torch.all(t_f32 == 1.0)
    assert torch.all(t_f16 == 2.0)
    assert torch.all(t_i32 == 3)


# ---------------------------------------------------------------------------
# owns_tensor
# ---------------------------------------------------------------------------


def test_vmem_pow2_owns_tensor():
    """owns_tensor correctly identifies heap vs. non-heap tensors."""
    ctx = iris.iris(8 << 20, allocator_type="vmem_pow2")

    heap_tensor = ctx.zeros(100, dtype=torch.float32)
    assert ctx.heap.allocator.owns_tensor(heap_tensor), "Heap tensor should be owned."

    external = torch.zeros(100, dtype=torch.float32, device=ctx.device)
    assert not ctx.heap.allocator.owns_tensor(external), "External tensor should not be owned."

    # Zero-element tensors NOT from the heap must NOT be claimed as owned.
    external_empty = torch.zeros(0, dtype=torch.float32, device=ctx.device)
    assert not ctx.heap.allocator.owns_tensor(external_empty), (
        "External zero-element tensor must not be claimed as owned."
    )

    del heap_tensor, external, external_empty
    torch.cuda.synchronize()
    torch.cuda.empty_cache()


# ---------------------------------------------------------------------------
# Free-list reuse
# ---------------------------------------------------------------------------


def test_vmem_pow2_free_reuse():
    """After free(), the same VA is returned on the next allocate() of the same size class."""
    ctx = iris.iris(16 << 20, allocator_type="vmem_pow2")
    allocator = ctx.heap.allocator

    t1 = ctx.zeros(512, dtype=torch.float32)
    ptr1 = t1.data_ptr()

    allocator.free(t1)

    t2 = ctx.zeros(512, dtype=torch.float32)
    ptr2 = t2.data_ptr()

    assert ptr2 == ptr1, f"Expected VA reuse 0x{ptr1:x}, got 0x{ptr2:x}."


def test_vmem_pow2_free_reuse_multiple():
    """Multiple free + realloc cycles for different size classes all reuse VAs."""
    ctx = iris.iris(64 << 20, allocator_type="vmem_pow2")
    allocator = ctx.heap.allocator

    for num_elems in [64, 256, 1024]:
        t = ctx.zeros(num_elems, dtype=torch.float32)
        ptr = t.data_ptr()
        allocator.free(t)

        t2 = ctx.zeros(num_elems, dtype=torch.float32)
        assert t2.data_ptr() == ptr, f"Expected VA reuse for {num_elems} elements."


def test_vmem_pow2_free_wrong_tensor_raises():
    """Freeing a tensor not allocated by this allocator raises ValueError."""
    ctx = iris.iris(8 << 20, allocator_type="vmem_pow2")
    allocator = ctx.heap.allocator

    external = torch.zeros(64, dtype=torch.float32, device=ctx.device)
    with pytest.raises(ValueError):
        allocator.free(external)


# ---------------------------------------------------------------------------
# GC-based auto-free
# ---------------------------------------------------------------------------


def test_vmem_pow2_gc_auto_free():
    """Tensors that go out of scope are automatically returned to the free list."""
    ctx = iris.iris(16 << 20, allocator_type="vmem_pow2")
    allocator = ctx.heap.allocator

    def alloc_and_drop():
        t = ctx.zeros(512, dtype=torch.float32)
        return t.data_ptr()  # tensor dies at function exit (CPython refcount → 0)

    ptr = alloc_and_drop()
    gc.collect()  # ensure finalizer has run across all Python implementations

    # The next allocation of the same size class must reuse the freed VA.
    t2 = ctx.zeros(512, dtype=torch.float32)
    assert t2.data_ptr() == ptr, f"Expected GC-freed VA 0x{ptr:x} to be reused, got 0x{t2.data_ptr():x}."


# ---------------------------------------------------------------------------
# Heap bases
# ---------------------------------------------------------------------------


def test_vmem_pow2_heap_bases():
    """Heap bases are properly initialised."""
    ctx = iris.iris(4 << 20, allocator_type="vmem_pow2")

    assert ctx.heap_bases.shape == (ctx.num_ranks,)
    assert int(ctx.heap_bases[ctx.cur_rank].item()) > 0

    if ctx.num_ranks > 1:
        for peer in range(ctx.num_ranks):
            if peer != ctx.cur_rank:
                assert int(ctx.heap_bases[peer].item()) > 0
                assert int(ctx.heap_bases[peer].item()) != int(ctx.heap_bases[ctx.cur_rank].item())


# ---------------------------------------------------------------------------
# Granularity alignment
# ---------------------------------------------------------------------------


def test_vmem_pow2_granularity_alignment():
    """The aligned heap size must be a multiple of the HIP granularity."""
    from iris.hip import get_allocation_granularity

    torch.cuda.synchronize()
    torch.cuda.empty_cache()

    ctx = iris.iris(4 << 20, allocator_type="vmem_pow2")
    granularity = get_allocation_granularity(ctx.gpu_id)

    assert ctx.heap.allocator.aligned_heap_size % granularity == 0


# ---------------------------------------------------------------------------
# Size-class rounding
# ---------------------------------------------------------------------------


def test_vmem_pow2_size_class_rounding():
    """
    Two allocation sizes that map to the same power-of-two size class can
    interchangeably reuse each other's freed VA block.
    """
    ctx = iris.iris(64 << 20, allocator_type="vmem_pow2")
    allocator = ctx.heap.allocator
    granularity = allocator.granularity

    # Both sizes round up to 2*granularity.
    size_a = granularity + 1  # bytes
    size_b = granularity + granularity // 2  # bytes

    elems_a = size_a  # dtype=torch.int8 → element_size == 1
    elems_b = size_b

    t_a = allocator.allocate(elems_a, torch.int8)
    ptr_a = t_a.data_ptr()
    allocator.free(t_a)

    t_b = allocator.allocate(elems_b, torch.int8)
    ptr_b = t_b.data_ptr()

    assert ptr_b == ptr_a, f"Expected VA reuse: both sizes share size class. ptr_a=0x{ptr_a:x}, ptr_b=0x{ptr_b:x}"


# ---------------------------------------------------------------------------
# stats()
# ---------------------------------------------------------------------------


def test_vmem_pow2_stats():
    """stats() returns sensible values before, during, and after an allocation."""
    ctx = iris.iris(8 << 20, allocator_type="vmem_pow2")
    allocator = ctx.heap.allocator

    s0 = allocator.stats()
    assert s0["heap_size"] == 8 << 20
    assert s0["granularity"] > 0
    assert s0["num_live_allocations"] == 0

    t = ctx.zeros(512, dtype=torch.float32)
    s1 = allocator.stats()
    assert s1["num_live_allocations"] == 1

    allocator.free(t)
    s2 = allocator.stats()
    assert s2["num_live_allocations"] == 0


# ---------------------------------------------------------------------------
# get_allocation_segments() and generation counter
# ---------------------------------------------------------------------------


def test_vmem_pow2_allocation_segments_grow():
    """
    get_allocation_segments() grows when new physical segments are mapped
    and does NOT grow when free-listed VAs are reused (remap same entry).
    """
    ctx = iris.iris(64 << 20, allocator_type="vmem_pow2")
    allocator = ctx.heap.allocator

    seg_count_0 = len(allocator.get_allocation_segments())

    # First allocation: new physical segment.
    t1 = ctx.zeros(512, dtype=torch.float32)
    seg_count_1 = len(allocator.get_allocation_segments())
    assert seg_count_1 == seg_count_0 + 1

    # Free + reallocate: reuse VA, no new entry in allocation_order.
    allocator.free(t1)
    t2 = ctx.zeros(512, dtype=torch.float32)
    seg_count_2 = len(allocator.get_allocation_segments())
    assert seg_count_2 == seg_count_1, "Free-list reuse must not create a new segment entry."


def test_vmem_pow2_generation_increments_on_remap():
    """Generation counter increases when a freed VA is remapped."""
    ctx = iris.iris(16 << 20, allocator_type="vmem_pow2")
    allocator = ctx.heap.allocator

    t = ctx.zeros(512, dtype=torch.float32)
    va = t.data_ptr()
    offset = va - allocator.base_va

    gen_before = allocator._segment_generation[offset]

    allocator.free(t)
    _ = ctx.zeros(512, dtype=torch.float32)  # remap same offset

    gen_after = allocator._segment_generation[offset]
    assert gen_after == gen_before + 1, "Generation must increment on remap."


# ---------------------------------------------------------------------------
# OOM / heap exhaustion
# ---------------------------------------------------------------------------


def test_vmem_pow2_oom():
    """Allocating beyond the VA space raises RuntimeError."""
    # A heap of exactly 4 granules; bootstrap uses 1, so we have 3 left.
    ctx = iris.iris(4 << 20, allocator_type="vmem_pow2")
    allocator = ctx.heap.allocator

    tensors = []
    with pytest.raises(RuntimeError, match="out of VA space"):
        while True:
            tensors.append(allocator.allocate(1, torch.int8))


# ---------------------------------------------------------------------------
# Thread safety
# ---------------------------------------------------------------------------


def test_vmem_pow2_thread_safety():
    """Concurrent alloc/free from multiple threads does not corrupt state."""
    ctx = iris.iris(256 << 20, allocator_type="vmem_pow2")
    allocator = ctx.heap.allocator
    errors: list = []

    def worker():
        try:
            for _ in range(20):
                t = allocator.allocate(16, torch.float32)
                allocator.free(t)
        except Exception as exc:  # noqa: BLE001 – capture any error from any thread
            errors.append(exc)

    threads = [threading.Thread(target=worker) for _ in range(4)]
    for th in threads:
        th.start()
    for th in threads:
        th.join()

    assert not errors, f"Thread-safety errors: {errors}"


# ---------------------------------------------------------------------------
# close() and resource cleanup
# ---------------------------------------------------------------------------


def test_vmem_pow2_close():
    """close() releases all resources and is idempotent."""
    ctx = iris.iris(8 << 20, allocator_type="vmem_pow2")
    allocator = ctx.heap.allocator

    t = ctx.zeros(512, dtype=torch.float32)
    allocator.free(t)

    allocator.close()
    assert allocator._closed
    assert allocator.base_va == 0

    # close() must be safe to call multiple times.
    allocator.close()
    assert allocator._closed


def test_vmem_pow2_close_disables_finalizers():
    """close() detaches GC finalizers so they cannot run after the allocator is gone."""
    ctx = iris.iris(8 << 20, allocator_type="vmem_pow2")
    allocator = ctx.heap.allocator

    t = ctx.zeros(512, dtype=torch.float32)
    va = t.data_ptr()

    assert va in allocator._finalizers
    allocator.close()
    assert not allocator._finalizers, "All finalizers must be cleared by close()."


# ---------------------------------------------------------------------------
# as_symmetric() (import_external_tensor)
# ---------------------------------------------------------------------------


def test_vmem_pow2_import_external_tensor():
    """
    Importing an external tensor gives a symmetric-heap view that shares
    physical memory; the original tensor remains valid after ctx is destroyed.
    """
    ctx = iris.iris(8 << 20, allocator_type="vmem_pow2")

    original = torch.randn(64, dtype=torch.float32, device=ctx.device)
    original_data = original.clone()

    imported = ctx.as_symmetric(original)
    assert torch.allclose(imported, original_data), "Imported data should match original."

    imported.fill_(7.0)
    assert torch.all(original == 7.0), "Original should see changes through shared memory."

    original.fill_(13.0)
    assert torch.all(imported == 13.0), "Imported should see changes through shared memory."

    del ctx, imported
    gc.collect()
    torch.cuda.synchronize()

    assert torch.all(original == 13.0), "Original tensor should survive ctx destruction."
    original.fill_(99.0)
    assert torch.all(original == 99.0), "Original tensor should still be writable."


# ---------------------------------------------------------------------------
# Multi-rank tests
# ---------------------------------------------------------------------------


def test_vmem_pow2_multirank_heap_bases():
    """Multi-rank: each rank sees all peers' heap bases after setup."""
    ctx = iris.iris(4 << 20, allocator_type="vmem_pow2")

    tensor = ctx.zeros(1024, dtype=torch.float32)
    tensor.fill_(float(ctx.cur_rank * 100))

    assert ctx.heap_bases.shape == (ctx.num_ranks,)
    assert int(ctx.heap_bases[ctx.cur_rank].item()) > 0

    if ctx.num_ranks > 1:
        for peer in range(ctx.num_ranks):
            if peer != ctx.cur_rank:
                assert int(ctx.heap_bases[peer].item()) > 0
                assert int(ctx.heap_bases[peer].item()) != int(ctx.heap_bases[ctx.cur_rank].item())

    ctx.barrier()
    tensor.fill_(float(ctx.cur_rank * 100))
    ctx.barrier()
    assert torch.all(tensor == float(ctx.cur_rank * 100))


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    test_next_pow2_small()
    test_next_pow2_large()
    test_next_pow2_one()
    test_vmem_pow2_allocator_creation()
    test_vmem_pow2_basic_allocation()
    test_vmem_pow2_multiple_allocations()
    test_vmem_pow2_different_dtypes()
    test_vmem_pow2_owns_tensor()
    test_vmem_pow2_free_reuse()
    test_vmem_pow2_free_reuse_multiple()
    test_vmem_pow2_gc_auto_free()
    test_vmem_pow2_heap_bases()
    test_vmem_pow2_granularity_alignment()
    test_vmem_pow2_size_class_rounding()
    test_vmem_pow2_stats()
    test_vmem_pow2_allocation_segments_grow()
    test_vmem_pow2_generation_increments_on_remap()
    test_vmem_pow2_oom()
    test_vmem_pow2_thread_safety()
    test_vmem_pow2_close()
    test_vmem_pow2_close_disables_finalizers()
    test_vmem_pow2_import_external_tensor()
    test_vmem_pow2_multirank_heap_bases()
    print("All VMemPow2Allocator tests passed.")
