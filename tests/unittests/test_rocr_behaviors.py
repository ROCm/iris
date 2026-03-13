# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.

"""
Test specific ROCr/HIP behaviors that Iris relies on.

These tests validate undocumented or non-obvious behaviors in the ROCr runtime
that are critical for correctness.

Known ROCm issues tested:
- hipMemSetAccess cumulative bug: https://github.com/ROCm/rocm-systems/issues/2667
- hipMemExportToShareableHandle returns base allocation (undocumented behavior)
"""

import torch
import os


def test_dmabuf_export_returns_base_allocation():
    """
    Test that hipMemExportToShareableHandle returns the BASE allocation,
    not the suballocated pointer.

    This is critical for import_external_tensor() to work correctly.
    Uses tensor slicing to guarantee an offset.
    """
    from iris.hip import export_dmabuf_handle, get_address_range

    # Create a tensor and slice it to guarantee an offset
    full_tensor = torch.randn(1000, 1000, device="cuda", dtype=torch.float32)

    # Slice to create an offset pointer (skip 100 rows = 100*1000*4 = 400KB)
    sliced_tensor = full_tensor[100:200, :]

    ptr = sliced_tensor.data_ptr()
    size = sliced_tensor.element_size() * sliced_tensor.numel()

    # Get the base allocation
    alloc_base, alloc_size = get_address_range(ptr)
    offset = ptr - alloc_base

    # Verify we have an offset (slice guarantees this)
    assert offset > 0, f"Sliced tensor should have non-zero offset, got {offset}"

    # CRITICAL TEST: Export DMA-BUF from the sliced (offset) pointer
    fd, export_base, export_size = export_dmabuf_handle(ptr, size)

    try:
        # ASSERTION 1: Export should return the BASE allocation, not ptr
        assert export_base == alloc_base, (
            f"DMA-BUF export should return base allocation {hex(alloc_base)}, but got {hex(export_base)}"
        )

        # ASSERTION 2: Export size should be the full allocation, not just slice size
        assert export_size == alloc_size, (
            f"DMA-BUF export size should be full allocation {alloc_size}, but got {export_size}"
        )

        assert export_base < ptr, f"For offset pointer, export_base {hex(export_base)} should be < pointer {hex(ptr)}"

    finally:
        os.close(fd)


def test_dmabuf_export_with_large_offset():
    """
    Test DMA-BUF export with a guaranteed large offset (using slicing).

    This ensures the export returns base allocation even for pointers
    far from the allocation start.
    """
    from iris.hip import export_dmabuf_handle, get_address_range

    # Create a large tensor
    large_tensor = torch.randn(10000, 1000, device="cuda", dtype=torch.float32)

    # Create a slice with guaranteed large offset
    # Skip 5000 rows = 5000 * 1000 * 4 bytes = 20MB offset
    slice_tensor = large_tensor[5000:6000, :]

    slice_ptr = slice_tensor.data_ptr()
    slice_size = slice_tensor.element_size() * slice_tensor.numel()

    # Get base allocation
    alloc_base, alloc_size = get_address_range(slice_ptr)
    offset = slice_ptr - alloc_base

    # Verify we have a significant offset
    expected_offset = 5000 * 1000 * 4  # 20MB
    assert offset >= expected_offset, f"Expected offset >= {expected_offset}, got {offset}"

    # Export DMA-BUF from the slice pointer (which is offset)
    fd, export_base, export_size = export_dmabuf_handle(slice_ptr, slice_size)

    try:
        # CRITICAL: Export should return base, not slice_ptr
        assert export_base == alloc_base, (
            f"DMA-BUF export from offset ptr {hex(slice_ptr)} should return "
            f"base {hex(alloc_base)}, but got {hex(export_base)}"
        )

        # Export size should be full allocation
        assert export_size == alloc_size, f"DMA-BUF export size should be {alloc_size}, got {export_size}"

        # Verify offset calculation is correct
        assert slice_ptr == alloc_base + offset, "Offset calculation mismatch"

    finally:
        os.close(fd)


def test_get_address_range_consistency():
    """
    Test that get_address_range consistently returns the same base for
    multiple pointers within the same allocation.
    """
    from iris.hip import get_address_range

    # Create a large tensor
    tensor = torch.randn(1000, 1000, device="cuda", dtype=torch.float32)

    # Get base for various offsets within the same tensor
    bases = []
    sizes = []

    # Test multiple slice offsets
    for start_row in [0, 100, 500, 900]:
        slice_tensor = tensor[start_row : start_row + 10, :]
        ptr = slice_tensor.data_ptr()
        base, size = get_address_range(ptr)
        bases.append(base)
        sizes.append(size)

    assert len(set(bases)) == 1, f"All slices should have same base, got {[hex(b) for b in bases]}"
    assert len(set(sizes)) == 1, f"All slices should have same size, got {sizes}"

    base = bases[0]
    for start_row in [0, 100, 500, 900]:
        slice_tensor = tensor[start_row : start_row + 10, :]
        ptr = slice_tensor.data_ptr()
        assert ptr >= base, f"Pointer {hex(ptr)} should be >= base {hex(base)}"


def test_get_address_range_with_guaranteed_offsets():
    """
    Test that get_address_range() correctly handles offset pointers.

    Uses tensor slicing to guarantee offsets (not dependent on caching allocator).
    """
    from iris.hip import get_address_range

    # Create a large tensor
    large_tensor = torch.randn(1000, 1000, device="cuda", dtype=torch.float32)

    # Get base for the full tensor
    full_ptr = large_tensor.data_ptr()
    full_base, full_size = get_address_range(full_ptr)

    # Create multiple slices with different offsets
    test_cases = [
        (100, 200, 100 * 1000 * 4),  # Skip 100 rows
        (500, 600, 500 * 1000 * 4),  # Skip 500 rows
        (900, 950, 900 * 1000 * 4),  # Skip 900 rows
    ]

    for start_row, end_row, expected_min_offset in test_cases:
        slice_tensor = large_tensor[start_row:end_row, :]
        slice_ptr = slice_tensor.data_ptr()
        slice_base, slice_size = get_address_range(slice_ptr)

        # ASSERTION 1: Slice should report same base as full tensor
        assert slice_base == full_base, (
            f"Slice [{start_row}:{end_row}] should have base {hex(full_base)}, got {hex(slice_base)}"
        )

        # ASSERTION 2: Slice should report same size as full tensor
        assert slice_size == full_size, f"Slice [{start_row}:{end_row}] should have size {full_size}, got {slice_size}"

        # ASSERTION 3: Slice pointer should have expected offset
        actual_offset = slice_ptr - slice_base
        assert actual_offset >= expected_min_offset, (
            f"Slice [{start_row}:{end_row}] should have offset >= {expected_min_offset}, got {actual_offset}"
        )


def test_vmem_allocator_uses_cumulative_access():
    """
    Test that VMemAllocator's cumulative access pattern works correctly.

    This validates the ROCr workaround where hipMemSetAccess must be called
    from base_va with cumulative size, not on individual sub-regions.

    ROCm bug: https://github.com/ROCm/rocm-systems/issues/2667

    We test this indirectly by using VMemAllocator and verifying multiple
    allocations work correctly.
    """
    import torch.distributed as dist

    from iris.allocators.vmem_allocator import VMemAllocator

    rank = dist.get_rank()
    world_size = dist.get_world_size()
    device_id = rank  # num_ranks == num_gpus: one GPU per rank
    heap_size = 1 << 20  # 1 MB

    allocator = VMemAllocator(heap_size, device_id, rank=rank, world_size=world_size)

    try:
        # Test multiple allocations (each triggers cumulative mem_set_access)
        tensors = []
        for i in range(5):
            t = allocator.allocate(100 * 100, torch.float32)
            t.fill_(float(i))
            tensors.append(t)

        for i, t in enumerate(tensors):
            torch.cuda.synchronize()
            assert torch.all(t == float(i)), f"Tensor {i} should be accessible and have value {i}"

        # If cumulative access pattern didn't work, some tensors
        # would fail to be accessible (segfault or wrong values)

        # Verify cumulative_mapped_size is tracking correctly (ROCm workaround)
        assert allocator.cumulative_mapped_size >= allocator.current_offset, (
            "cumulative_mapped_size should be >= current_offset"
        )

    finally:
        allocator.close()
        torch.cuda.synchronize()
        torch.cuda.empty_cache()
        import gc

        gc.collect()
        torch.cuda.synchronize()


def test_dmabuf_import_cleanup_preserves_original():
    """
    Test that importing a DMA-BUF and then cleaning up the import
    does NOT corrupt the original PyTorch tensor.

    This is the test for as_symmetric() lifetime contract:
    - Import creates a new mapping to the same physical memory
    - Cleanup destroys the import mapping
    - Original tensor should remain valid

    Uses only HIP functions, no Iris dependency.
    """
    from iris.hip import (
        export_dmabuf_handle,
        import_dmabuf_handle,
        destroy_external_memory,
        get_address_range,
    )
    import torch.distributed as dist

    # Use the same device selection logic as Iris: rank % num_gpus
    if dist.is_initialized():
        rank = dist.get_rank()
        num_gpus = torch.cuda.device_count()
        device_id = rank % num_gpus
    else:
        device_id = torch.cuda.current_device()

    device = f"cuda:{device_id}"

    # Create original PyTorch tensor
    original_tensor = torch.randn(100, dtype=torch.float32, device=device)
    original_tensor.fill_(42.0)

    # Verify original works
    assert torch.all(original_tensor == 42.0), "Original tensor should work"

    # Get original pointer
    original_ptr = original_tensor.data_ptr()
    alloc_base, alloc_size = get_address_range(original_ptr)

    # Export as DMA-BUF
    dmabuf_fd, export_base, export_size = export_dmabuf_handle(alloc_base, alloc_size)

    # Import DMA-BUF (creates new mapping and returns handle for cleanup)
    remapped_ptr, ext_mem_handle = import_dmabuf_handle(dmabuf_fd, export_size, original_ptr, export_base)

    # Create imported tensor from remapped pointer
    class CUDAArrayInterface:
        def __init__(self, ptr, size_bytes):
            self.ptr = ptr
            self.size_bytes = size_bytes

        @property
        def __cuda_array_interface__(self):
            return {
                "shape": (self.size_bytes,),
                "typestr": "|u1",
                "data": (self.ptr, False),
                "version": 3,
            }

    tensor_size = original_tensor.numel() * original_tensor.element_size()
    cuda_array = CUDAArrayInterface(remapped_ptr, tensor_size)
    imported_bytes = torch.as_tensor(cuda_array, device=device)
    imported_tensor = imported_bytes.view(torch.float32)

    # Verify shared memory works
    assert torch.all(imported_tensor == 42.0), "Imported tensor should see original data"

    imported_tensor.fill_(99.0)
    assert torch.all(original_tensor == 99.0), "Original should see imported changes"
    assert torch.all(imported_tensor == 99.0), "Imported should see its own changes"

    # Destroy imported tensor and external memory
    # This should clean up the import mapping but NOT affect original
    del imported_tensor, imported_bytes, cuda_array
    import gc

    gc.collect()
    torch.cuda.synchronize()

    # Destroy the external memory handle (this is what close() should do)
    print("Destroying external memory handle...")
    destroy_external_memory(ext_mem_handle)
    torch.cuda.synchronize()

    # Original tensor should still be valid and functional
    print("Testing original tensor after import cleanup...")

    # Can we read it?
    value = original_tensor[0].item()
    assert value == 99.0, f"Original tensor corrupted! Got {value}, expected 99.0"

    # Can we check all values?
    assert torch.all(original_tensor == 99.0), "Original tensor should still have correct values!"

    # Can we modify it?
    original_tensor.fill_(123.0)
    assert torch.all(original_tensor == 123.0), "Original tensor should still be modifiable!"

    # Can we do operations?
    result = original_tensor + 1.0
    assert torch.all(result == 124.0), "Original tensor should still support operations!"

    print("DMA-BUF import cleanup preserves original tensor")
