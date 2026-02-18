# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.

"""
Test PyTorch tensor import mechanism and as_symmetric() functionality.
"""

import torch
import pytest
import iris


def test_dmabuf_import_with_offset():
    """Validate PyTorch tensor import with offset preservation (mechanism used by as_symmetric)."""
    from iris.hip import (
        export_dmabuf_handle,
        get_address_range,
        import_dmabuf_handle,
        destroy_external_memory,
    )

    device = torch.device("cuda", torch.cuda.current_device())
    large_tensor = torch.randn(10000, dtype=torch.float32, device=device)
    sliced_tensor = large_tensor[1000:2000]
    sliced_tensor.fill_(42.0)
    torch.cuda.set_device(device)
    original_ptr = sliced_tensor.data_ptr()
    original_size = sliced_tensor.element_size() * sliced_tensor.numel()

    alloc_base, alloc_size = get_address_range(original_ptr)
    offset_in_alloc = original_ptr - alloc_base
    assert offset_in_alloc > 0, f"Expected non-zero offset, got {offset_in_alloc}"
    expected_min_offset = 1000 * 4
    assert offset_in_alloc >= expected_min_offset, f"Expected offset >= {expected_min_offset}, got {offset_in_alloc}"

    dmabuf_fd, export_base, export_size = export_dmabuf_handle(original_ptr, original_size)
    assert export_base == alloc_base, f"Export should return base {hex(alloc_base)}, got {hex(export_base)}"
    assert export_size == alloc_size, f"Export should return full size {alloc_size}, got {export_size}"

    try:
        remapped_ptr, ext_mem_handle = import_dmabuf_handle(dmabuf_fd, export_size, original_ptr, export_base)

        class CUDAArrayInterface:
            def __init__(self, ptr, size):
                self.ptr = ptr
                self.size = size

            @property
            def __cuda_array_interface__(self):
                return {
                    "shape": (self.size // 4,),  # float32 = 4 bytes
                    "typestr": "<f4",
                    "data": (self.ptr, False),
                    "version": 3,
                }

        cuda_array = CUDAArrayInterface(remapped_ptr, original_size)
        remapped_tensor = torch.as_tensor(cuda_array, device=device)

        torch.cuda.synchronize(device)
        assert remapped_tensor.shape == sliced_tensor.shape, (
            f"Shape mismatch: remapped={remapped_tensor.shape}, original={sliced_tensor.shape}"
        )

        assert torch.allclose(remapped_tensor, sliced_tensor), (
            "Remapped tensor should have same data as original!\n"
            f"Original: {sliced_tensor[:10].tolist()}\n"
            f"Remapped: {remapped_tensor[:10].tolist()}"
        )
        assert torch.all(remapped_tensor == 42.0), "Remapped tensor should have value 42.0"

        remapped_tensor.fill_(99.0)
        torch.cuda.synchronize(device)
        torch.cuda.synchronize(device)
        assert torch.all(sliced_tensor == 99.0), (
            "Original tensor should reflect write to remapped tensor (same physical memory)"
        )
        assert torch.all(remapped_tensor == 99.0), "Remapped tensor should have new value"

    finally:
        destroy_external_memory(ext_mem_handle)


def test_dmabuf_import_no_offset():
    """Import with no offset (tensor at start of allocation)."""
    from iris.hip import (
        export_dmabuf_handle,
        get_address_range,
        import_dmabuf_handle,
        destroy_external_memory,
    )

    torch.cuda.empty_cache()
    tensor = torch.zeros(1000, dtype=torch.float32, device="cuda")
    tensor.fill_(123.0)

    original_ptr = tensor.data_ptr()
    original_size = tensor.element_size() * tensor.numel()
    alloc_base, alloc_size = get_address_range(original_ptr)
    offset_in_alloc = original_ptr - alloc_base
    dmabuf_fd, export_base, export_size = export_dmabuf_handle(original_ptr, original_size)

    assert export_base == alloc_base

    try:
        remapped_ptr, ext_mem_handle = import_dmabuf_handle(dmabuf_fd, export_size, original_ptr, export_base)

        class CUDAArrayInterface:
            def __init__(self, ptr, size):
                self.ptr = ptr
                self.size = size

            @property
            def __cuda_array_interface__(self):
                return {
                    "shape": (self.size // 4,),
                    "typestr": "<f4",
                    "data": (self.ptr, False),
                    "version": 3,
                }

        cuda_array = CUDAArrayInterface(remapped_ptr, original_size)
        remapped_tensor = torch.as_tensor(cuda_array, device="cuda")

        torch.cuda.synchronize()
        assert torch.all(remapped_tensor == 123.0)

    finally:
        destroy_external_memory(ext_mem_handle)


def test_as_symmetric_basic():
    """Basic as_symmetric(): native VMem tensor and imported external tensor in same VA space."""
    ctx = iris.iris(64 << 20, allocator_type="vmem")

    native_tensor = ctx.zeros(1000, dtype=torch.float32)
    native_tensor.fill_(42.0)
    native_ptr = native_tensor.data_ptr()
    heap_base = ctx.heap.allocator.get_base_address()
    assert native_ptr >= heap_base, "Native tensor should be in our VA space"

    external_tensor = torch.randn(500, dtype=torch.float32, device="cuda")
    external_tensor.fill_(99.0)
    external_ptr = external_tensor.data_ptr()
    imported_tensor = ctx.as_symmetric(external_tensor)
    imported_ptr = imported_tensor.data_ptr()

    assert imported_ptr != external_ptr, (
        f"Imported tensor should be remapped, external={hex(external_ptr)}, imported={hex(imported_ptr)}"
    )
    assert imported_ptr >= heap_base, "Imported tensor should be in our VA space"

    torch.cuda.synchronize()
    assert torch.all(native_tensor == 42.0)
    assert torch.all(imported_tensor == 99.0)

    imported_tensor.fill_(123.0)
    torch.cuda.synchronize()
    assert torch.all(imported_tensor == 123.0)
    assert torch.all(external_tensor == 123.0)


def test_as_symmetric_with_offset():
    """as_symmetric() with sliced tensor (guaranteed offset)."""
    ctx = iris.iris(64 << 20, allocator_type="vmem")

    large_external = torch.randn(10000, dtype=torch.float32, device="cuda")
    sliced_external = large_external[1000:2000]
    sliced_external.fill_(55.0)
    imported_slice = ctx.as_symmetric(sliced_external)

    assert imported_slice.data_ptr() != sliced_external.data_ptr()
    torch.cuda.synchronize()
    assert torch.all(imported_slice == 55.0)

    imported_slice.fill_(77.0)
    torch.cuda.synchronize()
    assert torch.all(imported_slice == 77.0)


def test_as_symmetric_multiple_imports():
    """Import multiple external tensors; verify they don't interfere with each other."""
    ctx = iris.iris(64 << 20, allocator_type="vmem")

    native1 = ctx.zeros(500, dtype=torch.float32)
    native1.fill_(1.0)

    external_tensors = []
    imported_tensors = []
    for i in range(5):
        ext = torch.zeros(200, dtype=torch.float32, device="cuda")
        ext.fill_(float(10 + i))
        external_tensors.append(ext)

        imp = ctx.as_symmetric(ext)
        imported_tensors.append(imp)

    native2 = ctx.zeros(500, dtype=torch.float32)
    native2.fill_(2.0)

    torch.cuda.synchronize()

    assert torch.all(native1 == 1.0), "Native1 should be accessible"
    assert torch.all(native2 == 2.0), "Native2 should be accessible"

    for i, imp in enumerate(imported_tensors):
        expected_val = float(10 + i)
        assert torch.all(imp == expected_val), f"Imported tensor {i} should have value {expected_val}"


def test_as_symmetric_with_different_dtypes():
    """as_symmetric() with various dtypes."""
    ctx = iris.iris(64 << 20, allocator_type="vmem")

    test_cases = [
        (torch.float32, 42.0),
        (torch.float16, 16.0),
        (torch.int32, 123),
        (torch.int64, 456),
    ]

    for dtype, fill_value in test_cases:
        if dtype in [torch.float32, torch.float16]:
            ext = torch.randn(100, dtype=dtype, device="cuda")
        else:
            ext = torch.randint(0, 1000, (100,), dtype=dtype, device="cuda")

        ext.fill_(fill_value)
        imp = ctx.as_symmetric(ext)
        torch.cuda.synchronize()
        assert torch.all(imp == fill_value), f"Failed for dtype {dtype}"


def test_as_symmetric_preserves_shape():
    """as_symmetric() preserves tensor shape and strides."""
    ctx = iris.iris(64 << 20, allocator_type="vmem")

    shapes = [
        (100,),
        (10, 20),
        (5, 10, 4),
    ]

    for shape in shapes:
        ext = torch.randn(*shape, dtype=torch.float32, device="cuda")
        imp = ctx.as_symmetric(ext)
        assert imp.shape == ext.shape, f"Shape mismatch for {shape}"
        assert imp.is_contiguous(), f"Imported tensor should be contiguous for {shape}"


def test_as_symmetric_error_non_cuda():
    """as_symmetric() raises for non-CUDA tensors."""
    ctx = iris.iris(64 << 20, allocator_type="vmem")
    cpu_tensor = torch.randn(100, dtype=torch.float32)

    with pytest.raises(RuntimeError, match="CUDA"):
        ctx.as_symmetric(cpu_tensor)


def test_as_symmetric_integration():
    """
    Full integration test: native allocations + imports + computations.

    Tests the real use case: mixing native and imported tensors in computations.
    """
    # Heap must fit PyTorch caching allocator blocks (often 2MB)
    ctx = iris.iris(64 << 20, allocator_type="vmem")  # 64 MB heap

    # Step 1: Create some native tensors
    a = ctx.zeros(1000, dtype=torch.float32)
    a.fill_(1.0)

    b = ctx.zeros(1000, dtype=torch.float32)
    b.fill_(2.0)

    # Step 2: Import an external tensor
    external = torch.ones(1000, dtype=torch.float32, device="cuda") * 3.0
    c = ctx.as_symmetric(external)

    # Step 3: Do computations mixing native and imported
    result = a + b + c  # 1 + 2 + 3 = 6

    torch.cuda.synchronize()
    assert torch.all(result == 6.0), "Mixed computation should work correctly"

    # Step 4: Verify original tensors are still correct
    assert torch.all(a == 1.0), "Native tensor a should be unchanged"
    assert torch.all(b == 2.0), "Native tensor b should be unchanged"
    assert torch.all(c == 3.0), "Imported tensor c should be unchanged"
    assert torch.all(external == 3.0), "External tensor should be unchanged"
