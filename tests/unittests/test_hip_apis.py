# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.

"""
Test HIP API functionality with actual GPU operations.
"""

import torch
import pytest


def test_get_address_range_basic():
    """Test get_address_range with a simple tensor."""
    from iris.hip import get_address_range

    # Allocate a tensor
    tensor = torch.randn(1000, 1000, device="cuda", dtype=torch.float32)
    ptr = tensor.data_ptr()

    # Query the base allocation
    base_ptr, size = get_address_range(ptr)

    # Verify results
    assert base_ptr > 0, f"Expected valid base pointer, got {base_ptr}"
    assert size > 0, f"Expected valid size, got {size}"

    # For a freshly allocated tensor, ptr should be at or near the base
    # (may have small offset due to PyTorch's caching allocator)
    offset = ptr - base_ptr
    assert offset >= 0, f"Pointer {hex(ptr)} is before base {hex(base_ptr)}"
    assert offset < size, f"Offset {offset} exceeds allocation size {size}"

    # Size should be at least as large as the tensor
    tensor_size = tensor.element_size() * tensor.numel()
    assert size >= tensor_size, f"Allocation size {size} < tensor size {tensor_size}"


def test_get_address_range_offset_pointer():
    """Test get_address_range with offset pointers (sliced tensors)."""
    from iris.hip import get_address_range

    # Allocate a large tensor
    large_tensor = torch.randn(5000, 1000, device="cuda", dtype=torch.float32)
    large_ptr = large_tensor.data_ptr()
    large_base, large_size = get_address_range(large_ptr)

    # Create a slice (offset pointer)
    # Skip first 1000 rows = 1000 * 1000 * 4 bytes = 4,000,000 bytes
    slice_tensor = large_tensor[1000:2000, :]
    slice_ptr = slice_tensor.data_ptr()
    slice_base, slice_size = get_address_range(slice_ptr)

    # The slice should report the same base allocation
    assert slice_base == large_base, f"Slice base {hex(slice_base)} != large base {hex(large_base)}"
    assert slice_size == large_size, f"Slice size {slice_size} != large size {large_size}"

    # Verify offset calculation
    offset = slice_ptr - slice_base
    expected_offset = 1000 * 1000 * 4  # Skip 1000 rows of float32
    assert offset >= expected_offset, f"Offset {offset} < expected {expected_offset}"


def test_get_address_range_multiple_allocations():
    """Test get_address_range with multiple allocations (caching allocator)."""
    from iris.hip import get_address_range

    # Allocate multiple small tensors
    # PyTorch's caching allocator may allocate these from the same base buffer
    tensors = []
    for i in range(3):
        t = torch.zeros(128, 128, device="cuda", dtype=torch.float32)
        t.fill_(float(i))
        tensors.append(t)

    # Query each tensor's base allocation
    for i, tensor in enumerate(tensors):
        ptr = tensor.data_ptr()
        base_ptr, size = get_address_range(ptr)
        offset = ptr - base_ptr

        assert base_ptr > 0, f"Tensor {i}: Invalid base pointer"
        assert size > 0, f"Tensor {i}: Invalid size"
        assert offset >= 0, f"Tensor {i}: Negative offset"
        assert offset < size, f"Tensor {i}: Offset exceeds size"


def test_get_address_range_contiguous():
    """Test get_address_range with contiguous and non-contiguous tensors."""
    from iris.hip import get_address_range

    # Create contiguous tensor
    contiguous = torch.randn(5000, 1000, device="cuda", dtype=torch.float32)
    assert contiguous.is_contiguous(), "Tensor should be contiguous"

    cont_ptr = contiguous.data_ptr()
    cont_base, cont_size = get_address_range(cont_ptr)

    # Create non-contiguous tensor (transpose)
    transposed = contiguous.t()
    assert not transposed.is_contiguous(), "Transposed tensor should be non-contiguous"

    trans_ptr = transposed.data_ptr()
    trans_base, trans_size = get_address_range(trans_ptr)

    # Both should report the same base (share underlying storage)
    assert trans_base == cont_base, "Non-contiguous tensor should share same base"
    assert trans_size == cont_size, "Non-contiguous tensor should share same size"
    assert trans_ptr == cont_ptr, "Transpose shares same data pointer"


@pytest.mark.parametrize("dtype", [torch.float32, torch.float16, torch.int64, torch.int32])
def test_get_address_range_dtypes(dtype):
    """Test get_address_range with different data types."""
    from iris.hip import get_address_range

    # Use randint for integer types, randn for float types
    if dtype in [torch.int64, torch.int32]:
        tensor = torch.randint(0, 100, (100, 100), device="cuda", dtype=dtype)
    else:
        tensor = torch.randn(100, 100, device="cuda", dtype=dtype)

    ptr = tensor.data_ptr()
    base_ptr, size = get_address_range(ptr)

    assert base_ptr > 0, f"Invalid base pointer for dtype {dtype}"
    assert size > 0, f"Invalid size for dtype {dtype}"

    # Verify size is at least tensor size
    tensor_size = tensor.element_size() * tensor.numel()
    assert size >= tensor_size, f"Size {size} < tensor size {tensor_size} for dtype {dtype}"


@pytest.mark.parametrize("shape", [(10,), (10, 20), (5, 10, 15), (2, 3, 4, 5)])
def test_get_address_range_shapes(shape):
    """Test get_address_range with different tensor shapes."""
    from iris.hip import get_address_range

    tensor = torch.randn(*shape, device="cuda", dtype=torch.float32)
    ptr = tensor.data_ptr()
    base_ptr, size = get_address_range(ptr)

    assert base_ptr > 0, f"Invalid base pointer for shape {shape}"
    assert size > 0, f"Invalid size for shape {shape}"

    # Verify size is sufficient
    tensor_size = tensor.element_size() * tensor.numel()
    assert size >= tensor_size, f"Size {size} < tensor size {tensor_size} for shape {shape}"
