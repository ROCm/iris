# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.

"""
Test is_symmetric() functionality for checking if tensors are on symmetric heap.
"""

import torch
import iris


def test_is_symmetric_basic():
    """Test basic is_symmetric() functionality with torch allocator."""
    ctx = iris.iris(1 << 20, allocator_type="torch")

    # Create a symmetric tensor
    symmetric_tensor = ctx.zeros(1000, dtype=torch.float32)
    assert ctx.is_symmetric(symmetric_tensor), "Tensor allocated by ctx should be on symmetric heap"

    # Create an external tensor (not on symmetric heap)
    external_tensor = torch.zeros(1000, dtype=torch.float32, device="cuda")
    assert not ctx.is_symmetric(external_tensor), "External tensor should not be on symmetric heap"


def test_is_symmetric_various_dtypes():
    """Test is_symmetric() with different data types."""
    ctx = iris.iris(1 << 20, allocator_type="torch")

    dtypes = [torch.float16, torch.float32, torch.float64, torch.int32, torch.int64]

    for dtype in dtypes:
        # Symmetric tensor
        symmetric = ctx.zeros(100, dtype=dtype)
        assert ctx.is_symmetric(symmetric), f"Symmetric tensor with dtype {dtype} should return True"

        # External tensor
        external = torch.zeros(100, dtype=dtype, device="cuda")
        assert not ctx.is_symmetric(external), f"External tensor with dtype {dtype} should return False"


def test_is_symmetric_various_shapes():
    """Test is_symmetric() with different tensor shapes."""
    ctx = iris.iris(1 << 20, allocator_type="torch")

    shapes = [
        (100,),
        (10, 20),
        (5, 10, 4),
        (2, 3, 4, 5),
    ]

    for shape in shapes:
        # Symmetric tensor
        symmetric = ctx.zeros(shape, dtype=torch.float32)
        assert ctx.is_symmetric(symmetric), f"Symmetric tensor with shape {shape} should return True"

        # External tensor
        external = torch.zeros(shape, dtype=torch.float32, device="cuda")
        assert not ctx.is_symmetric(external), f"External tensor with shape {shape} should return False"


def test_is_symmetric_multiple_allocations():
    """Test is_symmetric() with multiple allocations."""
    ctx = iris.iris(1 << 20, allocator_type="torch")

    # Create multiple symmetric tensors
    symmetric_tensors = [ctx.zeros(100, dtype=torch.float32) for _ in range(10)]

    # All should be on symmetric heap
    for i, tensor in enumerate(symmetric_tensors):
        assert ctx.is_symmetric(tensor), f"Symmetric tensor {i} should return True"

    # Create multiple external tensors
    external_tensors = [torch.zeros(100, dtype=torch.float32, device="cuda") for _ in range(5)]

    # None should be on symmetric heap
    for i, tensor in enumerate(external_tensors):
        assert not ctx.is_symmetric(tensor), f"External tensor {i} should return False"


def test_is_symmetric_vmem_allocator():
    """Test is_symmetric() with vmem allocator."""
    ctx = iris.iris(64 << 20, allocator_type="vmem")

    # Create a symmetric tensor
    symmetric_tensor = ctx.zeros(1000, dtype=torch.float32)
    assert ctx.is_symmetric(symmetric_tensor), "Tensor allocated by ctx with vmem should be on symmetric heap"

    # Create an external tensor
    external_tensor = torch.zeros(1000, dtype=torch.float32, device="cuda")
    assert not ctx.is_symmetric(external_tensor), "External tensor should not be on symmetric heap with vmem"


def test_is_symmetric_imported_tensor():
    """Test is_symmetric() with imported external tensor (vmem allocator)."""
    ctx = iris.iris(64 << 20, allocator_type="vmem")

    # Create an external tensor
    external_tensor = torch.randn(500, dtype=torch.float32, device="cuda")
    external_tensor.fill_(99.0)

    # External tensor should not be on symmetric heap
    assert not ctx.is_symmetric(external_tensor), "External tensor before import should return False"

    # Import it
    imported_tensor = ctx.as_symmetric(external_tensor)

    # Imported tensor should be on symmetric heap
    assert ctx.is_symmetric(imported_tensor), "Imported tensor should return True"

    # Original external tensor still not on symmetric heap
    assert not ctx.is_symmetric(external_tensor), "Original external tensor should still return False"


def test_is_symmetric_different_allocators():
    """Test is_symmetric() with different allocator instances."""
    ctx1 = iris.iris(1 << 20, allocator_type="torch")
    ctx2 = iris.iris(1 << 20, allocator_type="torch")

    # Create tensor on ctx1
    tensor1 = ctx1.zeros(100, dtype=torch.float32)

    # Tensor should be symmetric on ctx1
    assert ctx1.is_symmetric(tensor1), "Tensor should be on ctx1's symmetric heap"

    # Tensor should not be symmetric on ctx2 (different heap)
    assert not ctx2.is_symmetric(tensor1), "Tensor from ctx1 should not be on ctx2's symmetric heap"

    # Create tensor on ctx2
    tensor2 = ctx2.zeros(100, dtype=torch.float32)

    # Tensor should be symmetric on ctx2
    assert ctx2.is_symmetric(tensor2), "Tensor should be on ctx2's symmetric heap"

    # Tensor should not be symmetric on ctx1 (different heap)
    assert not ctx1.is_symmetric(tensor2), "Tensor from ctx2 should not be on ctx1's symmetric heap"


def test_is_symmetric_consistency():
    """Test that is_symmetric() is consistent with internal validation."""
    ctx = iris.iris(1 << 20, allocator_type="torch")

    # Create a symmetric tensor
    symmetric_tensor = ctx.zeros(100, dtype=torch.float32)

    # is_symmetric should return True
    assert ctx.is_symmetric(symmetric_tensor), "Symmetric tensor should return True"

    # This should work without raising an error (internal validation uses is_symmetric)
    result = ctx.zeros(100, dtype=torch.float32, out=symmetric_tensor)
    assert result is symmetric_tensor


def test_is_symmetric_zeros_ones_full():
    """Test is_symmetric() with tensors from zeros, ones, and full."""
    ctx = iris.iris(1 << 20, allocator_type="torch")

    # Test with zeros
    zeros_tensor = ctx.zeros(100, dtype=torch.float32)
    assert ctx.is_symmetric(zeros_tensor), "Tensor from zeros() should be on symmetric heap"

    # Test with ones
    ones_tensor = ctx.ones(100, dtype=torch.float32)
    assert ctx.is_symmetric(ones_tensor), "Tensor from ones() should be on symmetric heap"

    # Test with full
    full_tensor = ctx.full((100,), 42.0, dtype=torch.float32)
    assert ctx.is_symmetric(full_tensor), "Tensor from full() should be on symmetric heap"


def test_is_symmetric_zeros_like():
    """Test is_symmetric() with zeros_like."""
    ctx = iris.iris(1 << 20, allocator_type="torch")

    # Create an input tensor on symmetric heap
    input_tensor = ctx.zeros(100, dtype=torch.float32)

    # Create a zeros_like tensor
    zeros_like_tensor = ctx.zeros_like(input_tensor)

    # Both should be on symmetric heap
    assert ctx.is_symmetric(input_tensor), "Input tensor should be on symmetric heap"
    assert ctx.is_symmetric(zeros_like_tensor), "zeros_like tensor should be on symmetric heap"
