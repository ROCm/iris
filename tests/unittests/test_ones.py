# SPDX-License-Identifier: MIT
# Copyright (c) 2025 Advanced Micro Devices, Inc. All rights reserved.

import torch
import pytest
import iris


@pytest.mark.parametrize(
    "dtype",
    [
        torch.int8,
        torch.int16,
        torch.int32,
        torch.int64,
        torch.float16,
        torch.float32,
        torch.float64,
        torch.bool,
    ],
)
@pytest.mark.parametrize(
    "size",
    [
        (1,),
        (5,),
        (2, 3),
        (3, 4, 5),
        (1, 1, 1),
        (10, 20),
    ],
)
def test_ones_basic(dtype, size):
    shmem = iris.iris(1 << 20)

    # Test basic ones
    result = shmem.ones(*size, dtype=dtype)

    # Verify shape matches
    assert result.shape == size
    assert result.dtype == dtype

    # Verify all values are one
    assert torch.all(result == 1)

    # Verify tensor is on symmetric heap
    assert shmem.is_symmetric(result)


def test_ones_default_dtype():
    shmem = iris.iris(1 << 20)

    # Test with default dtype (should use torch.get_default_dtype())
    result = shmem.ones(2, 3)
    expected_dtype = torch.get_default_dtype()
    assert result.dtype == expected_dtype
    assert torch.all(result == 1)
    assert shmem.is_symmetric(result)


@pytest.mark.parametrize(
    "requires_grad",
    [
        True,
        False,
    ],
)
def test_ones_requires_grad(requires_grad):
    shmem = iris.iris(1 << 20)

    # Test with requires_grad parameter
    result = shmem.ones(2, 2, dtype=torch.float32, requires_grad=requires_grad)

    # Verify requires_grad is set
    assert result.requires_grad == requires_grad
    assert torch.all(result == 1)
    assert shmem.is_symmetric(result)


def test_ones_device_handling():
    shmem = iris.iris(1 << 20)

    # Test default behavior (should use Iris device)
    result = shmem.ones(3, 3)
    assert str(result.device) == str(shmem.get_device())
    assert torch.all(result == 1)
    assert shmem.is_symmetric(result)

    # Test explicit device
    result = shmem.ones(3, 3, device=shmem.device)
    assert str(result.device) == str(shmem.get_device())
    assert torch.all(result == 1)
    assert shmem.is_symmetric(result)

    # Test that "cuda" shorthand works (should use current CUDA device)
    if shmem.device.startswith("cuda:"):
        result = shmem.ones(3, 3, device="cuda")
        assert str(result.device) == str(shmem.get_device())
        assert torch.all(result == 1)
        assert shmem.is_symmetric(result)

    # Test None device defaults to Iris device
    result = shmem.ones(3, 3, device=None)
    assert str(result.device) == str(shmem.get_device())
    assert torch.all(result == 1)
    assert shmem.is_symmetric(result)

    # Test that different device throws error
    different_device = "cpu"  # CPU is always different from CUDA
    with pytest.raises(RuntimeError):
        shmem.ones(3, 3, device=different_device)

    # Test that different CUDA device throws error
    if shmem.device.startswith("cuda:") and torch.cuda.device_count() >= 2:
        current_device = torch.device(shmem.device)
        different_cuda = f"cuda:{(current_device.index + 1) % torch.cuda.device_count()}"  # Use next GPU
        with pytest.raises(RuntimeError):
            shmem.ones(3, 3, device=different_cuda)


def test_ones_layout_handling():
    shmem = iris.iris(1 << 20)

    # Test with strided layout (default)
    result = shmem.ones(2, 4, layout=torch.strided)
    assert result.layout == torch.strided
    assert torch.all(result == 1)
    assert shmem.is_symmetric(result)

    # Test that unsupported layout throws error
    with pytest.raises(ValueError):
        shmem.ones(2, 4, layout=torch.sparse_coo)


def test_ones_out_parameter():
    shmem = iris.iris(1 << 20)

    # Test with out parameter
    out_tensor = shmem.heap.allocate(6, torch.float32)
    result = shmem.ones(2, 3, out=out_tensor)

    # Should share the same underlying data (same data_ptr)
    assert result.data_ptr() == out_tensor.data_ptr()
    assert torch.all(result == 1)
    assert result.shape == (2, 3)
    assert shmem.is_symmetric(result)

    # Test with different dtype out tensor
    out_tensor_int = shmem.heap.allocate(6, torch.int32)
    result_int = shmem.ones(2, 3, dtype=torch.int32, out=out_tensor_int)
    assert result_int.data_ptr() == out_tensor_int.data_ptr()
    assert result_int.dtype == torch.int32
    assert torch.all(result_int == 1)
    assert shmem.is_symmetric(result_int)


def test_ones_size_variations():
    shmem = iris.iris(1 << 20)

    # Test single dimension
    result1 = shmem.ones(5)
    assert result1.shape == (5,)
    assert torch.all(result1 == 1)
    assert shmem.is_symmetric(result1)

    # Test multiple dimensions
    result2 = shmem.ones(2, 3, 4)
    assert result2.shape == (2, 3, 4)
    assert torch.all(result2 == 1)
    assert shmem.is_symmetric(result2)

    # Test with tuple as single argument
    result3 = shmem.ones((3, 4))
    assert result3.shape == (3, 4)
    assert torch.all(result3 == 1)
    assert shmem.is_symmetric(result3)

    # Test with list as single argument
    result4 = shmem.ones([2, 5])
    assert result4.shape == (2, 5)
    assert torch.all(result4 == 1)
    assert shmem.is_symmetric(result4)


def test_ones_edge_cases():
    shmem = iris.iris(1 << 20)

    # Empty tensor
    empty_result = shmem.ones(0)
    assert empty_result.shape == (0,)
    assert empty_result.numel() == 0
    assert shmem.is_symmetric(empty_result)

    # Single element tensor
    single_result = shmem.ones(1)
    assert single_result.shape == (1,)
    assert single_result.numel() == 1
    assert single_result[0] == 1
    assert shmem.is_symmetric(single_result)

    # Large tensor
    large_result = shmem.ones(100, 100)
    assert large_result.shape == (100, 100)
    assert large_result.numel() == 10000
    assert torch.all(large_result == 1)
    assert shmem.is_symmetric(large_result)

    # Zero-dimensional tensor (scalar)
    scalar_result = shmem.ones(())
    assert scalar_result.shape == ()
    assert scalar_result.numel() == 1
    assert scalar_result.item() == 1
    assert shmem.is_symmetric(scalar_result)


def test_ones_pytorch_equivalence():
    shmem = iris.iris(1 << 20)

    # Test basic equivalence
    iris_result = shmem.ones(4, 3)
    pytorch_result = torch.ones(4, 3, device="cuda")

    assert iris_result.shape == pytorch_result.shape
    assert iris_result.dtype == pytorch_result.dtype
    assert torch.all(iris_result == pytorch_result)

    # Test with explicit dtype
    iris_result = shmem.ones(2, 2, dtype=torch.float64)
    pytorch_result = torch.ones(2, 2, dtype=torch.float64, device="cuda")

    assert iris_result.shape == pytorch_result.shape
    assert iris_result.dtype == pytorch_result.dtype
    assert torch.all(iris_result == pytorch_result)

    # Test with requires_grad
    iris_result = shmem.ones(3, 3, dtype=torch.float32, requires_grad=True)
    pytorch_result = torch.ones(3, 3, dtype=torch.float32, device="cuda", requires_grad=True)

    assert iris_result.shape == pytorch_result.shape
    assert iris_result.dtype == pytorch_result.dtype
    assert iris_result.requires_grad == pytorch_result.requires_grad
    assert torch.all(iris_result == pytorch_result)


@pytest.mark.parametrize(
    "params",
    [
        {"dtype": torch.float32, "requires_grad": True},
        {"dtype": torch.float64, "requires_grad": False},
        {"dtype": torch.int32},
        {"dtype": torch.float16},
        {"layout": torch.strided},
        {},
    ],
)
def test_ones_parameter_combinations(params):
    shmem = iris.iris(1 << 20)

    # Test various combinations of parameters
    result = shmem.ones(3, 3, **params)

    # Verify basic functionality
    assert result.shape == (3, 3)
    assert torch.all(result == 1)
    assert shmem.is_symmetric(result)

    # Verify dtype if specified
    if "dtype" in params:
        assert result.dtype == params["dtype"]

    # Verify requires_grad if specified
    if "requires_grad" in params:
        assert result.requires_grad == params["requires_grad"]

    # Verify layout if specified
    if "layout" in params:
        assert result.layout == params["layout"]


@pytest.mark.parametrize(
    "size,dtype",
    [
        ((1,), torch.float32),
        ((5,), torch.int32),
        ((2, 3), torch.float64),
        ((3, 4, 5), torch.float16),
        ((0,), torch.float32),  # Empty tensor
        ((100, 100), torch.float32),  # Large tensor
        ((), torch.float32),  # Scalar tensor
    ],
)
def test_ones_symmetric_heap_shapes_dtypes(size, dtype):
    """Test that ones returns tensors on symmetric heap for various shapes and dtypes."""
    shmem = iris.iris(1 << 20)

    # Test ones with this size and dtype
    result = shmem.ones(*size, dtype=dtype)

    # Verify tensor is on symmetric heap
    assert shmem.is_symmetric(result), f"Tensor with size {size}, dtype {dtype} is NOT on symmetric heap!"

    # Also verify basic functionality
    assert result.shape == size
    assert result.dtype == dtype
    assert torch.all(result == 1)


@pytest.mark.parametrize("dtype", [torch.float16, torch.float32, torch.float64, torch.int32, torch.int64])
def test_ones_symmetric_heap_dtype_override(dtype):
    """Test that ones with dtype override returns tensors on symmetric heap."""
    shmem = iris.iris(1 << 20)

    result = shmem.ones(3, 3, dtype=dtype)
    assert shmem.is_symmetric(result), f"Tensor with dtype {dtype} is NOT on symmetric heap!"
    assert result.dtype == dtype


def test_ones_symmetric_heap_other_params():
    """Test that ones with other parameters returns tensors on symmetric heap."""
    shmem = iris.iris(1 << 20)

    # Test with requires_grad
    result = shmem.ones(3, 3, dtype=torch.float32, requires_grad=True)
    assert shmem.is_symmetric(result), "Tensor with requires_grad=True is NOT on symmetric heap!"

    # Test with device override
    result = shmem.ones(3, 3, device=shmem.device)
    assert shmem.is_symmetric(result), "Tensor with device override is NOT on symmetric heap!"

    # Test with layout override (only strided is supported)
    result = shmem.ones(3, 3, layout=torch.strided)
    assert shmem.is_symmetric(result), "Tensor with layout override is NOT on symmetric heap!"

    # Test with out parameter
    out_tensor = shmem.heap.allocate(9, torch.float32)
    result = shmem.ones(3, 3, out=out_tensor)
    assert shmem.is_symmetric(result), "Tensor with out parameter is NOT on symmetric heap!"


def test_ones_invalid_output_tensor():
    """Test error handling for invalid output tensors."""
    shmem = iris.iris(1 << 20)

    # Test with wrong size output tensor
    wrong_size_tensor = shmem.heap.allocate(4, torch.float32)  # Wrong size for (3, 3)
    with pytest.raises(RuntimeError):
        shmem.ones(3, 3, out=wrong_size_tensor)

    # Test with wrong dtype output tensor
    wrong_dtype_tensor = shmem.heap.allocate(9, torch.int32)  # Wrong dtype
    with pytest.raises(RuntimeError):
        shmem.ones(3, 3, dtype=torch.float32, out=wrong_dtype_tensor)

    # Test with tensor not on symmetric heap (create a regular PyTorch tensor)
    regular_tensor = torch.ones(3, 3, device="cuda")
    with pytest.raises(RuntimeError):
        shmem.ones(3, 3, out=regular_tensor)


def test_ones_default_dtype_behavior():
    """Test that ones uses the global default dtype when dtype=None."""
    shmem = iris.iris(1 << 20)

    # Save original default dtype
    original_default = torch.get_default_dtype()

    try:
        # Test with float32 default
        torch.set_default_dtype(torch.float32)
        result1 = shmem.ones(2, 2)
        assert result1.dtype == torch.float32

        # Test with float64 default
        torch.set_default_dtype(torch.float64)
        result2 = shmem.ones(2, 2)
        assert result2.dtype == torch.float64

    finally:
        # Restore original default dtype
        torch.set_default_dtype(original_default)


def test_ones_size_parsing():
    """Test various ways of specifying size."""
    shmem = iris.iris(1 << 20)

    # Test individual arguments
    result1 = shmem.ones(2, 3, 4)
    assert result1.shape == (2, 3, 4)

    # Test single tuple argument
    result2 = shmem.ones((2, 3, 4))
    assert result2.shape == (2, 3, 4)

    # Test single list argument
    result3 = shmem.ones([2, 3, 4])
    assert result3.shape == (2, 3, 4)

    # Test nested tuple (should be flattened)
    result4 = shmem.ones(((2, 3, 4),))
    assert result4.shape == (2, 3, 4)

    # All should produce the same result
    assert torch.all(result1 == result2)
    assert torch.all(result2 == result3)
    assert torch.all(result3 == result4)


def test_ones_examples():
    """Test the examples from PyTorch documentation."""
    shmem = iris.iris(1 << 20)

    # Example 1: torch.ones(2, 3)
    result1 = shmem.ones(2, 3)
    expected1 = torch.tensor([[1.0, 1.0, 1.0], [1.0, 1.0, 1.0]], device=result1.device)
    assert result1.shape == (2, 3)
    assert torch.all(result1 == expected1)
    assert shmem.is_symmetric(result1)

    # Example 2: torch.ones(5)
    result2 = shmem.ones(5)
    expected2 = torch.tensor([1.0, 1.0, 1.0, 1.0, 1.0], device=result2.device)
    assert result2.shape == (5,)
    assert torch.all(result2 == expected2)
    assert shmem.is_symmetric(result2)
