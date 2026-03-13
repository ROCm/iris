# SPDX-License-Identifier: MIT
# Copyright (c) 2025 Advanced Micro Devices, Inc. All rights reserved.

import torch
import pytest
import iris


@pytest.mark.parametrize(
    "fill_value",
    [
        0,
        1,
        -1,
        3.141592,
        -2.718,
        42,
        -100,
        0.5,
        -0.25,
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
def test_full_basic(fill_value, size):
    shmem = iris.iris(1 << 20)

    # Test basic full
    result = shmem.full(size, fill_value)

    # Verify shape matches
    assert result.shape == size

    # Verify all values are the fill_value
    assert torch.all(result == fill_value)

    # Verify tensor is on symmetric heap
    assert shmem.is_symmetric(result)


def test_full_dtype_inference():
    shmem = iris.iris(1 << 20)

    # Test integer fill_value (should infer int64)
    result_int = shmem.full((2, 3), 42)
    assert result_int.dtype == torch.int64
    assert torch.all(result_int == 42)
    assert shmem.is_symmetric(result_int)

    # Test float fill_value (should infer default float dtype)
    result_float = shmem.full((2, 3), 3.141592)
    assert result_float.dtype == torch.get_default_dtype()
    assert torch.allclose(result_float, torch.tensor(3.141592))
    assert shmem.is_symmetric(result_float)

    # Test explicit dtype override
    result_explicit = shmem.full((2, 3), 42, dtype=torch.float32)
    assert result_explicit.dtype == torch.float32
    assert torch.all(result_explicit == 42)
    assert shmem.is_symmetric(result_explicit)


@pytest.mark.parametrize(
    "requires_grad",
    [
        True,
        False,
    ],
)
def test_full_requires_grad(requires_grad):
    shmem = iris.iris(1 << 20)

    # Test with requires_grad parameter
    result = shmem.full((2, 2), 1.5, dtype=torch.float32, requires_grad=requires_grad)

    # Verify requires_grad is set
    assert result.requires_grad == requires_grad
    assert torch.all(result == 1.5)
    assert shmem.is_symmetric(result)


def test_full_device_handling():
    shmem = iris.iris(1 << 20)

    # Test default behavior (should use Iris device)
    result = shmem.full((3, 3), 2.5)
    assert str(result.device) == str(shmem.get_device())
    assert torch.all(result == 2.5)
    assert shmem.is_symmetric(result)

    # Test explicit device
    result = shmem.full((3, 3), 2.5, device=shmem.device)
    assert str(result.device) == str(shmem.get_device())
    assert torch.all(result == 2.5)
    assert shmem.is_symmetric(result)

    # Test that "cuda" shorthand works (should use current CUDA device)
    if shmem.device.startswith("cuda:"):
        result = shmem.full((3, 3), 2.5, device="cuda")
        assert str(result.device) == str(shmem.get_device())
        assert torch.all(result == 2.5)
        assert shmem.is_symmetric(result)

    # Test None device defaults to Iris device
    result = shmem.full((3, 3), 2.5, device=None)
    assert str(result.device) == str(shmem.get_device())
    assert torch.all(result == 2.5)
    assert shmem.is_symmetric(result)

    # Test that different device throws error
    different_device = "cpu"  # CPU is always different from CUDA
    with pytest.raises(RuntimeError):
        shmem.full((3, 3), 2.5, device=different_device)

    # Test that different CUDA device throws error
    if shmem.device.startswith("cuda:") and torch.cuda.device_count() >= 2:
        current_device = torch.device(shmem.device)
        different_cuda = f"cuda:{(current_device.index + 1) % torch.cuda.device_count()}"  # Use next GPU
        with pytest.raises(RuntimeError):
            shmem.full((3, 3), 2.5, device=different_cuda)


def test_full_layout_handling():
    shmem = iris.iris(1 << 20)

    # Test with strided layout (default)
    result = shmem.full((2, 4), 1.0, layout=torch.strided)
    assert result.layout == torch.strided
    assert torch.all(result == 1.0)
    assert shmem.is_symmetric(result)

    # Test that unsupported layout throws error
    with pytest.raises(ValueError):
        shmem.full((2, 4), 1.0, layout=torch.sparse_coo)


def test_full_out_parameter():
    shmem = iris.iris(1 << 20)

    # Test with out parameter
    out_tensor = shmem.heap.allocate(6, torch.float32)
    result = shmem.full((2, 3), 3.141592, out=out_tensor)

    # Should share the same underlying data (same data_ptr)
    assert result.data_ptr() == out_tensor.data_ptr()
    assert torch.allclose(result, torch.tensor(3.141592))
    assert result.shape == (2, 3)
    assert shmem.is_symmetric(result)

    # Test with different dtype out tensor
    out_tensor_int = shmem.heap.allocate(6, torch.int32)
    result_int = shmem.full((2, 3), 42, dtype=torch.int32, out=out_tensor_int)
    assert result_int.data_ptr() == out_tensor_int.data_ptr()
    assert result_int.dtype == torch.int32
    assert torch.all(result_int == 42)
    assert shmem.is_symmetric(result_int)


def test_full_size_variations():
    shmem = iris.iris(1 << 20)

    # Test single dimension
    result1 = shmem.full((5,), 2.0)
    assert result1.shape == (5,)
    assert torch.all(result1 == 2.0)
    assert shmem.is_symmetric(result1)

    # Test multiple dimensions
    result2 = shmem.full((2, 3, 4), 1.5)
    assert result2.shape == (2, 3, 4)
    assert torch.all(result2 == 1.5)
    assert shmem.is_symmetric(result2)

    # Test with tuple as single argument
    result3 = shmem.full((3, 4), 0.5)
    assert result3.shape == (3, 4)
    assert torch.all(result3 == 0.5)
    assert shmem.is_symmetric(result3)

    # Test with list as single argument
    result4 = shmem.full([2, 5], -1.0)
    assert result4.shape == (2, 5)
    assert torch.all(result4 == -1.0)
    assert shmem.is_symmetric(result4)


def test_full_edge_cases():
    shmem = iris.iris(1 << 20)

    # Empty tensor
    empty_result = shmem.full((0,), 1.0)
    assert empty_result.shape == (0,)
    assert empty_result.numel() == 0
    assert shmem.is_symmetric(empty_result)

    # Single element tensor
    single_result = shmem.full((1,), 5.0)
    assert single_result.shape == (1,)
    assert single_result.numel() == 1
    assert single_result[0] == 5.0
    assert shmem.is_symmetric(single_result)

    # Large tensor
    large_result = shmem.full((100, 100), 0.1)
    assert large_result.shape == (100, 100)
    assert large_result.numel() == 10000
    assert torch.all(large_result == 0.1)
    assert shmem.is_symmetric(large_result)

    # Zero-dimensional tensor (scalar)
    scalar_result = shmem.full((), 2.718)
    assert scalar_result.shape == ()
    assert scalar_result.numel() == 1
    assert torch.allclose(scalar_result, torch.tensor(2.718))
    assert shmem.is_symmetric(scalar_result)


def test_full_pytorch_equivalence():
    shmem = iris.iris(1 << 20)

    # Test basic equivalence
    iris_result = shmem.full((4, 3), 3.141592)
    pytorch_result = torch.full((4, 3), 3.141592, device="cuda")

    assert iris_result.shape == pytorch_result.shape
    assert iris_result.dtype == pytorch_result.dtype
    assert torch.allclose(iris_result, pytorch_result)

    # Test with explicit dtype
    iris_result = shmem.full((2, 2), 42, dtype=torch.float64)
    pytorch_result = torch.full((2, 2), 42, dtype=torch.float64, device="cuda")

    assert iris_result.shape == pytorch_result.shape
    assert iris_result.dtype == pytorch_result.dtype
    assert torch.allclose(iris_result, pytorch_result)

    # Test with requires_grad
    iris_result = shmem.full((3, 3), 1.5, dtype=torch.float32, requires_grad=True)
    pytorch_result = torch.full((3, 3), 1.5, dtype=torch.float32, device="cuda", requires_grad=True)

    assert iris_result.shape == pytorch_result.shape
    assert iris_result.dtype == pytorch_result.dtype
    assert iris_result.requires_grad == pytorch_result.requires_grad
    assert torch.allclose(iris_result, pytorch_result)


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
def test_full_parameter_combinations(params):
    shmem = iris.iris(1 << 20)

    # Test various combinations of parameters
    result = shmem.full((3, 3), 2.5, **params)

    # Verify basic functionality
    assert result.shape == (3, 3)
    # Use appropriate comparison based on dtype
    if torch.is_floating_point(result):
        # For float dtypes, use close comparison with matching dtype
        expected = torch.tensor(2.5, dtype=result.dtype, device=result.device)
        assert torch.allclose(result, expected)
    else:
        # For integer dtypes, the fill value gets truncated
        assert torch.all(result == 2)
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
    "size,fill_value,dtype",
    [
        ((1,), 1.0, torch.float32),
        ((5,), 42, torch.int32),
        ((2, 3), 3.141592, torch.float64),
        ((3, 4, 5), 0.5, torch.float16),
        ((0,), 1.0, torch.float32),  # Empty tensor
        ((100, 100), 0.1, torch.float32),  # Large tensor
        ((), 2.718, torch.float32),  # Scalar tensor
    ],
)
def test_full_symmetric_heap_shapes_dtypes(size, fill_value, dtype):
    """Test that full returns tensors on symmetric heap for various shapes and dtypes."""
    shmem = iris.iris(1 << 20)

    # Test full with this size, fill_value, and dtype
    result = shmem.full(size, fill_value, dtype=dtype)

    # Verify tensor is on symmetric heap
    assert shmem.is_symmetric(result), (
        f"Tensor with size {size}, fill_value {fill_value}, dtype {dtype} is NOT on symmetric heap!"
    )

    # Also verify basic functionality
    assert result.shape == size
    assert result.dtype == dtype
    assert torch.allclose(result, torch.tensor(fill_value, dtype=dtype))


@pytest.mark.parametrize("dtype", [torch.float16, torch.float32, torch.float64, torch.int32, torch.int64])
def test_full_symmetric_heap_dtype_override(dtype):
    """Test that full with dtype override returns tensors on symmetric heap."""
    shmem = iris.iris(1 << 20)

    result = shmem.full((3, 3), 1.5, dtype=dtype)
    assert shmem.is_symmetric(result), f"Tensor with dtype {dtype} is NOT on symmetric heap!"
    assert result.dtype == dtype


def test_full_symmetric_heap_other_params():
    """Test that full with other parameters returns tensors on symmetric heap."""
    shmem = iris.iris(1 << 20)

    # Test with requires_grad
    result = shmem.full((3, 3), 1.5, dtype=torch.float32, requires_grad=True)
    assert shmem.is_symmetric(result), "Tensor with requires_grad=True is NOT on symmetric heap!"

    # Test with device override
    result = shmem.full((3, 3), 1.5, device=shmem.device)
    assert shmem.is_symmetric(result), "Tensor with device override is NOT on symmetric heap!"

    # Test with layout override (only strided is supported)
    result = shmem.full((3, 3), 1.5, layout=torch.strided)
    assert shmem.is_symmetric(result), "Tensor with layout override is NOT on symmetric heap!"

    # Test with out parameter
    out_tensor = shmem.heap.allocate(9, torch.float32)
    result = shmem.full((3, 3), 1.5, out=out_tensor)
    assert shmem.is_symmetric(result), "Tensor with out parameter is NOT on symmetric heap!"


def test_full_invalid_output_tensor():
    """Test error handling for invalid output tensors."""
    shmem = iris.iris(1 << 20)

    # Test with wrong size output tensor
    wrong_size_tensor = shmem.heap.allocate(4, torch.float32)  # Wrong size for (3, 3)
    with pytest.raises(RuntimeError):
        shmem.full((3, 3), 1.5, out=wrong_size_tensor)

    # Test with wrong dtype output tensor
    wrong_dtype_tensor = shmem.heap.allocate(9, torch.int32)  # Wrong dtype
    with pytest.raises(RuntimeError):
        shmem.full((3, 3), 1.5, dtype=torch.float32, out=wrong_dtype_tensor)

    # Test with tensor not on symmetric heap (create a regular PyTorch tensor)
    regular_tensor = torch.full((3, 3), 1.5, device="cuda")
    with pytest.raises(RuntimeError):
        shmem.full((3, 3), 1.5, out=regular_tensor)


def test_full_size_parsing():
    """Test various ways of specifying size."""
    shmem = iris.iris(1 << 20)

    # Test individual arguments
    result1 = shmem.full((2, 3, 4), 1.0)
    assert result1.shape == (2, 3, 4)

    # Test single tuple argument
    result2 = shmem.full((2, 3, 4), 1.0)
    assert result2.shape == (2, 3, 4)

    # Test single list argument
    result3 = shmem.full([2, 3, 4], 1.0)
    assert result3.shape == (2, 3, 4)

    # Test nested tuple (should be flattened)
    result4 = shmem.full(((2, 3, 4),), 1.0)
    assert result4.shape == (2, 3, 4)

    # All should produce the same result
    assert torch.all(result1 == result2)
    assert torch.all(result2 == result3)
    assert torch.all(result3 == result4)


def test_full_examples():
    """Test the examples from PyTorch documentation."""
    shmem = iris.iris(1 << 20)

    # Example: torch.full((2, 3), 3.141592)
    result = shmem.full((2, 3), 3.141592)
    expected = torch.tensor([[3.141592, 3.141592, 3.141592], [3.141592, 3.141592, 3.141592]], device=result.device)
    assert result.shape == (2, 3)
    assert torch.allclose(result, expected)
    assert shmem.is_symmetric(result)


def test_full_different_fill_values():
    """Test various fill values to ensure they work correctly."""
    shmem = iris.iris(1 << 20)

    # Test different numeric types
    test_cases = [
        (0, torch.int64),
        (1, torch.int64),
        (-1, torch.int64),
        (42, torch.int64),
        (0.0, torch.get_default_dtype()),
        (1.0, torch.get_default_dtype()),
        (-1.0, torch.get_default_dtype()),
        (3.141592, torch.get_default_dtype()),
        (-2.718, torch.get_default_dtype()),
    ]

    for fill_value, expected_dtype in test_cases:
        result = shmem.full((2, 2), fill_value)
        assert result.dtype == expected_dtype
        assert torch.allclose(result, torch.tensor(fill_value, dtype=expected_dtype))
        assert shmem.is_symmetric(result)


def test_full_dtype_override():
    """Test that explicit dtype overrides inference."""
    shmem = iris.iris(1 << 20)

    # Integer fill_value with float dtype
    result = shmem.full((2, 2), 42, dtype=torch.float32)
    assert result.dtype == torch.float32
    assert torch.allclose(result, torch.tensor(42.0, dtype=torch.float32))
    assert shmem.is_symmetric(result)

    # Float fill_value with int dtype
    result = shmem.full((2, 2), 3.14, dtype=torch.int32)
    assert result.dtype == torch.int32
    assert torch.all(result == 3)  # Truncated to int
    assert shmem.is_symmetric(result)
