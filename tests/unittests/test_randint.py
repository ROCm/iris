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
        torch.uint8,
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
def test_randint_basic(dtype, size):
    shmem = iris.iris(1 << 20)

    # Test basic randint with low, high, size
    result = shmem.randint(0, 10, size, dtype=dtype)

    # Verify shape matches
    assert result.shape == size
    assert result.dtype == dtype

    # Verify values are within range [0, 10)
    assert torch.all(result >= 0)
    assert torch.all(result < 10)

    # Verify tensor is on symmetric heap
    assert shmem.is_symmetric(result)


def test_randint_default_dtype():
    shmem = iris.iris(1 << 20)

    # Test with default dtype (should use torch.int64)
    result = shmem.randint(0, 10, (2, 3))
    assert result.dtype == torch.int64
    assert shmem.is_symmetric(result)


@pytest.mark.parametrize(
    "requires_grad",
    [
        True,
        False,
    ],
)
def test_randint_requires_grad(requires_grad):
    shmem = iris.iris(1 << 20)

    # Test with requires_grad parameter
    result = shmem.randint(0, 10, (2, 2), dtype=torch.float32, requires_grad=requires_grad)

    # Verify requires_grad is set
    assert result.requires_grad == requires_grad
    assert shmem.is_symmetric(result)


def test_randint_device_handling():
    shmem = iris.iris(1 << 20)

    # Test default behavior (should use Iris device)
    result = shmem.randint(0, 10, (3, 3))
    assert str(result.device) == str(shmem.get_device())
    assert shmem.is_symmetric(result)

    # Test explicit device
    result = shmem.randint(0, 10, (3, 3), device=shmem.device)
    assert str(result.device) == str(shmem.get_device())
    assert shmem.is_symmetric(result)

    # Test that "cuda" shorthand works (should use current CUDA device)
    if shmem.device.startswith("cuda:"):
        result = shmem.randint(0, 10, (3, 3), device="cuda")
        assert str(result.device) == str(shmem.get_device())
        assert shmem.is_symmetric(result)

    # Test None device defaults to Iris device
    result = shmem.randint(0, 10, (3, 3), device=None)
    assert str(result.device) == str(shmem.get_device())
    assert shmem.is_symmetric(result)

    # Test that different device throws error
    different_device = "cpu"  # CPU is always different from CUDA
    with pytest.raises(RuntimeError):
        shmem.randint(0, 10, (3, 3), device=different_device)

    # Test that different CUDA device throws error
    if shmem.device.startswith("cuda:") and torch.cuda.device_count() >= 2:
        current_device = torch.device(shmem.device)
        different_cuda = f"cuda:{(current_device.index + 1) % torch.cuda.device_count()}"  # Use next GPU
        with pytest.raises(RuntimeError):
            shmem.randint(0, 10, (3, 3), device=different_cuda)


def test_randint_layout_handling():
    shmem = iris.iris(1 << 20)

    # Test with strided layout (default)
    result = shmem.randint(0, 10, (2, 4), layout=torch.strided)
    assert result.layout == torch.strided
    assert shmem.is_symmetric(result)

    # Test that unsupported layout throws error
    with pytest.raises(ValueError):
        shmem.randint(0, 10, (2, 4), layout=torch.sparse_coo)


def test_randint_out_parameter():
    shmem = iris.iris(1 << 20)

    # Test with out parameter
    out_tensor = shmem.heap.allocate(6, torch.int64)
    result = shmem.randint(0, 10, (2, 3), out=out_tensor)

    # Should share the same underlying data (same data_ptr)
    assert result.data_ptr() == out_tensor.data_ptr()
    assert result.shape == (2, 3)
    assert torch.all(result >= 0)
    assert torch.all(result < 10)
    assert shmem.is_symmetric(result)

    # Test with explicit dtype
    out_tensor_int32 = shmem.heap.allocate(6, torch.int32)
    result_int32 = shmem.randint(0, 10, (2, 3), dtype=torch.int32, out=out_tensor_int32)
    assert result_int32.data_ptr() == out_tensor_int32.data_ptr()
    assert result_int32.dtype == torch.int32
    assert shmem.is_symmetric(result_int32)


def test_randint_size_variations():
    shmem = iris.iris(1 << 20)

    # Test single dimension
    result1 = shmem.randint(0, 5, (5,))
    assert result1.shape == (5,)
    assert torch.all(result1 >= 0)
    assert torch.all(result1 < 5)
    assert shmem.is_symmetric(result1)

    # Test multiple dimensions
    result2 = shmem.randint(0, 10, (2, 3, 4))
    assert result2.shape == (2, 3, 4)
    assert torch.all(result2 >= 0)
    assert torch.all(result2 < 10)
    assert shmem.is_symmetric(result2)

    # Test with tuple as single argument
    result3 = shmem.randint(0, 10, (3, 4))
    assert result3.shape == (3, 4)
    assert torch.all(result3 >= 0)
    assert torch.all(result3 < 10)
    assert shmem.is_symmetric(result3)

    # Test with list as single argument
    result4 = shmem.randint(0, 10, [2, 5])
    assert result4.shape == (2, 5)
    assert torch.all(result4 >= 0)
    assert torch.all(result4 < 10)
    assert shmem.is_symmetric(result4)


def test_randint_edge_cases():
    shmem = iris.iris(1 << 20)

    # Empty tensor
    empty_result = shmem.randint(0, 5, (0,))
    assert empty_result.shape == (0,)
    assert empty_result.numel() == 0
    assert shmem.is_symmetric(empty_result)

    # Single element tensor
    single_result = shmem.randint(0, 10, (1,))
    assert single_result.shape == (1,)
    assert single_result.numel() == 1
    assert torch.all(single_result >= 0)
    assert torch.all(single_result < 10)
    assert shmem.is_symmetric(single_result)

    # Large tensor
    large_result = shmem.randint(0, 100, (100, 100))
    assert large_result.shape == (100, 100)
    assert large_result.numel() == 10000
    assert torch.all(large_result >= 0)
    assert torch.all(large_result < 100)
    assert shmem.is_symmetric(large_result)

    # Zero-dimensional tensor (scalar)
    scalar_result = shmem.randint(0, 10, ())
    assert scalar_result.shape == ()
    assert scalar_result.numel() == 1
    assert torch.all(scalar_result >= 0)
    assert torch.all(scalar_result < 10)
    assert shmem.is_symmetric(scalar_result)


def test_randint_pytorch_equivalence():
    shmem = iris.iris(1 << 20)

    # Test basic equivalence
    iris_result = shmem.randint(0, 10, (4, 3))
    pytorch_result = torch.randint(0, 10, (4, 3), device="cuda")

    assert iris_result.shape == pytorch_result.shape
    assert iris_result.dtype == pytorch_result.dtype

    # Test with explicit dtype
    iris_result = shmem.randint(0, 10, (2, 2), dtype=torch.int32)
    pytorch_result = torch.randint(0, 10, (2, 2), dtype=torch.int32, device="cuda")

    assert iris_result.shape == pytorch_result.shape
    assert iris_result.dtype == pytorch_result.dtype

    # Test with requires_grad
    iris_result = shmem.randint(0, 10, (3, 3), dtype=torch.float32, requires_grad=True)
    pytorch_result = torch.randint(0, 10, (3, 3), dtype=torch.float32, device="cuda", requires_grad=True)

    assert iris_result.shape == pytorch_result.shape
    assert iris_result.dtype == pytorch_result.dtype
    assert iris_result.requires_grad == pytorch_result.requires_grad


@pytest.mark.parametrize(
    "params",
    [
        {"dtype": torch.float32, "requires_grad": True},
        {"dtype": torch.int64, "requires_grad": False},
        {"dtype": torch.int8},
        {"dtype": torch.uint8},
        {"layout": torch.strided},
        {},
    ],
)
def test_randint_parameter_combinations(params):
    shmem = iris.iris(1 << 20)

    # Test various combinations of parameters
    result = shmem.randint(0, 10, (3, 3), **params)

    # Verify basic functionality
    assert result.shape == (3, 3)
    assert torch.all(result >= 0)
    assert torch.all(result < 10)
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
        ((1,), torch.int32),
        ((5,), torch.int64),
        ((2, 3), torch.int8),
        ((3, 4, 5), torch.uint8),
        ((0,), torch.int32),  # Empty tensor
        ((100, 100), torch.int32),  # Large tensor
        ((), torch.int32),  # Scalar tensor
    ],
)
def test_randint_symmetric_heap_shapes_dtypes(size, dtype):
    """Test that randint returns tensors on symmetric heap for various shapes and dtypes."""
    shmem = iris.iris(1 << 20)

    # Test randint with this size and dtype
    result = shmem.randint(0, 10, size, dtype=dtype)

    # Verify tensor is on symmetric heap
    assert shmem.is_symmetric(result), f"Tensor with size {size}, dtype {dtype} is NOT on symmetric heap!"

    # Also verify basic functionality
    assert result.shape == size
    assert result.dtype == dtype
    assert torch.all(result >= 0)
    assert torch.all(result < 10)


@pytest.mark.parametrize("dtype", [torch.int8, torch.int16, torch.int32, torch.int64, torch.uint8])
def test_randint_symmetric_heap_dtype_override(dtype):
    """Test that randint with dtype override returns tensors on symmetric heap."""
    shmem = iris.iris(1 << 20)

    result = shmem.randint(0, 10, (3, 3), dtype=dtype)
    assert shmem.is_symmetric(result), f"Tensor with dtype {dtype} is NOT on symmetric heap!"
    assert result.dtype == dtype


def test_randint_symmetric_heap_other_params():
    """Test that randint with other parameters returns tensors on symmetric heap."""
    shmem = iris.iris(1 << 20)

    # Test with requires_grad
    result = shmem.randint(0, 10, (3, 3), dtype=torch.float32, requires_grad=True)
    assert shmem.is_symmetric(result), "Tensor with requires_grad=True is NOT on symmetric heap!"

    # Test with device override
    result = shmem.randint(0, 10, (3, 3), device=shmem.device)
    assert shmem.is_symmetric(result), "Tensor with device override is NOT on symmetric heap!"

    # Test with layout override (only strided is supported)
    result = shmem.randint(0, 10, (3, 3), layout=torch.strided)
    assert shmem.is_symmetric(result), "Tensor with layout override is NOT on symmetric heap!"

    # Test with out parameter
    out_tensor = shmem.heap.allocate(9, torch.int64)  # Use default dtype
    result = shmem.randint(0, 10, (3, 3), out=out_tensor)
    assert shmem.is_symmetric(result), "Tensor with out parameter is NOT on symmetric heap!"


def test_randint_invalid_output_tensor():
    """Test error handling for invalid output tensors."""
    shmem = iris.iris(1 << 20)

    # Test with wrong size output tensor
    wrong_size_tensor = shmem.heap.allocate(4, torch.int32)  # Wrong size for (3, 3)
    with pytest.raises(RuntimeError):
        shmem.randint(0, 10, (3, 3), out=wrong_size_tensor)

    # Test with wrong dtype output tensor
    wrong_dtype_tensor = shmem.heap.allocate(9, torch.float32)  # Wrong dtype
    with pytest.raises(RuntimeError):
        shmem.randint(0, 10, (3, 3), dtype=torch.int32, out=wrong_dtype_tensor)

    # Test with tensor not on symmetric heap (create a regular PyTorch tensor)
    regular_tensor = torch.randint(0, 10, (3, 3), device="cuda")
    with pytest.raises(RuntimeError):
        shmem.randint(0, 10, (3, 3), out=regular_tensor)


def test_randint_default_dtype_behavior():
    """Test that randint uses torch.int64 when dtype=None."""
    shmem = iris.iris(1 << 20)

    # Test with default dtype (should be torch.int64)
    result = shmem.randint(0, 10, (2, 2))
    assert result.dtype == torch.int64


def test_randint_size_parsing():
    """Test various ways of specifying size."""
    shmem = iris.iris(1 << 20)

    # Test individual arguments
    result1 = shmem.randint(0, 10, (2, 3, 4))
    assert result1.shape == (2, 3, 4)

    # Test single tuple argument
    result2 = shmem.randint(0, 10, (2, 3, 4))
    assert result2.shape == (2, 3, 4)

    # Test single list argument
    result3 = shmem.randint(0, 10, [2, 3, 4])
    assert result3.shape == (2, 3, 4)

    # Test nested tuple (should be flattened)
    result4 = shmem.randint(0, 10, ((2, 3, 4),))
    assert result4.shape == (2, 3, 4)

    # All should produce the same result shape
    assert result1.shape == result2.shape
    assert result2.shape == result3.shape
    assert result3.shape == result4.shape


def test_randint_generator():
    """Test generator parameter."""
    shmem = iris.iris(1 << 20)

    # Test with generator
    generator = torch.Generator(device="cuda")
    generator.manual_seed(42)
    result1 = shmem.randint(0, 10, (3, 3), generator=generator)
    assert result1.shape == (3, 3)
    assert torch.all(result1 >= 0)
    assert torch.all(result1 < 10)
    assert shmem.is_symmetric(result1)

    # Test without generator (should still work)
    result2 = shmem.randint(0, 10, (3, 3))
    assert result2.shape == (3, 3)
    assert torch.all(result2 >= 0)
    assert torch.all(result2 < 10)
    assert shmem.is_symmetric(result2)


def test_randint_argument_validation():
    """Test argument validation."""
    shmem = iris.iris(1 << 20)

    # Test with wrong number of arguments
    with pytest.raises(ValueError):
        shmem.randint(10)  # Missing size

    with pytest.raises(ValueError):
        shmem.randint(0, 10, (2, 3), (4, 5))  # Too many arguments

    # Test with invalid range (should throw error)
    with pytest.raises(RuntimeError):
        shmem.randint(10, 5, (2, 3))  # low > high should throw error


def test_randint_range_validation():
    """Test that randint respects the range [low, high)."""
    shmem = iris.iris(1 << 20)

    # Test positive range
    result = shmem.randint(5, 15, (100,))
    assert torch.all(result >= 5)
    assert torch.all(result < 15)

    # Test negative range
    result = shmem.randint(-10, -5, (100,))
    assert torch.all(result >= -10)
    assert torch.all(result < -5)

    # Test zero range
    result = shmem.randint(0, 1, (100,))
    assert torch.all(result == 0)

    # Test single value range
    result = shmem.randint(42, 43, (100,))
    assert torch.all(result == 42)


def test_randint_pytorch_signatures():
    """Test that randint supports both PyTorch signatures."""
    shmem = iris.iris(1 << 20)

    # Test randint(high, size) signature
    result1 = shmem.randint(10, (2, 3))
    assert result1.shape == (2, 3)
    assert torch.all(result1 >= 0)
    assert torch.all(result1 < 10)
    assert shmem.is_symmetric(result1)

    # Test randint(low, high, size) signature
    result2 = shmem.randint(5, 15, (2, 3))
    assert result2.shape == (2, 3)
    assert torch.all(result2 >= 5)
    assert torch.all(result2 < 15)
    assert shmem.is_symmetric(result2)

    # Both should work correctly
    assert result1.shape == result2.shape
    assert result1.dtype == result2.dtype


def test_randint_deterministic_behavior():
    """Test that randint works with deterministic settings."""
    shmem = iris.iris(1 << 20)

    # Test that randint works regardless of deterministic settings
    result = shmem.randint(0, 10, (2, 3))
    assert result.shape == (2, 3)
    assert torch.all(result >= 0)
    assert torch.all(result < 10)
    assert shmem.is_symmetric(result)
