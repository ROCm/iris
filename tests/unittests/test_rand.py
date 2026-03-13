# SPDX-License-Identifier: MIT
# Copyright (c) 2025 Advanced Micro Devices, Inc. All rights reserved.

import torch
import pytest
import iris


@pytest.mark.parametrize(
    "dtype",
    [
        torch.float16,
        torch.float32,
        torch.float64,
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
def test_rand_basic(dtype, size):
    shmem = iris.iris(1 << 20)

    # Test basic rand
    result = shmem.rand(*size, dtype=dtype)

    # Verify shape matches
    assert result.shape == size
    assert result.dtype == dtype

    # Verify values are within range [0, 1)
    assert torch.all(result >= 0)
    assert torch.all(result < 1)

    # Verify tensor is on symmetric heap
    assert shmem.is_symmetric(result)


def test_rand_default_dtype():
    shmem = iris.iris(1 << 20)

    # Test with default dtype (should use torch.get_default_dtype())
    result = shmem.rand(2, 3)
    expected_dtype = torch.get_default_dtype()
    assert result.dtype == expected_dtype
    assert shmem.is_symmetric(result)


@pytest.mark.parametrize(
    "requires_grad",
    [
        True,
        False,
    ],
)
def test_rand_requires_grad(requires_grad):
    shmem = iris.iris(1 << 20)

    # Test with requires_grad parameter
    result = shmem.rand(2, 2, dtype=torch.float32, requires_grad=requires_grad)

    # Verify requires_grad is set
    assert result.requires_grad == requires_grad
    assert shmem.is_symmetric(result)


def test_rand_device_handling():
    shmem = iris.iris(1 << 20)

    # Test default behavior (should use Iris device)
    result = shmem.rand(3, 3)
    assert str(result.device) == str(shmem.get_device())
    assert shmem.is_symmetric(result)

    # Test explicit device
    result = shmem.rand(3, 3, device=shmem.device)
    assert str(result.device) == str(shmem.get_device())
    assert shmem.is_symmetric(result)

    # Test that "cuda" shorthand works (should use current CUDA device)
    if shmem.device.startswith("cuda:"):
        result = shmem.rand(3, 3, device="cuda")
        assert str(result.device) == str(shmem.get_device())
        assert shmem.is_symmetric(result)

    # Test None device defaults to Iris device
    result = shmem.rand(3, 3, device=None)
    assert str(result.device) == str(shmem.get_device())
    assert shmem.is_symmetric(result)

    # Test that different device throws error
    different_device = "cpu"  # CPU is always different from CUDA
    with pytest.raises(RuntimeError):
        shmem.rand(3, 3, device=different_device)

    # Test that different CUDA device throws error
    if shmem.device.startswith("cuda:") and torch.cuda.device_count() >= 2:
        current_device = torch.device(shmem.device)
        different_cuda = f"cuda:{(current_device.index + 1) % torch.cuda.device_count()}"  # Use next GPU
        with pytest.raises(RuntimeError):
            shmem.rand(3, 3, device=different_cuda)


def test_rand_layout_handling():
    shmem = iris.iris(1 << 20)

    # Test with strided layout (default)
    result = shmem.rand(2, 4, layout=torch.strided)
    assert result.layout == torch.strided
    assert shmem.is_symmetric(result)

    # Test that unsupported layout throws error
    with pytest.raises(ValueError):
        shmem.rand(2, 4, layout=torch.sparse_coo)


def test_rand_out_parameter():
    shmem = iris.iris(1 << 20)

    # Test with out parameter
    out_tensor = shmem.heap.allocate(6, torch.float32)
    result = shmem.rand(2, 3, out=out_tensor)

    # Should share the same underlying data (same data_ptr)
    assert result.data_ptr() == out_tensor.data_ptr()
    assert result.shape == (2, 3)
    assert torch.all(result >= 0)
    assert torch.all(result < 1)
    assert shmem.is_symmetric(result)

    # Test with different dtype out tensor
    out_tensor_float64 = shmem.heap.allocate(6, torch.float64)
    result_float64 = shmem.rand(2, 3, dtype=torch.float64, out=out_tensor_float64)
    assert result_float64.data_ptr() == out_tensor_float64.data_ptr()
    assert result_float64.dtype == torch.float64
    assert shmem.is_symmetric(result_float64)


def test_rand_size_variations():
    shmem = iris.iris(1 << 20)

    # Test single dimension
    result1 = shmem.rand(5)
    assert result1.shape == (5,)
    assert torch.all(result1 >= 0)
    assert torch.all(result1 < 1)
    assert shmem.is_symmetric(result1)

    # Test multiple dimensions
    result2 = shmem.rand(2, 3, 4)
    assert result2.shape == (2, 3, 4)
    assert torch.all(result2 >= 0)
    assert torch.all(result2 < 1)
    assert shmem.is_symmetric(result2)

    # Test with tuple as single argument
    result3 = shmem.rand((3, 4))
    assert result3.shape == (3, 4)
    assert torch.all(result3 >= 0)
    assert torch.all(result3 < 1)
    assert shmem.is_symmetric(result3)

    # Test with list as single argument
    result4 = shmem.rand([2, 5])
    assert result4.shape == (2, 5)
    assert torch.all(result4 >= 0)
    assert torch.all(result4 < 1)
    assert shmem.is_symmetric(result4)


def test_rand_edge_cases():
    shmem = iris.iris(1 << 20)

    # Empty tensor
    empty_result = shmem.rand(0)
    assert empty_result.shape == (0,)
    assert empty_result.numel() == 0
    assert shmem.is_symmetric(empty_result)

    # Single element tensor
    single_result = shmem.rand(1)
    assert single_result.shape == (1,)
    assert single_result.numel() == 1
    assert torch.all(single_result >= 0)
    assert torch.all(single_result < 1)
    assert shmem.is_symmetric(single_result)

    # Large tensor
    large_result = shmem.rand(50, 50)
    assert large_result.shape == (50, 50)
    assert large_result.numel() == 2500
    assert torch.all(large_result >= 0)
    assert torch.all(large_result < 1)
    assert shmem.is_symmetric(large_result)

    # Zero-dimensional tensor (scalar)
    scalar_result = shmem.rand(())
    assert scalar_result.shape == ()
    assert scalar_result.numel() == 1
    assert torch.all(scalar_result >= 0)
    assert torch.all(scalar_result < 1)
    assert shmem.is_symmetric(scalar_result)


def test_rand_pytorch_equivalence():
    shmem = iris.iris(1 << 20)

    # Test basic equivalence
    iris_result = shmem.rand(4, 3)
    pytorch_result = torch.rand(4, 3, device="cuda")

    assert iris_result.shape == pytorch_result.shape
    assert iris_result.dtype == pytorch_result.dtype

    # Test with explicit dtype
    iris_result = shmem.rand(2, 2, dtype=torch.float64)
    pytorch_result = torch.rand(2, 2, dtype=torch.float64, device="cuda")

    assert iris_result.shape == pytorch_result.shape
    assert iris_result.dtype == pytorch_result.dtype

    # Test with requires_grad
    iris_result = shmem.rand(3, 3, dtype=torch.float32, requires_grad=True)
    pytorch_result = torch.rand(3, 3, dtype=torch.float32, device="cuda", requires_grad=True)

    assert iris_result.shape == pytorch_result.shape
    assert iris_result.dtype == pytorch_result.dtype
    assert iris_result.requires_grad == pytorch_result.requires_grad


@pytest.mark.parametrize(
    "params",
    [
        {"dtype": torch.float32, "requires_grad": True},
        {"dtype": torch.float64, "requires_grad": False},
        {"dtype": torch.float16},
        {"layout": torch.strided},
        {},
    ],
)
def test_rand_parameter_combinations(params):
    shmem = iris.iris(1 << 20)

    # Test various combinations of parameters
    result = shmem.rand(3, 3, **params)

    # Verify basic functionality
    assert result.shape == (3, 3)
    assert torch.all(result >= 0)
    assert torch.all(result < 1)
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
        ((5,), torch.float64),
        ((2, 3), torch.float16),
        ((3, 4, 5), torch.float32),
        ((0,), torch.float32),  # Empty tensor
        ((50, 50), torch.float32),  # Large tensor
        ((), torch.float32),  # Scalar tensor
    ],
)
def test_rand_symmetric_heap_shapes_dtypes(size, dtype):
    """Test that rand returns tensors on symmetric heap for various shapes and dtypes."""
    shmem = iris.iris(1 << 20)

    # Test rand with this size and dtype
    result = shmem.rand(*size, dtype=dtype)

    # Verify tensor is on symmetric heap
    assert shmem.is_symmetric(result), f"Tensor with size {size}, dtype {dtype} is NOT on symmetric heap!"

    # Also verify basic functionality
    assert result.shape == size
    assert result.dtype == dtype
    assert torch.all(result >= 0)
    assert torch.all(result < 1)


@pytest.mark.parametrize("dtype", [torch.float16, torch.float32, torch.float64])
def test_rand_symmetric_heap_dtype_override(dtype):
    """Test that rand with dtype override returns tensors on symmetric heap."""
    shmem = iris.iris(1 << 20)

    result = shmem.rand(3, 3, dtype=dtype)
    assert shmem.is_symmetric(result), f"Tensor with dtype {dtype} is NOT on symmetric heap!"
    assert result.dtype == dtype


def test_rand_symmetric_heap_other_params():
    """Test that rand with other parameters returns tensors on symmetric heap."""
    shmem = iris.iris(1 << 20)

    # Test with requires_grad
    result = shmem.rand(3, 3, dtype=torch.float32, requires_grad=True)
    assert shmem.is_symmetric(result), "Tensor with requires_grad=True is NOT on symmetric heap!"

    # Test with device override
    result = shmem.rand(3, 3, device=shmem.device)
    assert shmem.is_symmetric(result), "Tensor with device override is NOT on symmetric heap!"

    # Test with layout override (only strided is supported)
    result = shmem.rand(3, 3, layout=torch.strided)
    assert shmem.is_symmetric(result), "Tensor with layout override is NOT on symmetric heap!"

    # Test with out parameter
    out_tensor = shmem.heap.allocate(9, torch.float32)
    result = shmem.rand(3, 3, out=out_tensor)
    assert shmem.is_symmetric(result), "Tensor with out parameter is NOT on symmetric heap!"


def test_rand_invalid_output_tensor():
    """Test error handling for invalid output tensors."""
    shmem = iris.iris(1 << 20)

    # Test with wrong size output tensor
    wrong_size_tensor = shmem.heap.allocate(4, torch.float32)  # Wrong size for (3, 3)
    with pytest.raises(RuntimeError):
        shmem.rand(3, 3, out=wrong_size_tensor)

    # Test with wrong dtype output tensor
    wrong_dtype_tensor = shmem.heap.allocate(9, torch.int32)  # Wrong dtype
    with pytest.raises(RuntimeError):
        shmem.rand(3, 3, dtype=torch.float32, out=wrong_dtype_tensor)

    # Test with tensor not on symmetric heap (create a regular PyTorch tensor)
    regular_tensor = torch.rand(3, 3, device="cuda")
    with pytest.raises(RuntimeError):
        shmem.rand(3, 3, out=regular_tensor)


def test_rand_default_dtype_behavior():
    """Test that rand uses the global default dtype when dtype=None."""
    shmem = iris.iris(1 << 20)

    # Save original default dtype
    original_default = torch.get_default_dtype()

    try:
        # Test with float32 default
        torch.set_default_dtype(torch.float32)
        result1 = shmem.rand(2, 2)
        assert result1.dtype == torch.float32

        # Test with float64 default
        torch.set_default_dtype(torch.float64)
        result2 = shmem.rand(2, 2)
        assert result2.dtype == torch.float64

    finally:
        # Restore original default dtype
        torch.set_default_dtype(original_default)


def test_rand_size_parsing():
    """Test various ways of specifying size."""
    shmem = iris.iris(1 << 20)

    # Test individual arguments
    result1 = shmem.rand(2, 3, 4)
    assert result1.shape == (2, 3, 4)

    # Test single tuple argument
    result2 = shmem.rand((2, 3, 4))
    assert result2.shape == (2, 3, 4)

    # Test single list argument
    result3 = shmem.rand([2, 3, 4])
    assert result3.shape == (2, 3, 4)

    # Test nested tuple (should be flattened)
    result4 = shmem.rand(((2, 3, 4),))
    assert result4.shape == (2, 3, 4)

    # All should produce the same result shape
    assert result1.shape == result2.shape
    assert result2.shape == result3.shape
    assert result3.shape == result4.shape


def test_rand_generator():
    """Test generator parameter."""
    shmem = iris.iris(1 << 20)

    # Test with generator
    generator = torch.Generator(device="cuda")
    generator.manual_seed(42)
    result1 = shmem.rand(3, 3, generator=generator)
    assert result1.shape == (3, 3)
    assert torch.all(result1 >= 0)
    assert torch.all(result1 < 1)
    assert shmem.is_symmetric(result1)

    # Test without generator (should still work)
    result2 = shmem.rand(3, 3)
    assert result2.shape == (3, 3)
    assert torch.all(result2 >= 0)
    assert torch.all(result2 < 1)
    assert shmem.is_symmetric(result2)

    # Test that generator produces reproducible results
    generator1 = torch.Generator(device="cuda")
    generator1.manual_seed(123)
    result3 = shmem.rand(3, 3, generator=generator1)

    generator2 = torch.Generator(device="cuda")
    generator2.manual_seed(123)
    result4 = shmem.rand(3, 3, generator=generator2)

    # Results should be identical with same seed
    assert torch.allclose(result3, result4)


def test_rand_pin_memory():
    """Test pin_memory parameter (should be ignored for Iris tensors)."""
    shmem = iris.iris(1 << 20)

    # Test with pin_memory=True (should work but be ignored since Iris tensors are on GPU)
    result = shmem.rand(2, 3, pin_memory=True)
    assert result.shape == (2, 3)
    assert torch.all(result >= 0)
    assert torch.all(result < 1)
    assert shmem.is_symmetric(result)
    # Note: pin_memory is ignored for GPU tensors, so we just verify it doesn't cause errors


def test_rand_distribution():
    """Test that rand produces values in the correct range [0, 1)."""
    shmem = iris.iris(1 << 20)

    # Test with reasonably sized tensor to get good statistical coverage
    result = shmem.rand(100, 100)
    assert result.shape == (100, 100)

    # All values should be >= 0 and < 1
    assert torch.all(result >= 0)
    assert torch.all(result < 1)

    # Check that we have some values close to 0 and close to 1
    # (this is a statistical test, so we check for reasonable bounds)
    min_val = torch.min(result).item()
    max_val = torch.max(result).item()

    # Should have some values close to 0
    assert min_val < 0.1, f"Minimum value {min_val} is too high"
    # Should have some values close to 1
    assert max_val > 0.9, f"Maximum value {max_val} is too low"

    assert shmem.is_symmetric(result)


def test_rand_deterministic_behavior():
    """Test that rand works with deterministic settings."""
    shmem = iris.iris(1 << 20)

    # Test that rand works regardless of deterministic settings
    result = shmem.rand(2, 3)
    assert result.shape == (2, 3)
    assert torch.all(result >= 0)
    assert torch.all(result < 1)
    assert shmem.is_symmetric(result)
