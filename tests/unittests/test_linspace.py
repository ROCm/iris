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
        torch.complex64,
        torch.complex128,
    ],
)
@pytest.mark.parametrize(
    "start,end,steps",
    [
        (0.0, 1.0, 5),
        (-10.0, 10.0, 11),
        (3.0, 10.0, 5),
        (0.0, 100.0, 101),
        (1.0, 2.0, 2),
        (0.0, 0.0, 5),
    ],
)
def test_linspace_basic(dtype, start, end, steps):
    shmem = iris.iris(1 << 20)

    # Test basic linspace
    result = shmem.linspace(start, end, steps, dtype=dtype)

    # Verify shape matches
    assert result.shape == (steps,)
    assert result.dtype == dtype

    # Verify first and last values
    assert torch.allclose(result[0], torch.tensor(start, dtype=dtype))
    assert torch.allclose(result[-1], torch.tensor(end, dtype=dtype))

    # Verify tensor is on symmetric heap
    assert shmem.is_symmetric(result)


def test_linspace_default_dtype():
    shmem = iris.iris(1 << 20)

    # Test with default dtype (should use torch.get_default_dtype())
    result = shmem.linspace(0.0, 1.0, 5)
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
def test_linspace_requires_grad(requires_grad):
    shmem = iris.iris(1 << 20)

    # Test with requires_grad parameter
    result = shmem.linspace(0.0, 1.0, 5, dtype=torch.float32, requires_grad=requires_grad)

    # Verify requires_grad is set
    assert result.requires_grad == requires_grad
    assert shmem.is_symmetric(result)


def test_linspace_device_handling():
    shmem = iris.iris(1 << 20)

    # Test default behavior (should use Iris device)
    result = shmem.linspace(0.0, 1.0, 5)
    assert str(result.device) == str(shmem.get_device())
    assert shmem.is_symmetric(result)

    # Test explicit device
    result = shmem.linspace(0.0, 1.0, 5, device=shmem.device)
    assert str(result.device) == str(shmem.get_device())
    assert shmem.is_symmetric(result)

    # Test that "cuda" shorthand works (should use current CUDA device)
    if shmem.device.startswith("cuda:"):
        result = shmem.linspace(0.0, 1.0, 5, device="cuda")
        assert str(result.device) == str(shmem.get_device())
        assert shmem.is_symmetric(result)

    # Test None device defaults to Iris device
    result = shmem.linspace(0.0, 1.0, 5, device=None)
    assert str(result.device) == str(shmem.get_device())
    assert shmem.is_symmetric(result)

    # Test that different device throws error
    different_device = "cpu"  # CPU is always different from CUDA
    with pytest.raises(RuntimeError):
        shmem.linspace(0.0, 1.0, 5, device=different_device)

    # Test that different CUDA device throws error
    if shmem.device.startswith("cuda:") and torch.cuda.device_count() >= 2:
        current_device = torch.device(shmem.device)
        different_cuda = f"cuda:{(current_device.index + 1) % torch.cuda.device_count()}"  # Use next GPU
        with pytest.raises(RuntimeError):
            shmem.linspace(0.0, 1.0, 5, device=different_cuda)


def test_linspace_layout_handling():
    shmem = iris.iris(1 << 20)

    # Test with strided layout (default)
    result = shmem.linspace(0.0, 1.0, 5, layout=torch.strided)
    assert result.layout == torch.strided
    assert shmem.is_symmetric(result)

    # Test that unsupported layout throws error
    with pytest.raises(ValueError):
        shmem.linspace(0.0, 1.0, 5, layout=torch.sparse_coo)


def test_linspace_out_parameter():
    shmem = iris.iris(1 << 20)

    # Test with out parameter
    out_tensor = shmem._Iris__allocate(5, torch.float32)
    result = shmem.linspace(0.0, 1.0, 5, out=out_tensor)

    # Should share the same underlying data (same data_ptr)
    assert result.data_ptr() == out_tensor.data_ptr()
    assert result.shape == (5,)
    assert torch.allclose(result[0], torch.tensor(0.0))
    assert torch.allclose(result[-1], torch.tensor(1.0))
    assert shmem.is_symmetric(result)

    # Test with different dtype out tensor
    out_tensor_float64 = shmem._Iris__allocate(5, torch.float64)
    result_float64 = shmem.linspace(0.0, 1.0, 5, dtype=torch.float64, out=out_tensor_float64)
    assert result_float64.data_ptr() == out_tensor_float64.data_ptr()
    assert result_float64.dtype == torch.float64
    assert shmem.is_symmetric(result_float64)


def test_linspace_steps_variations():
    shmem = iris.iris(1 << 20)

    # Test single step
    result1 = shmem.linspace(0.0, 1.0, 1)
    assert result1.shape == (1,)
    assert torch.allclose(result1[0], torch.tensor(0.0))
    assert shmem.is_symmetric(result1)

    # Test multiple steps
    result2 = shmem.linspace(0.0, 1.0, 10)
    assert result2.shape == (10,)
    assert torch.allclose(result2[0], torch.tensor(0.0))
    assert torch.allclose(result2[-1], torch.tensor(1.0))
    assert shmem.is_symmetric(result2)

    # Test with tuple as steps argument
    result3 = shmem.linspace(0.0, 1.0, (5,))
    assert result3.shape == (5,)
    assert shmem.is_symmetric(result3)

    # Test with list as steps argument
    result4 = shmem.linspace(0.0, 1.0, [5])
    assert result4.shape == (5,)
    assert shmem.is_symmetric(result4)


def test_linspace_edge_cases():
    shmem = iris.iris(1 << 20)

    # Single step (start == end)
    single_result = shmem.linspace(5.0, 5.0, 1)
    assert single_result.shape == (1,)
    assert torch.allclose(single_result[0], torch.tensor(5.0))
    assert shmem.is_symmetric(single_result)

    # Two steps
    two_result = shmem.linspace(0.0, 1.0, 2)
    assert two_result.shape == (2,)
    assert torch.allclose(two_result[0], torch.tensor(0.0))
    assert torch.allclose(two_result[1], torch.tensor(1.0))
    assert shmem.is_symmetric(two_result)

    # Large number of steps
    large_result = shmem.linspace(0.0, 100.0, 1000)
    assert large_result.shape == (1000,)
    assert torch.allclose(large_result[0], torch.tensor(0.0))
    assert torch.allclose(large_result[-1], torch.tensor(100.0))
    assert shmem.is_symmetric(large_result)

    # Negative range
    neg_result = shmem.linspace(-10.0, -5.0, 6)
    assert neg_result.shape == (6,)
    assert torch.allclose(neg_result[0], torch.tensor(-10.0))
    assert torch.allclose(neg_result[-1], torch.tensor(-5.0))
    assert shmem.is_symmetric(neg_result)


def test_linspace_pytorch_equivalence():
    shmem = iris.iris(1 << 20)

    # Test basic equivalence
    iris_result = shmem.linspace(0.0, 1.0, 5)
    pytorch_result = torch.linspace(0.0, 1.0, 5, device="cuda")

    assert iris_result.shape == pytorch_result.shape
    assert iris_result.dtype == pytorch_result.dtype
    assert torch.allclose(iris_result, pytorch_result)

    # Test with explicit dtype
    iris_result = shmem.linspace(0.0, 1.0, 5, dtype=torch.float64)
    pytorch_result = torch.linspace(0.0, 1.0, 5, dtype=torch.float64, device="cuda")

    assert iris_result.shape == pytorch_result.shape
    assert iris_result.dtype == pytorch_result.dtype
    assert torch.allclose(iris_result, pytorch_result)

    # Test with requires_grad
    iris_result = shmem.linspace(0.0, 1.0, 5, dtype=torch.float32, requires_grad=True)
    pytorch_result = torch.linspace(0.0, 1.0, 5, dtype=torch.float32, device="cuda", requires_grad=True)

    assert iris_result.shape == pytorch_result.shape
    assert iris_result.dtype == pytorch_result.dtype
    assert iris_result.requires_grad == pytorch_result.requires_grad


@pytest.mark.parametrize(
    "params",
    [
        {"dtype": torch.float32, "requires_grad": True},
        {"dtype": torch.float64, "requires_grad": False},
        {"dtype": torch.complex64},
        {"dtype": torch.complex128},
        {"layout": torch.strided},
        {},
    ],
)
def test_linspace_parameter_combinations(params):
    shmem = iris.iris(1 << 20)

    # Test various combinations of parameters
    result = shmem.linspace(0.0, 1.0, 5, **params)

    # Verify basic functionality
    assert result.shape == (5,)
    assert torch.allclose(result[0], torch.tensor(0.0, dtype=result.dtype))
    assert torch.allclose(result[-1], torch.tensor(1.0, dtype=result.dtype))
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
    "start,end,steps,dtype",
    [
        (0.0, 1.0, 5, torch.float32),
        (-10.0, 10.0, 11, torch.float64),
        (3.0, 10.0, 5, torch.float16),
        (0.0, 100.0, 101, torch.complex64),
        (1.0, 2.0, 2, torch.complex128),
    ],
)
def test_linspace_symmetric_heap_shapes_dtypes(start, end, steps, dtype):
    """Test that linspace returns tensors on symmetric heap for various shapes and dtypes."""
    shmem = iris.iris(1 << 20)

    # Test linspace with these parameters
    result = shmem.linspace(start, end, steps, dtype=dtype)

    # Verify tensor is on symmetric heap
    assert shmem.is_symmetric(result), (
        f"Tensor with start={start}, end={end}, steps={steps}, dtype={dtype} is NOT on symmetric heap!"
    )

    # Also verify basic functionality
    assert result.shape == (steps,)
    assert result.dtype == dtype
    assert torch.allclose(result[0], torch.tensor(start, dtype=dtype))
    assert torch.allclose(result[-1], torch.tensor(end, dtype=dtype))


@pytest.mark.parametrize("dtype", [torch.float16, torch.float32, torch.float64, torch.complex64, torch.complex128])
def test_linspace_symmetric_heap_dtype_override(dtype):
    """Test that linspace with dtype override returns tensors on symmetric heap."""
    shmem = iris.iris(1 << 20)

    result = shmem.linspace(0.0, 1.0, 5, dtype=dtype)
    assert shmem.is_symmetric(result), f"Tensor with dtype {dtype} is NOT on symmetric heap!"
    assert result.dtype == dtype


def test_linspace_symmetric_heap_other_params():
    """Test that linspace with other parameters returns tensors on symmetric heap."""
    shmem = iris.iris(1 << 20)

    # Test with requires_grad
    result = shmem.linspace(0.0, 1.0, 5, dtype=torch.float32, requires_grad=True)
    assert shmem.is_symmetric(result), "Tensor with requires_grad=True is NOT on symmetric heap!"

    # Test with device override
    result = shmem.linspace(0.0, 1.0, 5, device=shmem.device)
    assert shmem.is_symmetric(result), "Tensor with device override is NOT on symmetric heap!"

    # Test with layout override (only strided is supported)
    result = shmem.linspace(0.0, 1.0, 5, layout=torch.strided)
    assert shmem.is_symmetric(result), "Tensor with layout override is NOT on symmetric heap!"

    # Test with out parameter
    out_tensor = shmem._Iris__allocate(5, torch.float32)
    result = shmem.linspace(0.0, 1.0, 5, out=out_tensor)
    assert shmem.is_symmetric(result), "Tensor with out parameter is NOT on symmetric heap!"


def test_linspace_invalid_output_tensor():
    """Test error handling for invalid output tensors."""
    shmem = iris.iris(1 << 20)

    # Test with wrong size output tensor
    wrong_size_tensor = shmem._Iris__allocate(3, torch.float32)  # Wrong size for 5 steps
    with pytest.raises(RuntimeError):
        shmem.linspace(0.0, 1.0, 5, out=wrong_size_tensor)

    # Test with wrong dtype output tensor
    wrong_dtype_tensor = shmem._Iris__allocate(5, torch.int32)  # Wrong dtype
    with pytest.raises(RuntimeError):
        shmem.linspace(0.0, 1.0, 5, dtype=torch.float32, out=wrong_dtype_tensor)

    # Test with tensor not on symmetric heap (create a regular PyTorch tensor)
    regular_tensor = torch.linspace(0.0, 1.0, 5, device="cuda")
    with pytest.raises(RuntimeError):
        shmem.linspace(0.0, 1.0, 5, out=regular_tensor)


def test_linspace_default_dtype_behavior():
    """Test that linspace uses the global default dtype when dtype=None."""
    shmem = iris.iris(1 << 20)

    # Save original default dtype
    original_default = torch.get_default_dtype()

    try:
        # Test with float32 default
        torch.set_default_dtype(torch.float32)
        result1 = shmem.linspace(0.0, 1.0, 5)
        assert result1.dtype == torch.float32

        # Test with float64 default
        torch.set_default_dtype(torch.float64)
        result2 = shmem.linspace(0.0, 1.0, 5)
        assert result2.dtype == torch.float64

    finally:
        # Restore original default dtype
        torch.set_default_dtype(original_default)


def test_linspace_steps_parsing():
    """Test various ways of specifying steps."""
    shmem = iris.iris(1 << 20)

    # Test integer argument
    result1 = shmem.linspace(0.0, 1.0, 5)
    assert result1.shape == (5,)

    # Test single tuple argument
    result2 = shmem.linspace(0.0, 1.0, (5,))
    assert result2.shape == (5,)

    # Test single list argument
    result3 = shmem.linspace(0.0, 1.0, [5])
    assert result3.shape == (5,)

    # Test nested tuple (should be flattened)
    result4 = shmem.linspace(0.0, 1.0, ((5,),))
    assert result4.shape == (5,)

    # All should produce the same result shape
    assert result1.shape == result2.shape
    assert result2.shape == result3.shape
    assert result3.shape == result4.shape


def test_linspace_complex_numbers():
    """Test linspace with complex numbers."""
    shmem = iris.iris(1 << 20)

    # Test with complex start and end
    result = shmem.linspace(0.0 + 0.0j, 1.0 + 1.0j, 5, dtype=torch.complex64)
    assert result.shape == (5,)
    assert result.dtype == torch.complex64
    assert torch.allclose(result[0], torch.tensor(0.0 + 0.0j, dtype=torch.complex64))
    assert torch.allclose(result[-1], torch.tensor(1.0 + 1.0j, dtype=torch.complex64))
    assert shmem.is_symmetric(result)

    # Test with complex dtype inference
    result = shmem.linspace(0.0 + 0.0j, 1.0 + 1.0j, 5)
    assert result.dtype == torch.complex64  # Should infer complex dtype
    assert shmem.is_symmetric(result)


def test_linspace_tensor_inputs():
    """Test linspace with tensor inputs."""
    shmem = iris.iris(1 << 20)

    # Test with 0-dimensional tensor inputs
    start_tensor = torch.tensor(0.0, device="cuda")
    end_tensor = torch.tensor(1.0, device="cuda")

    result = shmem.linspace(start_tensor, end_tensor, 5)
    assert result.shape == (5,)
    assert torch.allclose(result[0], torch.tensor(0.0))
    assert torch.allclose(result[-1], torch.tensor(1.0))
    assert shmem.is_symmetric(result)

    # Test with complex tensor inputs
    start_complex = torch.tensor(0.0 + 0.0j, device="cuda")
    end_complex = torch.tensor(1.0 + 1.0j, device="cuda")

    result_complex = shmem.linspace(start_complex, end_complex, 5)
    assert result_complex.shape == (5,)
    assert result_complex.dtype == torch.complex64
    assert shmem.is_symmetric(result_complex)


def test_linspace_accuracy():
    """Test that linspace produces accurate results."""
    shmem = iris.iris(1 << 20)

    # Test with simple range
    result = shmem.linspace(0.0, 1.0, 5)
    expected = torch.tensor([0.0, 0.25, 0.5, 0.75, 1.0], device="cuda")
    assert torch.allclose(result, expected, atol=1e-6)

    # Test with negative range
    result = shmem.linspace(-10.0, 10.0, 5)
    expected = torch.tensor([-10.0, -5.0, 0.0, 5.0, 10.0], device="cuda")
    assert torch.allclose(result, expected, atol=1e-6)

    # Test with many steps
    result = shmem.linspace(0.0, 1.0, 100)
    assert result.shape == (100,)
    assert torch.allclose(result[0], torch.tensor(0.0))
    assert torch.allclose(result[-1], torch.tensor(1.0))
    # Check that step size is correct
    step_size = result[1] - result[0]
    expected_step = 1.0 / 99.0  # (end - start) / (steps - 1)
    assert torch.allclose(step_size, torch.tensor(expected_step), atol=1e-6)


def test_linspace_deterministic_behavior():
    """Test that linspace works with deterministic settings."""
    shmem = iris.iris(1 << 20)

    # Test that linspace works regardless of deterministic settings
    result = shmem.linspace(0.0, 1.0, 5)
    assert result.shape == (5,)
    assert torch.allclose(result[0], torch.tensor(0.0))
    assert torch.allclose(result[-1], torch.tensor(1.0))
    assert shmem.is_symmetric(result)
