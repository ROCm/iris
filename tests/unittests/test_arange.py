# SPDX-License-Identifier: MIT
# Copyright (c) 2025 Advanced Micro Devices, Inc. All rights reserved.

import pytest
import torch
import iris


def test_arange_basic_functionality():
    """Test basic arange functionality with various argument combinations."""
    shmem = iris.iris(1 << 20)

    # Test 1: arange(end) - single argument
    result1 = shmem.arange(5)
    assert result1.shape == (5,)
    assert torch.all(result1 == torch.tensor([0, 1, 2, 3, 4], device=result1.device))
    assert result1.dtype == torch.int64
    assert shmem.is_symmetric(result1)

    # Test 2: arange(start, end) - two arguments
    result2 = shmem.arange(1, 4)
    assert result2.shape == (3,)
    assert torch.all(result2 == torch.tensor([1, 2, 3], device=result2.device))
    assert result2.dtype == torch.int64
    assert shmem.is_symmetric(result2)

    # Test 3: arange(start, end, step) - three arguments
    result3 = shmem.arange(1, 2.5, 0.5)
    assert result3.shape == (3,)
    assert torch.allclose(result3, torch.tensor([1.0, 1.5, 2.0], device=result3.device))
    assert result3.dtype == torch.float32
    assert shmem.is_symmetric(result3)

    # Test 4: arange with negative step
    result4 = shmem.arange(5, 0, -1)
    assert result4.shape == (5,)
    assert torch.all(result4 == torch.tensor([5, 4, 3, 2, 1], device=result4.device))
    assert shmem.is_symmetric(result4)


def test_arange_dtype_inference():
    """Test dtype inference logic."""
    shmem = iris.iris(1 << 20)

    # Test integer dtype inference
    result_int = shmem.arange(3)
    assert result_int.dtype == torch.int64
    assert shmem.is_symmetric(result_int)

    # Test float dtype inference
    result_float = shmem.arange(1.0, 3.0)
    assert result_float.dtype == torch.float32
    assert shmem.is_symmetric(result_float)

    # Test explicit dtype override
    result_explicit = shmem.arange(3, dtype=torch.float64)
    assert result_explicit.dtype == torch.float64
    assert shmem.is_symmetric(result_explicit)

    # Test mixed types (should infer float)
    result_mixed = shmem.arange(1, 3.5, 0.5)
    assert result_mixed.dtype == torch.float32
    assert shmem.is_symmetric(result_mixed)


def test_arange_device_handling():
    """Test device parameter handling."""
    shmem = iris.iris(1 << 20)

    # Test default device (should use Iris device)
    result_default = shmem.arange(3)
    assert str(result_default.device) == str(shmem.get_device())
    assert shmem.is_symmetric(result_default)

    # Test explicit device
    iris_device = str(shmem.get_device())
    result_explicit = shmem.arange(3, device=iris_device)
    assert str(result_explicit.device) == iris_device
    assert shmem.is_symmetric(result_explicit)

    # Test device=None (should use Iris device)
    result_none = shmem.arange(3, device=None)
    assert str(result_none.device) == str(shmem.get_device())
    assert shmem.is_symmetric(result_none)


def test_arange_layout_handling():
    """Test layout parameter handling."""
    shmem = iris.iris(1 << 20)

    # Test default layout (strided)
    result_strided = shmem.arange(3, layout=torch.strided)
    assert result_strided.layout == torch.strided
    assert shmem.is_symmetric(result_strided)


def test_arange_requires_grad():
    """Test requires_grad parameter."""
    shmem = iris.iris(1 << 20)

    # Test default (False)
    result_default = shmem.arange(3)
    assert not result_default.requires_grad
    assert shmem.is_symmetric(result_default)

    # Test True
    result_true = shmem.arange(3, dtype=torch.float32, requires_grad=True)
    assert result_true.requires_grad
    assert shmem.is_symmetric(result_true)

    # Test False explicitly
    result_false = shmem.arange(3, requires_grad=False)
    assert not result_false.requires_grad
    assert shmem.is_symmetric(result_false)


def test_arange_out_parameter():
    """Test out parameter functionality."""
    shmem = iris.iris(1 << 20)

    # Test with out parameter
    out_tensor = shmem._Iris__allocate(3, torch.int64)
    result = shmem.arange(3, out=out_tensor)

    # Should return the same tensor object
    assert result is out_tensor
    assert torch.all(result == torch.tensor([0, 1, 2], device=result.device))
    assert shmem.is_symmetric(result)

    # Test with different dtype out tensor
    out_tensor_float = shmem._Iris__allocate(3, torch.float32)
    result_float = shmem.arange(3, dtype=torch.float32, out=out_tensor_float)
    assert result_float is out_tensor_float
    assert result_float.dtype == torch.float32
    assert shmem.is_symmetric(result_float)


def test_arange_error_handling():
    """Test error handling for invalid inputs."""
    shmem = iris.iris(1 << 20)

    # Test step = 0 (should raise ValueError)
    with pytest.raises(ValueError, match="step must be non-zero"):
        shmem.arange(1, 5, 0)

    # Test invalid device (should raise RuntimeError)
    with pytest.raises(RuntimeError):
        shmem.arange(3, device="cpu")  # Iris only supports CUDA


def test_arange_edge_cases():
    """Test edge cases and boundary conditions."""
    shmem = iris.iris(1 << 20)

    # Test invalid ranges (should throw ValueError like PyTorch)
    with pytest.raises(ValueError):
        shmem.arange(5, 1)  # start > end with positive step

    with pytest.raises(ValueError):
        shmem.arange(1, 5, -1)  # start < end with negative step

    # Test single element result
    result_single = shmem.arange(1, 2)
    assert result_single.shape == (1,)
    assert result_single.numel() == 1
    assert result_single[0] == 1
    assert shmem.is_symmetric(result_single)

    # Test large tensor
    result_large = shmem.arange(1000)
    assert result_large.shape == (1000,)
    assert result_large.numel() == 1000
    assert result_large[0] == 0
    assert result_large[-1] == 999
    assert shmem.is_symmetric(result_large)

    # Test floating point precision
    result_float = shmem.arange(0, 1, 0.1)
    assert result_float.shape == (10,)
    assert torch.allclose(result_float[0], torch.tensor(0.0))
    assert torch.allclose(result_float[-1], torch.tensor(0.9))
    assert shmem.is_symmetric(result_float)


def test_arange_pytorch_equivalence():
    """Test that Iris arange produces equivalent results to PyTorch arange."""
    shmem = iris.iris(1 << 20)

    # Test basic equivalence
    iris_result = shmem.arange(5)
    pytorch_result = torch.arange(5, device="cuda")

    assert iris_result.shape == pytorch_result.shape
    assert iris_result.dtype == pytorch_result.dtype
    assert torch.all(iris_result == pytorch_result)

    # Test with start, end, step
    iris_result = shmem.arange(1, 4, 0.5)
    pytorch_result = torch.arange(1, 4, 0.5, device="cuda")

    assert iris_result.shape == pytorch_result.shape
    assert iris_result.dtype == pytorch_result.dtype
    assert torch.allclose(iris_result, pytorch_result)

    # Test dtype inference equivalence
    iris_result = shmem.arange(1.0, 3.0)
    pytorch_result = torch.arange(1.0, 3.0, device="cuda")

    assert iris_result.dtype == pytorch_result.dtype
    assert torch.allclose(iris_result, pytorch_result)


@pytest.mark.parametrize(
    "params",
    [
        {"start": 0, "end": 5, "step": 1, "dtype": torch.int64},
        {"start": 1, "end": 4, "step": 1, "dtype": torch.int64},
        {"start": 0, "end": 1, "step": 0.1, "dtype": torch.float32},
        {"start": 5, "end": 0, "step": -1, "dtype": torch.int64},
        {"start": 0, "end": 10, "step": 2, "dtype": torch.int64},
        {"start": 1.0, "end": 2.0, "step": 0.25, "dtype": torch.float32},
    ],
)
def test_arange_parameter_combinations(params):
    """Test arange with various parameter combinations."""
    shmem = iris.iris(1 << 20)

    result = shmem.arange(start=params["start"], end=params["end"], step=params["step"], dtype=params["dtype"])

    # Verify basic properties
    assert result.dtype == params["dtype"]
    assert shmem.is_symmetric(result)

    # Verify values match PyTorch
    pytorch_result = torch.arange(
        start=params["start"], end=params["end"], step=params["step"], dtype=params["dtype"], device="cuda"
    )

    assert result.shape == pytorch_result.shape
    assert torch.allclose(result, pytorch_result)


@pytest.mark.parametrize(
    "arange_args",
    [
        (5,),  # arange(end)
        (1, 4),  # arange(start, end)
        (0, 1, 0.1),  # arange(start, end, step)
        (10,),  # arange(end) with default dtype
        (3,),  # arange(end) for device test
        (5,),  # arange(end) for requires_grad test
        (3,),  # arange(end) for layout test
    ],
)
@pytest.mark.parametrize(
    "kwargs",
    [
        {},  # No kwargs
        {"dtype": torch.float64},  # dtype override
        {"device": "cuda:0"},  # device override (will be replaced with actual Iris device)
        {"dtype": torch.float32, "requires_grad": True},  # requires_grad True with float dtype
        {"layout": torch.strided},  # strided layout
    ],
)
def test_arange_symmetric_heap_verification(arange_args, kwargs):
    """Test that all arange results are on the symmetric heap."""
    shmem = iris.iris(1 << 20)

    # Replace hardcoded device with actual Iris device
    if "device" in kwargs and kwargs["device"] == "cuda:0":
        kwargs["device"] = str(shmem.get_device())

    # Call arange with the given arguments and kwargs
    result = shmem.arange(*arange_args, **kwargs)

    # Verify symmetric heap allocation
    assert shmem.is_symmetric(result), (
        f"Tensor {result} with args={arange_args}, kwargs={kwargs} is not on symmetric heap"
    )

    # Verify CUDA device
    assert result.device.type == "cuda", (
        f"Tensor {result} with args={arange_args}, kwargs={kwargs} is not on CUDA device"
    )
