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
    "shape",
    [
        (1,),
        (5,),
        (2, 3),
        (3, 4, 5),
        (1, 1, 1),
        (10, 20),
    ],
)
def test_zeros_like_basic(dtype, shape):
    shmem = iris.iris(1 << 20)

    # Create input tensor with various shapes and dtypes
    input_tensor = shmem.full(shape, 5, dtype=dtype)

    # Test basic zeros_like
    result = shmem.zeros_like(input_tensor)

    # Verify shape matches
    assert result.shape == input_tensor.shape
    assert result.dtype == input_tensor.dtype

    # Verify all values are zero
    assert torch.all(result == 0)


@pytest.mark.parametrize(
    "input_dtype",
    [
        torch.int32,
        torch.float32,
    ],
)
@pytest.mark.parametrize(
    "output_dtype",
    [
        torch.float32,
        torch.float64,
        torch.int64,
    ],
)
def test_zeros_like_dtype_override(input_dtype, output_dtype):
    shmem = iris.iris(1 << 20)

    input_tensor = shmem.full((2, 3), 10, dtype=input_dtype)

    # Override dtype
    result = shmem.zeros_like(input_tensor, dtype=output_dtype)

    # Verify dtype is overridden
    assert result.dtype == output_dtype
    assert result.shape == input_tensor.shape
    assert torch.all(result == 0)


@pytest.mark.parametrize(
    "requires_grad",
    [
        True,
        False,
    ],
)
def test_zeros_like_requires_grad(requires_grad):
    shmem = iris.iris(1 << 20)

    input_tensor = shmem.full((2, 2), 1, dtype=torch.float32)

    # Test with requires_grad parameter
    result = shmem.zeros_like(input_tensor, requires_grad=requires_grad)

    # Verify requires_grad is set
    assert result.requires_grad == requires_grad
    assert torch.all(result == 0)


def test_zeros_like_device_override():
    shmem = iris.iris(1 << 20)
    input_tensor = shmem.full((3, 3), 2, dtype=torch.float32)

    # Test default behavior
    result = shmem.zeros_like(input_tensor)
    assert str(result.device) == shmem.device
    assert torch.all(result == 0)

    # Test same device works
    result = shmem.zeros_like(input_tensor, device=shmem.device)
    assert str(result.device) == shmem.device
    assert torch.all(result == 0)

    # Test that "cuda" shorthand works (should use current CUDA device)
    if shmem.device.startswith("cuda:"):
        result = shmem.zeros_like(input_tensor, device="cuda")
        assert str(result.device) == shmem.device
        assert torch.all(result == 0)

    # Test None device defaults to Iris device
    result = shmem.zeros_like(input_tensor, device=None)
    assert str(result.device) == shmem.device
    assert torch.all(result == 0)

    # Test that different device throws error
    different_device = "cpu"  # CPU is always different from CUDA
    with pytest.raises(RuntimeError):
        shmem.zeros_like(input_tensor, device=different_device)

    # Test that different CUDA device throws error
    if shmem.device.startswith("cuda:"):
        current_device = torch.device(shmem.device)
        different_cuda = f"cuda:{(current_device.index + 1) % 4}"  # Use next GPU
        with pytest.raises(RuntimeError):
            shmem.zeros_like(input_tensor, device=different_cuda)


def test_zeros_like_layout_override():
    shmem = iris.iris(1 << 20)

    input_tensor = shmem.full((2, 4), 3, dtype=torch.float32)

    # Test with different layout (should default to input layout)
    result = shmem.zeros_like(input_tensor, layout=torch.strided)

    # Verify layout and values
    assert result.layout == input_tensor.layout
    assert torch.all(result == 0)


def test_zeros_like_memory_format():
    shmem = iris.iris(1 << 20)

    input_tensor = shmem.full((4, 2), 1, dtype=torch.float32)

    # Test with default memory_format
    result = shmem.zeros_like(input_tensor, memory_format=torch.contiguous_format)
    assert result.shape == input_tensor.shape
    assert torch.all(result == 0)

    # Test that unsupported memory formats throw an error
    with pytest.raises(RuntimeError):
        shmem.zeros_like(input_tensor, memory_format=torch.channels_last)

    # Test that preserve_format fails if input is not contiguous
    non_contiguous_tensor = input_tensor.transpose(0, 1)  # This makes it non-contiguous
    with pytest.raises(RuntimeError):
        shmem.zeros_like(non_contiguous_tensor, memory_format=torch.preserve_format)


def test_zeros_like_pytorch_equivalence():
    shmem = iris.iris(1 << 20)

    # Create input tensor
    input_tensor = shmem.full((4, 3), 7, dtype=torch.float32)

    # Get Iris result
    iris_result = shmem.zeros_like(input_tensor)

    # Create equivalent PyTorch tensor and get PyTorch result
    pytorch_input = torch.full((4, 3), 7, dtype=torch.float32, device="cuda")
    pytorch_result = torch.zeros_like(pytorch_input)

    # Verify shapes and dtypes match
    assert iris_result.shape == pytorch_result.shape
    assert iris_result.dtype == pytorch_result.dtype

    # Verify values match (both should be all zeros)
    assert torch.all(iris_result == 0)
    assert torch.all(pytorch_result == 0)


def test_zeros_like_edge_cases():
    shmem = iris.iris(1 << 20)

    # Empty tensor
    empty_tensor = shmem.full((0,), 1, dtype=torch.float32)
    empty_result = shmem.zeros_like(empty_tensor)
    assert empty_result.shape == (0,)
    assert empty_result.numel() == 0

    # Single element tensor
    single_tensor = shmem.full((1,), 5, dtype=torch.int32)
    single_result = shmem.zeros_like(single_tensor)
    assert single_result.shape == (1,)
    assert single_result.numel() == 1
    assert single_result[0] == 0

    # Large tensor
    large_tensor = shmem.full((100, 100), 10, dtype=torch.float32)
    large_result = shmem.zeros_like(large_tensor)
    assert large_result.shape == (100, 100)
    assert large_result.numel() == 10000
    assert torch.all(large_result == 0)


@pytest.mark.parametrize(
    "params",
    [
        {"dtype": torch.float32, "requires_grad": True},
        {"dtype": torch.float64, "requires_grad": False},
        {"dtype": torch.float32, "requires_grad": True},
        {"dtype": torch.float16},
        {},
    ],
)
def test_zeros_like_parameter_combinations(params):
    shmem = iris.iris(1 << 20)

    # Use float32 input tensor to support requires_grad
    input_tensor = shmem.full((3, 3), 1, dtype=torch.float32)

    # Test various combinations of parameters
    result = shmem.zeros_like(input_tensor, **params)

    # Verify basic functionality
    assert result.shape == input_tensor.shape
    assert torch.all(result == 0)

    # Verify dtype if specified
    if "dtype" in params:
        assert result.dtype == params["dtype"]

    # Verify requires_grad if specified
    if "requires_grad" in params:
        assert result.requires_grad == params["requires_grad"]
