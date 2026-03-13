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
    assert str(result.device) == str(input_tensor.device)
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

    # Test None device defaults to input tensor's device
    result = shmem.zeros_like(input_tensor, device=None)
    assert str(result.device) == str(input_tensor.device)
    assert torch.all(result == 0)

    # Test that different device throws error
    different_device = "cpu"  # CPU is always different from CUDA
    with pytest.raises(RuntimeError):
        shmem.zeros_like(input_tensor, device=different_device)

    # Test that different CUDA device throws error
    if shmem.device.startswith("cuda:") and torch.cuda.device_count() >= 2:
        current_device = torch.device(shmem.device)
        different_cuda = f"cuda:{(current_device.index + 1) % torch.cuda.device_count()}"  # Use next GPU
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

    # Test channels_last format (should work for 4D tensors)
    # Create a 4D tensor (NCHW format)
    input_4d = shmem.full((2, 3, 4, 5), 1, dtype=torch.float32)
    result_4d = shmem.zeros_like(input_4d, memory_format=torch.channels_last)

    # For channels_last format, the shape remains (N, C, H, W); only the memory layout (strides) changes.
    # Input: (2, 3, 4, 5) -> Output: (2, 3, 4, 5) with channels_last strides
    expected_shape = input_4d.shape
    assert result_4d.shape == expected_shape, f"Expected {expected_shape}, got {result_4d.shape}"
    assert torch.all(result_4d == 0)

    # Compare with PyTorch's channels_last implementation
    pytorch_input_4d = torch.full((2, 3, 4, 5), 1, dtype=torch.float32, device="cuda")
    pytorch_result_4d = torch.zeros_like(pytorch_input_4d, memory_format=torch.channels_last)

    # Verify it's actually in channels_last format
    strides = result_4d.stride()
    assert strides[0] > strides[2] > strides[3] > strides[1] == 1, (
        f"Expected channels_last format strides, got {strides}"
    )

    # Test channels_last_3d format (should work for 5D tensors)
    input_5d = shmem.full((2, 3, 4, 5, 6), 1, dtype=torch.float32)
    result_5d = shmem.zeros_like(input_5d, memory_format=torch.channels_last_3d)

    # For channels_last_3d format, the shape remains (N, C, D, H, W); only the memory layout (strides) changes.
    # Input: (2, 3, 4, 5, 6) -> Output: (2, 3, 4, 5, 6) with channels_last_3d strides
    expected_shape_5d = input_5d.shape
    assert result_5d.shape == expected_shape_5d, f"Expected {expected_shape_5d}, got {result_5d.shape}"
    assert torch.all(result_5d == 0)

    # Compare with PyTorch's channels_last_3d implementation
    pytorch_input_5d = torch.full((2, 3, 4, 5, 6), 1, dtype=torch.float32, device="cuda")
    pytorch_result_5d = torch.zeros_like(pytorch_input_5d, memory_format=torch.channels_last_3d)

    # Verify it's actually in channels_last_3d format
    strides_5d = result_5d.stride()
    assert strides_5d[0] > strides_5d[2] > strides_5d[3] > strides_5d[4] > strides_5d[1] == 1, (
        f"Expected channels_last_3d format strides, got {strides_5d}"
    )

    # Test preserve_format with contiguous input
    result_preserve = shmem.zeros_like(input_tensor, memory_format=torch.preserve_format)
    assert result_preserve.shape == input_tensor.shape
    assert torch.all(result_preserve == 0)

    # Test preserve_format with non-contiguous input (should now work)
    non_contiguous_tensor = input_tensor.transpose(0, 1)  # This makes it non-contiguous
    result_non_contig = shmem.zeros_like(non_contiguous_tensor, memory_format=torch.preserve_format)
    assert result_non_contig.shape == non_contiguous_tensor.shape
    assert torch.all(result_non_contig == 0)

    # Test preserve_format with channels_last input (should copy the format)
    # Create input tensor directly in channels_last format using Iris
    input_4d_channels_last = shmem.zeros_like(
        shmem.full((2, 3, 4, 5), 1, dtype=torch.float32), memory_format=torch.channels_last
    )
    result_preserve_channels_last = shmem.zeros_like(input_4d_channels_last, memory_format=torch.preserve_format)

    # Compare with PyTorch's preserve_format behavior
    pytorch_input_4d_cl = torch.full((2, 3, 4, 5), 1, dtype=torch.float32, device="cuda")
    pytorch_input_4d_cl = pytorch_input_4d_cl.to(memory_format=torch.channels_last)
    pytorch_result_preserve = torch.zeros_like(pytorch_input_4d_cl, memory_format=torch.preserve_format)

    # Verify strides match exactly (preserve_format should copy the input's memory format)
    assert result_preserve_channels_last.stride() == pytorch_result_preserve.stride(), (
        f"Preserve format strides don't match: {result_preserve_channels_last.stride()} vs {pytorch_result_preserve.stride()}"
    )

    # Verify all results are on the symmetric heap
    assert shmem.is_symmetric(result_4d)
    assert shmem.is_symmetric(result_5d)
    assert shmem.is_symmetric(result_preserve_channels_last)


def test_channels_last_format_shape_preservation():
    """Test that channels_last format preserves shape and only changes strides."""
    shmem = iris.iris(1 << 20)

    # Test 4D tensor
    input_4d = shmem.full((2, 3, 4, 5), 1, dtype=torch.float32)
    result_4d = shmem.zeros_like(input_4d, memory_format=torch.channels_last)

    # Verify shape is preserved
    assert result_4d.shape == input_4d.shape, f"Shape changed: {input_4d.shape} -> {result_4d.shape}"
    assert result_4d.shape == (2, 3, 4, 5), f"Expected shape (2, 3, 4, 5), got {result_4d.shape}"

    # Verify strides indicate channels_last format
    strides = result_4d.stride()
    N, C, H, W = 2, 3, 4, 5
    expected_strides = (C * H * W, 1, C * W, C)  # (60, 1, 15, 3)
    assert strides == expected_strides, f"Expected strides {expected_strides}, got {strides}"

    # Verify channels_last format characteristics: strides[1] == 1 (channels dimension is contiguous)
    assert strides[1] == 1, f"Channels dimension should be contiguous (stride=1), got {strides[1]}"

    # Test 5D tensor
    input_5d = shmem.full((2, 3, 4, 5, 6), 1, dtype=torch.float32)
    result_5d = shmem.zeros_like(input_5d, memory_format=torch.channels_last_3d)

    # Verify shape is preserved
    assert result_5d.shape == input_5d.shape, f"Shape changed: {input_5d.shape} -> {result_5d.shape}"
    assert result_5d.shape == (2, 3, 4, 5, 6), f"Expected shape (2, 3, 4, 5, 6), got {result_5d.shape}"

    # Verify strides indicate channels_last_3d format
    strides_5d = result_5d.stride()
    N, C, D, H, W = 2, 3, 4, 5, 6
    expected_strides_5d = (C * D * H * W, 1, C * D * W, C * W, C)  # (360, 1, 90, 18, 3)
    assert strides_5d == expected_strides_5d, f"Expected strides {expected_strides_5d}, got {strides_5d}"

    # Verify channels_last_3d format characteristics: strides[1] == 1 (channels dimension is contiguous)
    assert strides_5d[1] == 1, f"Channels dimension should be contiguous (stride=1), got {strides_5d[1]}"

    # Compare with PyTorch's behavior to ensure consistency
    pytorch_input_4d = torch.full((2, 3, 4, 5), 1, dtype=torch.float32, device="cuda")
    pytorch_result_4d = torch.zeros_like(pytorch_input_4d, memory_format=torch.channels_last)

    # Verify Iris and PyTorch have same shape
    assert result_4d.shape == pytorch_result_4d.shape, (
        f"Shape mismatch: Iris {result_4d.shape} vs PyTorch {pytorch_result_4d.shape}"
    )

    # Verify Iris and PyTorch have same strides
    assert result_4d.stride() == pytorch_result_4d.stride(), (
        f"Strides mismatch: Iris {result_4d.stride()} vs PyTorch {pytorch_result_4d.stride()}"
    )

    # Verify tensors are on symmetric heap
    assert shmem.is_symmetric(result_4d)
    assert shmem.is_symmetric(result_5d)


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

    # Test that device defaults work like PyTorch
    # PyTorch: device=None defaults to input.device
    # Iris: should do the same
    iris_result_default = shmem.zeros_like(input_tensor, device=None)
    pytorch_result_default = torch.zeros_like(pytorch_input, device=None)

    # Both should default to their input tensor's device
    assert str(iris_result_default.device) == str(input_tensor.device)
    assert str(pytorch_result_default.device) == str(pytorch_input.device)


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

    # Verify all edge case results are on symmetric heap
    assert shmem.is_symmetric(empty_result)
    assert shmem.is_symmetric(single_result)
    assert shmem.is_symmetric(large_result)


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

    # Verify tensor is on symmetric heap
    assert shmem.is_symmetric(result)


@pytest.mark.parametrize(
    "shape,dtype",
    [
        ((1,), torch.float32),
        ((5,), torch.int32),
        ((2, 3), torch.float64),
        ((3, 4, 5), torch.float16),
        ((2, 3, 4, 5), torch.float32),  # 4D for channels_last
        ((2, 3, 4, 5, 6), torch.float32),  # 5D for channels_last_3d
        ((0,), torch.float32),  # Empty tensor
        ((100, 100), torch.float32),  # Large tensor
    ],
)
def test_zeros_like_symmetric_heap_shapes_dtypes(shape, dtype):
    """Test that zeros_like returns tensors on symmetric heap for various shapes and dtypes."""
    shmem = iris.iris(1 << 20)

    # Create input tensor
    input_tensor = shmem.full(shape, 5, dtype=dtype)

    # Test all compatible memory formats
    memory_formats = [
        torch.contiguous_format,
        torch.preserve_format,
    ]

    # Add dimension-specific formats
    if len(shape) == 4:
        memory_formats.append(torch.channels_last)
    elif len(shape) == 5:
        memory_formats.append(torch.channels_last_3d)

    for memory_format in memory_formats:
        # Test zeros_like with this memory format
        result = shmem.zeros_like(input_tensor, memory_format=memory_format)

        # Verify tensor is on symmetric heap
        assert shmem.is_symmetric(result), (
            f"Tensor with shape {shape}, dtype {dtype}, memory_format {memory_format} is NOT on symmetric heap!"
        )

        # Also verify basic functionality
        # Memory formats preserve the logical shape, only changing the memory layout (strides)
        assert result.shape == shape
        assert result.dtype == dtype
        assert torch.all(result == 0)


@pytest.mark.parametrize("dtype", [torch.float16, torch.float32, torch.float64, torch.int32, torch.int64])
def test_zeros_like_symmetric_heap_dtype_override(dtype):
    """Test that zeros_like with dtype override returns tensors on symmetric heap."""
    shmem = iris.iris(1 << 20)
    input_tensor = shmem.full((3, 3), 1, dtype=torch.float32)

    result = shmem.zeros_like(input_tensor, dtype=dtype)
    assert shmem.is_symmetric(result), f"Tensor with dtype {dtype} is NOT on symmetric heap!"
    assert result.dtype == dtype


def test_zeros_like_symmetric_heap_other_params():
    """Test that zeros_like with other parameters returns tensors on symmetric heap."""
    shmem = iris.iris(1 << 20)
    input_tensor = shmem.full((3, 3), 1, dtype=torch.float32)

    # Test with requires_grad
    result = shmem.zeros_like(input_tensor, requires_grad=True)
    assert shmem.is_symmetric(result), "Tensor with requires_grad=True is NOT on symmetric heap!"

    # Test with device override
    result = shmem.zeros_like(input_tensor, device=shmem.device)
    assert shmem.is_symmetric(result), "Tensor with device override is NOT on symmetric heap!"

    # Test with layout override (only strided is supported)
    result = shmem.zeros_like(input_tensor, layout=torch.strided)
    assert shmem.is_symmetric(result), "Tensor with layout override is NOT on symmetric heap!"
