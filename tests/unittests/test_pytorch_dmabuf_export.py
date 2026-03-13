# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.

"""
Test PyTorch tensor → DMA-BUF export.

Verifies:
- get_address_range() correctly finds base allocation
- export_dmabuf_handle() returns base, not offset pointer
- Works with sliced tensors and different dtypes
"""

import torch
import torch.distributed as dist
import os
import pytest
from iris.hip import (
    get_address_range,
    export_dmabuf_handle,
)


def _get_device_id():
    """Get device ID for current rank."""
    if dist.is_initialized():
        rank = dist.get_rank()
        num_gpus = torch.cuda.device_count()
        device_id = rank % num_gpus
        torch.cuda.set_device(device_id)
    else:
        device_id = torch.cuda.current_device()
    return device_id


def test_get_address_range_basic():
    """Test get_address_range with simple tensor."""
    device_id = _get_device_id()

    torch.cuda.empty_cache()
    tensor = torch.randn(1000, dtype=torch.float32, device="cuda")
    ptr = tensor.data_ptr()

    base_ptr, size = get_address_range(ptr)

    assert base_ptr > 0
    assert size > 0
    assert ptr >= base_ptr

    tensor_size = tensor.element_size() * tensor.numel()
    assert size >= tensor_size

    torch.cuda.synchronize()
    if dist.is_initialized():
        dist.barrier()


def test_get_address_range_with_offset():
    """Test get_address_range with sliced tensor (offset pointer)."""
    device_id = _get_device_id()

    large_tensor = torch.randn(10000, dtype=torch.float32, device="cuda")
    large_ptr = large_tensor.data_ptr()
    large_base, large_size = get_address_range(large_ptr)

    sliced_tensor = large_tensor[1000:2000]
    slice_ptr = sliced_tensor.data_ptr()
    slice_base, slice_size = get_address_range(slice_ptr)

    # Slice should report same base as full tensor
    assert slice_base == large_base
    assert slice_size == large_size

    # Verify offset
    offset = slice_ptr - slice_base
    expected_min_offset = 1000 * 4
    assert offset >= expected_min_offset

    torch.cuda.synchronize()
    if dist.is_initialized():
        dist.barrier()


def test_export_returns_base():
    """Test that export_dmabuf_handle returns BASE allocation, not offset pointer."""
    device_id = _get_device_id()

    large_tensor = torch.randn(10000, dtype=torch.float32, device="cuda")
    sliced_tensor = large_tensor[1000:2000]
    sliced_tensor.fill_(42.0)

    slice_ptr = sliced_tensor.data_ptr()
    slice_size = sliced_tensor.element_size() * sliced_tensor.numel()

    alloc_base, alloc_size = get_address_range(slice_ptr)
    offset = slice_ptr - alloc_base

    assert offset > 0

    fd, export_base, export_size = export_dmabuf_handle(alloc_base, alloc_size)

    try:
        # Export returns BASE, not slice pointer
        assert export_base == alloc_base
        assert export_size == alloc_size
        assert export_base < slice_ptr

    finally:
        os.close(fd)
        torch.cuda.synchronize()
        if dist.is_initialized():
            dist.barrier()


def test_export_large_offset():
    """Test DMA-BUF export with large offset."""
    device_id = _get_device_id()

    large_tensor = torch.randn(10000, 1000, dtype=torch.float32, device="cuda")
    slice_tensor = large_tensor[5000:6000, :]
    slice_tensor.fill_(99.0)

    slice_ptr = slice_tensor.data_ptr()
    slice_size = slice_tensor.element_size() * slice_tensor.numel()

    alloc_base, alloc_size = get_address_range(slice_ptr)
    offset = slice_ptr - alloc_base

    expected_min_offset = 5000 * 1000 * 4
    assert offset >= expected_min_offset

    fd, export_base, export_size = export_dmabuf_handle(alloc_base, alloc_size)

    try:
        assert export_base == alloc_base
        assert export_size == alloc_size

    finally:
        os.close(fd)
        torch.cuda.synchronize()
        if dist.is_initialized():
            dist.barrier()


def test_get_address_range_consistency():
    """Test multiple pointers in same allocation report same base."""
    device_id = _get_device_id()

    tensor = torch.randn(1000, 1000, dtype=torch.float32, device="cuda")

    test_cases = [
        (0, 100),
        (100, 200),
        (500, 600),
        (900, 1000),
    ]

    bases = []
    sizes = []

    for start, end in test_cases:
        slice_tensor = tensor[start:end, :]
        ptr = slice_tensor.data_ptr()
        base, size = get_address_range(ptr)
        bases.append(base)
        sizes.append(size)

    assert len(set(bases)) == 1
    assert len(set(sizes)) == 1

    torch.cuda.synchronize()
    if dist.is_initialized():
        dist.barrier()


@pytest.mark.parametrize("dtype", [torch.float32, torch.float16, torch.int32, torch.int64])
def test_export_different_dtypes(dtype):
    """Test DMA-BUF export with different data types."""
    device_id = _get_device_id()

    if dtype in [torch.float32, torch.float16]:
        tensor = torch.randn(1000, dtype=dtype, device="cuda")
    else:
        tensor = torch.randint(0, 100, (1000,), dtype=dtype, device="cuda")

    ptr = tensor.data_ptr()
    size = tensor.element_size() * tensor.numel()

    alloc_base, alloc_size = get_address_range(ptr)
    fd, export_base, export_size = export_dmabuf_handle(alloc_base, alloc_size)

    try:
        assert export_base == alloc_base
        assert export_size == alloc_size
    finally:
        os.close(fd)
        torch.cuda.synchronize()
        if dist.is_initialized():
            dist.barrier()
