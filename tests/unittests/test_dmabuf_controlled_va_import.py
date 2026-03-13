# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.

"""
Test DMA-BUF import with controlled VA (export → import → map → set_access → access).
"""

import torch
import torch.distributed as dist
import os
from iris.hip import (
    get_allocation_granularity,
    get_address_range,
    export_dmabuf_handle,
    mem_address_reserve,
    mem_address_free,
    mem_import_from_shareable_handle,
    mem_map,
    mem_unmap,
    mem_release,
    mem_set_access,
    hipMemAccessDesc,
    hipMemLocationTypeDevice,
    hipMemAccessFlagsProtReadWrite,
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


def test_import_to_controlled_va():
    """Basic test: Import PyTorch tensor DMA-BUF to controlled VA."""
    device_id = _get_device_id()
    granularity = get_allocation_granularity(device_id)
    va_size = 4 << 20

    base_va = mem_address_reserve(va_size, granularity, 0)

    try:
        tensor = torch.randn(1024, dtype=torch.float32, device="cuda")
        tensor.fill_(42.0)
        original_ptr = tensor.data_ptr()

        alloc_base, alloc_size = get_address_range(original_ptr)
        dmabuf_fd, export_base, export_size = export_dmabuf_handle(alloc_base, alloc_size)

        imported_handle = mem_import_from_shareable_handle(dmabuf_fd)
        target_va = base_va
        mem_map(target_va, export_size, 0, imported_handle)

        access_desc = hipMemAccessDesc()
        access_desc.location.type = hipMemLocationTypeDevice
        access_desc.location.id = device_id
        access_desc.flags = hipMemAccessFlagsProtReadWrite
        mem_set_access(target_va, export_size, access_desc)

        offset_in_alloc = original_ptr - alloc_base
        tensor_va = target_va + offset_in_alloc
        tensor_bytes = tensor.numel() * tensor.element_size()

        class CUDAArrayInterface:
            def __init__(self, ptr, size_bytes):
                self.ptr = ptr
                self.size_bytes = size_bytes

            @property
            def __cuda_array_interface__(self):
                return {
                    "shape": (self.size_bytes // 4,),
                    "typestr": "<f4",
                    "data": (self.ptr, False),
                    "version": 3,
                }

        cuda_array = CUDAArrayInterface(tensor_va, tensor_bytes)
        imported_tensor = torch.as_tensor(cuda_array, device="cuda")

        torch.cuda.synchronize()
        assert torch.all(imported_tensor == 42.0)
        assert target_va == base_va

        del imported_tensor, tensor
        os.close(dmabuf_fd)
        mem_unmap(target_va, export_size)
        mem_release(imported_handle)

    finally:
        mem_address_free(base_va, va_size)
        torch.cuda.synchronize()
        if dist.is_initialized():
            dist.barrier()


def test_import_with_offset():
    """Test import with offset preservation (sliced tensor)."""
    device_id = _get_device_id()
    granularity = get_allocation_granularity(device_id)
    va_size = 4 << 20

    base_va = mem_address_reserve(va_size, granularity, 0)

    try:
        large_tensor = torch.randn(10000, dtype=torch.float32, device="cuda")
        sliced_tensor = large_tensor[1000:2000]
        sliced_tensor.fill_(99.0)

        slice_ptr = sliced_tensor.data_ptr()
        slice_size = sliced_tensor.element_size() * sliced_tensor.numel()

        alloc_base, alloc_size = get_address_range(slice_ptr)
        offset_in_alloc = slice_ptr - alloc_base

        assert offset_in_alloc > 0

        dmabuf_fd, export_base, export_size = export_dmabuf_handle(alloc_base, alloc_size)
        assert export_base == alloc_base

        imported_handle = mem_import_from_shareable_handle(dmabuf_fd)
        target_va = base_va
        mem_map(target_va, export_size, 0, imported_handle)

        access_desc = hipMemAccessDesc()
        access_desc.location.type = hipMemLocationTypeDevice
        access_desc.location.id = device_id
        access_desc.flags = hipMemAccessFlagsProtReadWrite
        mem_set_access(target_va, export_size, access_desc)

        tensor_va = target_va + offset_in_alloc

        class CUDAArrayInterface:
            def __init__(self, ptr, size_bytes):
                self.ptr = ptr
                self.size_bytes = size_bytes

            @property
            def __cuda_array_interface__(self):
                return {
                    "shape": (self.size_bytes // 4,),
                    "typestr": "<f4",
                    "data": (self.ptr, False),
                    "version": 3,
                }

        cuda_array = CUDAArrayInterface(tensor_va, slice_size)
        imported_tensor = torch.as_tensor(cuda_array, device="cuda")

        torch.cuda.synchronize()
        assert torch.all(imported_tensor == 99.0)

        del imported_tensor, sliced_tensor, large_tensor
        os.close(dmabuf_fd)
        mem_unmap(target_va, export_size)
        mem_release(imported_handle)

    finally:
        mem_address_free(base_va, va_size)
        torch.cuda.synchronize()
        if dist.is_initialized():
            dist.barrier()


def test_import_memory_sharing():
    """Test that imported memory shares physical memory with original tensor."""
    device_id = _get_device_id()
    granularity = get_allocation_granularity(device_id)
    va_size = 4 << 20

    base_va = mem_address_reserve(va_size, granularity, 0)

    try:
        original_tensor = torch.randn(1024, dtype=torch.float32, device="cuda")
        original_tensor.fill_(42.0)
        original_ptr = original_tensor.data_ptr()

        alloc_base, alloc_size = get_address_range(original_ptr)
        dmabuf_fd, export_base, export_size = export_dmabuf_handle(alloc_base, alloc_size)

        imported_handle = mem_import_from_shareable_handle(dmabuf_fd)
        target_va = base_va
        mem_map(target_va, export_size, 0, imported_handle)

        access_desc = hipMemAccessDesc()
        access_desc.location.type = hipMemLocationTypeDevice
        access_desc.location.id = device_id
        access_desc.flags = hipMemAccessFlagsProtReadWrite
        mem_set_access(target_va, export_size, access_desc)

        offset_in_alloc = original_ptr - alloc_base
        tensor_va = target_va + offset_in_alloc
        tensor_bytes = original_tensor.numel() * original_tensor.element_size()

        class CUDAArrayInterface:
            def __init__(self, ptr, size_bytes):
                self.ptr = ptr
                self.size_bytes = size_bytes

            @property
            def __cuda_array_interface__(self):
                return {
                    "shape": (self.size_bytes // 4,),
                    "typestr": "<f4",
                    "data": (self.ptr, False),
                    "version": 3,
                }

        cuda_array = CUDAArrayInterface(tensor_va, tensor_bytes)
        imported_tensor = torch.as_tensor(cuda_array, device="cuda")

        torch.cuda.synchronize()
        assert torch.all(imported_tensor == 42.0)
        assert torch.all(original_tensor == 42.0)

        imported_tensor.fill_(99.0)
        torch.cuda.synchronize()
        assert torch.all(original_tensor == 99.0)
        assert torch.all(imported_tensor == 99.0)

        original_tensor.fill_(123.0)
        torch.cuda.synchronize()
        assert torch.all(imported_tensor == 123.0)
        assert torch.all(original_tensor == 123.0)

        del imported_tensor, original_tensor
        os.close(dmabuf_fd)
        mem_unmap(target_va, export_size)
        mem_release(imported_handle)

    finally:
        mem_address_free(base_va, va_size)
        torch.cuda.synchronize()
        if dist.is_initialized():
            dist.barrier()


def test_import_multiple_tensors():
    """Test importing multiple PyTorch tensors into same VA range."""
    device_id = _get_device_id()
    granularity = get_allocation_granularity(device_id)
    va_size = 8 << 20

    base_va = mem_address_reserve(va_size, granularity, 0)

    try:
        tensor1 = torch.randn(1024, dtype=torch.float32, device="cuda")
        tensor1.fill_(11.0)

        tensor2 = torch.randn(1024, dtype=torch.float32, device="cuda")
        tensor2.fill_(22.0)

        alloc_base1, alloc_size1 = get_address_range(tensor1.data_ptr())
        fd1, export_base1, export_size1 = export_dmabuf_handle(alloc_base1, alloc_size1)

        alloc_base2, alloc_size2 = get_address_range(tensor2.data_ptr())
        fd2, export_base2, export_size2 = export_dmabuf_handle(alloc_base2, alloc_size2)

        handle1 = mem_import_from_shareable_handle(fd1)
        va1 = base_va
        mem_map(va1, export_size1, 0, handle1)

        handle2 = mem_import_from_shareable_handle(fd2)
        va2 = base_va + (4 << 20)
        mem_map(va2, export_size2, 0, handle2)

        access_desc = hipMemAccessDesc()
        access_desc.location.type = hipMemLocationTypeDevice
        access_desc.location.id = device_id
        access_desc.flags = hipMemAccessFlagsProtReadWrite
        mem_set_access(va1, export_size1, access_desc)
        mem_set_access(va2, export_size2, access_desc)

        class CUDAArrayInterface:
            def __init__(self, ptr, size_bytes):
                self.ptr = ptr
                self.size_bytes = size_bytes

            @property
            def __cuda_array_interface__(self):
                return {
                    "shape": (self.size_bytes // 4,),
                    "typestr": "<f4",
                    "data": (self.ptr, False),
                    "version": 3,
                }

        offset1 = tensor1.data_ptr() - alloc_base1
        tensor_va1 = va1 + offset1
        tensor_bytes1 = tensor1.numel() * tensor1.element_size()
        cuda_array1 = CUDAArrayInterface(tensor_va1, tensor_bytes1)
        imported1 = torch.as_tensor(cuda_array1, device="cuda")

        offset2 = tensor2.data_ptr() - alloc_base2
        tensor_va2 = va2 + offset2
        tensor_bytes2 = tensor2.numel() * tensor2.element_size()
        cuda_array2 = CUDAArrayInterface(tensor_va2, tensor_bytes2)
        imported2 = torch.as_tensor(cuda_array2, device="cuda")

        torch.cuda.synchronize()
        assert torch.all(imported1 == 11.0)
        assert torch.all(imported2 == 22.0)

        del imported1, imported2, tensor1, tensor2
        os.close(fd1)
        os.close(fd2)
        mem_unmap(va1, export_size1)
        mem_unmap(va2, export_size2)
        mem_release(handle1)
        mem_release(handle2)

    finally:
        mem_address_free(base_va, va_size)
        torch.cuda.synchronize()
        if dist.is_initialized():
            dist.barrier()


def test_cleanup_preserves_original():
    """Verify cleanup doesn't corrupt original PyTorch tensor."""
    device_id = _get_device_id()
    granularity = get_allocation_granularity(device_id)
    va_size = 4 << 20

    original_tensor = torch.randn(1024, dtype=torch.float32, device="cuda")
    original_tensor.fill_(42.0)

    assert torch.all(original_tensor == 42.0)

    base_va = mem_address_reserve(va_size, granularity, 0)

    try:
        original_ptr = original_tensor.data_ptr()
        alloc_base, alloc_size = get_address_range(original_ptr)
        dmabuf_fd, export_base, export_size = export_dmabuf_handle(alloc_base, alloc_size)

        imported_handle = mem_import_from_shareable_handle(dmabuf_fd)
        target_va = base_va
        mem_map(target_va, export_size, 0, imported_handle)

        access_desc = hipMemAccessDesc()
        access_desc.location.type = hipMemLocationTypeDevice
        access_desc.location.id = device_id
        access_desc.flags = hipMemAccessFlagsProtReadWrite
        mem_set_access(target_va, export_size, access_desc)

        offset = original_ptr - alloc_base
        tensor_va = target_va + offset
        tensor_bytes = original_tensor.numel() * original_tensor.element_size()

        class CUDAArrayInterface:
            def __init__(self, ptr, size_bytes):
                self.ptr = ptr
                self.size_bytes = size_bytes

            @property
            def __cuda_array_interface__(self):
                return {
                    "shape": (self.size_bytes // 4,),
                    "typestr": "<f4",
                    "data": (self.ptr, False),
                    "version": 3,
                }

        cuda_array = CUDAArrayInterface(tensor_va, tensor_bytes)
        imported_tensor = torch.as_tensor(cuda_array, device="cuda")

        imported_tensor.fill_(99.0)
        assert torch.all(original_tensor == 99.0)

        del imported_tensor, cuda_array
        import gc

        gc.collect()
        torch.cuda.synchronize()

        os.close(dmabuf_fd)
        mem_unmap(target_va, export_size)
        mem_release(imported_handle)
        torch.cuda.synchronize()

        assert torch.all(original_tensor == 99.0)
        original_tensor.fill_(123.0)
        assert torch.all(original_tensor == 123.0)
        result = original_tensor + 1.0
        assert torch.all(result == 124.0)

    finally:
        mem_address_free(base_va, va_size)
        torch.cuda.synchronize()
        if dist.is_initialized():
            dist.barrier()
