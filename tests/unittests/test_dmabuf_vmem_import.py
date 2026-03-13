# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.

"""
Test importing DMA-BUF handles into controlled VMem address ranges.

Verifies:
1. Import PyTorch tensors (exported as DMA-BUF) into reserved VA space
2. Mix imported and native VMem allocations in the same VA range
"""

import torch
import torch.distributed as dist
from iris.hip import (
    get_allocation_granularity,
    get_address_range,
    export_dmabuf_handle,
    mem_address_reserve,
    mem_address_free,
    mem_create,
    mem_map,
    mem_unmap,
    mem_release,
    mem_set_access,
    mem_import_from_shareable_handle,
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


def test_dmabuf_import_into_reserved_va():
    """Test importing DMA-BUF into reserved VMem VA range."""
    device_id = _get_device_id()
    granularity = get_allocation_granularity(device_id)
    va_size = 4 << 20

    base_va = mem_address_reserve(va_size, granularity, 0)

    try:
        tensor_size = 1024
        external_tensor = torch.randn(tensor_size, dtype=torch.float32, device="cuda")
        external_tensor.fill_(42.0)

        external_ptr = external_tensor.data_ptr()
        alloc_base, alloc_size = get_address_range(external_ptr)
        dmabuf_fd, export_base, export_size = export_dmabuf_handle(alloc_base, alloc_size)

        imported_handle = mem_import_from_shareable_handle(dmabuf_fd)
        target_va = base_va
        mem_map(target_va, export_size, 0, imported_handle)

        access_desc = hipMemAccessDesc()
        access_desc.location.type = hipMemLocationTypeDevice
        access_desc.location.id = device_id
        access_desc.flags = hipMemAccessFlagsProtReadWrite
        mem_set_access(target_va, export_size, access_desc)

        class CUDAArrayInterface:
            def __init__(self, ptr, size_bytes):
                self.ptr = ptr
                self.size_bytes = size_bytes

            @property
            def __cuda_array_interface__(self):
                return {
                    "shape": (self.size_bytes,),
                    "typestr": "|u1",
                    "data": (self.ptr, False),
                    "version": 3,
                }

        offset_in_alloc = external_ptr - alloc_base
        tensor_ptr = target_va + offset_in_alloc
        tensor_bytes = tensor_size * 4

        cuda_array = CUDAArrayInterface(tensor_ptr, tensor_bytes)
        imported_tensor = torch.as_tensor(cuda_array, device="cuda").view(torch.float32)

        assert torch.all(imported_tensor == 42.0)
        assert target_va == base_va

        del imported_tensor, external_tensor
        mem_unmap(target_va, export_size)
        mem_release(imported_handle)

    finally:
        mem_address_free(base_va, va_size)
        torch.cuda.synchronize()
        if dist.is_initialized():
            dist.barrier()


def test_dmabuf_import_after_native_allocation():
    """Test importing DMA-BUF into VA range with existing native allocations."""
    device_id = _get_device_id()
    granularity = get_allocation_granularity(device_id)
    va_size = 4 << 20
    native_size = 2 << 20

    base_va = mem_address_reserve(va_size, granularity, 0)

    try:
        native_handle = mem_create(native_size, device_id)
        mem_map(base_va, native_size, 0, native_handle)

        access_desc = hipMemAccessDesc()
        access_desc.location.type = hipMemLocationTypeDevice
        access_desc.location.id = device_id
        access_desc.flags = hipMemAccessFlagsProtReadWrite
        mem_set_access(base_va, native_size, access_desc)

        class CUDAArrayInterface:
            def __init__(self, ptr, size_bytes):
                self.ptr = ptr
                self.size_bytes = size_bytes

            @property
            def __cuda_array_interface__(self):
                return {
                    "shape": (self.size_bytes,),
                    "typestr": "|u1",
                    "data": (self.ptr, False),
                    "version": 3,
                }

        native_elements = 1024
        native_bytes = native_elements * 4
        cuda_array_native = CUDAArrayInterface(base_va, native_bytes)
        native_tensor = torch.as_tensor(cuda_array_native, device="cuda").view(torch.float32)
        native_tensor.fill_(123.0)
        assert torch.all(native_tensor == 123.0)

        external_tensor = torch.randn(1024, dtype=torch.float32, device="cuda")
        external_tensor.fill_(456.0)

        external_ptr = external_tensor.data_ptr()
        alloc_base, alloc_size = get_address_range(external_ptr)
        dmabuf_fd, export_base, export_size = export_dmabuf_handle(alloc_base, alloc_size)

        desired_va = base_va + native_size
        imported_handle = mem_import_from_shareable_handle(dmabuf_fd)
        mem_map(desired_va, export_size, 0, imported_handle)

        access_desc_import = hipMemAccessDesc()
        access_desc_import.location.type = hipMemLocationTypeDevice
        access_desc_import.location.id = device_id
        access_desc_import.flags = hipMemAccessFlagsProtReadWrite
        mem_set_access(desired_va, export_size, access_desc_import)

        offset_in_alloc = external_ptr - alloc_base
        imported_tensor_ptr = desired_va + offset_in_alloc
        imported_bytes = 1024 * 4
        cuda_array_imported = CUDAArrayInterface(imported_tensor_ptr, imported_bytes)
        imported_tensor = torch.as_tensor(cuda_array_imported, device="cuda").view(torch.float32)

        assert torch.all(native_tensor == 123.0)
        assert torch.all(imported_tensor == 456.0)
        assert desired_va == base_va + native_size

        del native_tensor, imported_tensor, external_tensor
        mem_unmap(desired_va, export_size)
        mem_release(imported_handle)
        mem_unmap(base_va, native_size)
        mem_release(native_handle)

    finally:
        mem_address_free(base_va, va_size)
        torch.cuda.synchronize()
        if dist.is_initialized():
            dist.barrier()
