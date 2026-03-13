# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.

"""
Test HIP VMem primitive operations.

These tests verify the foundational VMem APIs:
- mem_address_reserve / mem_address_free
- mem_create / mem_release
- mem_map / mem_unmap
- mem_set_access
"""

import torch
import torch.distributed as dist
from iris.hip import (
    get_allocation_granularity,
    mem_address_reserve,
    mem_address_free,
    mem_create,
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


def test_vmem_reserve_and_free():
    """Test basic VA reservation and cleanup."""
    device_id = _get_device_id()
    granularity = get_allocation_granularity(device_id)
    va_size = 4 << 20  # 4 MB

    base_va = mem_address_reserve(va_size, granularity, 0)
    assert base_va > 0
    mem_address_free(base_va, va_size)

    torch.cuda.synchronize()
    if dist.is_initialized():
        dist.barrier()


def test_vmem_create_map_access():
    """Test complete VMem workflow: reserve → create → map → access."""
    device_id = _get_device_id()
    granularity = get_allocation_granularity(device_id)
    alloc_size = 2 << 20  # 2 MB
    va_size = 4 << 20  # 4 MB

    base_va = mem_address_reserve(va_size, granularity, 0)

    try:
        handle = mem_create(alloc_size, device_id)
        mem_map(base_va, alloc_size, 0, handle)

        access_desc = hipMemAccessDesc()
        access_desc.location.type = hipMemLocationTypeDevice
        access_desc.location.id = device_id
        access_desc.flags = hipMemAccessFlagsProtReadWrite
        mem_set_access(base_va, alloc_size, access_desc)

        # Verify we can use the memory
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

        num_floats = 1024
        cuda_array = CUDAArrayInterface(base_va, num_floats * 4)
        tensor = torch.as_tensor(cuda_array, device="cuda")

        tensor.fill_(42.0)
        torch.cuda.synchronize()
        assert torch.all(tensor == 42.0)

        del tensor
        mem_unmap(base_va, alloc_size)
        mem_release(handle)

    finally:
        mem_address_free(base_va, va_size)
        torch.cuda.synchronize()
        if dist.is_initialized():
            dist.barrier()


def test_vmem_multiple_mappings():
    """Test multiple allocations: map, set_access (cumulative), map, set_access (cumulative).

    Calls mem_set_access twice, each time with cumulative size from base_va so far.
    """
    device_id = _get_device_id()
    granularity = get_allocation_granularity(device_id)
    alloc_size = 2 << 20  # 2 MB each
    va_size = 4 << 20  # 4 MB total

    base_va = mem_address_reserve(va_size, granularity, 0)

    try:
        handle1 = mem_create(alloc_size, device_id)
        handle2 = mem_create(alloc_size, device_id)

        va1 = base_va
        mem_map(va1, alloc_size, 0, handle1)
        access_desc = hipMemAccessDesc()
        access_desc.location.type = hipMemLocationTypeDevice
        access_desc.location.id = device_id
        access_desc.flags = hipMemAccessFlagsProtReadWrite
        mem_set_access(base_va, alloc_size, access_desc)

        va2 = base_va + alloc_size
        mem_map(va2, alloc_size, 0, handle2)
        mem_set_access(base_va, alloc_size * 2, access_desc)

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

        num_floats = 1024
        cuda_array1 = CUDAArrayInterface(va1, num_floats * 4)
        tensor1 = torch.as_tensor(cuda_array1, device="cuda")

        cuda_array2 = CUDAArrayInterface(va2, num_floats * 4)
        tensor2 = torch.as_tensor(cuda_array2, device="cuda")

        tensor1.fill_(11.0)
        tensor2.fill_(22.0)
        torch.cuda.synchronize()

        assert torch.all(tensor1 == 11.0)
        assert torch.all(tensor2 == 22.0)

        del tensor1, tensor2
        mem_unmap(va1, alloc_size)
        mem_unmap(va2, alloc_size)
        mem_release(handle1)
        mem_release(handle2)

    finally:
        mem_address_free(base_va, va_size)
        torch.cuda.synchronize()
        if dist.is_initialized():
            dist.barrier()


def test_vmem_granularity():
    """Test allocation granularity is valid."""
    device_id = _get_device_id()
    granularity = get_allocation_granularity(device_id)

    assert granularity > 0
    assert granularity >= 4096
    assert (granularity & (granularity - 1)) == 0  # Power of 2

    if dist.is_initialized():
        dist.barrier()


def test_vmem_remap():
    """Test unmap and remap to same VA."""
    device_id = _get_device_id()
    granularity = get_allocation_granularity(device_id)
    alloc_size = 2 << 20
    va_size = 4 << 20

    base_va = mem_address_reserve(va_size, granularity, 0)

    try:
        handle1 = mem_create(alloc_size, device_id)
        handle2 = mem_create(alloc_size, device_id)

        access_desc = hipMemAccessDesc()
        access_desc.location.type = hipMemLocationTypeDevice
        access_desc.location.id = device_id
        access_desc.flags = hipMemAccessFlagsProtReadWrite

        # Map first handle
        mem_map(base_va, alloc_size, 0, handle1)
        mem_set_access(base_va, alloc_size, access_desc)

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

        num_floats = 1024
        cuda_array = CUDAArrayInterface(base_va, num_floats * 4)
        tensor = torch.as_tensor(cuda_array, device="cuda")
        tensor.fill_(11.0)
        torch.cuda.synchronize()
        assert torch.all(tensor == 11.0)
        del tensor

        # Unmap and remap with different handle
        mem_unmap(base_va, alloc_size)
        mem_map(base_va, alloc_size, 0, handle2)
        mem_set_access(base_va, alloc_size, access_desc)

        cuda_array2 = CUDAArrayInterface(base_va, num_floats * 4)
        tensor2 = torch.as_tensor(cuda_array2, device="cuda")
        tensor2.fill_(22.0)
        torch.cuda.synchronize()
        assert torch.all(tensor2 == 22.0)

        del tensor2
        mem_unmap(base_va, alloc_size)
        mem_release(handle1)
        mem_release(handle2)

    finally:
        mem_address_free(base_va, va_size)
        torch.cuda.synchronize()
        if dist.is_initialized():
            dist.barrier()
