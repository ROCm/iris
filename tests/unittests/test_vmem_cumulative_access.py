# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.

"""
Test cumulative mem_set_access pattern for VMem allocations.

ROCm bug workaround: hipMemSetAccess must be called cumulatively from base_va
with total allocated size, not on individual sub-regions.

See: https://github.com/ROCm/rocm-systems/issues/2667
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


def test_vmem_cumulative_access_pattern():
    """
    Test cumulative mem_set_access with 3 allocations of different sizes.

    Workflow:
    1. Map alloc1 (2MB) → mem_set_access(base, 2MB)
    2. Map alloc2 (2MB) → mem_set_access(base, 4MB) [CUMULATIVE!]
    3. Map alloc3 (4KB) → mem_set_access(base, 4MB+4KB) [CUMULATIVE!]
    4. Verify all 3 work
    5. Unmap alloc3, verify alloc1+2 still work
    6. Unmap alloc2, verify alloc1 still works
    7. Unmap alloc1
    """
    device_id = _get_device_id()
    granularity = get_allocation_granularity(device_id)

    heap_size = 64 << 20  # 64 MB total VA

    # Allocation sizes (different sizes to test mixed pattern)
    size1 = 2 << 20  # 2 MB
    size2 = 2 << 20  # 2 MB
    size3_bytes = 64  # 64 bytes
    size3 = (size3_bytes + granularity - 1) & ~(granularity - 1)  # Align to 4KB

    base_va = mem_address_reserve(heap_size, granularity, 0)

    access_desc = hipMemAccessDesc()
    access_desc.location.type = hipMemLocationTypeDevice
    access_desc.location.id = device_id
    access_desc.flags = hipMemAccessFlagsProtReadWrite

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

    try:
        # Step 1: Allocate and map first (2MB)
        va1 = base_va
        handle1 = mem_create(size1, device_id)
        mem_map(va1, size1, 0, handle1)

        # CUMULATIVE: set access from base_va with total size so far
        cumulative_size = size1
        mem_set_access(base_va, cumulative_size, access_desc)

        array1 = CUDAArrayInterface(va1, 1024 * 4)
        tensor1 = torch.as_tensor(array1, device="cuda")
        tensor1.fill_(111.0)
        torch.cuda.synchronize()
        assert torch.all(tensor1 == 111.0)
        print("✓ Alloc1 (2MB) at offset 0 works")

        # Step 2: Allocate and map second (2MB)
        va2 = base_va + size1
        handle2 = mem_create(size2, device_id)
        mem_map(va2, size2, 0, handle2)

        # CUMULATIVE: set access from base_va with NEW cumulative size
        cumulative_size = size1 + size2
        mem_set_access(base_va, cumulative_size, access_desc)

        array2 = CUDAArrayInterface(va2, 1024 * 4)
        tensor2 = torch.as_tensor(array2, device="cuda")
        tensor2.fill_(222.0)
        torch.cuda.synchronize()
        assert torch.all(tensor2 == 222.0)
        assert torch.all(tensor1 == 111.0), "Alloc1 corrupted after cumulative access!"
        print("✓ Alloc2 (2MB) at offset 2MB works, alloc1 still valid")

        # Step 3: Allocate and map third (4KB)
        va3 = base_va + size1 + size2
        handle3 = mem_create(size3, device_id)
        mem_map(va3, size3, 0, handle3)

        # CUMULATIVE: set access from base_va with NEW cumulative size
        cumulative_size = size1 + size2 + size3
        mem_set_access(base_va, cumulative_size, access_desc)

        array3 = CUDAArrayInterface(va3, 64)
        tensor3 = torch.as_tensor(array3, device="cuda")
        tensor3.fill_(333.0)
        torch.cuda.synchronize()
        assert torch.all(tensor3 == 333.0)
        assert torch.all(tensor2 == 222.0), "Alloc2 corrupted after cumulative access!"
        assert torch.all(tensor1 == 111.0), "Alloc1 corrupted after cumulative access!"
        print("✓ Alloc3 (4KB) at offset 4MB works, alloc1+2 still valid")

        # Step 4: Unmap last allocation, verify first 2 still work
        del tensor3
        mem_unmap(va3, size3)
        mem_release(handle3)

        torch.cuda.synchronize()
        assert torch.all(tensor1 == 111.0), "Alloc1 corrupted after unmap3!"
        assert torch.all(tensor2 == 222.0), "Alloc2 corrupted after unmap3!"
        print("✓ After unmapping alloc3, alloc1+2 still valid")

        # Step 5: Unmap second allocation, verify first still works
        del tensor2
        mem_unmap(va2, size2)
        mem_release(handle2)

        torch.cuda.synchronize()
        assert torch.all(tensor1 == 111.0), "Alloc1 corrupted after unmap2!"
        print("✓ After unmapping alloc2, alloc1 still valid")

        # Step 6: Unmap first allocation
        del tensor1
        mem_unmap(va1, size1)
        mem_release(handle1)
        print("✓ Cleanup complete")

    finally:
        mem_address_free(base_va, heap_size)
        torch.cuda.synchronize()
        if dist.is_initialized():
            dist.barrier()


def test_vmem_cumulative_access_with_import():
    """
    Test cumulative mem_set_access with native and imported allocations.

    Workflow:
    1. Map minimal (2MB) → mem_set_access(base, 2MB)
    2. Map import (2MB) → mem_set_access(base, 4MB) [CUMULATIVE!]
    3. Map native (4KB) → mem_set_access(base, 4MB+4KB) [CUMULATIVE!]
    4. Verify all 3 work
    """
    device_id = _get_device_id()
    granularity = get_allocation_granularity(device_id)

    heap_size = 64 << 20  # 64 MB total VA
    minimal_size = 2 << 20  # 2 MB

    base_va = mem_address_reserve(heap_size, granularity, 0)

    access_desc = hipMemAccessDesc()
    access_desc.location.type = hipMemLocationTypeDevice
    access_desc.location.id = device_id
    access_desc.flags = hipMemAccessFlagsProtReadWrite

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

    try:
        # Step 1: Minimal allocation (2MB)
        va_minimal = base_va
        handle_minimal = mem_create(minimal_size, device_id)
        mem_map(va_minimal, minimal_size, 0, handle_minimal)

        cumulative_size = minimal_size
        mem_set_access(base_va, cumulative_size, access_desc)

        array_minimal = CUDAArrayInterface(va_minimal, 1024 * 4)
        tensor_minimal = torch.as_tensor(array_minimal, device="cuda")
        tensor_minimal.fill_(111.0)
        torch.cuda.synchronize()
        assert torch.all(tensor_minimal == 111.0)
        print("✓ Minimal (2MB) at offset 0 works")

        # Step 2: Import external tensor (2MB)
        external_tensor = torch.randn(16, dtype=torch.float32, device="cuda")
        external_tensor.fill_(999.0)

        from iris.hip import get_address_range, export_dmabuf_handle, mem_import_from_shareable_handle

        external_ptr = external_tensor.data_ptr()
        alloc_base, alloc_size = get_address_range(external_ptr)
        dmabuf_fd, export_base, export_size = export_dmabuf_handle(alloc_base, alloc_size)

        imported_handle = mem_import_from_shareable_handle(dmabuf_fd)
        import os

        os.close(dmabuf_fd)

        va_import = base_va + cumulative_size
        aligned_import_size = (export_size + granularity - 1) & ~(granularity - 1)
        mem_map(va_import, aligned_import_size, 0, imported_handle)

        # CUMULATIVE: set access from base_va with NEW total size
        cumulative_size += aligned_import_size
        mem_set_access(base_va, cumulative_size, access_desc)

        offset_in_alloc = external_ptr - alloc_base
        imported_ptr = va_import + offset_in_alloc
        array_import = CUDAArrayInterface(imported_ptr, 16 * 4)
        tensor_import = torch.as_tensor(array_import, device="cuda")
        torch.cuda.synchronize()
        assert torch.all(tensor_import == 999.0)
        assert torch.all(tensor_minimal == 111.0), "Minimal corrupted after import access!"
        print("✓ Import (2MB) at offset 2MB works, minimal still valid")

        # Step 3: Native allocation (4KB)
        va_native = base_va + cumulative_size
        native_size_bytes = 64
        native_size = (native_size_bytes + granularity - 1) & ~(granularity - 1)
        handle_native = mem_create(native_size, device_id)
        mem_map(va_native, native_size, 0, handle_native)

        # CUMULATIVE: set access from base_va with NEW total size
        cumulative_size += native_size
        mem_set_access(base_va, cumulative_size, access_desc)

        array_native = CUDAArrayInterface(va_native, native_size_bytes)
        tensor_native = torch.as_tensor(array_native, device="cuda")
        tensor_native.fill_(777.0)
        torch.cuda.synchronize()
        assert torch.all(tensor_native == 777.0)
        assert torch.all(tensor_import == 999.0), "Import corrupted after native access!"
        assert torch.all(tensor_minimal == 111.0), "Minimal corrupted after native access!"
        print("✓ Native (4KB) at offset 4MB works, minimal+import still valid")

        # Step 4: Unmap last (native), verify first 2 remain
        del tensor_native
        mem_unmap(va_native, native_size)
        mem_release(handle_native)

        torch.cuda.synchronize()
        assert torch.all(tensor_minimal == 111.0), "Minimal corrupted after native unmap!"
        assert torch.all(tensor_import == 999.0), "Import corrupted after native unmap!"
        print("✓ After unmapping native, minimal+import still valid")

        # Step 5: Unmap import, verify minimal remains
        del tensor_import
        mem_unmap(va_import, aligned_import_size)
        mem_release(imported_handle)

        torch.cuda.synchronize()
        assert torch.all(tensor_minimal == 111.0), "Minimal corrupted after import unmap!"
        print("✓ After unmapping import, minimal still valid")

        # Step 6: Unmap minimal
        del tensor_minimal
        mem_unmap(va_minimal, minimal_size)
        mem_release(handle_minimal)
        print("✓ All cleanup complete")

    finally:
        mem_address_free(base_va, heap_size)
        torch.cuda.synchronize()
        if dist.is_initialized():
            dist.barrier()
