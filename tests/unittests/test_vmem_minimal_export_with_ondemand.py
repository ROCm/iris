# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.

"""
Test VMem allocator strategy: minimal export + on-demand mapping.

Verifies the multi-rank allocator workflow:
1. Reserve large VA space
2. Map minimal initial allocation (for export requirement)
3. Export DMA-BUF from the mapped region
4. Map additional native allocations on-demand
5. Map imported external tensors on-demand

This strategy avoids pre-mapping the entire heap while satisfying
SymmetricHeap's requirement to export a physically-backed VA.
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


def test_vmem_small_native_after_import():
    """
    Test small native allocation (64 bytes) after import.

    This mimics the VMemAllocator pattern:
    1. Minimal 2MB mapping
    2. Import 2MB
    3. Small native 64 bytes (aligned to 4KB)
    """
    device_id = _get_device_id()
    granularity = get_allocation_granularity(device_id)

    heap_size = 64 << 20  # 64 MB total VA
    minimal_size = 2 << 20  # 2 MB minimal

    base_va = mem_address_reserve(heap_size, granularity, 0)

    access_desc = hipMemAccessDesc()
    access_desc.location.type = hipMemLocationTypeDevice
    access_desc.location.id = device_id
    access_desc.flags = hipMemAccessFlagsProtReadWrite

    try:
        # Step 1: Map minimal at offset 0
        minimal_handle = mem_create(minimal_size, device_id)
        mem_map(base_va, minimal_size, 0, minimal_handle)
        cumulative_size = minimal_size
        mem_set_access(base_va, cumulative_size, access_desc)
        print(f"✓ Minimal mapped at 0, size={minimal_size}")

        # Step 2: Import external tensor at offset 2MB
        external_tensor = torch.randn(16, dtype=torch.float32, device="cuda")
        external_tensor.fill_(999.0)

        external_ptr = external_tensor.data_ptr()
        alloc_base, alloc_size = get_address_range(external_ptr)
        dmabuf_fd, export_base, export_size = export_dmabuf_handle(alloc_base, alloc_size)

        imported_handle = mem_import_from_shareable_handle(dmabuf_fd)
        import os

        os.close(dmabuf_fd)

        import_offset = minimal_size
        import_va = base_va + import_offset
        aligned_import_size = (export_size + granularity - 1) & ~(granularity - 1)

        mem_map(import_va, aligned_import_size, 0, imported_handle)
        cumulative_size = import_offset + aligned_import_size
        mem_set_access(base_va, cumulative_size, access_desc)
        print(f"✓ Import mapped at {import_offset}, size={aligned_import_size}")

        # Step 3: Small native allocation (64 bytes) at offset 4MB
        native_offset = minimal_size + aligned_import_size
        native_va = base_va + native_offset
        native_size_bytes = 16 * 4  # 64 bytes
        aligned_native_size = (native_size_bytes + granularity - 1) & ~(granularity - 1)

        native_handle = mem_create(aligned_native_size, device_id)
        print(f"Attempting native alloc: offset={native_offset}, va={hex(native_va)}, size={aligned_native_size}")
        mem_map(native_va, aligned_native_size, 0, native_handle)
        print("✓ Native mapped")
        cumulative_size = native_offset + aligned_native_size
        mem_set_access(base_va, cumulative_size, access_desc)
        print("✓ Native mem_set_access succeeded!")

        # Verify all work
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

        native_array = CUDAArrayInterface(native_va, native_size_bytes)
        native_tensor = torch.as_tensor(native_array, device="cuda")
        native_tensor.fill_(777.0)
        torch.cuda.synchronize()
        assert torch.all(native_tensor == 777.0)
        print("✓ Small native allocation works!")

        # Cleanup
        mem_unmap(base_va, minimal_size)
        mem_unmap(import_va, aligned_import_size)
        mem_unmap(native_va, aligned_native_size)
        mem_release(minimal_handle)
        mem_release(imported_handle)
        mem_release(native_handle)

    finally:
        mem_address_free(base_va, heap_size)
        torch.cuda.synchronize()
        if dist.is_initialized():
            dist.barrier()


def test_vmem_minimal_export_then_ondemand_allocations():
    """
    Test the minimal export + on-demand mapping strategy.

    Workflow:
    1. Reserve VA (e.g., 64MB)
    2. Map minimal space (e.g., 2MB) at offset 0
    3. Export DMA-BUF from the minimal mapping
    4. Map native allocation at offset 2MB
    5. Map imported external tensor at offset 4MB

    All mappings are separate and non-overlapping - no unmapping needed!
    """
    device_id = _get_device_id()
    granularity = get_allocation_granularity(device_id)

    heap_size = 64 << 20  # 64 MB total VA
    minimal_size = 2 << 20  # 2 MB minimal initial mapping
    native_size = 2 << 20  # 2 MB native allocation

    # Step 1: Reserve VA space
    base_va = mem_address_reserve(heap_size, granularity, 0)

    access_desc = hipMemAccessDesc()
    access_desc.location.type = hipMemLocationTypeDevice
    access_desc.location.id = device_id
    access_desc.flags = hipMemAccessFlagsProtReadWrite

    try:
        # Step 2: Map minimal space at offset 0 (for export requirement)
        minimal_handle = mem_create(minimal_size, device_id)
        minimal_va = base_va
        mem_map(minimal_va, minimal_size, 0, minimal_handle)
        cumulative_size = minimal_size
        mem_set_access(base_va, cumulative_size, access_desc)

        # Verify minimal mapping works
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

        minimal_array = CUDAArrayInterface(minimal_va, 1024 * 4)
        minimal_tensor = torch.as_tensor(minimal_array, device="cuda")
        minimal_tensor.fill_(111.0)
        torch.cuda.synchronize()
        assert torch.all(minimal_tensor == 111.0)

        # Step 3: Export DMA-BUF from minimal mapping (satisfies SymmetricHeap requirement)
        dmabuf_fd, export_base, export_size = export_dmabuf_handle(minimal_va, minimal_size)
        import os

        os.close(dmabuf_fd)

        # Step 4: Map native allocation on-demand at offset 2MB
        native_offset = minimal_size
        native_va = base_va + native_offset
        native_handle = mem_create(native_size, device_id)
        mem_map(native_va, native_size, 0, native_handle)
        cumulative_size = native_offset + native_size
        mem_set_access(base_va, cumulative_size, access_desc)

        native_array = CUDAArrayInterface(native_va, 1024 * 4)
        native_tensor = torch.as_tensor(native_array, device="cuda")
        native_tensor.fill_(222.0)
        torch.cuda.synchronize()
        assert torch.all(native_tensor == 222.0)

        # Step 5: Create external tensor
        external_tensor = torch.randn(1024, dtype=torch.float32, device="cuda")
        external_tensor.fill_(333.0)

        external_ptr = external_tensor.data_ptr()
        alloc_base, alloc_size = get_address_range(external_ptr)
        ext_dmabuf_fd, ext_export_base, ext_export_size = export_dmabuf_handle(alloc_base, alloc_size)

        # Step 6: Import external tensor on-demand at offset 4MB
        import_offset = minimal_size + native_size
        import_va = base_va + import_offset

        imported_handle = mem_import_from_shareable_handle(ext_dmabuf_fd)
        os.close(ext_dmabuf_fd)

        aligned_import_size = (ext_export_size + granularity - 1) & ~(granularity - 1)
        mem_map(import_va, aligned_import_size, 0, imported_handle)
        cumulative_size = import_offset + aligned_import_size
        mem_set_access(base_va, cumulative_size, access_desc)

        # Verify imported tensor access with offset preservation
        offset_in_alloc = external_ptr - alloc_base
        imported_ptr = import_va + offset_in_alloc
        imported_array = CUDAArrayInterface(imported_ptr, 1024 * 4)
        imported_tensor = torch.as_tensor(imported_array, device="cuda")
        torch.cuda.synchronize()
        assert torch.all(imported_tensor == 333.0)

        # Verify all three allocations are independent and working
        assert torch.all(minimal_tensor == 111.0), "Minimal mapping corrupted"
        assert torch.all(native_tensor == 222.0), "Native allocation corrupted"
        assert torch.all(imported_tensor == 333.0), "Imported tensor corrupted"

        # Cleanup
        del minimal_tensor, native_tensor, imported_tensor
        mem_unmap(minimal_va, minimal_size)
        mem_unmap(native_va, native_size)
        mem_unmap(import_va, aligned_import_size)
        mem_release(minimal_handle)
        mem_release(native_handle)
        mem_release(imported_handle)

    finally:
        mem_address_free(base_va, heap_size)
        torch.cuda.synchronize()
        if dist.is_initialized():
            dist.barrier()


def test_vmem_multiple_natives_and_imports():
    """
    Test multiple native allocations and imports in the same VA space.

    This simulates a realistic allocator usage pattern:
    - Initial minimal mapping for export
    - Multiple native allocations
    - Multiple imported tensors
    - All coexist in the same VA range without conflicts
    """
    device_id = _get_device_id()
    granularity = get_allocation_granularity(device_id)

    heap_size = 128 << 20  # 128 MB total VA
    minimal_size = 2 << 20  # 2 MB minimal

    base_va = mem_address_reserve(heap_size, granularity, 0)

    access_desc = hipMemAccessDesc()
    access_desc.location.type = hipMemLocationTypeDevice
    access_desc.location.id = device_id
    access_desc.flags = hipMemAccessFlagsProtReadWrite

    allocations = []  # Track all allocations for cleanup
    current_offset = 0

    try:
        # Initial minimal mapping
        minimal_handle = mem_create(minimal_size, device_id)
        mem_map(base_va, minimal_size, 0, minimal_handle)
        current_offset += minimal_size
        mem_set_access(base_va, current_offset, access_desc)
        allocations.append(("minimal", base_va, minimal_size, minimal_handle))

        # Export for SymmetricHeap
        dmabuf_fd, _, _ = export_dmabuf_handle(base_va, minimal_size)
        import os

        os.close(dmabuf_fd)

        # Create 3 native allocations
        for i in range(3):
            alloc_size = 2 << 20  # 2 MB each
            handle = mem_create(alloc_size, device_id)
            va = base_va + current_offset
            mem_map(va, alloc_size, 0, handle)
            current_offset += alloc_size
            mem_set_access(base_va, current_offset, access_desc)
            allocations.append((f"native_{i}", va, alloc_size, handle))

        # Create 3 imported tensors
        external_tensors = []
        for i in range(3):
            ext_tensor = torch.randn(1024, dtype=torch.float32, device="cuda")
            ext_tensor.fill_(float(i + 100))
            external_tensors.append(ext_tensor)

            ext_ptr = ext_tensor.data_ptr()
            alloc_base, alloc_size = get_address_range(ext_ptr)
            ext_fd, ext_base, ext_size = export_dmabuf_handle(alloc_base, alloc_size)

            imported_handle = mem_import_from_shareable_handle(ext_fd)
            os.close(ext_fd)

            aligned_size = (ext_size + granularity - 1) & ~(granularity - 1)
            va = base_va + current_offset
            mem_map(va, aligned_size, 0, imported_handle)
            current_offset += aligned_size
            mem_set_access(base_va, current_offset, access_desc)
            allocations.append((f"import_{i}", va, aligned_size, imported_handle))

        # Verify all allocations are accessible and independent
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

        # Test native allocations
        for name, va, size, handle in allocations:
            if name.startswith("native") or name == "minimal":
                array = CUDAArrayInterface(va, 1024 * 4)
                tensor = torch.as_tensor(array, device="cuda")
                tensor.fill_(42.0)
                torch.cuda.synchronize()
                assert torch.all(tensor == 42.0), f"{name} allocation failed"

        # Test imported allocations with offset preservation
        import_idx = 0
        for name, va, size, handle in allocations:
            if name.startswith("import"):
                ext_tensor = external_tensors[import_idx]
                ext_ptr = ext_tensor.data_ptr()
                alloc_base, _ = get_address_range(ext_ptr)
                offset = ext_ptr - alloc_base

                imported_ptr = va + offset
                array = CUDAArrayInterface(imported_ptr, 1024 * 4)
                tensor = torch.as_tensor(array, device="cuda")
                torch.cuda.synchronize()

                expected = float(import_idx + 100)
                assert torch.all(tensor == expected), f"{name} has wrong value"
                import_idx += 1

        # Cleanup all allocations
        for name, va, size, handle in allocations:
            mem_unmap(va, size)
            mem_release(handle)

    finally:
        mem_address_free(base_va, heap_size)
        torch.cuda.synchronize()
        if dist.is_initialized():
            dist.barrier()
