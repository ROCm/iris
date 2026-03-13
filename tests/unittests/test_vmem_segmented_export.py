# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.

"""
Test segmented DMA-BUF export/import for multi-allocation VMem.

This test demonstrates that we CAN achieve symmetric addressing by
exporting/importing each allocation separately to matching VA offsets.
"""

import torch
import os
from iris.hip import (
    get_allocation_granularity,
    mem_address_reserve,
    mem_address_free,
    mem_create,
    mem_map,
    mem_unmap,
    mem_release,
    mem_set_access,
    export_dmabuf_handle,
    mem_import_from_shareable_handle,
    hipMemAccessDesc,
    hipMemLocationTypeDevice,
    hipMemAccessFlagsProtReadWrite,
)


def test_vmem_segmented_export_import():
    """
    Test that exporting/importing each allocation separately works.

    Strategy:
    - Create 3 separate allocations at VA offsets 0, 2MB, 4MB
    - Export EACH allocation individually (3 DMA-BUFs)
    - Import each to corresponding offsets in a separate VA range
    - Verify all data is accessible
    """
    device_id = torch.cuda.current_device()
    granularity = get_allocation_granularity(device_id)

    heap_size = 64 << 20  # 64 MB VA
    size1 = 2 << 20  # 2 MB
    size2 = 2 << 20  # 2 MB
    size3 = granularity  # 4KB

    # Source heap (like Rank 0's heap)
    src_base_va = mem_address_reserve(heap_size, granularity, 0)

    # Destination heap (like Rank 1's view of Rank 0's heap)
    dst_base_va = mem_address_reserve(heap_size, granularity, 0)

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

    src_handles = []
    dst_handles = []
    cumulative_size = 0

    try:
        print("=== Creating source allocations ===")

        # Allocation 1: 2MB at offset 0
        va1 = src_base_va
        handle1 = mem_create(size1, device_id)
        mem_map(va1, size1, 0, handle1)
        cumulative_size += size1
        mem_set_access(src_base_va, cumulative_size, access_desc)  # Cumulative!
        src_handles.append((va1, size1, handle1, 0))

        array1 = CUDAArrayInterface(va1, 1024 * 4)
        tensor1 = torch.as_tensor(array1, device="cuda")
        tensor1.fill_(111.0)
        print("✓ Alloc1: 2MB at offset 0, value=111.0")

        # Allocation 2: 2MB at offset 2MB
        va2 = src_base_va + size1
        handle2 = mem_create(size2, device_id)
        mem_map(va2, size2, 0, handle2)
        cumulative_size += size2
        mem_set_access(src_base_va, cumulative_size, access_desc)  # Cumulative!
        src_handles.append((va2, size2, handle2, size1))

        array2 = CUDAArrayInterface(va2, 1024 * 4)
        tensor2 = torch.as_tensor(array2, device="cuda")
        tensor2.fill_(222.0)
        print(f"✓ Alloc2: 2MB at offset {size1}, value=222.0")

        # Allocation 3: 4KB at offset 4MB
        va3 = src_base_va + size1 + size2
        handle3 = mem_create(size3, device_id)
        mem_map(va3, size3, 0, handle3)
        cumulative_size += size3
        mem_set_access(src_base_va, cumulative_size, access_desc)  # Cumulative!
        src_handles.append((va3, size3, handle3, size1 + size2))

        array3 = CUDAArrayInterface(va3, 64)
        tensor3 = torch.as_tensor(array3, device="cuda")
        tensor3.fill_(333.0)
        print(f"✓ Alloc3: 4KB at offset {size1 + size2}, value=333.0")

        torch.cuda.synchronize()

        print("\n=== Exporting each allocation separately ===")

        # Export each allocation individually
        exported_fds = []
        for i, (va, size, handle, offset) in enumerate(src_handles, 1):
            dmabuf_fd, export_base, export_size = export_dmabuf_handle(va, size)
            exported_fds.append((dmabuf_fd, export_size, offset))
            print(f"✓ Exported alloc{i}: size={export_size} bytes (requested {size})")

        print("\n=== Importing to matching VA offsets ===")

        # Import each to the SAME offset in destination VA range
        dst_cumulative_size = 0
        for i, (fd, export_size, offset) in enumerate(exported_fds, 1):
            imported_handle = mem_import_from_shareable_handle(fd)
            os.close(fd)

            # Map to SAME offset in destination
            dst_va = dst_base_va + offset
            mem_map(dst_va, export_size, 0, imported_handle)
            dst_cumulative_size += export_size
            mem_set_access(dst_base_va, dst_cumulative_size, access_desc)  # Cumulative!
            dst_handles.append((dst_va, export_size, imported_handle))

            print(f"✓ Imported alloc{i} to offset {offset} ({hex(dst_va)})")

        print("\n=== Verifying data through imports ===")

        # Read alloc1 through destination
        dst_array1 = CUDAArrayInterface(dst_base_va, 1024 * 4)
        dst_tensor1 = torch.as_tensor(dst_array1, device="cuda")
        torch.cuda.synchronize()
        val1 = dst_tensor1[0].item()
        assert val1 == 111.0, f"Alloc1 mismatch: {val1} != 111.0"
        print(f"✓ Alloc1 through import: {val1} (expected 111.0)")

        # Read alloc2 through destination
        dst_array2 = CUDAArrayInterface(dst_base_va + size1, 1024 * 4)
        dst_tensor2 = torch.as_tensor(dst_array2, device="cuda")
        torch.cuda.synchronize()
        val2 = dst_tensor2[0].item()
        assert val2 == 222.0, f"Alloc2 mismatch: {val2} != 222.0"
        print(f"✓ Alloc2 through import: {val2} (expected 222.0)")

        # Read alloc3 through destination
        dst_array3 = CUDAArrayInterface(dst_base_va + size1 + size2, 64)
        dst_tensor3 = torch.as_tensor(dst_array3, device="cuda")
        torch.cuda.synchronize()
        val3 = dst_tensor3[0].item()
        assert val3 == 333.0, f"Alloc3 mismatch: {val3} != 333.0"
        print(f"✓ Alloc3 through import: {val3} (expected 333.0)")

        print("\n✅ SUCCESS: Segmented export/import maintains symmetric addressing!")

    finally:
        # Cleanup destination imports
        for dst_va, size, handle in dst_handles:
            mem_unmap(dst_va, size)
            mem_release(handle)

        # Cleanup source allocations
        for va, size, handle, _ in src_handles:
            mem_unmap(va, size)
            mem_release(handle)

        mem_address_free(dst_base_va, heap_size)
        mem_address_free(src_base_va, heap_size)
        torch.cuda.synchronize()


if __name__ == "__main__":
    test_vmem_segmented_export_import()
