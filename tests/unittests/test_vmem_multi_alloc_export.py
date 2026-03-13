# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.

"""
Test DMA-BUF export of multiple separate VMem allocations.

This test demonstrates that export_dmabuf_handle() on a VA range with
multiple separate physical allocations only exports the first one.
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


def test_vmem_multi_alloc_export_fails():
    """
    Test that exporting a VA range with multiple physical allocations fails.

    Expected: export_dmabuf_handle should fail or only export first allocation.

    Setup:
    - Reserve 64MB VA range
    - Map 3 separate allocations: 2MB, 2MB, 4KB
    - Try to export entire cumulative range (4MB + 4KB)
    - Try to import and access beyond first allocation
    """
    device_id = torch.cuda.current_device()
    granularity = get_allocation_granularity(device_id)

    heap_size = 64 << 20  # 64 MB VA
    size1 = 2 << 20  # 2 MB
    size2 = 2 << 20  # 2 MB
    size3 = granularity  # 4KB (one granule)

    base_va = mem_address_reserve(heap_size, granularity, 0)

    access_desc = hipMemAccessDesc()
    access_desc.location.type = hipMemLocationTypeDevice
    access_desc.location.id = device_id
    access_desc.flags = hipMemAccessFlagsProtReadWrite

    cumulative_size = 0

    try:
        # Allocation 1: 2MB at offset 0
        va1 = base_va
        handle1 = mem_create(size1, device_id)
        mem_map(va1, size1, 0, handle1)
        cumulative_size += size1
        mem_set_access(base_va, cumulative_size, access_desc)
        print(f"✓ Mapped alloc1: 2MB at offset 0, cumulative={cumulative_size}")

        # Allocation 2: 2MB at offset 2MB (separate physical allocation!)
        va2 = base_va + size1
        handle2 = mem_create(size2, device_id)
        mem_map(va2, size2, 0, handle2)
        cumulative_size += size2
        mem_set_access(base_va, cumulative_size, access_desc)
        print(f"✓ Mapped alloc2: 2MB at offset 2MB, cumulative={cumulative_size}")

        # Allocation 3: 4KB at offset 4MB (separate physical allocation!)
        va3 = base_va + size1 + size2
        handle3 = mem_create(size3, device_id)
        mem_map(va3, size3, 0, handle3)
        cumulative_size += size3
        mem_set_access(base_va, cumulative_size, access_desc)
        print(f"✓ Mapped alloc3: 4KB at offset 4MB, cumulative={cumulative_size}")

        # Fill each allocation with distinct values
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

        array1 = CUDAArrayInterface(va1, 1024 * 4)
        tensor1 = torch.as_tensor(array1, device="cuda")
        tensor1.fill_(111.0)

        array2 = CUDAArrayInterface(va2, 1024 * 4)
        tensor2 = torch.as_tensor(array2, device="cuda")
        tensor2.fill_(222.0)

        array3 = CUDAArrayInterface(va3, 64)
        tensor3 = torch.as_tensor(array3, device="cuda")
        tensor3.fill_(333.0)

        torch.cuda.synchronize()
        print("✓ Filled all allocations with test values")

        # NOW: Try to export the entire cumulative range
        print("\n=== Attempting to export cumulative range ===")
        print(f"Base VA: {hex(base_va)}")
        print(f"Cumulative size: {cumulative_size} bytes ({cumulative_size >> 20}MB + {cumulative_size & 0xFFF} bytes)")

        try:
            dmabuf_fd, export_base, export_size = export_dmabuf_handle(base_va, cumulative_size)
            print("✓ Export succeeded!")
            print(f"  Export base: {hex(export_base)}")
            print(f"  Export size: {export_size} bytes ({export_size >> 20}MB)")

            # Import it back
            imported_handle = mem_import_from_shareable_handle(dmabuf_fd)
            os.close(dmabuf_fd)

            # Map to a different VA to test
            import_va = mem_address_reserve(heap_size, granularity, 0)

            try:
                # Try to map the full export size
                mem_map(import_va, export_size, 0, imported_handle)
                mem_set_access(import_va, export_size, access_desc)
                print(f"✓ Imported and mapped {export_size} bytes")

                # Try to read from each allocation through the import
                print("\n=== Testing access through import ===")

                # Read alloc1 (should work - it's the first one)
                import_array1 = CUDAArrayInterface(import_va, 1024 * 4)
                import_tensor1 = torch.as_tensor(import_array1, device="cuda")
                torch.cuda.synchronize()
                val1 = import_tensor1[0].item()
                print(f"  Alloc1 through import: {val1} (expected 111.0) - {'✓ PASS' if val1 == 111.0 else '✗ FAIL'}")

                # Read alloc2 (might fail - separate allocation)
                if export_size >= size1 + 4096:  # At least some of alloc2
                    try:
                        import_array2 = CUDAArrayInterface(import_va + size1, 1024 * 4)
                        import_tensor2 = torch.as_tensor(import_array2, device="cuda")
                        torch.cuda.synchronize()
                        val2 = import_tensor2[0].item()
                        print(
                            f"  Alloc2 through import: {val2} (expected 222.0) - {'✓ PASS' if val2 == 222.0 else '✗ FAIL'}"
                        )
                    except Exception as e:
                        print(f"  Alloc2 through import: ✗ FAIL - {e}")

                # Read alloc3 (might fail - separate allocation)
                if export_size >= size1 + size2 + 64:
                    try:
                        import_array3 = CUDAArrayInterface(import_va + size1 + size2, 64)
                        import_tensor3 = torch.as_tensor(import_array3, device="cuda")
                        torch.cuda.synchronize()
                        val3 = import_tensor3[0].item()
                        print(
                            f"  Alloc3 through import: {val3} (expected 333.0) - {'✓ PASS' if val3 == 333.0 else '✗ FAIL'}"
                        )
                    except Exception as e:
                        print(f"  Alloc3 through import: ✗ FAIL - {e}")

                mem_unmap(import_va, export_size)
                mem_release(imported_handle)

            finally:
                mem_address_free(import_va, heap_size)

        except RuntimeError as e:
            print(f"✗ Export FAILED: {e}")
            print("  This is expected if ROCm can't export multiple allocations as one!")

        # Cleanup
        mem_unmap(va3, size3)
        mem_release(handle3)
        mem_unmap(va2, size2)
        mem_release(handle2)
        mem_unmap(va1, size1)
        mem_release(handle1)

    finally:
        mem_address_free(base_va, heap_size)
        torch.cuda.synchronize()


if __name__ == "__main__":
    test_vmem_multi_alloc_export_fails()
