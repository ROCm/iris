# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.

"""
VMem-based allocator using HIP's virtual memory management APIs.

This allocator provides fine-grained control over virtual and physical memory,
enabling features like memory oversubscription and on-demand paging.
"""

import torch
import os
from typing import Dict
from threading import Lock

from .base import BaseAllocator
from ..hip import (
    get_allocation_granularity,
    get_address_range,
    export_dmabuf_handle,
    mem_import_from_shareable_handle,
    mem_create,
    mem_address_reserve,
    mem_map,
    mem_unmap,
    mem_address_free,
    mem_release,
    mem_set_access,
    hipMemAccessDesc,
    hipMemLocationTypeDevice,
    hipMemAccessFlagsProtReadWrite,
)


class VMemAllocator(BaseAllocator):
    """
    Virtual Memory allocator using HIP's VMem APIs.

    Features:
    - Reserve large virtual address (VA) space upfront
    - Map physical memory on demand
    - Support memory oversubscription
    - Fine-grained control over allocations

    Args:
        heap_size: Total size of the heap in bytes
        device: PyTorch device (e.g., "cuda:0")
        rank: Current rank ID
        world_size: Total number of ranks
        va_multiplier: VA space multiplier (reserve more VA than physical)
    """

    def __init__(
        self,
        heap_size: int,
        device_id: int,
        rank: int,
        world_size: int,
        va_multiplier: float = 1.0,
    ):
        super().__init__(heap_size, device_id, rank, world_size)
        self.va_multiplier = va_multiplier
        self.device = torch.device(f"cuda:{device_id}")
        self.lock = Lock()
        self.granularity = get_allocation_granularity(self.device_id)
        self.aligned_heap_size = (heap_size + self.granularity - 1) & ~(self.granularity - 1)
        self.va_size = self.aligned_heap_size
        self.base_va = mem_address_reserve(self.va_size, self.granularity, 0)

        self.minimal_size = min(2 << 20, self.aligned_heap_size // 2)
        if self.minimal_size < self.granularity:
            self.minimal_size = self.granularity

        self.minimal_handle = mem_create(self.minimal_size, self.device_id)
        mem_map(self.base_va, self.minimal_size, 0, self.minimal_handle)

        # ROCm: mem_set_access must be called cumulatively from base_va (see rocm-systems#2667)
        self.access_descs = []
        for peer_device_id in range(world_size):
            desc = hipMemAccessDesc()
            desc.location.type = hipMemLocationTypeDevice
            desc.location.id = peer_device_id
            desc.flags = hipMemAccessFlagsProtReadWrite
            self.access_descs.append(desc)

        self.cumulative_mapped_size = self.minimal_size
        mem_set_access(self.base_va, self.cumulative_mapped_size, self.access_descs)

        self.allocations: Dict[int, tuple] = {}
        self.allocation_order = []
        self._track_allocation(0, self.minimal_size, False, self.minimal_handle, self.base_va)
        self.current_offset = self.minimal_size

        self.world_size = world_size

    def get_base_address(self) -> int:
        """Get the base address of the heap."""
        return self.base_va

    def _track_allocation(self, offset: int, size: int, is_imported: bool, handle, va: int):
        """Track a new allocation for cleanup and segmented export."""
        self.allocations[offset] = (size, is_imported, handle, va)
        self.allocation_order.append((offset, size))

    def get_allocation_segments(self):
        """
        Get list of allocation segments for segmented DMA-BUF export.

        Returns:
            List of (offset, size, va) tuples for each allocation in order.
            Each tuple describes one physically-backed segment that needs
            to be exported/imported separately.
        """
        segments = []
        for offset, size in self.allocation_order:
            va = self.base_va + offset
            segments.append((offset, size, va))
        return segments

    def get_minimum_allocation_size(self) -> int:
        """Minimum allocation size in bytes (one granule; hipMemCreate(0) is invalid)."""
        return self.granularity

    def allocate(self, num_elements: int, dtype: torch.dtype, alignment: int = 1024) -> torch.Tensor:
        """
        Allocate memory from the VMem heap.

        Args:
            num_elements: Number of elements to allocate
            dtype: PyTorch data type
            alignment: Alignment requirement in bytes

        Returns:
            PyTorch tensor wrapping the allocated memory

        Raises:
            RuntimeError: If allocation fails or heap is full
        """
        with self.lock:
            element_size = torch.tensor([], dtype=dtype).element_size()
            size_bytes = num_elements * element_size
            actual_size_bytes = max(size_bytes, self.get_minimum_allocation_size())
            aligned_size = (actual_size_bytes + self.granularity - 1) & ~(self.granularity - 1)
            aligned_offset = (self.current_offset + alignment - 1) & ~(alignment - 1)

            if aligned_offset + aligned_size > self.aligned_heap_size:
                raise RuntimeError(
                    f"Out of VMem address space for allocation: "
                    f"need {aligned_size} bytes at offset {aligned_offset}, "
                    f"but heap size is {self.aligned_heap_size}. "
                    f"Current offset: {self.current_offset}, "
                    f"available: {self.aligned_heap_size - aligned_offset} bytes"
                )

            target_va = self.base_va + aligned_offset
            handle = mem_create(aligned_size, self.device_id)
            mem_map(target_va, aligned_size, 0, handle)

            new_cumulative_size = aligned_offset + aligned_size
            if new_cumulative_size > self.cumulative_mapped_size:
                self.cumulative_mapped_size = new_cumulative_size
                mem_set_access(self.base_va, self.cumulative_mapped_size, self.access_descs)

            self._track_allocation(aligned_offset, aligned_size, False, handle, target_va)
            self.current_offset = aligned_offset + aligned_size

            interface_size = (aligned_size // element_size) * element_size

            class CUDAArrayInterface:
                def __init__(self, ptr, size_bytes, device):
                    self.ptr = ptr
                    self.size_bytes = size_bytes
                    self.device = device

                @property
                def __cuda_array_interface__(self):
                    return {
                        "shape": (self.size_bytes,),
                        "typestr": "|u1",
                        "data": (self.ptr, False),
                        "version": 3,
                    }

            cuda_array = CUDAArrayInterface(target_va, interface_size, self.device)
            tensor_bytes = torch.as_tensor(cuda_array, device=self.device)
            full = tensor_bytes.view(dtype)
            if num_elements == 0:
                tensor = full.narrow(0, 1, 0)
            else:
                tensor = full.narrow(0, 0, num_elements)
            return tensor

    def get_device(self) -> torch.device:
        """
        Get the PyTorch device for this allocator.

        Returns:
            PyTorch device object
        """
        return self.device

    def owns_tensor(self, tensor: torch.Tensor) -> bool:
        """
        Check if a tensor's memory belongs to this allocator's heap.

        Args:
            tensor: Tensor to check

        Returns:
            True if tensor is within this allocator's heap, False otherwise
        """
        if not tensor.is_cuda:
            return False
        if tensor.numel() == 0:
            return True

        ptr = tensor.data_ptr()
        return self.base_va <= ptr < self.base_va + self.aligned_heap_size

    def import_external_tensor(self, external_tensor: torch.Tensor) -> torch.Tensor:
        """
        Import an external PyTorch tensor into the symmetric heap (as_symmetric).

        This creates a view into the symmetric heap that shares physical memory
        with the external tensor, handling PyTorch caching allocator offsets.

        Args:
            external_tensor: External PyTorch tensor to import

        Returns:
            New tensor view in symmetric heap that shares memory with external tensor

        Raises:
            RuntimeError: If import fails or tensor is not contiguous
        """

        with self.lock:
            if not external_tensor.is_cuda:
                raise RuntimeError("Can only import CUDA tensors")
            if not external_tensor.is_contiguous():
                raise RuntimeError("Only contiguous tensors can be imported; call .contiguous() before as_symmetric()")

            external_ptr = external_tensor.data_ptr()
            alloc_base, alloc_size = get_address_range(external_ptr)
            offset_in_alloc = external_ptr - alloc_base
            aligned_size = (alloc_size + self.granularity - 1) & ~(self.granularity - 1)
            aligned_offset = (self.current_offset + self.granularity - 1) & ~(self.granularity - 1)

            if aligned_offset + aligned_size > self.aligned_heap_size:
                raise RuntimeError(
                    f"Out of VMem address space for import: "
                    f"need {aligned_size} bytes at offset {aligned_offset}, "
                    f"but heap size is {self.aligned_heap_size}. "
                    f"Current offset: {self.current_offset}, "
                    f"available: {self.aligned_heap_size - aligned_offset} bytes"
                )

            dmabuf_fd, export_base, export_size = export_dmabuf_handle(alloc_base, alloc_size)
            aligned_export_size = (export_size + self.granularity - 1) & ~(self.granularity - 1)
            target_va = self.base_va + aligned_offset
            imported_handle = mem_import_from_shareable_handle(dmabuf_fd)
            os.close(dmabuf_fd)

            mem_map(target_va, aligned_export_size, 0, imported_handle)

            new_cumulative_size = aligned_offset + aligned_export_size
            if new_cumulative_size > self.cumulative_mapped_size:
                self.cumulative_mapped_size = new_cumulative_size
                mem_set_access(self.base_va, self.cumulative_mapped_size, self.access_descs)

            tensor_va = target_va + offset_in_alloc
            self._track_allocation(aligned_offset, aligned_export_size, True, imported_handle, target_va)
            self.current_offset = aligned_offset + aligned_export_size

            tensor_size = external_tensor.numel() * external_tensor.element_size()

            class CUDAArrayInterface:
                def __init__(self, ptr, size_bytes, device):
                    self.ptr = ptr
                    self.size_bytes = size_bytes
                    self.device = device

                @property
                def __cuda_array_interface__(self):
                    return {
                        "shape": (self.size_bytes,),
                        "typestr": "|u1",
                        "data": (self.ptr, False),
                        "version": 3,
                    }

            cuda_array = CUDAArrayInterface(tensor_va, tensor_size, self.device)
            tensor_bytes = torch.as_tensor(cuda_array, device=self.device)
            imported_tensor = tensor_bytes.view(external_tensor.dtype).reshape(external_tensor.shape)

            return imported_tensor

    def close(self):
        """Explicitly release VMem resources."""
        if hasattr(self, "_closed") and self._closed:
            return

        with self.lock:
            for offset, alloc_info in self.allocations.items():
                if len(alloc_info) == 4:
                    size, is_imported, handle, va = alloc_info

                    if handle is not None:
                        aligned_size = (size + self.granularity - 1) & ~(self.granularity - 1)
                        mem_unmap(va, aligned_size)
                        mem_release(handle)

            self.allocations.clear()

            if hasattr(self, "base_va") and self.base_va:
                mem_address_free(self.base_va, self.va_size)
                self.base_va = 0

            self._closed = True

    def __del__(self):
        """Cleanup VMem resources on deletion."""
        self.close()
