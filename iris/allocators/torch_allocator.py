# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.

"""
PyTorch-based allocator for Iris symmetric heap.

Uses torch.empty() to allocate a large memory pool and manages
sub-allocations within it using bump allocation.
"""

import math
import numpy as np
import torch
from typing import Optional, Dict
import struct

from .base import BaseAllocator
from iris.hip import export_dmabuf_handle, import_dmabuf_handle, destroy_external_memory
from iris.fd_passing import send_fd, recv_fd, managed_fd
from iris.util import is_simulation_env


class TorchAllocator(BaseAllocator):
    """
    PyTorch-based memory allocator using a pre-allocated memory pool.

    This allocator creates a single large torch.empty() buffer and
    manages sub-allocations within it using bump allocation.
    """

    def __init__(self, heap_size: int, device_id: int, cur_rank: int, num_ranks: int):
        """
        Initialize the PyTorch allocator.

        Args:
            heap_size: Size of the heap in bytes
            device_id: GPU device ID
            cur_rank: Current process rank
            num_ranks: Total number of ranks
        """
        super().__init__(heap_size, device_id, cur_rank, num_ranks)

        self.device = f"cuda:{device_id}"
        if is_simulation_env():
            import json

            # In simulation, each rank allocates n distinct buffers; memory_pool is a shallow view of the ith.
            self.rank_bools = [torch.empty(heap_size, device=self.device, dtype=torch.int8) for _ in range(num_ranks)]
            self.memory_pool = self.rank_bools[cur_rank]

            heap_views = [self.rank_bools[r].data_ptr() for r in range(num_ranks)]
            out_path = f"iris_rank_{cur_rank}_allocator_views.json"
            with open(out_path, "w") as f:
                json.dump(
                    {
                        "rank": cur_rank,
                        "num_ranks": num_ranks,
                        "heap_views": [hex(b) for b in heap_views],
                    },
                    f,
                    indent=2,
                )
        else:
            self.rank_bools = None
            self.memory_pool = torch.empty(heap_size, device=self.device, dtype=torch.int8)

        self._peer_ext_mem_handles: Dict[int, object] = {}

    def get_minimum_allocation_size(self) -> int:
        """Minimum allocation size in bytes (PyTorch allows 0-size views)."""
        return 0

    def get_base_address(self) -> int:
        """Get the base address of the memory pool."""
        return self.memory_pool.data_ptr()

    def allocate(self, num_elements: int, dtype: torch.dtype, alignment: int = 1024) -> torch.Tensor:
        """
        Allocate a tensor from the memory pool using bump allocation.

        Args:
            num_elements: Number of elements to allocate
            dtype: PyTorch data type
            alignment: Memory alignment in bytes (default: 1024)

        Returns:
            Tensor view into the memory pool

        Raises:
            MemoryError: If heap is out of space
        """
        element_size = torch.tensor([], dtype=dtype).element_size()
        size_in_bytes = num_elements * element_size
        aligned_size = math.ceil(size_in_bytes / alignment) * alignment

        if self.heap_offset + aligned_size > self.heap_size:
            raise MemoryError("Heap out of memory")

        start = self.heap_offset
        self.heap_offset += aligned_size

        sub_buffer = self.memory_pool[start : start + size_in_bytes].view(dtype)
        return sub_buffer.reshape((num_elements,))

    def get_shareable_handle(self) -> tuple:
        """
        Get a shareable handle for the memory pool.

        Returns:
            tuple: (fd, base_ptr, base_size) from export_dmabuf_handle
        """
        heap_base = self.get_base_address()
        return export_dmabuf_handle(heap_base, self.heap_size)

    def establish_peer_access(self, all_bases: Dict[int, int], connections: Optional[Dict] = None):
        """
        Establish access to peer memory for symmetric addressing.

        Args:
            all_bases: Dictionary mapping rank -> base address
            connections: Optional peer connections for handle exchange
        """
        heap_bases_array = np.zeros(self.num_ranks, dtype=np.uint64)

        if connections is not None:
            for handle in self._peer_ext_mem_handles.values():
                try:
                    destroy_external_memory(handle)
                except Exception:
                    pass
            self._peer_ext_mem_handles.clear()

            my_fd, my_base, my_size = self.get_shareable_handle()
            heap_base = self.get_base_address()
            my_metadata = struct.pack("QQQ", my_base, my_size, heap_base)

            with managed_fd(my_fd):
                for peer, sock in connections.items():
                    if peer == self.cur_rank:
                        continue

                    # Higher rank sends first to avoid deadlock
                    if self.cur_rank > peer:
                        send_fd(sock, my_fd, payload=my_metadata)
                        peer_handle, peer_metadata = recv_fd(sock, payload_size=24)
                    else:
                        peer_handle, peer_metadata = recv_fd(sock, payload_size=24)
                        send_fd(sock, my_fd, payload=my_metadata)

                    peer_base, peer_size, peer_heap = struct.unpack("QQQ", peer_metadata)

                    with managed_fd(peer_handle):
                        mapped_ptr, ext_mem_handle = import_dmabuf_handle(peer_handle, peer_size, peer_heap, peer_base)
                        heap_bases_array[peer] = mapped_ptr
                        self._peer_ext_mem_handles[peer] = ext_mem_handle

            heap_bases_array[self.cur_rank] = all_bases[self.cur_rank]
        else:
            heap_bases_array[self.cur_rank] = all_bases[self.cur_rank]

        self.heap_bases_array = heap_bases_array

    def close(self):
        """Release peer external memory handles."""
        for handle in self._peer_ext_mem_handles.values():
            try:
                destroy_external_memory(handle)
            except Exception:
                pass
        self._peer_ext_mem_handles.clear()

    def get_device(self) -> torch.device:
        """Get the torch device."""
        return self.memory_pool.device

    def import_external_tensor(self, external_tensor: torch.Tensor) -> torch.Tensor:
        """
        Place an external tensor's data on the symmetric heap by copying.

        Unlike the VMem allocator, this does not share memory with the external
        tensor: it allocates on the heap and copies. Subsequent changes to the
        external tensor are not visible in the returned tensor.

        Args:
            external_tensor: External PyTorch tensor to copy from (must be CUDA, contiguous)

        Returns:
            New tensor on the symmetric heap with the same data and shape.
        """
        if not external_tensor.is_cuda:
            raise RuntimeError("Can only import CUDA tensors")
        if not external_tensor.is_contiguous():
            raise RuntimeError("Only contiguous tensors can be imported; call .contiguous() before as_symmetric()")
        num_elements = external_tensor.numel()
        dtype = external_tensor.dtype
        shape = external_tensor.shape
        heap_tensor = self.allocate(num_elements, dtype)
        heap_tensor = heap_tensor.reshape(shape).copy_(external_tensor)
        return heap_tensor

    def owns_tensor(self, tensor: torch.Tensor) -> bool:
        """
        Check if a tensor is within the allocator's managed heap.

        Args:
            tensor: PyTorch tensor to check

        Returns:
            True if tensor is within the heap, False otherwise
        """
        if tensor.numel() == 0:
            return True

        ptr = int(tensor.data_ptr())
        heap_base = self.get_base_address()
        return ptr >= heap_base and ptr < heap_base + self.heap_size
