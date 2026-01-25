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

from .base import BaseAllocator
from iris.hip import export_dmabuf_handle, import_dmabuf_handle
from iris.fd_passing import send_fd, recv_fd, managed_fd


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
        self.memory_pool = torch.empty(heap_size, device=self.device, dtype=torch.int8)
        self.heap_bases_array = None  # Will be set in establish_peer_access

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

    def get_shareable_handle(self) -> int:
        """
        Get a shareable handle for the memory pool.

        Returns:
            DMA-BUF file descriptor
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
        # Use the original heap bases (no remapping for TorchAllocator)
        heap_bases_array = np.zeros(self.num_ranks, dtype=np.uint64)

        if connections is not None:
            # Get shareable handle for our memory pool
            my_handle = self.get_shareable_handle()

            # Use context manager for automatic cleanup
            with managed_fd(my_handle):
                # Exchange handles with all peers
                for peer, sock in connections.items():
                    if peer == self.cur_rank:
                        continue

                    # To avoid deadlock, higher rank sends first
                    if self.cur_rank > peer:
                        send_fd(sock, my_handle)
                        peer_handle, _ = recv_fd(sock)
                    else:
                        peer_handle, _ = recv_fd(sock)
                        send_fd(sock, my_handle)

                    # Use context manager for peer handle and import the DMA-BUF
                    with managed_fd(peer_handle):
                        # Import peer's memory via DMA-BUF and get mapped address
                        mapped_addr = import_dmabuf_handle(peer_handle, self.heap_size)
                        heap_bases_array[peer] = mapped_addr

            # Set our own base
            heap_bases_array[self.cur_rank] = all_bases[self.cur_rank]
        else:
            # Single rank, just set our own base
            heap_bases_array[self.cur_rank] = all_bases[self.cur_rank]

        self.heap_bases_array = heap_bases_array

    def get_device(self) -> torch.device:
        """Get the torch device."""
        return self.memory_pool.device

    def get_heap_bases(self) -> torch.Tensor:
        """Get heap base addresses as a tensor."""
        return torch.from_numpy(self.heap_bases_array).to(device=self.device, dtype=torch.uint64)
