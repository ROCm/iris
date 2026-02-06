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
        # Use the original heap bases (no remapping for TorchAllocator)
        heap_bases_array = np.zeros(self.num_ranks, dtype=np.uint64)

        if connections is not None:
            # Get shareable handle for our memory pool
            my_fd, my_base, my_size = self.get_shareable_handle()
            heap_base = self.get_base_address()

            # Pack metadata: (base_ptr, base_size, heap_ptr) as three 64-bit unsigned ints
            my_metadata = struct.pack("QQQ", my_base, my_size, heap_base)

            # Use context manager for automatic cleanup
            with managed_fd(my_fd):
                # Exchange handles with all peers
                for peer, sock in connections.items():
                    if peer == self.cur_rank:
                        continue

                    # To avoid deadlock, higher rank sends first
                    # Send FD along with metadata (base_ptr, base_size, heap_ptr)
                    if self.cur_rank > peer:
                        send_fd(sock, my_fd, payload=my_metadata)
                        peer_handle, peer_metadata = recv_fd(sock, payload_size=24)  # 3 * 8 bytes
                    else:
                        peer_handle, peer_metadata = recv_fd(sock, payload_size=24)  # 3 * 8 bytes
                        send_fd(sock, my_fd, payload=my_metadata)

                    # Unpack peer's metadata
                    peer_base, peer_size, peer_heap = struct.unpack("QQQ", peer_metadata)

                    # Use context manager for peer handle and import the DMA-BUF
                    with managed_fd(peer_handle):
                        # Import peer's memory via DMA-BUF with proper offset correction
                        # peer_heap is where their heap starts (what they want us to use)
                        # peer_base is the base of their allocation buffer
                        # peer_size is the size of their allocation buffer
                        mapped_addr = import_dmabuf_handle(
                            peer_handle,
                            peer_size,  # Import the full base allocation
                            peer_heap,  # Original heap pointer (for offset calculation)
                            peer_base,  # Base of allocation (for offset calculation)
                        )
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

    def owns_tensor(self, tensor: torch.Tensor) -> bool:
        """
        Check if a tensor is within the allocator's managed heap.

        Args:
            tensor: PyTorch tensor to check

        Returns:
            True if tensor is within the heap, False otherwise
        """
        # Special case for empty tensors - they might not have a valid data_ptr
        if tensor.numel() == 0:
            return True

        ptr = int(tensor.data_ptr())
        heap_base = self.get_base_address()
        return ptr >= heap_base and ptr < heap_base + self.heap_size
