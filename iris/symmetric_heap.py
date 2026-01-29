# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.

"""
Symmetric heap abstraction for Iris.

Provides a high-level interface for distributed symmetric memory management,
hiding the details of allocators and inter-process memory sharing.
"""

import numpy as np
import torch

from iris.allocators import TorchAllocator
from iris.fd_passing import setup_fd_infrastructure
from iris._distributed_helpers import distributed_allgather


class SymmetricHeap:
    """
    High-level symmetric heap abstraction.

    Manages distributed memory with symmetric addressing across ranks,
    handling all allocator coordination and memory sharing internally.
    """

    def __init__(self, heap_size: int, device_id: int, cur_rank: int, num_ranks: int):
        """
        Initialize symmetric heap.

        Args:
            heap_size: Size of the heap in bytes
            device_id: GPU device ID
            cur_rank: Current process rank
            num_ranks: Total number of ranks
        """
        self.heap_size = heap_size
        self.device_id = device_id
        self.cur_rank = cur_rank
        self.num_ranks = num_ranks

        # Create allocator
        self.allocator = TorchAllocator(heap_size, device_id, cur_rank, num_ranks)

        # All-gather heap bases for pointer translation
        heap_base = self.allocator.get_base_address()
        local_base_arr = np.array([heap_base], dtype=np.uint64)
        all_bases_arr = distributed_allgather(local_base_arr).reshape(num_ranks).astype(np.uint64)
        all_bases = {rank: int(all_bases_arr[rank]) for rank in range(num_ranks)}

        # Setup FD passing infrastructure
        fd_conns = setup_fd_infrastructure(cur_rank, num_ranks)

        # Establish access to peer memory
        self.allocator.establish_peer_access(all_bases, fd_conns)

        # Get final heap bases
        self.heap_bases = self.allocator.get_heap_bases()

    def allocate(self, num_elements: int, dtype: torch.dtype) -> torch.Tensor:
        """
        Allocate a tensor on the symmetric heap.

        Args:
            num_elements: Number of elements to allocate
            dtype: PyTorch data type

        Returns:
            Allocated tensor on the symmetric heap
        """
        return self.allocator.allocate(num_elements, dtype)

    def get_device(self) -> torch.device:
        """Get the torch device for this heap."""
        return self.allocator.get_device()

    def on_symmetric_heap(self, tensor: torch.Tensor) -> bool:
        """
        Check if a tensor is allocated on the symmetric heap.

        Args:
            tensor: PyTorch tensor to check

        Returns:
            True if tensor is on the symmetric heap, False otherwise
        """
        # Delegate to allocator to check if tensor is in heap
        return self.allocator.owns_tensor(tensor)

    def get_heap_bases(self) -> torch.Tensor:
        """Get heap base addresses for all ranks as a tensor."""
        return self.heap_bases
