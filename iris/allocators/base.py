# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.

"""
Base allocator interface for Iris symmetric heap management.
"""

from abc import ABC, abstractmethod
from typing import Optional, Dict, Any
import torch


class BaseAllocator(ABC):
    """
    Abstract base class for Iris memory allocators.

    Allocators manage GPU memory for the symmetric heap and handle
    inter-process memory sharing.
    """

    def __init__(self, heap_size: int, device_id: int, cur_rank: int, num_ranks: int):
        """
        Initialize the allocator.

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
        self.heap_offset = 0

    @abstractmethod
    def get_base_address(self) -> int:
        """
        Get the base address of the heap.

        Returns:
            Integer pointer to the base of the heap
        """
        pass

    @abstractmethod
    def allocate(self, num_elements: int, dtype: torch.dtype, alignment: int = 1024) -> torch.Tensor:
        """
        Allocate a tensor on the symmetric heap.

        Args:
            num_elements: Number of elements to allocate
            dtype: PyTorch data type
            alignment: Memory alignment in bytes (default: 1024)

        Returns:
            Allocated tensor on the symmetric heap
        """
        pass

    @abstractmethod
    def get_shareable_handle(self) -> Any:
        """
        Get a shareable handle for inter-process communication.

        Returns:
            Shareable handle (implementation-specific: FD, IPC handle, etc.)
        """
        pass

    @abstractmethod
    def establish_peer_access(self, all_bases: Dict[int, int], connections: Optional[Dict] = None):
        """
        Establish access to peer memory for symmetric addressing.

        Args:
            all_bases: Dictionary mapping rank -> base address
            connections: Optional peer connections for handle exchange
        """
        pass

    @abstractmethod
    def get_device(self) -> torch.device:
        """
        Get the torch device for this allocator.

        Returns:
            PyTorch device object
        """
        pass

    @abstractmethod
    def get_heap_bases(self) -> torch.Tensor:
        """
        Get heap base addresses for all ranks as a tensor.

        Returns:
            Tensor of shape (num_ranks,) with base addresses
        """
        pass
