# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.

"""
PyTorch-based allocator for Iris symmetric heap.

Uses torch.empty() to allocate a large memory pool and manages
sub-allocations within it using bump allocation.
"""

import logging
import math
import numpy as np
import torch
from typing import Optional, Dict
import struct

from .base import BaseAllocator
from iris.host.logging.logging import _log_rank
from iris.host.platform.hip import export_dmabuf_handle, import_dmabuf_handle, destroy_external_memory
from iris.host.distributed.fd_passing import send_fd, recv_fd, managed_fd
from iris.host.platform.utils import is_simulation_env


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
        _log_rank(
            logging.INFO,
            "TorchAllocator: init heap_size=%.1fGB device=%d",
            heap_size / (1 << 30),
            device_id,
            rank=cur_rank,
            num_ranks=num_ranks,
        )
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
        self._buffer_registry: Dict[int, torch.Tensor] = {}
        self._buffer_registry_sizes: Dict[int, int] = {}
        self._buffer_ext_mem_handles: Dict[int, list] = {}
        self._fd_conns = None

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

        _log_rank(
            logging.DEBUG,
            "TorchAllocator.allocate: num_elements=%d dtype=%s size_bytes=%d offset=%d",
            num_elements,
            dtype,
            size_in_bytes,
            self.heap_offset,
            rank=self.cur_rank,
            num_ranks=self.num_ranks,
        )

        if self.heap_offset + aligned_size > self.heap_size:
            _log_rank(
                logging.ERROR,
                "TorchAllocator: OOM requested=%d available=%d",
                aligned_size,
                self.heap_size - self.heap_offset,
                rank=self.cur_rank,
                num_ranks=self.num_ranks,
            )
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
        for data_ptr in list(self._buffer_registry.keys()):
            self._unregister_buffer(data_ptr)

    def get_device(self) -> torch.device:
        """Get the torch device."""
        return self.memory_pool.device

    def import_external_tensor(self, external_tensor: torch.Tensor, force_copy: bool = False) -> torch.Tensor:
        """
        Make an external tensor accessible across ranks.

        When force_copy=False (default): exchanges IPC handles with all peers
        so each rank can read this tensor directly. No copy. The original tensor
        is returned and a per-rank pointer table is stored in the buffer registry.

        When force_copy=True: allocates on the heap and copies (legacy behavior).

        Args:
            external_tensor: External PyTorch tensor (must be CUDA, contiguous)
            force_copy: If True, use legacy copy-to-heap behavior

        Returns:
            The original tensor (IPC mode) or a heap copy (force_copy mode).
        """
        if not external_tensor.is_cuda:
            raise RuntimeError("Can only import CUDA tensors")
        if not external_tensor.is_contiguous():
            raise RuntimeError("Only contiguous tensors can be imported; call .contiguous() before as_symmetric()")

        if force_copy:
            num_elements = external_tensor.numel()
            dtype = external_tensor.dtype
            shape = external_tensor.shape
            heap_tensor = self.allocate(num_elements, dtype)
            heap_tensor = heap_tensor.reshape(shape).copy_(external_tensor)
            return heap_tensor

        ptr = external_tensor.data_ptr()
        size = external_tensor.numel() * external_tensor.element_size()

        if ptr in self._buffer_registry:
            # Guard against caching allocator address reuse
            if self._buffer_registry_sizes[ptr] != size:
                self._unregister_buffer(ptr)
            else:
                return external_tensor
        remote_ptrs = self._exchange_buffer_fds(ptr, size)

        input_bases = torch.tensor(remote_ptrs, dtype=torch.int64, device=external_tensor.device)
        self._buffer_registry[ptr] = input_bases
        self._buffer_registry_sizes[ptr] = size

        _log_rank(
            logging.INFO,
            "TorchAllocator: registered external buffer ptr=0x%x size=%d",
            ptr,
            size,
            rank=self.cur_rank,
            num_ranks=self.num_ranks,
        )

        return external_tensor

    def _exchange_buffer_fds(self, data_ptr: int, size: int) -> list:
        """
        Exchange dmabuf handles for a single buffer with all peers.

        Same pattern as establish_peer_access but for one tensor.
        Requires fd_conns to be set via set_fd_conns().

        Returns:
            List of mapped pointers indexed by rank.
        """
        if self._fd_conns is None:
            raise RuntimeError("fd_conns not set; call set_fd_conns() first")

        fd, base, export_size = export_dmabuf_handle(data_ptr, size)
        my_metadata = struct.pack("QQQ", base, export_size, data_ptr)

        remote_ptrs = [0] * self.num_ranks
        remote_ptrs[self.cur_rank] = data_ptr
        ext_mem_handles = []

        with managed_fd(fd):
            for peer, sock in self._fd_conns.items():
                if peer == self.cur_rank:
                    continue

                if self.cur_rank > peer:
                    send_fd(sock, fd, payload=my_metadata)
                    peer_fd, peer_metadata = recv_fd(sock, payload_size=24)
                else:
                    peer_fd, peer_metadata = recv_fd(sock, payload_size=24)
                    send_fd(sock, fd, payload=my_metadata)

                peer_base, peer_size, peer_data_ptr = struct.unpack("QQQ", peer_metadata)

                with managed_fd(peer_fd):
                    mapped_ptr, ext_mem_handle = import_dmabuf_handle(peer_fd, peer_size, peer_data_ptr, peer_base)
                    remote_ptrs[peer] = mapped_ptr
                    ext_mem_handles.append(ext_mem_handle)

        self._buffer_ext_mem_handles[data_ptr] = ext_mem_handles
        return remote_ptrs

    def _unregister_buffer(self, data_ptr: int):
        """Remove a buffer from the registry and clean up its handles."""
        self._buffer_registry.pop(data_ptr, None)
        self._buffer_registry_sizes.pop(data_ptr, None)
        handles = self._buffer_ext_mem_handles.pop(data_ptr, [])
        for handle in handles:
            try:
                destroy_external_memory(handle)
            except Exception:
                pass

    def set_fd_conns(self, fd_conns):
        """Store reference to fd connections for per-tensor IPC exchange."""
        self._fd_conns = fd_conns

    def get_remote_ptrs(self, tensor: torch.Tensor):
        """Look up per-rank pointer table for a registered tensor."""
        return self._buffer_registry.get(tensor.data_ptr())

    def is_registered(self, tensor: torch.Tensor) -> bool:
        """Check if a tensor has been registered for cross-rank access."""
        return tensor.data_ptr() in self._buffer_registry

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
