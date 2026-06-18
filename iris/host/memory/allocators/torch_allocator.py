# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.

"""
PyTorch-based allocator for Iris symmetric heap.

Uses torch.empty() to allocate a large memory pool and manages
sub-allocations within it using bump allocation.
"""

import ctypes
import ctypes.util
import logging
import math
import mmap
import numpy as np
import os
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
        self._shm_mmap = None
        self._shm_fd = None
        self._shm_name = None

        if is_simulation_env():
            self._shm_name = f"/iris-sim-heap-{os.environ.get('SLURM_JOB_ID', os.getppid())}"
            total_size = heap_size * num_ranks
            librt = ctypes.CDLL(ctypes.util.find_library("rt") or "librt.so.1", use_errno=True)
            librt.shm_open.argtypes = [ctypes.c_char_p, ctypes.c_int, ctypes.c_uint]
            librt.shm_open.restype = ctypes.c_int
            self._librt = librt

            if cur_rank == 0:
                librt.shm_unlink(self._shm_name.encode())
                fd = librt.shm_open(self._shm_name.encode(), os.O_CREAT | os.O_RDWR, 0o600)
                if fd < 0:
                    raise OSError(ctypes.get_errno(), f"shm_open create failed: {os.strerror(ctypes.get_errno())}")
                os.ftruncate(fd, total_size)
            else:
                for _ in range(100):
                    fd = librt.shm_open(self._shm_name.encode(), os.O_RDWR, 0)
                    if fd >= 0:
                        break
                    import time

                    time.sleep(0.05)
                else:
                    raise OSError(f"shm_open failed: rank 0 never created {self._shm_name}")

            self._shm_fd = fd
            self._shm_mmap = mmap.mmap(fd, total_size, mmap.MAP_SHARED, mmap.PROT_READ | mmap.PROT_WRITE)

            my_offset = cur_rank * heap_size
            buf = (ctypes.c_int8 * heap_size).from_buffer(self._shm_mmap, my_offset)
            np_arr = np.ctypeslib.as_array(buf)
            self.memory_pool = torch.from_numpy(np_arr)

            _log_rank(
                logging.INFO,
                "TorchAllocator: sim shm_open %s, rank %d slice at offset %d",
                self._shm_name,
                cur_rank,
                my_offset,
                rank=cur_rank,
                num_ranks=num_ranks,
            )
        else:
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

        if is_simulation_env() and self._shm_mmap is not None:
            self._shm_peer_views = []
            for rank in range(self.num_ranks):
                peer_offset = rank * self.heap_size
                peer_buf = (ctypes.c_int8 * self.heap_size).from_buffer(self._shm_mmap, peer_offset)
                peer_np = np.ctypeslib.as_array(peer_buf)
                peer_tensor = torch.from_numpy(peer_np)
                self._shm_peer_views.append(peer_tensor)
                heap_bases_array[rank] = peer_tensor.data_ptr()
            self.heap_bases_array = heap_bases_array
            _log_rank(
                logging.INFO,
                "TorchAllocator: sim peer access via shm_open, %d ranks",
                self.num_ranks,
                rank=self.cur_rank,
                num_ranks=self.num_ranks,
            )
            return

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
        """Release peer external memory handles and shm resources."""
        for handle in self._peer_ext_mem_handles.values():
            try:
                destroy_external_memory(handle)
            except Exception:
                pass
        self._peer_ext_mem_handles.clear()

        if self._shm_mmap is not None:
            try:
                self._shm_mmap.close()
            except Exception:
                pass
            self._shm_mmap = None
        if self._shm_fd is not None:
            try:
                os.close(self._shm_fd)
            except Exception:
                pass
            self._shm_fd = None
        if self._shm_name is not None and self.cur_rank == 0:
            try:
                self._librt.shm_unlink(self._shm_name.encode())
            except Exception:
                pass

    def get_device(self) -> torch.device:
        """Get the torch device."""
        if is_simulation_env():
            return torch.device(self.device)
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
