# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.

"""
Symmetric heap abstraction for Iris.

Provides a high-level interface for distributed symmetric memory management,
hiding the details of allocators and inter-process memory sharing.
"""

import numpy as np
import torch
import os

from iris.allocators import TorchAllocator, VMemAllocator
from iris.fd_passing import setup_fd_infrastructure
from iris._distributed_helpers import distributed_allgather


class SymmetricHeap:
    """
    High-level symmetric heap abstraction.

    Manages distributed memory with symmetric addressing across ranks,
    handling all allocator coordination and memory sharing internally.

    Supports multiple allocator backends: 'torch' (default) and 'vmem'.
    """

    def __init__(
        self,
        heap_size: int,
        device_id: int,
        cur_rank: int,
        num_ranks: int,
        allocator_type: str = "torch",
    ):
        """
        Initialize symmetric heap.

        Args:
            heap_size: Size of the heap in bytes
            device_id: GPU device ID
            cur_rank: Current process rank
            num_ranks: Total number of ranks
            allocator_type: Type of allocator ("torch" or "vmem"); default "torch"

        Raises:
            ValueError: If allocator_type is not supported
        """
        self.heap_size = heap_size
        self.device_id = device_id
        self.cur_rank = cur_rank
        self.num_ranks = num_ranks
        allocator_type = os.environ.get("IRIS_ALLOCATOR", allocator_type).lower()

        if allocator_type == "torch":
            self.allocator = TorchAllocator(heap_size, device_id, cur_rank, num_ranks)
        elif allocator_type == "vmem":
            self.allocator = VMemAllocator(heap_size, device_id, cur_rank, num_ranks)
        else:
            raise ValueError(f"Unknown allocator type: {allocator_type}. Supported: 'torch', 'vmem'")

        self.fd_conns = setup_fd_infrastructure(cur_rank, num_ranks)
        device = self.allocator.get_device()
        self.heap_bases = torch.zeros(num_ranks, dtype=torch.uint64, device=device)
        self.refresh_peer_access()

    def allocate(self, num_elements: int, dtype: torch.dtype, alignment: int = 1024) -> torch.Tensor:
        """
        Allocate a tensor on the symmetric heap.

        Always allocates at least the allocator's minimum allocation size so that
        even zero-element requests get a buffer on the heap; for num_elements==0
        we return a zero-length slice of that buffer so the tensor is still on heap.

        Args:
            num_elements: Number of elements to allocate
            dtype: PyTorch data type
            alignment: Alignment requirement in bytes (default: 1024)

        Returns:
            Allocated tensor on the symmetric heap (shape (num_elements,) or (0,) for empty)

        Note:
            This should be called collectively across all ranks to maintain
            symmetric heap consistency. After allocation, peer access is refreshed.
        """
        min_bytes = self.allocator.get_minimum_allocation_size()
        element_size = torch.tensor([], dtype=dtype).element_size()
        min_elements = max(1, (min_bytes + element_size - 1) // element_size)
        actual_elements = max(num_elements, min_elements)
        tensor = self.allocator.allocate(actual_elements, dtype, alignment)
        tensor = tensor[:num_elements]
        self.refresh_peer_access()
        return tensor

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
        return self.allocator.owns_tensor(tensor)

    def is_symmetric(self, tensor: torch.Tensor) -> bool:
        """
        Check if a tensor is allocated on the symmetric heap.

        This method provides a public API to check whether a tensor resides in the
        symmetric heap, making it accessible for RMA operations across ranks.

        Args:
            tensor: PyTorch tensor to check

        Returns:
            True if tensor is on the symmetric heap, False otherwise

        Example:
            >>> ctx = iris.iris(heap_size=2**30)
            >>> symmetric_tensor = ctx.zeros(1000, dtype=torch.float32)
            >>> external_tensor = torch.zeros(1000, dtype=torch.float32, device='cuda')
            >>> ctx.heap.is_symmetric(symmetric_tensor)  # True
            >>> ctx.heap.is_symmetric(external_tensor)   # False
        """
        return self.on_symmetric_heap(tensor)

    def get_heap_bases(self) -> torch.Tensor:
        """Get heap base addresses for all ranks as a tensor."""
        return self.heap_bases

    def refresh_peer_access(self):
        """
        Refresh peer DMA-BUF imports using segmented export/import.
        Collective: all ranks must call together. Do not cache heap_bases.
        """
        import torch.distributed as dist
        from iris.fd_passing import send_fd, recv_fd
        from iris.hip import (
            export_dmabuf_handle,
            mem_import_from_shareable_handle,
            mem_map,
            mem_set_access,
            mem_address_reserve,
            hipMemAccessDesc,
            hipMemLocationTypeDevice,
            hipMemAccessFlagsProtReadWrite,
        )

        if dist.is_initialized():
            dist.barrier()

        my_base = self.allocator.get_base_address()
        local_base_arr = np.array([my_base], dtype=np.uint64)
        all_bases_arr = distributed_allgather(local_base_arr).reshape(self.num_ranks).astype(np.uint64)
        self.heap_bases[self.cur_rank] = int(all_bases_arr[self.cur_rank])

        if self.num_ranks == 1 or self.fd_conns is None:
            return

        if not hasattr(self.allocator, "get_allocation_segments"):
            if hasattr(self.allocator, "establish_peer_access"):
                all_bases = {r: int(all_bases_arr[r]) for r in range(self.num_ranks)}
                self.allocator.establish_peer_access(all_bases, self.fd_conns)
                for r in range(self.num_ranks):
                    self.heap_bases[r] = int(self.allocator.heap_bases_array[r])
            return

        my_segments = self.allocator.get_allocation_segments()
        my_exported_fds = []
        for offset, size, va in my_segments:
            dmabuf_fd, export_base, export_size = export_dmabuf_handle(va, size)
            my_exported_fds.append((dmabuf_fd, export_size, offset))

        access_desc = hipMemAccessDesc()
        access_desc.location.type = hipMemLocationTypeDevice
        access_desc.location.id = self.device_id
        access_desc.flags = hipMemAccessFlagsProtReadWrite

        for peer, sock in self.fd_conns.items():
            if peer == self.cur_rank:
                continue

            if not hasattr(self, "_peer_va_ranges"):
                self._peer_va_ranges = {}

            if peer not in self._peer_va_ranges:
                peer_va_base = mem_address_reserve(self.heap_size, self.allocator.granularity, 0)
                self._peer_va_ranges[peer] = peer_va_base
            else:
                peer_va_base = self._peer_va_ranges[peer]

            peer_fds = []
            for seg_idx, (my_fd, my_size, my_offset) in enumerate(my_exported_fds):
                # Exchange FDs (higher rank sends first to avoid deadlock)
                if self.cur_rank > peer:
                    send_fd(sock, my_fd)
                    peer_fd, _ = recv_fd(sock)
                else:
                    peer_fd, _ = recv_fd(sock)
                    send_fd(sock, my_fd)

                peer_fds.append((peer_fd, my_size, my_offset))

            if not hasattr(self, "_peer_cumulative_sizes"):
                self._peer_cumulative_sizes = {}
            cumulative_size = self._peer_cumulative_sizes.get(peer, 0)

            if not hasattr(self, "_peer_imported_segments"):
                self._peer_imported_segments = {}
            if peer not in self._peer_imported_segments:
                self._peer_imported_segments[peer] = set()

            for peer_fd, segment_size, offset in peer_fds:
                segment_key = (offset, segment_size)
                if segment_key in self._peer_imported_segments[peer]:
                    import os

                    os.close(peer_fd)
                    continue

                imported_handle = mem_import_from_shareable_handle(peer_fd)
                import os

                os.close(peer_fd)

                peer_va = peer_va_base + offset
                mem_map(peer_va, segment_size, 0, imported_handle)
                self._peer_imported_segments[peer].add(segment_key)

                new_cumulative = offset + segment_size
                if new_cumulative > cumulative_size:
                    cumulative_size = new_cumulative
                    mem_set_access(peer_va_base, cumulative_size, access_desc)

            self._peer_cumulative_sizes[peer] = cumulative_size
            self.heap_bases[peer] = peer_va_base

        for fd, _, _ in my_exported_fds:
            import os

            os.close(fd)

        if dist.is_initialized():
            dist.barrier()

    def as_symmetric(self, external_tensor: torch.Tensor) -> torch.Tensor:
        """
        Place an external PyTorch tensor on the symmetric heap.

        With the torch allocator: allocates on the heap and copies the data;
        the returned tensor is independent of the input. With the vmem
        allocator: imports the memory so both tensors share the same storage.

        Args:
            external_tensor: External PyTorch tensor (must be CUDA, contiguous)

        Returns:
            Tensor on the symmetric heap (same shape/dtype; copy or shared per allocator)

        Raises:
            RuntimeError: If allocator doesn't support imports or import fails
        """
        if not hasattr(self.allocator, "import_external_tensor"):
            raise RuntimeError(f"{type(self.allocator).__name__} does not support as_symmetric().")

        imported = self.allocator.import_external_tensor(external_tensor)
        self.refresh_peer_access()
        return imported
