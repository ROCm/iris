# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.

"""
Symmetric heap abstraction for Iris.

Provides a high-level interface for distributed symmetric memory management,
hiding the details of allocators and inter-process memory sharing.
"""

import logging

import numpy as np
import torch
import os

from iris.host.logging.logging import _log_rank, logger
from iris.host.memory.allocators import TorchAllocator, VMemAllocator, VMemChunkedAllocator
from iris.host.distributed.fd_passing import setup_fd_infrastructure
from iris.host.distributed.helpers import distributed_allgather
from iris.host.platform.utils import is_simulation_env


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
        _log_rank(
            logging.INFO,
            "SymmetricHeap: init heap_size=%.1fGB device=%d allocator=%s",
            heap_size / (1 << 30),
            device_id,
            allocator_type,
            rank=cur_rank,
            num_ranks=num_ranks,
        )

        if is_simulation_env():
            allocator_type = "torch"

        if allocator_type == "torch":
            self.allocator = TorchAllocator(heap_size, device_id, cur_rank, num_ranks)
        elif allocator_type == "vmem":
            self.allocator = VMemAllocator(heap_size, device_id, cur_rank, num_ranks)
        elif allocator_type == "vmem_chunked":
            self.allocator = VMemChunkedAllocator(heap_size, device_id, cur_rank, num_ranks)
        else:
            raise ValueError(f"Unknown allocator type: {allocator_type}. Supported: 'torch', 'vmem', 'vmem_chunked'")

        self.fd_conns = setup_fd_infrastructure(cur_rank, num_ranks)
        device = self.allocator.get_device()

        # Use int64 instead of uint64 for gloo backend compatibility
        # Create from numpy array to avoid kernel issue (torch.zeros on small tensors triggers problematic kernel)
        heap_bases_array = np.zeros(self.num_ranks, dtype=np.int64)
        # Create on CPU first, then move to device to avoid FFM ioctl issue
        if is_simulation_env():
            self.heap_bases = torch.tensor(heap_bases_array, device="cpu", dtype=torch.int64)
            self.heap_bases = self.heap_bases.to(device)
        else:
            self.heap_bases = torch.tensor(heap_bases_array, device=device, dtype=torch.int64)

        self.refresh_peer_access()

    def close_fd_conns(self):
        """Close all FD-passing sockets.

        Called from ``iris.__del__`` to eagerly release UNIX domain sockets
        before the allocator is torn down.  Without this, a fast rank can
        unlink and rebind a socket path while a slow rank's old ``fd_conns``
        still reference the previous listener, causing stale-connection hangs
        on the next ``setup_fd_mesh`` call when iris is reconstructed.
        """
        if self.fd_conns is not None:
            for sock in self.fd_conns.values():
                try:
                    sock.close()
                except OSError:
                    pass
            self.fd_conns = None

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
        _log_rank(
            logging.DEBUG,
            "SymmetricHeap.allocate: num_elements=%d dtype=%s",
            num_elements,
            dtype,
            rank=self.cur_rank,
            num_ranks=self.num_ranks,
        )
        min_bytes = self.allocator.get_minimum_allocation_size()
        element_size = torch.tensor([], dtype=dtype).element_size()
        min_elements = max(1, (min_bytes + element_size - 1) // element_size)
        actual_elements = max(num_elements, min_elements)

        # For chunked allocator, track chunk count to detect growth
        is_chunked = isinstance(self.allocator, VMemChunkedAllocator)
        chunks_before = self.allocator.get_num_chunks() if is_chunked else 0

        tensor = self.allocator.allocate(actual_elements, dtype, alignment)
        tensor = tensor[:num_elements]

        if is_chunked:
            # Only refresh peer access if new chunks were grown
            chunks_after = self.allocator.get_num_chunks()
            if chunks_after > chunks_before:
                self.refresh_peer_access()
        else:
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

        _log_rank(
            logging.DEBUG,
            "refresh_peer_access: start rank=%d num_ranks=%d",
            self.cur_rank,
            self.num_ranks,
            rank=self.cur_rank,
            num_ranks=self.num_ranks,
        )

        if dist.is_initialized():
            dist.barrier()

        my_base = self.allocator.get_base_address()
        # Use int64 instead of uint64 to avoid gloo issues with all_gather_object
        local_base_arr = np.array([my_base], dtype=np.int64)
        all_bases_arr = distributed_allgather(local_base_arr).reshape(self.num_ranks).astype(np.int64)
        self.heap_bases[self.cur_rank] = int(all_bases_arr[self.cur_rank])

        if self.num_ranks == 1 or self.fd_conns is None:
            return

        # Dispatch to allocator-specific peer access path
        if isinstance(self.allocator, VMemChunkedAllocator):
            self._refresh_peer_access_chunked(dist)
        elif hasattr(self.allocator, "get_allocation_segments"):
            self._refresh_peer_access_segmented(dist)
        elif hasattr(self.allocator, "establish_peer_access"):
            self._refresh_peer_access_torch(dist, all_bases_arr)
        else:
            return

        if dist.is_initialized():
            dist.barrier()

    def _refresh_peer_access_torch(self, dist, all_bases_arr):
        """Peer access for TorchAllocator (IPC-based)."""
        from iris.util import is_simulation_env

        if is_simulation_env():
            for r in range(self.num_ranks):
                self.heap_bases[r] = int(all_bases_arr[r])
        else:
            all_bases = {r: int(all_bases_arr[r]) for r in range(self.num_ranks)}
            self.allocator.establish_peer_access(all_bases, self.fd_conns)
            for r in range(self.num_ranks):
                self.heap_bases[r] = int(self.allocator.heap_bases_array[r])

    def _refresh_peer_access_chunked(self, dist):
        """
        Peer access for VMemChunkedAllocator (chunk-based).

        Only exchanges NEW chunks that haven't been shared yet.
        Each chunk is exported via hipMemExportToShareableHandle (VMem handle export),
        not hipMemGetHandleForAddressRange (address range export).
        """
        from iris.fd_passing import send_fd, recv_fd
        from iris.hip import (
            mem_export_to_shareable_handle,
            mem_import_from_shareable_handle,
            mem_map,
            mem_set_access,
            mem_address_reserve,
            hipMemAccessDesc,
            hipMemLocationTypeDevice,
            hipMemAccessFlagsProtReadWrite,
        )

        chunks = self.allocator.get_allocation_chunks()

        # Track which chunks have been shared already
        if not hasattr(self, "_shared_chunk_count"):
            self._shared_chunk_count = 0

        new_chunks = chunks[self._shared_chunk_count :]
        if not new_chunks and self._shared_chunk_count > 0:
            # Nothing new to share, but ensure heap_bases are set
            return

        # Export new chunk handles as DMA-BUF fds
        new_fds = []
        for chunk_idx, offset, size, handle in new_chunks:
            fd = mem_export_to_shareable_handle(handle)
            new_fds.append((fd, offset, size))

        access_desc = hipMemAccessDesc()
        access_desc.location.type = hipMemLocationTypeDevice
        access_desc.location.id = self.device_id
        access_desc.flags = hipMemAccessFlagsProtReadWrite

        for peer, sock in self.fd_conns.items():
            if peer == self.cur_rank:
                continue

            if not hasattr(self, "_peer_va_ranges"):
                self._peer_va_ranges = {}
            if not hasattr(self, "_peer_imported_mappings"):
                self._peer_imported_mappings = {}

            if peer not in self._peer_va_ranges:
                # Reserve VA for this peer's chunks -- same size as our VA
                peer_va_base = mem_address_reserve(self.allocator.va_size, self.allocator.granularity, 0)
                self._peer_va_ranges[peer] = peer_va_base
            if peer not in self._peer_imported_mappings:
                self._peer_imported_mappings[peer] = []
            peer_va_base = self._peer_va_ranges[peer]

            # Exchange new chunk fds
            for my_fd, my_offset, my_size in new_fds:
                if self.cur_rank > peer:
                    send_fd(sock, my_fd)
                    peer_fd, _ = recv_fd(sock)
                else:
                    peer_fd, _ = recv_fd(sock)
                    send_fd(sock, my_fd)

                # Import and map peer's chunk
                imported_handle = mem_import_from_shareable_handle(peer_fd)
                os.close(peer_fd)

                peer_va = peer_va_base + my_offset
                mem_map(peer_va, my_size, 0, imported_handle)
                mem_set_access(peer_va, my_size, access_desc)

                # Track for cleanup: must unmap + release before freeing VA
                self._peer_imported_mappings[peer].append((imported_handle, peer_va, my_size))

            self.heap_bases[peer] = peer_va_base

        # Close our exported fds
        for fd, _, _ in new_fds:
            os.close(fd)

        self._shared_chunk_count = len(chunks)

    def _refresh_peer_access_segmented(self, dist):
        """Peer access for VMemAllocator (segment-based, legacy)."""
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
            _log_rank(
                logging.DEBUG,
                "refresh_peer_access: peer=%d segments=%d",
                peer,
                len(my_exported_fds),
                rank=self.cur_rank,
                num_ranks=self.num_ranks,
            )

            if not hasattr(self, "_peer_va_ranges"):
                self._peer_va_ranges = {}

            if peer not in self._peer_va_ranges:
                peer_va_base = mem_address_reserve(self.heap_size, self.allocator.granularity, 0)
                self._peer_va_ranges[peer] = peer_va_base
            else:
                peer_va_base = self._peer_va_ranges[peer]

            peer_fds = []
            for seg_idx, (my_fd, my_size, my_offset) in enumerate(my_exported_fds):
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
            if not hasattr(self, "_peer_imported_mappings"):
                self._peer_imported_mappings = {}
            if peer not in self._peer_imported_mappings:
                self._peer_imported_mappings[peer] = []

            for peer_fd, segment_size, offset in peer_fds:
                segment_key = (offset, segment_size)
                if segment_key in self._peer_imported_segments[peer]:
                    os.close(peer_fd)
                    continue

                imported_handle = mem_import_from_shareable_handle(peer_fd)
                os.close(peer_fd)

                peer_va = peer_va_base + offset
                mem_map(peer_va, segment_size, 0, imported_handle)
                self._peer_imported_segments[peer].add(segment_key)

                # Track for cleanup
                self._peer_imported_mappings[peer].append((imported_handle, peer_va, segment_size))

                new_cumulative = offset + segment_size
                if new_cumulative > cumulative_size:
                    cumulative_size = new_cumulative
                    mem_set_access(peer_va_base, cumulative_size, access_desc)

            self._peer_cumulative_sizes[peer] = cumulative_size
            self.heap_bases[peer] = peer_va_base

        for fd, _, _ in my_exported_fds:
            os.close(fd)

        if logger.isEnabledFor(logging.DEBUG):
            _log_rank(
                logging.DEBUG,
                "refresh_peer_access: done heap_bases=[%s]",
                ", ".join(f"0x{int(self.heap_bases[r].item()):x}" for r in range(min(3, self.num_ranks))),
                rank=self.cur_rank,
                num_ranks=self.num_ranks,
            )

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

    def close(self):
        """Release resources held by the symmetric heap."""
        # Synchronize GPU to ensure all in-flight kernels complete before
        # unmapping memory.  Without this, async operations (NCCL collectives,
        # .zero_(), .fill_()) may still reference mapped addresses, causing
        # hipErrorUnknown when the VA ranges are freed.
        try:
            import torch

            if hasattr(self, "device_id"):
                torch.cuda.synchronize(self.device_id)
        except Exception:
            pass

        # Unmap and release peer-imported chunks BEFORE freeing VA ranges.
        # mem_address_free on a VA range with active mappings corrupts the
        # HIP runtime, causing hipErrorUnknown on subsequent GPU operations.
        if hasattr(self, "_peer_imported_mappings"):
            try:
                from iris.hip import mem_unmap, mem_release

                for peer, mappings in self._peer_imported_mappings.items():
                    for handle, va, size in mappings:
                        try:
                            mem_unmap(va, size)
                        except Exception:
                            pass
                        try:
                            mem_release(handle)
                        except Exception:
                            pass
                self._peer_imported_mappings.clear()
            except ImportError:
                pass

        # Free peer VA ranges (now safe -- all mappings have been removed)
        if hasattr(self, "_peer_va_ranges"):
            try:
                from iris.hip import mem_address_free

                for peer, va_base in self._peer_va_ranges.items():
                    try:
                        va_size = getattr(self.allocator, "va_size", self.heap_size)
                        mem_address_free(va_base, va_size)
                    except Exception:
                        pass
                self._peer_va_ranges.clear()
            except ImportError:
                pass

        # Close the allocator
        if hasattr(self, "allocator") and hasattr(self.allocator, "close"):
            self.allocator.close()

        # Close FD sockets
        if hasattr(self, "fd_conns") and self.fd_conns:
            for peer, sock in self.fd_conns.items():
                try:
                    sock.close()
                except Exception:
                    pass
            self.fd_conns = None

    def __del__(self):
        try:
            self.close()
        except Exception:
            pass
