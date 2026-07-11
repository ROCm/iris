# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.

"""
Symmetric heap abstraction for Iris.

Provides a high-level interface for distributed symmetric memory management,
hiding the details of allocators and inter-process memory sharing.
"""

import logging
import os
import struct

import numpy as np
import torch

from iris.host.logging.logging import _log_rank
from iris.host.memory.allocators import TorchAllocator, VMemAllocator, VMemChunkedAllocator
from iris.drivers.base import MappingPlacement, PeerMapping
from iris.host.distributed.fd_passing import setup_fd_infrastructure
from iris.host.distributed.helpers import distributed_allgather
from iris.host.platform.utils import is_simulation_env

logger = logging.getLogger("iris.host.memory.symmetric_heap")

# Layout of LocalHipDriver handle bytes: native int FD, then uint64 offset,
# then uint64 base_size. See LocalHipDriver.export_handle in
# iris/drivers/local/amd.py.
_LOCAL_HIP_HANDLE_FMT = "=iQQ"
_LOCAL_HIP_HANDLE_BYTES = struct.calcsize(_LOCAL_HIP_HANDLE_FMT)
_LOCAL_CUDA_HANDLE_FMT = "=i"
_LOCAL_CUDA_HANDLE_BYTES = struct.calcsize(_LOCAL_CUDA_HANDLE_FMT)


def _extract_fd_from_local_handle(handle_bytes: bytes) -> int:
    """Extract the FD from a local-driver handle byte string."""
    if len(handle_bytes) == _LOCAL_HIP_HANDLE_BYTES:
        fd, _offset, _base_size = struct.unpack(_LOCAL_HIP_HANDLE_FMT, handle_bytes)
        return fd
    if len(handle_bytes) == _LOCAL_CUDA_HANDLE_BYTES:
        (fd,) = struct.unpack(_LOCAL_CUDA_HANDLE_FMT, handle_bytes)
        return fd
    raise RuntimeError(
        "Unsupported local driver handle size: "
        f"expected {_LOCAL_HIP_HANDLE_BYTES} (HIP) or {_LOCAL_CUDA_HANDLE_BYTES} "
        f"(CUDA), got {len(handle_bytes)}"
    )


def _replace_fd_in_local_handle(handle_bytes: bytes, new_fd: int) -> bytes:
    """Return a copy of handle_bytes with the FD field replaced."""
    if len(handle_bytes) == _LOCAL_HIP_HANDLE_BYTES:
        _old_fd, offset, base_size = struct.unpack(_LOCAL_HIP_HANDLE_FMT, handle_bytes)
        return struct.pack(_LOCAL_HIP_HANDLE_FMT, new_fd, offset, base_size)
    if len(handle_bytes) == _LOCAL_CUDA_HANDLE_BYTES:
        return struct.pack(_LOCAL_CUDA_HANDLE_FMT, new_fd)
    raise RuntimeError(
        "Unsupported local driver handle size: "
        f"expected {_LOCAL_HIP_HANDLE_BYTES} (HIP) or {_LOCAL_CUDA_HANDLE_BYTES} "
        f"(CUDA), got {len(handle_bytes)}"
    )


def _validate_local_handle(handle_bytes: bytes) -> None:
    """Validate that handle_bytes matches a supported local-driver layout."""
    if len(handle_bytes) not in (
        _LOCAL_HIP_HANDLE_BYTES,
        _LOCAL_CUDA_HANDLE_BYTES,
    ):
        raise RuntimeError(
            "Unsupported local driver handle size: "
            f"expected {_LOCAL_HIP_HANDLE_BYTES} (HIP) or {_LOCAL_CUDA_HANDLE_BYTES} "
            f"(CUDA), got {len(handle_bytes)}"
        )


class SymmetricHeap:
    """
    High-level symmetric heap abstraction.

    Manages distributed memory with symmetric addressing across ranks,
    handling all allocator coordination and memory sharing internally.

    Supports multiple allocator backends: 'torch' (default), 'vmem', and
    'vmem_chunked'.
    """

    _PEER_REFRESH_FAILURE_MESSAGE = (
        "SymmetricHeap peer refresh failed unrecoverably. "
        "Peer refresh is intentionally fatal because partial peer mappings "
        "or VA reservations may remain; destroy the Iris context or restart "
        "the process instead of retrying."
    )

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
            allocator_type: Type of allocator ("torch", "vmem", or
                "vmem_chunked"); default "torch"

        Raises:
            ValueError: If allocator_type is not supported
        """
        self.heap_size = heap_size
        self.device_id = device_id
        self.cur_rank = cur_rank
        self.num_ranks = num_ranks
        self._peer_va_ranges = {}
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
            from iris.host.distributed.topology import TopologyDiscovery

            try:
                topology = TopologyDiscovery().discover()
            except Exception as exc:
                logger.warning(
                    "TopologyDiscovery.discover() failed (%s); VMemChunkedAllocator will default to INTRA_NODE driver.",
                    exc,
                )
                topology = None
            self.allocator = VMemChunkedAllocator(
                heap_size,
                device_id,
                cur_rank,
                num_ranks,
                topology=topology,
            )
        else:
            raise ValueError(f"Unknown allocator type: {allocator_type}. Supported: 'torch', 'vmem', 'vmem_chunked'")

        needs_fd_infra = True
        if isinstance(self.allocator, VMemChunkedAllocator):
            from iris.host.distributed.topology import InterconnectLevel

            if self.allocator.transport_tier == InterconnectLevel.INTRA_RACK_FABRIC:
                needs_fd_infra = False

        self.fd_conns = setup_fd_infrastructure(cur_rank, num_ranks) if needs_fd_infra else None
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

        self._peer_refresh_failed = False
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

    def _ensure_peer_refresh_healthy(self) -> None:
        if self._peer_refresh_failed:
            raise RuntimeError(self._PEER_REFRESH_FAILURE_MESSAGE)

    def _peer_va_size(self) -> int:
        if isinstance(self.allocator, (VMemAllocator, VMemChunkedAllocator)):
            return self.allocator.va_size
        return self.heap_size

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

        self._ensure_peer_refresh_healthy()

        min_bytes = self.allocator.get_minimum_allocation_size()
        element_size = torch.tensor([], dtype=dtype).element_size()
        min_elements = max(1, (min_bytes + element_size - 1) // element_size)
        actual_elements = max(num_elements, min_elements)

        is_chunked = isinstance(self.allocator, VMemChunkedAllocator)
        chunks_before = self.allocator.get_num_chunks() if is_chunked else 0

        tensor = self.allocator.allocate(actual_elements, dtype, alignment)
        tensor = tensor[:num_elements]

        if is_chunked:
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

        Returns True for any tensor whose memory falls inside either:
        - the local allocator's own heap (delegates to allocator.owns_tensor), or
        - any peer-VA region the symmetric heap has imported via
          _refresh_peer_access_chunked / _refresh_peer_access_fabric.

        The peer-region check is needed because the symmetric heap calls
        driver.import_and_map directly (rather than through the allocator's
        import_peer_chunk method), so peer mappings are tracked on the
        symmetric heap, not the allocator. Without this check, peer-imported
        tensors would incorrectly report as NOT on the symmetric heap.

        Args:
            tensor: PyTorch tensor to check

        Returns:
            True if tensor is on the symmetric heap, False otherwise
        """
        if self.allocator.owns_tensor(tensor):
            return True

        if not tensor.is_cuda or tensor.numel() == 0:
            return False

        if not self._peer_va_ranges:
            return False

        ptr = tensor.data_ptr()
        va_size = self._peer_va_size()
        for peer_va_base in self._peer_va_ranges.values():
            if peer_va_base <= ptr < peer_va_base + va_size:
                return True
        return False

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
        self._ensure_peer_refresh_healthy()
        return self.heap_bases

    def refresh_peer_access(self):
        """
        Refresh peer imports for the active allocator backend.
        Collective: all ranks must call together. Do not cache heap_bases.

        Failure policy: refresh is not recoverable. The low-level refresh paths
        are intentionally not transactional; if any error is raised, partial
        imports or VA reservations may remain. This method marks the heap as
        failed and rejects future heap operations instead of allowing retries
        against possibly-stale mapping state.
        """
        self._ensure_peer_refresh_healthy()
        try:
            self._refresh_peer_access_impl()
        except Exception as exc:
            self._peer_refresh_failed = True
            logger.critical(self._PEER_REFRESH_FAILURE_MESSAGE, exc_info=True)
            raise RuntimeError(self._PEER_REFRESH_FAILURE_MESSAGE) from exc

    def _refresh_peer_access_impl(self):
        """Implementation for refresh_peer_access; caller owns fatal handling."""
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

        if self.num_ranks == 1:
            return

        # Dispatch to allocator-specific peer access path
        if isinstance(self.allocator, VMemChunkedAllocator):
            from iris.host.distributed.topology import InterconnectLevel

            if self.allocator.transport_tier == InterconnectLevel.INTRA_RACK_FABRIC:
                self._refresh_peer_access_fabric(dist)
            else:
                if self.fd_conns is None:
                    return
                self._refresh_peer_access_chunked(dist)
        elif hasattr(self.allocator, "get_allocation_segments"):
            if self.fd_conns is None:
                return
            self._refresh_peer_access_segmented(dist)
        elif hasattr(self.allocator, "establish_peer_access"):
            if self.fd_conns is None:
                return
            self._refresh_peer_access_torch(dist, all_bases_arr)
        else:
            return

        if dist.is_initialized():
            dist.barrier()

    def _refresh_peer_access_torch(self, dist, all_bases_arr):
        """Peer access for TorchAllocator (IPC-based)."""
        from iris.host.platform.utils import is_simulation_env

        if is_simulation_env():
            for r in range(self.num_ranks):
                self.heap_bases[r] = int(all_bases_arr[r])
        else:
            all_bases = {r: int(all_bases_arr[r]) for r in range(self.num_ranks)}
            self.allocator.establish_peer_access(all_bases, self.fd_conns)
            for r in range(self.num_ranks):
                self.heap_bases[r] = int(self.allocator.heap_bases_array[r])

    def _get_collective_device(self, dist) -> torch.device:
        """Choose a safe device for small distributed tensor collectives."""
        if not dist.is_initialized():
            return self.allocator.get_device()
        try:
            backend = str(dist.get_backend()).lower()
        except Exception:
            backend = "gloo"
        if backend == "nccl":
            return self.allocator.get_device()
        return torch.device("cpu")

    def _ensure_chunk_tracking_state(self) -> None:
        if not hasattr(self, "_shared_chunk_counts"):
            self._shared_chunk_counts = [0] * self.num_ranks
        if not hasattr(self, "_peer_va_ranges"):
            self._peer_va_ranges = {}
        if not hasattr(self, "_peer_imported_mappings"):
            self._peer_imported_mappings = {}

    def _gather_total_chunk_counts(self, dist, local_total: int) -> list[int]:
        count_device = self._get_collective_device(dist)
        local_total_tensor = torch.tensor([local_total], dtype=torch.int64, device=count_device)
        gathered_counts = [torch.zeros(1, dtype=torch.int64, device=count_device) for _ in range(self.num_ranks)]
        dist.all_gather(gathered_counts, local_total_tensor)
        return [int(count.item()) for count in gathered_counts]

    def _refresh_peer_access_chunked(self, dist):
        """
        Peer access for VMemChunkedAllocator on local-host (FD-based) transports.

        Uses Unix-domain sockets for FD passing, but routes export/import
        through the driver abstraction. The handle-byte layout is driver-
        specific; this method only extracts and replaces the embedded FD
        field for socket transport.

        Supports both LocalHipDriver's 20-byte DMA-BUF layout and
        LocalCudaDriver's 4-byte POSIX-FD layout.
        """
        from iris.host.distributed.fd_passing import recv_fd, send_fd

        self._ensure_chunk_tracking_state()

        my_total = self.allocator.get_num_chunks()
        total_chunks_per_rank = self._gather_total_chunk_counts(dist, my_total)
        my_total = total_chunks_per_rank[self.cur_rank]
        my_already_shared = self._shared_chunk_counts[self.cur_rank]
        new_chunk_records = self.allocator.get_allocation_chunks_since(my_already_shared)

        my_metadata = [(offset, size, handle_bytes) for (_idx, offset, size, handle_bytes) in new_chunk_records]
        gathered_metadata = [None] * self.num_ranks
        dist.all_gather_object(gathered_metadata, my_metadata)

        new_fds = [_extract_fd_from_local_handle(handle_bytes) for (_offset, _size, handle_bytes) in my_metadata]

        pending_cloned_fds = []
        try:
            for peer, sock in self.fd_conns.items():
                if peer == self.cur_rank:
                    continue

                if peer not in self._peer_va_ranges:
                    peer_va_base = self.allocator.driver.reserve_va(
                        self.allocator.va_size,
                        self.allocator.granularity,
                    )
                    self._peer_va_ranges[peer] = peer_va_base
                else:
                    peer_va_base = self._peer_va_ranges[peer]
                self._peer_imported_mappings.setdefault(peer, [])

                peer_metadata = gathered_metadata[peer]
                if peer_metadata is None:
                    raise RuntimeError(f"Missing chunk metadata for peer {peer}")

                peer_new_count = total_chunks_per_rank[peer] - self._shared_chunk_counts[peer]
                if peer_new_count != len(peer_metadata):
                    raise RuntimeError(
                        f"Chunk metadata count mismatch for peer {peer}: "
                        f"expected {peer_new_count}, got {len(peer_metadata)}"
                    )

                for peer_offset, peer_size, peer_handle_bytes in peer_metadata:
                    if peer_offset + peer_size > self.allocator.va_size:
                        raise RuntimeError(
                            f"Peer {peer} chunk extends beyond va_size: "
                            f"offset={peer_offset}, size={peer_size}, "
                            f"va_size={self.allocator.va_size}"
                        )
                    _validate_local_handle(peer_handle_bytes)

                cloned_fds = []
                if self.cur_rank > peer:
                    for my_fd in new_fds:
                        send_fd(sock, my_fd)
                    for _ in range(peer_new_count):
                        cloned_fd, _ = recv_fd(sock)
                        cloned_fds.append(cloned_fd)
                        pending_cloned_fds.append(cloned_fd)
                else:
                    for _ in range(peer_new_count):
                        cloned_fd, _ = recv_fd(sock)
                        cloned_fds.append(cloned_fd)
                        pending_cloned_fds.append(cloned_fd)
                    for my_fd in new_fds:
                        send_fd(sock, my_fd)

                for cloned_fd, (peer_offset, peer_size, peer_handle_bytes) in zip(cloned_fds, peer_metadata):
                    # After the upfront metadata validation above, handing the
                    # FD to import_and_map transfers ownership to the driver.
                    # Remove it from the local pending list first so the
                    # cleanup path below only closes never-consumed FDs.
                    pending_cloned_fds.pop(0)
                    reconstructed_handle = _replace_fd_in_local_handle(peer_handle_bytes, cloned_fd)
                    placement = MappingPlacement(peer_va_base + peer_offset, arena_base=peer_va_base)
                    mapping = self.allocator.driver.import_and_map(
                        peer,
                        reconstructed_handle,
                        peer_size,
                        placement,
                    )
                    self._peer_imported_mappings[peer].append(mapping)

                self._shared_chunk_counts[peer] = total_chunks_per_rank[peer]
                self.heap_bases[peer] = peer_va_base
        finally:
            for fd in new_fds:
                try:
                    os.close(fd)
                except OSError:
                    pass
            for fd in pending_cloned_fds:
                try:
                    os.close(fd)
                except OSError:
                    pass

        self._shared_chunk_counts[self.cur_rank] = my_total

    def _refresh_peer_access_fabric(self, dist):
        """
        Peer access for VMemChunkedAllocator using fabric handles.

        Works across hosts in the same fabric domain. Handle bytes travel
        through torch.distributed collectives instead of AF_UNIX sockets.

        Note: there is no FD cleanup in this method (cf.
        _refresh_peer_access_chunked's try/finally). Fabric handles are
        plain bytes, not kernel resources, so there is nothing to close.
        """
        handle_bytes_size = 64
        record_bytes = 8 + 8 + handle_bytes_size

        self._ensure_chunk_tracking_state()

        my_total = self.allocator.get_num_chunks()
        total_chunks_per_rank = self._gather_total_chunk_counts(dist, my_total)
        my_total = total_chunks_per_rank[self.cur_rank]
        my_already_shared = self._shared_chunk_counts[self.cur_rank]
        new_chunks = self.allocator.get_allocation_chunks_since(my_already_shared)

        payload = bytearray(len(new_chunks) * record_bytes)
        for index, (_chunk_idx, offset, size, handle_bytes) in enumerate(new_chunks):
            if len(handle_bytes) != handle_bytes_size:
                raise RuntimeError(f"Expected {handle_bytes_size}-byte fabric handle, got {len(handle_bytes)}")
            record_offset = index * record_bytes
            payload[record_offset : record_offset + 8] = int(offset).to_bytes(8, "little", signed=False)
            payload[record_offset + 8 : record_offset + 16] = int(size).to_bytes(8, "little", signed=False)
            payload[record_offset + 16 : record_offset + record_bytes] = handle_bytes

        gathered_payloads = [None] * self.num_ranks
        dist.all_gather_object(gathered_payloads, bytes(payload))

        for peer in range(self.num_ranks):
            if peer == self.cur_rank:
                self._shared_chunk_counts[peer] = total_chunks_per_rank[peer]
                continue

            peer_new_count = total_chunks_per_rank[peer] - self._shared_chunk_counts[peer]
            if peer_new_count == 0:
                if peer in self._peer_va_ranges:
                    self.heap_bases[peer] = self._peer_va_ranges[peer]
                continue

            peer_payload = gathered_payloads[peer] or b""
            expected_size = peer_new_count * record_bytes
            if len(peer_payload) != expected_size:
                raise RuntimeError(
                    f"Fabric payload size mismatch for peer {peer}: expected {expected_size}, got {len(peer_payload)}"
                )

            if peer not in self._peer_va_ranges:
                peer_va_base = self.allocator.driver.reserve_va(
                    self.allocator.va_size,
                    self.allocator.granularity,
                )
                self._peer_va_ranges[peer] = peer_va_base
            else:
                peer_va_base = self._peer_va_ranges[peer]
            self._peer_imported_mappings.setdefault(peer, [])

            for index in range(peer_new_count):
                record_offset = index * record_bytes
                chunk_offset = int.from_bytes(
                    peer_payload[record_offset : record_offset + 8],
                    "little",
                    signed=False,
                )
                chunk_size = int.from_bytes(
                    peer_payload[record_offset + 8 : record_offset + 16],
                    "little",
                    signed=False,
                )
                handle_bytes = peer_payload[record_offset + 16 : record_offset + record_bytes]
                if chunk_offset + chunk_size > self.allocator.va_size:
                    raise RuntimeError(
                        f"Peer {peer} chunk extends beyond va_size: "
                        f"offset={chunk_offset}, size={chunk_size}, "
                        f"va_size={self.allocator.va_size}"
                    )
                mapping = self.allocator.driver.import_and_map(
                    peer,
                    handle_bytes,
                    chunk_size,
                    MappingPlacement(peer_va_base + chunk_offset, arena_base=peer_va_base),
                )
                self._peer_imported_mappings[peer].append(mapping)

            self._shared_chunk_counts[peer] = total_chunks_per_rank[peer]
            self.heap_bases[peer] = peer_va_base

    def _refresh_peer_access_segmented(self, dist):
        """Peer access for VMemAllocator (segment-based, legacy)."""
        from iris.host.distributed.fd_passing import recv_fd, send_fd
        from iris.host.platform.hip import (
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

    _symmetric_cache: dict = None

    def as_symmetric(self, external_tensor: torch.Tensor) -> torch.Tensor:
        """
        Place an external PyTorch tensor on the symmetric heap.

        Caches heap buffers by (shape, dtype) for stable addresses across
        calls. First call allocates on the heap and copies data. Subsequent
        calls with the same (shape, dtype) reuse the cached buffer and only
        copy data. This makes as_symmetric graph-capture safe — addresses
        are stable across replays.

        If the tensor is already on the symmetric heap, returns it directly.

        Args:
            external_tensor: External PyTorch tensor (must be CUDA, contiguous)

        Returns:
            Tensor on the symmetric heap (same shape/dtype)
        """
        if self.is_symmetric(external_tensor):
            return external_tensor

        if self._symmetric_cache is None:
            self._symmetric_cache = {}

        key = (external_tensor.shape, external_tensor.dtype)
        if key in self._symmetric_cache:
            buf = self._symmetric_cache[key]
            buf.copy_(external_tensor)
            return buf

        self._ensure_peer_refresh_healthy()

        if not hasattr(self.allocator, "import_external_tensor"):
            raise RuntimeError(f"{type(self.allocator).__name__} does not support as_symmetric().")

        imported = self.allocator.import_external_tensor(external_tensor)
        self.refresh_peer_access()
        self._symmetric_cache[key] = imported
        return imported

    def close(self):
        """Release resources held by the symmetric heap."""
        try:
            import torch

            if hasattr(self, "device_id"):
                torch.cuda.synchronize(self.device_id)
        except Exception:
            pass

        if hasattr(self, "_peer_imported_mappings"):
            has_driver = hasattr(self.allocator, "driver")
            for peer, mappings in self._peer_imported_mappings.items():
                for entry in mappings:
                    try:
                        if has_driver and isinstance(entry, PeerMapping):
                            self.allocator.driver.cleanup(entry)
                        else:
                            from iris.host.platform.hip import mem_release, mem_unmap

                            handle, va, size = entry
                            mem_unmap(va, size)
                            mem_release(handle)
                    except Exception as exc:
                        logger.warning("cleanup failed for peer %d: %s", peer, exc)
            self._peer_imported_mappings.clear()

        # Free peer VA ranges (now safe -- all mappings have been removed)
        if self._peer_va_ranges:
            va_size = self._peer_va_size()
            has_driver = hasattr(self.allocator, "driver")
            if has_driver:
                for peer, va_base in self._peer_va_ranges.items():
                    try:
                        self.allocator.driver.free_va(va_base, va_size)
                    except Exception as exc:
                        logger.warning("free_va failed for peer %d: %s", peer, exc)
            else:
                from iris.host.platform.hip import mem_address_free

                for peer, va_base in self._peer_va_ranges.items():
                    try:
                        mem_address_free(va_base, va_size)
                    except Exception as exc:
                        logger.warning("VA free failed for peer %d: %s", peer, exc)
            self._peer_va_ranges.clear()

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
