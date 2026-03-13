"""
Test VMem peer DMA-BUF exchange workflow.

This test verifies the multi-rank SymmetricHeap pattern:
1. Rank creates VMem allocation and exports as DMA-BUF
2. Peer rank receives DMA-BUF and imports into their VMem space
3. Both ranks can access the memory via controlled VA

This is the foundation for multi-rank RMA with VMem allocator.
"""

import torch
import torch.distributed as dist
import pytest
from iris.hip import (
    mem_address_reserve,
    mem_address_free,
    mem_create,
    mem_map,
    mem_unmap,
    mem_release,
    mem_set_access,
    get_allocation_granularity,
    export_dmabuf_handle,
    mem_import_from_shareable_handle,
    hipMemAccessDesc,
    hipMemLocationTypeDevice,
    hipMemAccessFlagsProtReadWrite,
)


def _get_device_id():
    """Get current device ID."""
    if torch.cuda.is_available():
        return torch.cuda.current_device()
    return 0


def test_vmem_peer_dmabuf_exchange_single_rank():
    """
    Test the export/import flow on a single rank (simulating peer exchange).

    Workflow:
    1. Create VMem allocation with physical backing
    2. Export as DMA-BUF
    3. Import into a different VA range (simulating peer)
    4. Verify both VAs access the same physical memory
    """
    device_id = _get_device_id()
    granularity = get_allocation_granularity(device_id)
    alloc_size = 2 << 20  # 2 MB

    # Reserve VA for "local" rank
    local_va = mem_address_reserve(alloc_size, granularity, 0)

    # Reserve VA for "peer" rank (simulated)
    peer_va = mem_address_reserve(alloc_size, granularity, 0)

    try:
        # Step 1: Local rank creates physical allocation and maps it
        local_handle = mem_create(alloc_size, device_id)
        mem_map(local_va, alloc_size, 0, local_handle)

        access_desc = hipMemAccessDesc()
        access_desc.location.type = hipMemLocationTypeDevice
        access_desc.location.id = device_id
        access_desc.flags = hipMemAccessFlagsProtReadWrite
        mem_set_access(local_va, alloc_size, access_desc)

        print(f"✓ Local rank mapped VMem: va={hex(local_va)}, size={alloc_size}")

        # Write data via local VA
        class CUDAArrayInterface:
            def __init__(self, ptr, size_bytes):
                self.ptr = ptr
                self.size_bytes = size_bytes

            @property
            def __cuda_array_interface__(self):
                return {
                    "shape": (self.size_bytes // 4,),
                    "typestr": "<f4",
                    "data": (self.ptr, False),
                    "version": 3,
                }

        local_array = CUDAArrayInterface(local_va, 1024 * 4)
        local_tensor = torch.as_tensor(local_array, device="cuda")
        local_tensor.fill_(12345.0)
        torch.cuda.synchronize()
        print("✓ Local rank wrote value: 12345.0")

        # Step 2: Export local allocation as DMA-BUF
        dmabuf_fd, export_base, export_size = export_dmabuf_handle(local_va, alloc_size)
        print(f"✓ Exported DMA-BUF: fd={dmabuf_fd}, base={hex(export_base)}, size={export_size}")

        # Step 3: "Peer" imports the DMA-BUF into their VA space
        imported_handle = mem_import_from_shareable_handle(dmabuf_fd)
        import os

        os.close(dmabuf_fd)

        # Map to peer VA
        mem_map(peer_va, export_size, 0, imported_handle)
        mem_set_access(peer_va, export_size, access_desc)
        print(f"✓ Peer rank imported and mapped: va={hex(peer_va)}")

        # Step 4: Verify peer can read the same data
        peer_array = CUDAArrayInterface(peer_va, 1024 * 4)
        peer_tensor = torch.as_tensor(peer_array, device="cuda")
        torch.cuda.synchronize()

        assert torch.all(peer_tensor == 12345.0), f"Peer read wrong value: {peer_tensor[0]}"
        print(f"✓ Peer rank read same value: {peer_tensor[0].item()}")

        # Step 5: Modify via peer, read via local
        peer_tensor.fill_(67890.0)
        torch.cuda.synchronize()

        assert torch.all(local_tensor == 67890.0), f"Local read wrong value after peer write: {local_tensor[0]}"
        print(f"✓ Local rank sees peer's write: {local_tensor[0].item()}")

        # Cleanup
        mem_unmap(peer_va, export_size)
        mem_release(imported_handle)

        mem_unmap(local_va, alloc_size)
        mem_release(local_handle)

    finally:
        mem_address_free(local_va, alloc_size)
        mem_address_free(peer_va, alloc_size)
        torch.cuda.synchronize()
        if dist.is_initialized():
            dist.barrier()


def test_vmem_peer_dmabuf_exchange_multi_rank():
    """
    Test actual multi-rank DMA-BUF exchange using torch.distributed.

    Workflow:
    1. Each rank creates VMem allocation
    2. Each rank exports as DMA-BUF
    3. Ranks exchange DMA-BUF FDs
    4. Each rank imports peer's DMA-BUF into their VMem space
    5. Verify cross-rank memory access
    """
    if not dist.is_initialized():
        pytest.skip("Test requires torch.distributed")

    device_id = _get_device_id()
    granularity = get_allocation_granularity(device_id)
    alloc_size = 2 << 20  # 2 MB

    cur_rank = dist.get_rank()
    num_ranks = dist.get_world_size()

    if num_ranks < 2:
        pytest.skip("Test requires at least 2 ranks")

    # Reserve VA for local allocation
    local_va = mem_address_reserve(alloc_size, granularity, 0)

    # Reserve VA for peer imports
    peer_vas = {}
    for peer in range(num_ranks):
        if peer != cur_rank:
            peer_vas[peer] = mem_address_reserve(alloc_size, granularity, 0)

    try:
        # Step 1: Create local VMem allocation
        local_handle = mem_create(alloc_size, device_id)
        mem_map(local_va, alloc_size, 0, local_handle)

        access_desc = hipMemAccessDesc()
        access_desc.location.type = hipMemLocationTypeDevice
        access_desc.location.id = device_id
        access_desc.flags = hipMemAccessFlagsProtReadWrite
        mem_set_access(local_va, alloc_size, access_desc)

        print(f"Rank {cur_rank}: Mapped local VMem at {hex(local_va)}")

        # Write rank-specific value
        class CUDAArrayInterface:
            def __init__(self, ptr, size_bytes):
                self.ptr = ptr
                self.size_bytes = size_bytes

            @property
            def __cuda_array_interface__(self):
                return {
                    "shape": (self.size_bytes // 4,),
                    "typestr": "<f4",
                    "data": (self.ptr, False),
                    "version": 3,
                }

        local_array = CUDAArrayInterface(local_va, 1024 * 4)
        local_tensor = torch.as_tensor(local_array, device="cuda")
        local_tensor.fill_(float(cur_rank * 100))
        torch.cuda.synchronize()
        print(f"Rank {cur_rank}: Wrote value {cur_rank * 100}")

        # Step 2: Export as DMA-BUF
        dmabuf_fd, export_base, export_size = export_dmabuf_handle(local_va, alloc_size)
        print(f"Rank {cur_rank}: Exported DMA-BUF fd={dmabuf_fd}")

        # Step 3: Exchange FDs using Unix domain sockets
        from iris.fd_passing import setup_fd_infrastructure, send_fd, recv_fd, managed_fd

        fd_conns = setup_fd_infrastructure(cur_rank, num_ranks)

        peer_handles = {}
        if fd_conns is not None:
            with managed_fd(dmabuf_fd):
                for peer, sock in fd_conns.items():
                    if peer == cur_rank:
                        continue

                    # Exchange FDs (higher rank sends first)
                    if cur_rank > peer:
                        send_fd(sock, dmabuf_fd)
                        peer_fd, _ = recv_fd(sock)
                    else:
                        peer_fd, _ = recv_fd(sock)
                        send_fd(sock, dmabuf_fd)

                    print(f"Rank {cur_rank}: Received DMA-BUF fd from rank {peer}")

                    # Import peer's DMA-BUF
                    with managed_fd(peer_fd):
                        peer_imported_handle = mem_import_from_shareable_handle(peer_fd)
                        peer_handles[peer] = peer_imported_handle

                        # Map to our VA space
                        peer_va = peer_vas[peer]
                        mem_map(peer_va, export_size, 0, peer_imported_handle)
                        mem_set_access(peer_va, export_size, access_desc)

                        print(f"Rank {cur_rank}: Imported peer {peer}'s memory at {hex(peer_va)}")

            # Cleanup FD connections
            for sock in fd_conns.values():
                sock.close()

        dist.barrier()

        # Step 4: Verify we can read peer's data
        for peer, peer_va in peer_vas.items():
            peer_array = CUDAArrayInterface(peer_va, 1024 * 4)
            peer_tensor = torch.as_tensor(peer_array, device="cuda")
            torch.cuda.synchronize()

            expected_value = float(peer * 100)
            actual_value = peer_tensor[0].item()

            print(f"Rank {cur_rank}: Read from peer {peer}: expected={expected_value}, actual={actual_value}")
            assert abs(actual_value - expected_value) < 0.1, (
                f"Rank {cur_rank} read wrong value from peer {peer}: {actual_value} != {expected_value}"
            )

        print(f"✓ Rank {cur_rank}: All peer reads successful!")

        dist.barrier()

        # Cleanup
        for peer, peer_va in peer_vas.items():
            mem_unmap(peer_va, export_size)
            mem_release(peer_handles[peer])
            mem_address_free(peer_va, alloc_size)

        mem_unmap(local_va, alloc_size)
        mem_release(local_handle)

    finally:
        mem_address_free(local_va, alloc_size)
        torch.cuda.synchronize()
        if dist.is_initialized():
            dist.barrier()
