# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.

"""
Test basic DMA-BUF export/import APIs.
"""

import torch
import iris


def test_dmabuf_export():
    """Test exporting a DMA-BUF file descriptor."""
    from iris.hip import export_dmabuf_handle

    # Create a simple GPU tensor
    tensor = torch.zeros(1024, dtype=torch.float32, device="cuda")
    ptr = tensor.data_ptr()
    size = tensor.element_size() * tensor.numel()

    # Export the DMA-BUF FD
    fd = export_dmabuf_handle(ptr, size)

    # Verify we got a valid file descriptor
    assert fd >= 0, f"Expected valid FD, got {fd}"

    # Close the FD
    import os

    os.close(fd)


def test_dmabuf_import():
    """Test importing a DMA-BUF file descriptor."""
    from iris.hip import export_dmabuf_handle, import_dmabuf_handle
    import os

    # Create a simple GPU tensor
    tensor = torch.zeros(1024, dtype=torch.float32, device="cuda")
    ptr = tensor.data_ptr()
    size = tensor.element_size() * tensor.numel()

    # Export the DMA-BUF FD
    fd = export_dmabuf_handle(ptr, size)
    assert fd >= 0

    try:
        # Import the DMA-BUF FD
        mapped_ptr = import_dmabuf_handle(fd, size)

        # Verify we got a valid pointer
        assert mapped_ptr > 0, f"Expected valid pointer, got {mapped_ptr}"

    finally:
        # Close the FD
        os.close(fd)


def test_dmabuf_export_import_roundtrip():
    """Test export/import roundtrip with actual memory access."""
    from iris.hip import export_dmabuf_handle, import_dmabuf_handle
    import os

    # Create a GPU tensor and fill it with test data
    tensor = torch.arange(1024, dtype=torch.float32, device="cuda")
    ptr = tensor.data_ptr()
    size = tensor.element_size() * tensor.numel()

    # Export the DMA-BUF FD
    fd = export_dmabuf_handle(ptr, size)
    assert fd >= 0

    try:
        # Import the DMA-BUF FD
        mapped_ptr = import_dmabuf_handle(fd, size)
        assert mapped_ptr > 0

        # Create a tensor view from the mapped pointer

        class CUDAArrayInterface:
            def __init__(self, ptr, size):
                self.ptr = ptr
                self.size = size

            @property
            def __cuda_array_interface__(self):
                return {
                    "shape": (self.size // 4,),  # float32 = 4 bytes
                    "typestr": "<f4",  # little-endian float32
                    "data": (self.ptr, False),
                    "version": 3,
                }

        cuda_array = CUDAArrayInterface(mapped_ptr, size)
        mapped_tensor = torch.as_tensor(cuda_array, device="cuda")

        # Verify the data matches
        torch.cuda.synchronize()
        assert torch.allclose(tensor, mapped_tensor), "Mapped data doesn't match original"

    finally:
        os.close(fd)


def test_iris_symmetric_heap_creation():
    """Test that Iris context can be created with the new allocator."""
    ctx = iris.iris(1 << 20)  # 1 MB heap

    # Basic sanity checks
    assert ctx.cur_rank >= 0
    assert ctx.num_ranks >= 1
    assert ctx.heap_size == 1 << 20

    # Test allocation works
    tensor = ctx.zeros(100, dtype=torch.float32)
    assert tensor.shape == (100,)
    assert tensor.device.type == "cuda"
    assert torch.all(tensor == 0)


def test_dmabuf_multirank_exchange():
    """Test FD export/import and RMA between multiple ranks."""
    ctx = iris.iris(1 << 20)  # 1 MB heap

    # Allocate and initialize tensor on each rank
    tensor = ctx.zeros(1024, dtype=torch.float32)
    tensor.fill_(float(ctx.cur_rank * 100))

    # Verify heap bases are set up correctly
    assert ctx.heap_bases.shape == (ctx.num_ranks,)
    assert int(ctx.heap_bases[ctx.cur_rank].item()) > 0

    # For multi-rank, verify we can see peer heap bases
    if ctx.num_ranks > 1:
        for peer in range(ctx.num_ranks):
            if peer != ctx.cur_rank:
                assert int(ctx.heap_bases[peer].item()) > 0, f"Peer {peer} heap base not set"
                # Verify heap bases are different addresses
                assert int(ctx.heap_bases[peer].item()) != int(ctx.heap_bases[ctx.cur_rank].item())

    # Verify local memory access still works after FD exchange
    ctx.barrier()
    tensor.fill_(float(ctx.cur_rank * 100))
    ctx.barrier()
    assert torch.all(tensor == float(ctx.cur_rank * 100))

    print(f"Rank {ctx.cur_rank}: Multi-rank FD exchange test passed!")
