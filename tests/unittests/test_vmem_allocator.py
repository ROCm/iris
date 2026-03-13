# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.

"""
Test VMem allocator functionality.
"""

import torch
import iris
import pytest


def test_vmem_allocator_creation():
    """Test that VMem allocator can be created."""
    # Create Iris context with VMem allocator
    ctx = iris.iris(1 << 20, allocator_type="vmem")  # 1 MB heap

    # Basic sanity checks
    assert ctx.cur_rank >= 0
    assert ctx.num_ranks >= 1
    assert ctx.heap_size == 1 << 20

    print(f"Rank {ctx.cur_rank}: VMem allocator created successfully!")


def test_vmem_basic_allocation():
    """Test basic memory allocation with VMem."""
    ctx = iris.iris(1 << 20, allocator_type="vmem")  # 1 MB heap

    # Allocate a tensor
    tensor = ctx.zeros(1024, dtype=torch.float32)

    assert tensor.shape == (1024,)
    assert tensor.device.type == "cuda"
    assert torch.all(tensor == 0)

    # Write some data
    tensor.fill_(42.0)
    assert torch.all(tensor == 42.0)

    print(f"Rank {ctx.cur_rank}: VMem basic allocation test passed!")


def test_vmem_multiple_allocations():
    """Test multiple allocations from VMem heap."""
    ctx = iris.iris(1 << 20, allocator_type="vmem")  # 1 MB heap

    # Allocate multiple tensors
    tensors = []
    for i in range(10):
        t = ctx.zeros(100, dtype=torch.float32)
        t.fill_(float(i))
        tensors.append(t)

    # Verify each tensor
    for i, t in enumerate(tensors):
        assert torch.all(t == float(i))

    print(f"Rank {ctx.cur_rank}: VMem multiple allocations test passed!")


def test_vmem_heap_bases():
    """Test that heap bases are properly set up with VMem."""
    ctx = iris.iris(1 << 20, allocator_type="vmem")  # 1 MB heap

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

    print(f"Rank {ctx.cur_rank}: VMem heap bases test passed!")


def test_vmem_multirank_exchange():
    """Test VMem FD exchange and heap base setup across multiple ranks."""
    ctx = iris.iris(1 << 20, allocator_type="vmem")  # 1 MB heap

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

    print(f"Rank {ctx.cur_rank}: VMem multi-rank FD exchange test passed!")


def test_vmem_owns_tensor():
    """Test owns_tensor detection with VMem allocator."""
    ctx = iris.iris(1 << 20, allocator_type="vmem")

    # Allocate tensor on symmetric heap
    heap_tensor = ctx.zeros(100, dtype=torch.float32)
    assert ctx.heap.allocator.owns_tensor(heap_tensor)

    # Allocate tensor NOT on symmetric heap
    external_tensor = torch.zeros(100, dtype=torch.float32, device=ctx.device)
    assert not ctx.heap.allocator.owns_tensor(external_tensor)

    print(f"Rank {ctx.cur_rank}: VMem owns_tensor test passed!")

    # Cleanup: Delete tensors and sync GPU before test ends
    del heap_tensor, external_tensor
    torch.cuda.synchronize()
    torch.cuda.empty_cache()


@pytest.mark.skipif(torch.cuda.device_count() < 2, reason="Requires at least 2 GPUs for RMA testing")
def test_vmem_rma_compatibility():
    """Test that VMem allocator works with RMA operations."""
    pytest.skip("RMA operations (ctx.get/ctx.put) not yet implemented")

    ctx = iris.iris(1 << 20, allocator_type="vmem")

    if ctx.num_ranks < 2:
        pytest.skip("Requires at least 2 ranks for RMA testing")

    # Allocate tensors
    local_tensor = ctx.zeros(100, dtype=torch.float32)
    remote_tensor = ctx.zeros(100, dtype=torch.float32)

    # Fill local tensor
    local_tensor.fill_(float(ctx.cur_rank))

    ctx.barrier()

    # Test get operation (read from peer)
    peer_rank = (ctx.cur_rank + 1) % ctx.num_ranks
    ctx.get(remote_tensor, peer_rank, local_tensor)

    ctx.barrier()

    # Verify we read the peer's value
    expected_value = float(peer_rank)
    assert torch.all(remote_tensor == expected_value), f"Expected {expected_value}, got {remote_tensor[0].item()}"

    print(f"Rank {ctx.cur_rank}: VMem RMA compatibility test passed!")


def test_vmem_granularity_alignment():
    """Test that VMem allocations respect granularity."""
    from iris.hip import get_allocation_granularity

    # Cleanup GPU state from previous tests
    torch.cuda.synchronize()
    torch.cuda.empty_cache()

    ctx = iris.iris(1 << 20, allocator_type="vmem")

    # Get granularity for the device
    granularity = get_allocation_granularity(ctx.gpu_id)
    print(f"Rank {ctx.cur_rank}: VMem granularity = {granularity} bytes")

    # Heap size should be aligned to granularity
    assert ctx.heap.allocator.aligned_heap_size % granularity == 0

    print(f"Rank {ctx.cur_rank}: VMem granularity alignment test passed!")


def test_vmem_import_external_tensor():
    """
    Test importing external PyTorch tensors via as_symmetric().

    This validates the critical lifetime contract:
    1. External tensors can be imported into the symmetric heap
    2. The imported tensor shares memory with the original (while ctx is alive)
    3. When ctx is destroyed, imported_tensor becomes invalid
    4. BUT the original tensor REMAINS VALID and fully usable

    This is THE KEY CONTRACT: imported tensors die with ctx, originals survive.
    """
    import gc

    # Create Iris context with large enough heap for PyTorch's 2MB allocations
    ctx = iris.iris(4 << 20, allocator_type="vmem")

    # Create original PyTorch tensor on the correct device for this rank
    original_tensor = torch.randn(100, dtype=torch.float32, device=ctx.device)
    original_data = original_tensor.clone()

    # Import the external tensor
    imported_tensor = ctx.as_symmetric(original_tensor)

    # Verify imported tensor has same data
    assert torch.allclose(imported_tensor, original_data), "Imported tensor should match original"

    # Modify via imported tensor
    imported_tensor.fill_(42.0)
    assert torch.all(imported_tensor == 42.0), "Imported tensor modifications should work"

    # Original tensor should see the change (shared memory)
    assert torch.all(original_tensor == 42.0), "Original tensor should see changes via shared memory"

    # Modify via original tensor
    original_tensor.fill_(99.0)
    assert torch.all(original_tensor == 99.0), "Original tensor modifications should work"
    assert torch.all(imported_tensor == 99.0), "Imported tensor should see changes via shared memory"

    # NOW THE CRITICAL PART: Destroy ctx
    # This invalidates imported_tensor, but original_tensor should survive!
    del ctx, imported_tensor
    gc.collect()
    torch.cuda.synchronize()

    # VERIFY: Original tensor is still valid and fully usable
    assert torch.all(original_tensor == 99.0), "Original tensor should still be valid after ctx destroyed!"

    # Can still modify it
    original_tensor.fill_(123.0)
    assert torch.all(original_tensor == 123.0), "Original tensor should still be modifiable!"

    # Can do operations on it
    result = original_tensor + 1.0
    assert torch.all(result == 124.0), "Original tensor should still support operations!"

    print(
        f"Rank {torch.distributed.get_rank() if torch.distributed.is_initialized() else 0}: "
        f"VMem import external tensor test passed!"
    )
    print("  ✓ Imported tensor shared memory with original (while ctx alive)")
    print("  ✓ Original tensor survived ctx destruction")
    print("  ✓ Original tensor still fully functional after ctx destroyed")


if __name__ == "__main__":
    # Run a quick test
    test_vmem_allocator_creation()
    test_vmem_basic_allocation()
    test_vmem_heap_bases()
    test_vmem_owns_tensor()
