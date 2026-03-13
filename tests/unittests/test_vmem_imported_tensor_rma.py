# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.

"""
End-to-end test for imported external tensors with Iris operations.

Tests the full workflow:
1. Create external PyTorch tensor (not on symmetric heap)
2. Import it via ctx.as_symmetric()
3. Use it in Triton kernels with load/store operations
4. Validate results across multiple ranks
"""

import torch
import triton
import triton.language as tl
import pytest
import iris


@triton.jit
def read_local_kernel(
    imported_data,
    results,
    BLOCK_SIZE: tl.constexpr,
):
    """
    Kernel that reads from local imported tensor (no RMA).
    """
    pid = tl.program_id(0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < BLOCK_SIZE

    # Simple local load
    data = tl.load(imported_data + offsets, mask=mask)

    # Store to results
    tl.store(results + offsets, data, mask=mask)


@triton.jit
def read_remote_kernel(
    imported_data,
    results,
    cur_rank: tl.constexpr,
    target_rank: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
    heap_bases: tl.tensor,
):
    """
    Kernel that reads from remote rank's imported tensor (RMA).
    """
    pid = tl.program_id(0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < BLOCK_SIZE

    # RMA load from target rank
    data = iris.load(imported_data + offsets, cur_rank, target_rank, heap_bases, mask=mask)

    # Store to local results
    tl.store(results + offsets, data, mask=mask)


def test_vmem_imported_tensor_local_read():
    """
    Test LOCAL read from imported tensor (no RMA, just local access).

    Workflow:
    1. Rank creates external tensor with specific value
    2. Import it via as_symmetric()
    3. Kernel reads from the LOCAL imported tensor
    4. Verify we can read our own imported tensor
    """
    BLOCK_SIZE = 16

    # Use VMem allocator (large heap for PyTorch caching allocator's 2MB blocks)
    ctx = iris.iris(64 << 20, allocator_type="vmem")  # 64 MB heap

    num_ranks = ctx.get_num_ranks()
    cur_rank = ctx.get_rank()

    # Step 1: Create EXTERNAL tensor (not on symmetric heap)
    external_tensor = torch.ones(BLOCK_SIZE, dtype=torch.float32, device=ctx.device)
    external_tensor.fill_(float(cur_rank + 100))  # Rank 0 -> 100.0, Rank 1 -> 101.0

    # Step 2: Import the external tensor into symmetric heap
    imported_tensor = ctx.as_symmetric(external_tensor)

    print(f"Rank {cur_rank}: External tensor ptr: {hex(external_tensor.data_ptr())}")
    print(f"Rank {cur_rank}: Imported tensor ptr: {hex(imported_tensor.data_ptr())}")

    # Allocate results tensor on symmetric heap
    results = ctx.zeros(BLOCK_SIZE, dtype=torch.float32)

    ctx.barrier()

    # Step 3: Read from LOCAL imported tensor (no RMA)
    grid = lambda meta: (1,)
    read_local_kernel[grid](imported_tensor, results, BLOCK_SIZE)

    ctx.barrier()

    # Step 4: Verify results - should see our own value
    expected_value = float(cur_rank + 100)
    expected = torch.full((BLOCK_SIZE,), expected_value, dtype=torch.float32, device=ctx.device)

    try:
        torch.testing.assert_close(results, expected, rtol=0, atol=1e-5)
        print(f"Rank {cur_rank}: Local read from imported tensor test passed!")
        print(f"  External tensor value: {external_tensor[0].item()}")
        print(f"  Imported tensor value: {imported_tensor[0].item()}")
        print(f"  Read result: {results[0].item()}")
        print(f"  Expected: {expected[0].item()}")
    except AssertionError as e:
        print(f"Rank {cur_rank}: Test failed!")
        print(f"  Expected: {expected}")
        print(f"  Got: {results}")
        raise
    finally:
        ctx.barrier()
        # Cleanup
        del imported_tensor, results, external_tensor
        import gc

        gc.collect()


def test_vmem_imported_tensor_remote_read():
    """
    Test REMOTE read from imported tensor (RMA across ranks).

    Workflow:
    1. Each rank creates external tensor with rank-specific value
    2. Each rank imports it via as_symmetric()
    3. Kernel reads from PEER rank's imported tensor using RMA
    4. Verify we can read peer's data correctly
    """
    BLOCK_SIZE = 16

    # Use VMem allocator (large heap for PyTorch caching allocator's 2MB blocks)
    ctx = iris.iris(64 << 20, allocator_type="vmem")  # 64 MB heap

    num_ranks = ctx.get_num_ranks()
    heap_bases = ctx.get_heap_bases()
    cur_rank = ctx.get_rank()

    if num_ranks < 2:
        pytest.skip("Test requires at least 2 ranks")

    # Step 1: Create EXTERNAL tensor (not on symmetric heap)
    external_tensor = torch.ones(BLOCK_SIZE, dtype=torch.float32, device=ctx.device)
    external_tensor.fill_(float(cur_rank + 100))  # Rank 0 -> 100.0, Rank 1 -> 101.0

    # Step 2: Import the external tensor into symmetric heap
    imported_tensor = ctx.as_symmetric(external_tensor)

    imported_ptr = imported_tensor.data_ptr()
    my_heap_base = int(heap_bases[cur_rank].item())
    my_offset = imported_ptr - my_heap_base

    print(f"Rank {cur_rank}: External tensor ptr: {hex(external_tensor.data_ptr())}")
    print(f"Rank {cur_rank}: Imported tensor ptr: {hex(imported_ptr)}")
    print(f"Rank {cur_rank}: My heap base: {hex(my_heap_base)}")
    print(f"Rank {cur_rank}: Offset in heap: {hex(my_offset)} ({my_offset} bytes)")

    # Show ALL heap bases
    for r in range(num_ranks):
        print(f"Rank {cur_rank} sees: heap_bases[{r}] = {hex(int(heap_bases[r].item()))}")

    # Allocate results tensor on symmetric heap
    results = ctx.zeros(BLOCK_SIZE, dtype=torch.float32)

    # Collect offset info BEFORE barrier (safer)
    offsets_info = []
    for r in range(num_ranks):
        offsets_info.append(int(heap_bases[r].item()))

    ctx.barrier()

    # NOW print after barrier when everyone is ready
    for r in range(num_ranks):
        if r == cur_rank:
            print(f"===== Rank {cur_rank} Offset Analysis =====")
            for peer in range(num_ranks):
                peer_base = offsets_info[peer]
                print(f"  heap_bases[{peer}] = {hex(peer_base)}")
            print(f"  My imported ptr = {hex(imported_ptr)}")
            print(f"  My offset = {hex(my_offset)}")
        ctx.barrier()

    # Step 3: Read from PEER rank's imported tensor using RMA
    peer_rank = (cur_rank + 1) % num_ranks
    grid = lambda meta: (1,)
    read_remote_kernel[grid](imported_tensor, results, cur_rank, peer_rank, BLOCK_SIZE, heap_bases)

    ctx.barrier()

    # Step 4: Verify results - should see peer's value
    expected_value = float(peer_rank + 100)
    expected = torch.full((BLOCK_SIZE,), expected_value, dtype=torch.float32, device=ctx.device)

    try:
        torch.testing.assert_close(results, expected, rtol=0, atol=1e-5)
        print(f"Rank {cur_rank}: Remote read from imported tensor test passed!")
        print(f"  My external tensor: {external_tensor[0].item()}")
        print(f"  Read from rank {peer_rank}: {results[0].item()}")
        print(f"  Expected: {expected[0].item()}")
    except AssertionError as e:
        print(f"Rank {cur_rank}: Test failed!")
        print(f"  Expected: {expected}")
        print(f"  Got: {results}")
        print(f"  My imported tensor: {imported_tensor}")
        raise
    finally:
        ctx.barrier()
        # Cleanup
        del imported_tensor, results, external_tensor
        import gc

        gc.collect()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
