# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.

"""
Simple test to check if imported tensor offsets are symmetric across ranks.
"""

import torch
import pytest
import iris


def test_vmem_imported_tensor_offset_symmetry():
    """
    Check if imported tensors get the SAME offset on all ranks.

    This is critical for RMA to work - offsets must be symmetric!
    """
    BLOCK_SIZE = 16

    # Use VMem allocator
    ctx = iris.iris(4 << 20, allocator_type="vmem")  # 4 MB heap

    num_ranks = ctx.get_num_ranks()
    heap_bases = ctx.get_heap_bases()
    cur_rank = ctx.get_rank()

    if num_ranks < 2:
        pytest.skip("Test requires at least 2 ranks")

    # Step 1: Create EXTERNAL tensor (same on all ranks)
    external_tensor = torch.zeros(BLOCK_SIZE, dtype=torch.float32, device=ctx.device)

    # Step 2: Import the external tensor
    imported_tensor = ctx.as_symmetric(external_tensor)

    # Calculate offset
    imported_ptr = imported_tensor.data_ptr()
    my_heap_base = int(heap_bases[cur_rank].item())
    my_offset = imported_ptr - my_heap_base

    print(f"Rank {cur_rank}: heap_base={hex(my_heap_base)}, imported_ptr={hex(imported_ptr)}, offset={hex(my_offset)}")

    # Gather offsets from all ranks (use ctx.device so backend matches process group)
    offset_tensor = torch.tensor([my_offset], dtype=torch.int64, device=ctx.device)
    all_offsets = [torch.zeros(1, dtype=torch.int64, device=ctx.device) for _ in range(num_ranks)]

    torch.distributed.all_gather(all_offsets, offset_tensor)

    # Convert to list
    offsets_list = [int(t.item()) for t in all_offsets]

    print(f"Rank {cur_rank}: All offsets = {[hex(o) for o in offsets_list]}")

    # Check if offsets are the same
    if len(set(offsets_list)) == 1:
        print(f"Rank {cur_rank}: ✅ OFFSETS ARE SYMMETRIC!")
    else:
        print(f"Rank {cur_rank}: ❌ OFFSETS ARE DIFFERENT!")
        for r, offset in enumerate(offsets_list):
            print(f"  Rank {r}: offset = {hex(offset)}")

    ctx.barrier()

    # Cleanup
    del imported_tensor, external_tensor
    import gc

    gc.collect()

    # Assert offsets must be the same for RMA to work
    assert len(set(offsets_list)) == 1, f"Offsets must be symmetric for RMA! Got: {[hex(o) for o in offsets_list]}"


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
