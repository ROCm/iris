# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.

"""
Test allocate_symmetric().

The kernel loads one already-translated pointer out of a table and uses it.
The rank index is a runtime value, so no unrolling over ranks is needed.
"""

import gc

import pytest
import torch
import triton
import triton.language as tl

import iris


@triton.jit
def _put_kernel(src, peer_ptrs, n_elements, target_rank, BLOCK_SIZE: tl.constexpr):
    """Copy src into the target rank's copy. target_rank is runtime."""
    offsets = tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements

    dst = tl.load(peer_ptrs + target_rank).to(src.dtype, bitcast=True)
    # Not optional: a pointer out of memory carries no alignment, and without
    # this the compiler emits 4x narrower stores. Silent -- still correct.
    dst = tl.multiple_of(dst, 16)

    tl.store(dst + offsets, tl.load(src + offsets, mask=mask), mask=mask)


def test_allocate_symmetric_returns_pointer_table():
    """The table is rank-indexed and holds this allocation's address per rank."""
    ctx = iris.iris(1 << 20)
    rank = ctx.get_rank()
    world_size = ctx.get_num_ranks()

    try:
        tensor, peer_ptrs = ctx.allocate_symmetric(1024, dtype=torch.float32)

        assert tensor.shape == (1024,)
        assert ctx.is_symmetric(tensor)

        assert peer_ptrs.numel() == world_size
        assert peer_ptrs.dtype in (torch.int64, torch.uint64)
        assert peer_ptrs.is_cuda

        # Our own entry is our own address. Translation subtracts nothing.
        assert int(peer_ptrs[rank].item()) == tensor.data_ptr()

        # Every entry sits at the same offset within its rank's heap.
        heap_bases = ctx.get_heap_bases()
        offsets = {int(peer_ptrs[r].item()) - int(heap_bases[r].item()) for r in range(world_size)}
        assert len(offsets) == 1

        if world_size > 1:
            assert int(peer_ptrs[(rank + 1) % world_size].item()) != tensor.data_ptr()
    finally:
        ctx.barrier()
        del ctx
        gc.collect()


def test_view_translates_against_allocation_root():
    """A view keeps its offset within the allocation across translation."""
    ctx = iris.iris(1 << 24)
    rank = ctx.get_rank()
    world_size = ctx.get_num_ranks()
    target_rank = (rank + 1) % world_size

    n_elements = 512
    view_start = 128

    try:
        src, _ = ctx.allocate_symmetric(n_elements, dtype=torch.float32)
        dst, dst_ptrs = ctx.allocate_symmetric(n_elements, dtype=torch.float32)

        # A view's table is the allocation's table shifted by the same amount.
        view = dst[view_start:]
        view_ptrs = dst_ptrs + view_start * dst.element_size()
        assert int(view_ptrs[rank].item()) == view.data_ptr()

        src.fill_(rank + 1)
        dst.fill_(-1)
        torch.cuda.synchronize()
        ctx.barrier()

        _put_kernel[(1,)](src, view_ptrs, view.numel(), target_rank, BLOCK_SIZE=512)
        torch.cuda.synchronize()
        ctx.barrier()

        source_rank = (rank - 1) % world_size
        torch.testing.assert_close(dst[view_start:], torch.full_like(dst[view_start:], source_rank + 1))
        torch.testing.assert_close(dst[:view_start], torch.full_like(dst[:view_start], -1.0))

        if world_size > 1:
            assert dst[view_start].item() != rank + 1
    finally:
        ctx.barrier()
        del ctx
        gc.collect()


@pytest.mark.parametrize("n_elements", [256, 200])
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16, torch.float32])
def test_remote_put_runtime_rank(dtype, n_elements):
    """A runtime rank index needs no unrolling: one load out of the table."""
    ctx = iris.iris(1 << 24)
    rank = ctx.get_rank()
    world_size = ctx.get_num_ranks()
    target_rank = (rank + 1) % world_size

    try:
        src, _ = ctx.allocate_symmetric(n_elements, dtype=dtype)
        dst, dst_ptrs = ctx.allocate_symmetric(n_elements, dtype=dtype)

        # Distinct per rank, so a write landing on the wrong rank gives a wrong
        # answer rather than a plausible one.
        src.fill_(rank + 1)
        dst.fill_(-1)
        torch.cuda.synchronize()
        ctx.barrier()

        _put_kernel[(1,)](src, dst_ptrs, n_elements, target_rank, BLOCK_SIZE=256)
        torch.cuda.synchronize()
        ctx.barrier()

        source_rank = (rank - 1) % world_size
        torch.testing.assert_close(dst, torch.full_like(dst, source_rank + 1))

        if world_size > 1:
            # A translation resolving locally would leave our own value here,
            # and the check above would still pass at world size 1.
            assert dst[0].item() != rank + 1
    finally:
        ctx.barrier()
        del ctx
        gc.collect()


def test_sweep_all_ranks():
    """The all-peers loop iris collectives actually use."""
    ctx = iris.iris(1 << 24)
    rank = ctx.get_rank()
    world_size = ctx.get_num_ranks()
    n_elements = 256

    try:
        src, _ = ctx.allocate_symmetric(n_elements, dtype=torch.bfloat16)
        dst, dst_ptrs = ctx.allocate_symmetric(world_size * n_elements, dtype=torch.bfloat16)

        src.fill_(rank + 1)
        dst.fill_(-1)
        torch.cuda.synchronize()
        ctx.barrier()

        shard = dst_ptrs + rank * n_elements * dst.element_size()
        for r in range(world_size):
            _put_kernel[(1,)](src, shard, n_elements, r, BLOCK_SIZE=256)
        torch.cuda.synchronize()
        ctx.barrier()

        expected = torch.cat(
            [torch.full((n_elements,), r + 1, dtype=torch.bfloat16, device=dst.device) for r in range(world_size)]
        )
        torch.testing.assert_close(dst, expected)
    finally:
        ctx.barrier()
        del ctx
        gc.collect()
