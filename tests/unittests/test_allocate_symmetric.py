# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.

"""
Test allocate_symmetric().

The kernel receives one pointer per rank as ordinary arguments and selects
among them. There is no address arithmetic and no base table to load from:
translation happened on the host, once per allocation.
"""

import gc

import pytest
import torch
import triton
import triton.language as tl

import iris


@triton.jit
def _put_kernel(src, peers, n_elements, target_rank, N_RANKS: tl.constexpr, BLOCK_SIZE: tl.constexpr):
    """Copy src into the target rank's view of the same allocation."""
    offsets = tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements

    # target_rank is runtime, so select over an unrolled tuple rather than
    # indexing it. The pointers are kernel arguments, not memory.
    dst = peers[0].to(tl.int64, bitcast=True)
    for r in tl.static_range(N_RANKS):
        dst = tl.where(r == target_rank, peers[r].to(tl.int64, bitcast=True), dst)
    dst = tl.multiple_of(dst.to(src.dtype, bitcast=True), 16)

    tl.store(dst + offsets, tl.load(src + offsets, mask=mask), mask=mask)


@triton.jit
def _put_kernel_static(src, peers, n_elements, TARGET: tl.constexpr, BLOCK_SIZE: tl.constexpr):
    """Same, when the destination is known at compile time: a direct index."""
    offsets = tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    dst = tl.multiple_of(peers[TARGET], 16)
    tl.store(dst + offsets, tl.load(src + offsets, mask=mask), mask=mask)


def test_allocate_symmetric_returns_peer_views():
    """One view per rank, and our own entry aliases the local tensor."""
    ctx = iris.iris(1 << 20)
    rank = ctx.get_rank()
    world_size = ctx.get_num_ranks()

    try:
        tensor, peers = ctx.allocate_symmetric(1024, dtype=torch.float32)

        assert len(peers) == world_size
        assert ctx.is_symmetric(tensor)

        assert peers[rank].data_ptr() == tensor.data_ptr()
        assert all(p.shape == tensor.shape and p.dtype == tensor.dtype for p in peers)

        # Writing through our own view is writing the local tensor.
        tensor.fill_(-1)
        peers[rank].fill_(7)
        assert tensor[0].item() == 7

        if world_size > 1:
            # Distinct allocations, so a peer view is a different address.
            assert peers[(rank + 1) % world_size].data_ptr() != tensor.data_ptr()
    finally:
        ctx.barrier()
        del ctx
        gc.collect()


def test_view_translates_against_allocation_root():
    """A view of the allocation maps to the same region on the peer."""
    ctx = iris.iris(1 << 24)
    rank = ctx.get_rank()
    world_size = ctx.get_num_ranks()
    target_rank = (rank + 1) % world_size

    n_elements = 512
    offset = 128

    try:
        src, _ = ctx.allocate_symmetric(n_elements, dtype=torch.float32)
        dst, _ = ctx.allocate_symmetric(n_elements, dtype=torch.float32)

        # The peer tuple for a view describes the view's region, not the
        # allocation base.
        view = dst[offset:]
        view_peers = ctx.peer_views(view)
        assert view_peers[rank].data_ptr() == view.data_ptr()

        src.fill_(rank + 1)
        dst.fill_(-1)
        ctx.barrier()

        _put_kernel[(1,)](
            src,
            view_peers,
            view.numel(),
            target_rank,
            N_RANKS=world_size,
            BLOCK_SIZE=512,
        )
        ctx.barrier()

        source_rank = (rank - 1) % world_size
        torch.testing.assert_close(dst[offset:], torch.full_like(dst[offset:], source_rank + 1))
        torch.testing.assert_close(dst[:offset], torch.full_like(dst[:offset], -1.0))

        if world_size > 1:
            assert dst[offset].item() != rank + 1
    finally:
        ctx.barrier()
        del ctx
        gc.collect()


@pytest.mark.parametrize("n_elements,block_size", [(256, 256), (200, 256), (1024, 1024)])
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16, torch.float32])
def test_remote_put_runtime_rank(dtype, n_elements, block_size):
    """Runtime destination, selected out of the unrolled peer tuple.

    block_size 1024 puts four elements in a lane, which is where the alignment
    hint changes codegen; at 256 it is inert.
    """
    ctx = iris.iris(1 << 24)
    rank = ctx.get_rank()
    world_size = ctx.get_num_ranks()
    target_rank = (rank + 1) % world_size

    try:
        src, _ = ctx.allocate_symmetric(n_elements, dtype=dtype)
        dst, dst_peers = ctx.allocate_symmetric(n_elements, dtype=dtype)

        src.fill_(rank + 1)
        dst.fill_(-1)
        ctx.barrier()

        _put_kernel[(1,)](
            src,
            dst_peers,
            n_elements,
            target_rank,
            N_RANKS=world_size,
            BLOCK_SIZE=block_size,
        )
        ctx.barrier()

        source_rank = (rank - 1) % world_size
        torch.testing.assert_close(dst, torch.full_like(dst, source_rank + 1))

        if world_size > 1:
            assert dst[0].item() != rank + 1
    finally:
        ctx.barrier()
        del ctx
        gc.collect()


def test_remote_put_static_rank():
    """Compile-time destination needs no select at all."""
    ctx = iris.iris(1 << 24)
    rank = ctx.get_rank()
    world_size = ctx.get_num_ranks()
    target_rank = (rank + 1) % world_size

    n_elements = 256

    try:
        src, _ = ctx.allocate_symmetric(n_elements, dtype=torch.float32)
        dst, dst_peers = ctx.allocate_symmetric(n_elements, dtype=torch.float32)

        src.fill_(rank + 1)
        dst.fill_(-1)
        ctx.barrier()

        _put_kernel_static[(1,)](src, dst_peers, n_elements, TARGET=target_rank, BLOCK_SIZE=256)
        ctx.barrier()

        source_rank = (rank - 1) % world_size
        torch.testing.assert_close(dst, torch.full_like(dst, source_rank + 1))

        if world_size > 1:
            assert dst[0].item() != rank + 1
    finally:
        ctx.barrier()
        del ctx
        gc.collect()
