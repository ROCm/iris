# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.

"""
Test allocate_symmetric().

The kernel takes a pointer and a peer-base table as two ordinary arguments and
inlines the translation, so the same device code works for a tensor from any
provider.
"""

import gc

import pytest
import torch
import triton
import triton.language as tl

import iris


@triton.jit
def _put_translated_kernel(
    src,
    dst,
    dst_peer_bases,
    n_elements,
    target_rank,
    CUR_RANK: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    """Copy src into dst on target_rank.

    target_rank is runtime: a collective loops over peers, and specializing on
    it would compile one kernel per destination.
    """
    offsets = tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements

    # Loaded once, outside the access path.
    local_base = tl.load(dst_peer_bases + CUR_RANK)
    remote_base = tl.load(dst_peer_bases + target_rank)

    # Same offset within the allocation, resolved against the peer's base.
    offset = tl.cast(dst, tl.uint64) - local_base
    remote_base_byte = tl.cast(remote_base, tl.pointer_type(tl.int8))
    remote_dst = tl.cast(remote_base_byte + offset, dst.dtype)

    values = tl.load(src + offsets, mask=mask)
    tl.store(remote_dst + offsets, values, mask=mask)


def test_allocate_symmetric_returns_peer_bases():
    """The table is device-resident, rank-indexed, and holds our own base."""
    ctx = iris.iris(1 << 20)

    try:
        tensor, peer_bases = ctx.allocate_symmetric(1024, dtype=torch.float32)

        assert tensor.shape == (1024,)
        assert tensor.dtype == torch.float32
        assert ctx.is_symmetric(tensor)

        assert peer_bases.numel() == ctx.get_num_ranks()
        assert peer_bases.dtype in (torch.int64, torch.uint64)
        assert peer_bases.is_cuda

        # peer_bases[cur_rank] is what translation subtracts, so the tensor has
        # to sit inside the heap it points at.
        local_base = int(peer_bases[ctx.get_rank()].item())
        assert local_base <= tensor.data_ptr()
        assert tensor.data_ptr() + tensor.nbytes <= local_base + ctx.heap_size
    finally:
        ctx.barrier()
        del ctx
        gc.collect()


def test_view_translates_against_allocation_root():
    """A view keeps its offset within the allocation across translation.

    Subtracting the view pointer instead of the allocation base would land at
    the start of the peer's allocation.
    """
    ctx = iris.iris(1 << 24)
    rank = ctx.get_rank()
    world_size = ctx.get_num_ranks()
    target_rank = (rank + 1) % world_size

    n_elements = 512
    offset = 128

    try:
        src, _ = ctx.allocate_symmetric(n_elements, dtype=torch.float32)
        dst, dst_peer_bases = ctx.allocate_symmetric(n_elements, dtype=torch.float32)

        view = dst[offset:]
        assert view.data_ptr() != int(dst_peer_bases[rank].item())

        src.fill_(rank + 1)
        dst.fill_(-1)
        ctx.barrier()

        _put_translated_kernel[(1,)](
            src,
            view,
            dst_peer_bases,
            view.numel(),
            target_rank,
            CUR_RANK=rank,
            BLOCK_SIZE=512,
        )
        ctx.barrier()

        source_rank = (rank - 1) % world_size
        # The view's region received the peer's data; everything before it did not.
        torch.testing.assert_close(dst[offset:], torch.full_like(dst[offset:], source_rank + 1))
        torch.testing.assert_close(dst[:offset], torch.full_like(dst[:offset], -1.0))

        if world_size > 1:
            # At one rank the sender is the receiver, so the assertions above
            # hold even if translation never left this rank.
            assert dst[offset].item() != rank + 1
    finally:
        ctx.barrier()
        del ctx
        gc.collect()


@pytest.mark.parametrize("n_elements", [256, 200])
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16, torch.float32])
def test_allocate_symmetric_remote_put(dtype, n_elements):
    """A kernel given (pointer, peer_bases) reaches the right peer allocation."""
    ctx = iris.iris(1 << 24)
    rank = ctx.get_rank()
    world_size = ctx.get_num_ranks()
    target_rank = (rank + 1) % world_size

    # 200 is not a multiple of the block, so the mask is exercised.
    block_size = 256

    try:
        src, _ = ctx.allocate_symmetric(n_elements, dtype=dtype)
        dst, dst_peer_bases = ctx.allocate_symmetric(n_elements, dtype=dtype)

        # Distinct per rank, so a write landing on the wrong rank gives a wrong
        # answer rather than a plausible one.
        src.fill_(rank + 1)
        dst.fill_(-1)
        ctx.barrier()

        _put_translated_kernel[(1,)](
            src,
            dst,
            dst_peer_bases,
            n_elements,
            target_rank,
            CUR_RANK=rank,
            BLOCK_SIZE=block_size,
        )
        ctx.barrier()

        # We received from the rank targeting us, not the one we target.
        source_rank = (rank - 1) % world_size
        torch.testing.assert_close(dst, torch.full_like(dst, source_rank + 1))

        if world_size > 1:
            # A translation resolving to the local allocation would leave our
            # own value here, and the check above would still pass at 1 rank.
            assert dst[0].item() != rank + 1
    finally:
        ctx.barrier()
        del ctx
        gc.collect()
