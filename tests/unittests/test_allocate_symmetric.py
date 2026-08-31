# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.

"""
Test the allocator-agnostic allocate_symmetric() API.

The kernel here deliberately does not call any Iris translation helper. It
takes a pointer and a peer-base table as two ordinary kernel arguments and
inlines the translation, which is the form the allocator-agnostic interface is
specified against: identical device code has to work for a tensor from any
provider, and hoisting the base loads out of the access path is only possible
when the translation is written by hand.
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
    """Copy src into dst on target_rank, translating dst by hand."""
    offsets = tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements

    # Hoisted: both base loads happen once, not once per access. target_rank is
    # a runtime argument because a collective loops over peers; specializing on
    # it would compile a separate kernel per destination.
    local_base = tl.load(dst_peer_bases + CUR_RANK)
    remote_base = tl.load(dst_peer_bases + target_rank)

    # Translate the allocation, then index into it.
    offset = tl.cast(dst, tl.uint64) - local_base
    remote_base_byte = tl.cast(remote_base, tl.pointer_type(tl.int8))
    remote_dst = tl.cast(remote_base_byte + offset, dst.dtype)

    # Re-apply the contiguity hint by hand. iris.load/store take a `hint` that
    # does this inside __translate, and every production collective passes one;
    # inlining the translation drops it unless the author puts it back. Manual
    # translation buys control over hoisting, not a free win over the helper.
    #
    # It has to go here rather than on remote_dst: translating the base once and
    # indexing after leaves the translated pointer scalar, and max_contiguous
    # takes a block whose shape matches the hint.
    remote_ptrs = remote_dst + offsets
    remote_ptrs = tl.max_contiguous(tl.multiple_of(remote_ptrs, BLOCK_SIZE), BLOCK_SIZE)

    values = tl.load(src + offsets, mask=mask)
    tl.store(remote_ptrs, values, mask=mask)


def test_allocate_symmetric_returns_peer_bases():
    """The table is device-resident, rank-indexed, and holds our own base."""
    shmem = iris.iris(1 << 20)

    try:
        tensor, peer_bases = shmem.allocate_symmetric(1024, dtype=torch.float32)

        assert tensor.shape == (1024,)
        assert tensor.dtype == torch.float32
        assert shmem.is_symmetric(tensor)

        assert peer_bases.numel() == shmem.get_num_ranks()
        assert peer_bases.dtype in (torch.int64, torch.uint64)
        assert peer_bases.is_cuda

        # peer_bases[cur_rank] is the base translation subtracts, so it has to
        # be this rank's own heap base and the tensor has to sit inside it.
        local_base = int(peer_bases[shmem.get_rank()].item())
        assert local_base == int(shmem.get_heap_bases()[shmem.get_rank()].item())
        assert tensor.data_ptr() >= local_base

        # An external tensor has no table to hand out.
        external = torch.zeros(1024, dtype=torch.float32, device=shmem.get_device())
        with pytest.raises(ValueError, match="not on the Iris symmetric heap"):
            shmem.get_peer_bases(external)
    finally:
        shmem.barrier()
        del shmem
        gc.collect()


def test_view_translates_against_allocation_root():
    """A view translates against the allocation root, not its own pointer.

    The kernel subtracts peer_bases[local_rank] -- the allocation base -- so a
    view's offset within the allocation survives translation. Subtracting the
    view pointer instead would land at the start of the peer's allocation.
    """
    shmem = iris.iris(1 << 24)
    rank = shmem.get_rank()
    world_size = shmem.get_num_ranks()
    target_rank = (rank + 1) % world_size

    n_elements = 512
    offset = 128

    try:
        src, _ = shmem.allocate_symmetric(n_elements, dtype=torch.float32)
        dst, dst_peer_bases = shmem.allocate_symmetric(n_elements, dtype=torch.float32)

        # A view into the middle of the allocation. Its data_ptr is not the
        # allocation base, but it translates against the same table.
        view = dst[offset:]
        assert view.data_ptr() != int(dst_peer_bases[rank].item())

        src.fill_(rank + 1)
        dst.fill_(-1)
        shmem.barrier()

        _put_translated_kernel[(1,)](
            src,
            view,
            dst_peer_bases,
            view.numel(),
            target_rank,
            CUR_RANK=rank,
            BLOCK_SIZE=512,
        )
        shmem.barrier()

        source_rank = (rank - 1) % world_size
        # The view's region received the peer's data; everything before it did not.
        torch.testing.assert_close(dst[offset:], torch.full_like(dst[offset:], source_rank + 1))
        torch.testing.assert_close(dst[:offset], torch.full_like(dst[:offset], -1.0))
    finally:
        shmem.barrier()
        del shmem
        gc.collect()


@pytest.mark.parametrize("n_elements", [256, 200])
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16, torch.float32])
def test_allocate_symmetric_remote_put(dtype, n_elements):
    """A kernel given (pointer, peer_bases) reaches the right peer allocation."""
    shmem = iris.iris(1 << 24)
    rank = shmem.get_rank()
    world_size = shmem.get_num_ranks()
    target_rank = (rank + 1) % world_size

    # 200 is not a multiple of the block, so the mask is actually exercised.
    block_size = 256

    try:
        src, _ = shmem.allocate_symmetric(n_elements, dtype=dtype)
        dst, dst_peer_bases = shmem.allocate_symmetric(n_elements, dtype=dtype)

        # Each rank stamps a distinct value so a write landing on the wrong
        # rank produces the wrong answer rather than a plausible one.
        src.fill_(rank + 1)
        dst.fill_(-1)
        shmem.barrier()

        _put_translated_kernel[(1,)](
            src,
            dst,
            dst_peer_bases,
            n_elements,
            target_rank,
            CUR_RANK=rank,
            BLOCK_SIZE=block_size,
        )
        shmem.barrier()

        # We received from the rank that targets us, not from the rank we target.
        source_rank = (rank - 1) % world_size
        expected = torch.full_like(dst, source_rank + 1)
        torch.testing.assert_close(dst, expected)

        if world_size > 1:
            # Pin the failure mode: a translation that silently resolved to the
            # local allocation would leave this rank's own value here, and
            # assert_close above would still pass at world_size 1.
            assert dst[0].item() != rank + 1
    finally:
        shmem.barrier()
        del shmem
        gc.collect()
