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

    values = tl.load(src + offsets, mask=mask)
    tl.store(remote_dst + offsets, values, mask=mask)


def test_allocate_symmetric_descriptor():
    """The descriptor describes the tensor's backing allocation."""
    shmem = iris.iris(1 << 20)

    try:
        tensor, address_map = shmem.allocate_symmetric(1024, dtype=torch.float32)

        assert tensor.shape == (1024,)
        assert tensor.dtype == torch.float32
        assert shmem.is_symmetric(tensor)

        # The invariant device translation depends on: subtracting the local
        # base and adding a peer base is only correct if these agree.
        local_base = int(address_map.peer_bases[address_map.local_rank].item())
        assert local_base == address_map.allocation_base

        assert address_map.local_rank == shmem.get_rank()
        assert address_map.world_size == shmem.get_num_ranks()
        assert address_map.owns(tensor)

        # A tensor outside the heap is not covered by this allocation.
        external = torch.zeros(1024, dtype=torch.float32, device=shmem.get_device())
        assert not address_map.owns(external)
    finally:
        shmem.barrier()
        del shmem
        gc.collect()


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16, torch.float32])
def test_allocate_symmetric_remote_put(dtype):
    """A kernel given (pointer, peer_bases) reaches the right peer allocation."""
    shmem = iris.iris(1 << 24)
    rank = shmem.get_rank()
    world_size = shmem.get_num_ranks()
    target_rank = (rank + 1) % world_size

    n_elements = 256
    block_size = 256

    try:
        src, _ = shmem.allocate_symmetric(n_elements, dtype=dtype)
        dst, dst_map = shmem.allocate_symmetric(n_elements, dtype=dtype)

        # Each rank stamps a distinct value so a write landing on the wrong
        # rank produces the wrong answer rather than a plausible one.
        src.fill_(rank + 1)
        dst.fill_(-1)
        shmem.barrier()

        _put_translated_kernel[(1,)](
            src,
            dst,
            dst_map.peer_bases,
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
