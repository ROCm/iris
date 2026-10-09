# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.

"""Iris device kernels driving PyTorch symmetric memory.

Run under the usual launcher, which sets up torch.distributed and the device:

    python tests/run_tests_distributed.py tests/unittests/test_torch_symm_mem_provider.py \
        --num_ranks 2 -v

Skips when torch symmetric memory cannot allocate on this build, when fewer than
2 ranks are present, or when peers are not directly addressable, so it is inert
rather than failing in a normal CI run.

The table invariant and per-allocation anchoring are covered for every provider
by test_provider_unified.py; this module holds what is specific to this one.
"""

import pytest
import torch
import triton
import triton.language as tl

import iris

BLOCK_SIZE = 1024


@triton.jit
def _broadcast_kernel(
    data, results, peer_bases, n_elements, cur_rank, num_ranks: tl.constexpr, BLOCK_SIZE: tl.constexpr
):
    """Ordinary Iris device code -- unaware the table came from torch."""
    offsets = tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    value = tl.load(data + offsets, mask=mask)
    for dst_rank in range(num_ranks):
        iris.store(results + offsets, value, cur_rank, dst_rank, peer_bases, mask=mask)


@pytest.fixture(scope="module")
def provider(torch_symm_mem_provider):
    # Probe and skips live in conftest, shared with test_provider_unified.py.
    return torch_symm_mem_provider


@pytest.fixture
def symmetric_pair(provider):
    """Two allocations, each with its own table.

    Both tables are yielded because a table built for one torch allocation need
    not translate another's pointers: on the default backend each allocation is
    a separate IPC mapping rather than a window onto one linear heap. A kernel
    must translate against the table of the allocation it is addressing.
    """
    data, data_bases = provider.allocate_symmetric(BLOCK_SIZE, dtype=torch.float32)
    results, results_bases = provider.allocate_symmetric(BLOCK_SIZE, dtype=torch.float32)
    yield provider, data, results, data_bases, results_bases
    provider.barrier()
    provider.free(data)
    provider.free(results)


def test_address_map_reports_reachability(symmetric_pair):
    """Per-peer reachability, and the 0 base that goes with it."""
    provider, _data, results, _data_bases, _results_bases = symmetric_pair
    amap = provider.symmetric_address_map(results)
    ws = provider.get_num_ranks()

    assert len(amap.direct) == ws
    assert amap.direct[provider.get_rank()], "a rank must be able to reach itself"
    assert amap.allocation_base == results.data_ptr()
    assert amap.allocation_bytes == results.numel() * results.element_size()
    # A non-direct peer's base is 0.
    for peer, is_direct in enumerate(amap.direct):
        assert (int(amap.peer_bases[peer].item()) != 0) == is_direct


def test_foreign_tensor_is_rejected(provider):
    """A tensor this provider never allocated has no handle to describe it."""
    stray = torch.zeros(8, dtype=torch.float32, device="cuda")
    with pytest.raises(KeyError):
        provider.symmetric_address_map(stray)


def test_iris_store_over_torch_symm_mem_memory(symmetric_pair):
    """Unmodified iris.store, on memory Iris did not allocate."""
    provider, data, results, _data_bases, results_bases = symmetric_pair
    me, ws = provider.get_rank(), provider.get_num_ranks()

    amap = provider.symmetric_address_map(results)
    if not amap.all_direct():
        pytest.skip(f"peers {amap.indirect_peers()} are not directly addressable; this path is intra-node only")

    data.fill_(float(me + 1))
    results.fill_(-1.0)
    torch.cuda.synchronize()
    provider.barrier()

    if me == 0:
        # results_bases, not data_bases: the store addresses `results`, and
        # each allocation has its own mapping. tl.load of `data` is local and
        # needs no translation.
        _broadcast_kernel[(1,)](
            data, results, results_bases, BLOCK_SIZE, me, num_ranks=ws, BLOCK_SIZE=BLOCK_SIZE, num_warps=4
        )
        torch.cuda.synchronize()
    provider.barrier()
    # Every rank, not just the writer: the barrier orders the ranks, but the
    # reading rank still has its own stream to drain before the inbound writes
    # are guaranteed visible to work it queues next.
    torch.cuda.synchronize()

    # Rank 0 pushed its value to every rank, including this one.
    assert torch.allclose(results, torch.full_like(results, 1.0))
