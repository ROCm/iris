# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.

"""Iris device kernels driving rocSHMEM-allocated memory.

Run under the usual launcher, which sets up torch.distributed and the device:

    python tests/run_tests_distributed.py tests/unittests/test_rocshmem_provider.py \
        --num_ranks 2 -v

Skips when rocshmem4py is absent, when fewer than 2 ranks are present, or when
peers are not directly addressable, so it is inert rather than failing in a
normal CI run. tests/manual_rocshmem_provider.py covers the multi-node case.
"""

import pytest
import torch
import torch.distributed as dist
import triton
import triton.language as tl

import iris

BLOCK_SIZE = 1024


@triton.jit
def _broadcast_kernel(data, results, peer_bases, n_elements, cur_rank,
                      num_ranks: tl.constexpr, BLOCK_SIZE: tl.constexpr):
    """Ordinary Iris device code -- unaware the table came from rocSHMEM."""
    offsets = tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    value = tl.load(data + offsets, mask=mask)
    for dst_rank in range(num_ranks):
        iris.store(results + offsets, value, cur_rank, dst_rank, peer_bases, mask=mask)


@pytest.fixture(scope="module")
def provider():
    if not dist.is_initialized():
        pytest.skip("needs torch.distributed; run via tests/run_tests_distributed.py")
    if dist.get_world_size() < 2:
        pytest.skip("needs at least 2 ranks (--num_ranks 2)")

    # Imported here rather than at module scope so the tests are collected and
    # individually skipped. A module-level importorskip collects zero items,
    # which makes pytest exit 5 (NO_TESTS_COLLECTED) and fails the whole run.
    rshmem = pytest.importorskip(
        "rocshmem4py", reason="rocSHMEM provider tests need rocshmem4py installed"
    )
    from iris.experimental.rocshmem_provider import RocshmemProvider

    # rocSHMEM initialises once per process, hence module scope. No finalize in
    # teardown: it would pull the runtime out from under anything else running.
    rshmem.init_rocshmem_by_uniqueid(dist.group.WORLD)
    return RocshmemProvider()


@pytest.fixture
def symmetric_pair(provider):
    data, peer_bases = provider.allocate_symmetric(BLOCK_SIZE, dtype=torch.float32)
    results, _ = provider.allocate_symmetric(BLOCK_SIZE, dtype=torch.float32)
    yield provider, data, results, peer_bases
    provider.barrier()
    provider.free(data)
    provider.free(results)


def test_peer_bases_shape_and_invariant(symmetric_pair):
    """The invariant Iris device code translates against."""
    provider, data, _results, peer_bases = symmetric_pair
    ws = provider.get_num_ranks()

    assert peer_bases.numel() == ws
    assert peer_bases.dtype == torch.int64
    assert peer_bases.is_cuda
    # peer_bases[local_rank] is the base translation subtracts.
    assert int(peer_bases[provider.get_rank()].item()) == data.data_ptr()


def test_peer_offsets_are_shared(provider):
    """Each allocation gets its own table, built from shared per-peer offsets.

    Equal offsets are what make a table from one allocation able to translate
    another's pointers, which iris.copy depends on.
    """
    a, bases_a = provider.allocate_symmetric(64, dtype=torch.float32)
    b, bases_b = provider.allocate_symmetric(64, dtype=torch.float32)
    try:
        assert int(bases_a[provider.get_rank()].item()) == a.data_ptr()
        assert int(bases_b[provider.get_rank()].item()) == b.data_ptr()
        direct = provider.symmetric_address_map(a).direct
        for peer in range(provider.get_num_ranks()):
            if not direct[peer]:
                # Unreachable peers are 0 in every table, not base + offset.
                assert int(bases_a[peer].item()) == 0
                assert int(bases_b[peer].item()) == 0
                continue
            da = int(bases_a[peer].item()) - a.data_ptr()
            db = int(bases_b[peer].item()) - b.data_ptr()
            assert da == db, f"peer {peer}: offset {da} != {db}"
    finally:
        provider.barrier()
        provider.free(a)
        provider.free(b)


def test_address_map_reports_reachability(symmetric_pair):
    """Per-peer reachability, and the 0 base that goes with it."""
    provider, _data, results, _peer_bases = symmetric_pair
    amap = provider.symmetric_address_map(results)
    ws = provider.get_num_ranks()

    assert len(amap.direct) == ws
    assert amap.direct[provider.get_rank()], "a rank must be able to reach itself"
    assert amap.allocation_base == results.data_ptr()
    assert amap.allocation_bytes == results.numel() * results.element_size()
    # A non-direct peer's base is 0.
    for peer, is_direct in enumerate(amap.direct):
        assert (int(amap.peer_bases[peer].item()) != 0) == is_direct


def test_iris_store_over_rocshmem_memory(symmetric_pair):
    """Unmodified iris.store, on memory Iris did not allocate."""
    provider, data, results, peer_bases = symmetric_pair
    me, ws = provider.get_rank(), provider.get_num_ranks()

    amap = provider.symmetric_address_map(results)
    if not amap.all_direct():
        pytest.skip(f"peers {amap.indirect_peers()} are not directly addressable; "
                    "this path is intra-node only")

    data.fill_(float(me + 1))
    results.fill_(-1.0)
    torch.cuda.synchronize()
    provider.barrier()

    if me == 0:
        _broadcast_kernel[(1,)](data, results, peer_bases, BLOCK_SIZE, me,
                                num_ranks=ws, BLOCK_SIZE=BLOCK_SIZE, num_warps=4)
        torch.cuda.synchronize()
    provider.barrier()

    # Rank 0 pushed its value to every rank, including this one.
    assert torch.allclose(results, torch.full_like(results, 1.0))
