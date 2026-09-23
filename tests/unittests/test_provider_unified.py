# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.

"""
One kernel, two tensors, two peer tables, across providers and backends.

Each tensor translates against its own table. Iris and rocSHMEM allocate from
one heap, so the peer delta is shared and any table happens to translate any
pointer; Torch Symmetric Memory allocates per tensor, so the deltas differ and
reusing a table mistranslates.

Run:
    python tests/run_tests_distributed.py tests/unittests/test_provider_unified.py --num_ranks 2
"""

import pytest
import torch
import torch.distributed as dist
import triton
import triton.language as tl
from triton.experimental import gluon
from triton.experimental.gluon import language as gl

import iris

BLOCK_SIZE = 256


@triton.jit
def _remote_read_scale_write_triton(
    a,
    a_peers,
    b,
    b_peers,
    n_elements,
    peer,
    BLOCK_SIZE: tl.constexpr,
):
    """Read the peer's ``a``, scale it, write into the peer's ``b``."""
    offsets = tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements

    # Entries are already the peer's address for that tensor, so there is
    # nothing to translate. a and b are here for their element type.
    src = tl.load(a_peers + peer).to(a.dtype, bitcast=True)
    dst = tl.load(b_peers + peer).to(b.dtype, bitcast=True)

    value = tl.load(src + offsets, mask=mask)
    tl.store(dst + offsets, value * 2, mask=mask)


@gluon.jit
def _remote_read_scale_write_gluon(
    a,
    a_peers,
    b,
    b_peers,
    n_elements,
    peer,
    WARP_SIZE: gl.constexpr,
    BLOCK_SIZE: gl.constexpr,
):
    """Gluon form of the same kernel."""
    # Warp size is 64 on CDNA, 32 on NVIDIA/RDNA; a layout that disagrees with
    # the block does not compile.
    layout: gl.constexpr = gl.BlockedLayout([BLOCK_SIZE // WARP_SIZE], [WARP_SIZE], [1], [0])
    offsets = gl.arange(0, BLOCK_SIZE, layout=layout)
    mask = offsets < n_elements

    src = gl.load(a_peers + peer).to(a.dtype, bitcast=True)
    dst = gl.load(b_peers + peer).to(b.dtype, bitcast=True)

    value = gl.load(src + offsets, mask=mask)
    gl.store(dst + offsets, value * 2, mask=mask)


BACKENDS = {
    "triton": _remote_read_scale_write_triton,
    "gluon": _remote_read_scale_write_gluon,
}


def _make_rocshmem(request):
    # rocshmem_runtime is session-scoped so the runtime is initialised once per
    # process, however many modules ask for it.
    request.getfixturevalue("rocshmem_runtime")
    from iris.experimental.rocshmem_provider import RocshmemProvider

    return RocshmemProvider()


# Needs allocate_symmetric / get_rank / get_num_ranks / barrier. An Iris
# context already has all four, so it goes in unwrapped.
PROVIDERS = {
    "iris": lambda request: iris.iris(1 << 24),
    "rocshmem": _make_rocshmem,
}


@pytest.fixture(params=sorted(PROVIDERS), scope="module")
def provider(request):
    if not dist.is_initialized():
        pytest.skip("needs torch.distributed; run via tests/run_tests_distributed.py")
    if dist.get_world_size() < 2:
        pytest.skip("needs at least 2 ranks (--num_ranks 2)")
    # Built here, not at module scope: importorskip there collects zero items,
    # which exits pytest 5 and fails the whole distributed run.
    return PROVIDERS[request.param](request)


def _warp_size():
    """Lanes per warp on the active target."""
    return triton.runtime.driver.active.get_current_target().warp_size


@pytest.fixture
def symmetric(provider):
    """Allocate through the provider; release on teardown, pass or fail."""
    allocated = []

    def alloc(*size, dtype=None):
        tensor, table = provider.allocate_symmetric(*size, dtype=dtype)
        allocated.append(tensor)
        return tensor, table

    yield alloc

    # rocshmem_free is collective, so every rank frees the same allocations in
    # the same order. Iris has no free; it releases on heap teardown.
    provider.barrier()
    free = getattr(provider, "free", None)
    if free is not None:
        for tensor in allocated:
            free(tensor)


def _deallocate(provider, tensor):
    """Release a symmetric allocation.

    Stub until Iris grows a deallocate. rocSHMEM frees explicitly because
    rocshmem_free is collective; Iris reclaims on heap teardown, except the
    chunked allocator which reclaims by GC finalizer.
    """
    free = getattr(provider, "free", None)
    if free is not None:
        free(tensor)


@pytest.fixture
def symmetric(provider):
    """Allocate through the provider; release on teardown, pass or fail."""
    allocated = []

    def alloc(*size, dtype=None):
        tensor, table = provider.allocate_symmetric(*size, dtype=dtype)
        allocated.append(tensor)
        return tensor, table

    yield alloc

    # Collective where implemented, so every rank releases the same
    # allocations in the same order.
    provider.barrier()
    for tensor in allocated:
        _deallocate(provider, tensor)


@pytest.mark.parametrize("backend", sorted(BACKENDS))
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float32])
def test_two_tensors_two_tables(provider, symmetric, dtype, backend):
    """Each tensor translates through its own table, in one kernel."""
    rank = provider.get_rank()
    world_size = provider.get_num_ranks()
    peer = (rank + 1) % world_size

    warp_size = _warp_size()
    assert BLOCK_SIZE % warp_size == 0, f"block {BLOCK_SIZE} must divide into warps of {warp_size}"

    # Keep the allocations together. A rank that dies between them frees a
    # different count than its peers, and a collective free then hangs.
    a, a_peers = symmetric(BLOCK_SIZE, dtype=dtype)
    b, b_peers = symmetric(BLOCK_SIZE, dtype=dtype)

    # The invariant device translation subtracts, per tensor not per heap.
    assert int(a_peers[rank].item()) == a.data_ptr()
    assert int(b_peers[rank].item()) == b.data_ptr()

    # Distinct per rank, so a wrong peer gives a wrong answer not a plausible one.
    a.fill_(rank + 1)
    b.fill_(-1)
    torch.cuda.synchronize()
    provider.barrier()

    extra = {"WARP_SIZE": warp_size} if backend == "gluon" else {}
    BACKENDS[backend][(1,)](
        a,
        a_peers,
        b,
        b_peers,
        BLOCK_SIZE,
        peer,
        BLOCK_SIZE=BLOCK_SIZE,
        num_warps=1,
        **extra,
    )
    torch.cuda.synchronize()
    provider.barrier()

    # Written by the rank targeting us, which read our a (rank + 1) and doubled
    # it -- so this proves both the remote read and the remote write landed here.
    torch.testing.assert_close(b, torch.full_like(b, 2 * (rank + 1)))
    # Nothing writes through a's table.
    torch.testing.assert_close(a, torch.full_like(a, rank + 1))

    provider.barrier()
