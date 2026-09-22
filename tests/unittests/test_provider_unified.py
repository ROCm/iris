# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.

"""
One kernel, two tensors, two peer tables, every provider.

The contract a provider owes Iris is a table whose entry ``r`` is that
tensor's address on rank ``r``. This exercises the part of that contract a
single-tensor test cannot reach: **each tensor is translated against its own
table, never against another tensor's.**

That distinction is not cosmetic. Iris and rocSHMEM both allocate from one
symmetric heap, so the peer delta is a constant of the heap and any table
happens to translate any pointer. Torch Symmetric Memory is symmetric
*memory*, not a symmetric heap — each tensor is its own allocation with its
own peer mapping, so it genuinely has one translation per tensor and the
deltas differ between allocations.

A kernel that reuses one table for two tensors therefore works on the heap
providers and silently mistranslates on the per-tensor one. Keeping the tables
paired with their tensors is what makes the same device code correct
everywhere, so that is what is tested here rather than assumed.

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
def _remote_read_scale_write(
    a,
    a_peers,
    b,
    b_peers,
    n_elements,
    peer,
    CUR_RANK: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    """Read the peer's ``a``, scale it, write into the peer's ``b``.

    Two tensors, two tables, one launch. ``a`` resolves only through
    ``a_peers`` and ``b`` only through ``b_peers``; swapping them is the
    mistake this kernel is shaped to avoid.
    """
    offsets = tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements

    a_local = tl.load(a_peers + CUR_RANK)
    a_remote = tl.load(a_peers + peer)
    src = tl.cast(
        tl.cast(a_remote, tl.pointer_type(tl.int8)) + (tl.cast(a, tl.uint64) - a_local),
        a.dtype,
    )

    b_local = tl.load(b_peers + CUR_RANK)
    b_remote = tl.load(b_peers + peer)
    dst = tl.cast(
        tl.cast(b_remote, tl.pointer_type(tl.int8)) + (tl.cast(b, tl.uint64) - b_local),
        b.dtype,
    )

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
    CUR_RANK: gl.constexpr,
    WARP_SIZE: gl.constexpr,
    BLOCK_SIZE: gl.constexpr,
):
    """Gluon form of the same kernel, translating by hand.

    The device-side contract is a plain table, so it does not depend on which
    backend consumes it. This asserts that: the same two tables drive Gluon
    and Triton to the same result.
    """
    # One warp of WARP_SIZE lanes, BLOCK_SIZE // WARP_SIZE elements each. The
    # warp size comes from the target rather than being assumed: it is 64 on
    # CDNA and 32 on NVIDIA and RDNA, and a layout that disagrees with the
    # block does not compile.
    layout: gl.constexpr = gl.BlockedLayout([BLOCK_SIZE // WARP_SIZE], [WARP_SIZE], [1], [0])
    offsets = gl.arange(0, BLOCK_SIZE, layout=layout)
    mask = offsets < n_elements

    a_local = gl.load(a_peers + CUR_RANK)
    a_remote = gl.load(a_peers + peer)
    src = tl.cast(
        tl.cast(a_remote, gl.pointer_type(gl.int8)) + (tl.cast(a, gl.uint64) - a_local),
        a.dtype,
    )

    b_local = gl.load(b_peers + CUR_RANK)
    b_remote = gl.load(b_peers + peer)
    dst = tl.cast(
        tl.cast(b_remote, gl.pointer_type(gl.int8)) + (tl.cast(b, gl.uint64) - b_local),
        b.dtype,
    )

    value = gl.load(src + offsets, mask=mask)
    gl.store(dst + offsets, value * 2, mask=mask)


# Backend -> kernel. Both take the same arguments and mean the same thing;
# only the dialect differs.
BACKENDS = {
    "triton": _remote_read_scale_write,
    "gluon": _remote_read_scale_write_gluon,
}


class _IrisProvider:
    """Adapter so an Iris context and a standalone provider look the same."""

    name = "iris"

    def __init__(self):
        self._ctx = iris.iris(1 << 24)

    def allocate_symmetric(self, *size, dtype=None):
        return self._ctx.allocate_symmetric(*size, dtype=dtype)

    def get_rank(self):
        return self._ctx.get_rank()

    def get_num_ranks(self):
        return self._ctx.get_num_ranks()

    def barrier(self):
        self._ctx.barrier()


def _make_rocshmem():
    rshmem = pytest.importorskip("rocshmem4py", reason="needs rocshmem4py installed")
    from iris.experimental.rocshmem_provider import RocshmemProvider

    rshmem.init_rocshmem_by_uniqueid(dist.group.WORLD)
    provider = RocshmemProvider()
    provider.name = "rocshmem"
    return provider


# Add new providers here. A provider qualifies if it exposes
# allocate_symmetric / get_rank / get_num_ranks / barrier. Torch Symmetric
# Memory joins this list when its provider lands.
PROVIDERS = {
    "iris": _IrisProvider,
    "rocshmem": _make_rocshmem,
}


@pytest.fixture(params=sorted(PROVIDERS), scope="module")
def provider(request):
    if not dist.is_initialized():
        pytest.skip("needs torch.distributed; run via tests/run_tests_distributed.py")
    if dist.get_world_size() < 2:
        pytest.skip("needs at least 2 ranks (--num_ranks 2)")
    # Built here rather than at module scope so an unavailable provider skips
    # its own parameters instead of collecting zero items, which would make
    # pytest exit 5 and fail the whole distributed run.
    return PROVIDERS[request.param]()


def _warp_size():
    """Lanes per warp on the active target. 64 on CDNA, 32 on NVIDIA/RDNA."""
    return triton.runtime.driver.active.get_current_target().warp_size


@pytest.mark.parametrize("backend", sorted(BACKENDS))
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float32])
def test_two_tensors_two_tables(provider, dtype, backend):
    """Each tensor translates through its own table, in one kernel."""
    rank = provider.get_rank()
    world_size = provider.get_num_ranks()
    peer = (rank + 1) % world_size

    warp_size = _warp_size()
    assert BLOCK_SIZE % warp_size == 0, f"block {BLOCK_SIZE} must divide into warps of {warp_size}"

    a, a_peers = provider.allocate_symmetric(BLOCK_SIZE, dtype=dtype)
    b, b_peers = provider.allocate_symmetric(BLOCK_SIZE, dtype=dtype)

    # Each table names its own tensor on this rank. This is the invariant
    # device translation subtracts, and it is per tensor, not per heap.
    assert int(a_peers[rank].item()) == a.data_ptr()
    assert int(b_peers[rank].item()) == b.data_ptr()

    # Distinct per rank, so data arriving from the wrong peer is a wrong
    # answer rather than a plausible one.
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
        CUR_RANK=rank,
        BLOCK_SIZE=BLOCK_SIZE,
        num_warps=1,
        **extra,
    )
    torch.cuda.synchronize()
    provider.barrier()

    # Our b was written by the rank that targets us. It read OUR a, which held
    # rank + 1, and doubled it -- so the value proves both the remote read and
    # the remote write resolved to this rank.
    torch.testing.assert_close(b, torch.full_like(b, 2 * (rank + 1)))

    # a is untouched: nothing in this kernel writes through a's table.
    torch.testing.assert_close(a, torch.full_like(a, rank + 1))

    provider.barrier()
