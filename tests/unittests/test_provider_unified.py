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
    """Read the peer's ``a``, scale it, write into the peer's ``b``."""
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
    """Gluon form of the same kernel."""
    # Warp size is 64 on CDNA, 32 on NVIDIA/RDNA; a layout that disagrees with
    # the block does not compile.
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


BACKENDS = {
    "triton": _remote_read_scale_write,
    "gluon": _remote_read_scale_write_gluon,
}


class _IrisProvider:
    """Gives an Iris context the same surface as a standalone provider."""

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


# Needs allocate_symmetric / get_rank / get_num_ranks / barrier.
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
    """Lanes per warp on the active target."""
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
        CUR_RANK=rank,
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
