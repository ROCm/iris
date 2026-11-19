# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
#
# RCCL offers its device API only with cuMem, which is off unless asked for, and
# reads the setting once, when torch's process group starts -- before this file
# runs. So .github/scripts/run_tests.sh sets it for this file alone: CI-wide it
# would also change RCCL's own collectives, which are timed as a baseline.
# iris-ci-env: NCCL_CUMEM_ENABLE=1

"""The RCCL GIN provider, the engines=("rdma",) backend over it, and wide EP.

Run under the usual launcher, which sets up torch.distributed and the device,
with cuMem on (see the iris-ci-env line above):

    NCCL_CUMEM_ENABLE=1 python tests/run_tests_distributed.py \\
        tests/unittests/test_rccl_gin_provider.py --num_ranks 2 -v

The shim is built by iris.experimental.rccl_gin itself on first use (into
IRIS_RCCL_GIN_CACHE), unless IRIS_RCCL_GIN_BUILD names a prebuilt one,
against RCCL_HOME or else the RCCL that ships with ROCm; a failed build fails
the tests. Skips when a prerequisite is missing -- no RCCL with
the GIN device API, as with ROCm 7.x -- so it is inert in an image without one. Only RcclGinUnavailable is turned into a skip; any other error
fails. Without a GIN backend (no RDMA NIC, as on CI runners) the load/store
tests still run, since on one node every peer is direct, and the tests that
send over GIN skip.

The launcher runs a single node, so every peer is direct. GIN is still
exercised: with full connectivity RCCL connects every rank pair, so
test_gin_put_signal_reaches_every_peer sends to same-node peers over the NIC.
The multi-node path is covered by examples/35_wide_ep_rccl_gin, not here.
"""

import os

import pytest
import torch
import torch.distributed as dist
import triton
import triton.language as tl

import iris
from iris.experimental import rccl_gin  # importing loads nothing; the shim loads on first use
from iris.rdma import device as rdma

BLOCK = 1024  # int32 words per block
MAX_POLLS = 20_000_000  # bounds every GIN wait, so a lost message fails rather than hangs
FLUSH_CYCLES = 4_000_000_000  # flush_timeout bound: about 2 s, vs microseconds when healthy
FLUSH_ROUNDS = 4  # the first put+flush in a process got through even when broken


def _build_shim():
    """Build the shim (every rank; one builds, the rest wait on its lock). No
    RCCL with GIN headers skips; a build that fails fails."""
    if not os.environ.get("IRIS_RCCL_GIN_BUILD") and rccl_gin._find_rccl_home() is None:
        pytest.skip("no RCCL with the GIN device API (set RCCL_HOME, or use ROCm 10.x)")
    try:
        problem = None
        rccl_gin._build_dir()
    except rccl_gin.RcclGinUnavailable as exc:
        problem = str(exc)
    problems = [None] * dist.get_world_size()
    dist.all_gather_object(problems, problem)
    failed = [p for p in problems if p]
    if failed:
        pytest.fail(failed[0])


@pytest.fixture(scope="module")
def provider():
    if not dist.is_initialized():
        pytest.skip("needs torch.distributed; run via tests/run_tests_distributed.py")
    if dist.get_world_size() < 2:
        pytest.skip("needs at least 2 ranks (--num_ranks 2)")
    _build_shim()
    try:
        prov = rccl_gin.RcclGinProvider()
    except rccl_gin.RcclGinUnavailable as exc:
        pytest.skip(f"RCCL GIN unavailable: {exc}")
    yield prov
    prov.close()


@pytest.fixture(scope="module")
def device_comm(provider):
    """For the tests that send over GIN. Without a GIN backend (e.g. no RDMA
    NIC, as on CI runners) they skip, while the load/store tests still run."""
    if not provider.gin_available:
        pytest.skip("no GIN backend (no RDMA NIC?); every peer is load/store reachable")
    return provider.device_comm(signal_count=provider.get_num_ranks())


# ── layout: pure arithmetic, no device ───────────────────────────────────────


@pytest.mark.parametrize("world,hidden,max_tokens,topk", [(2, 7, 3, 1), (16, 2048, 512, 8), (4, 1024, 1, 4)])
def test_plan_window_layout(world, hidden, max_tokens, topk):
    """Regions are disjoint, in order and aligned; metadata views are aligned
    even for an odd hidden size; each exchange has its own flags and signals."""
    from iris.experimental.wide_ep import _HEADER, _plan_window

    x_bytes = hidden * 2  # bf16
    dispatch, combine, meta_off, total = _plan_window(world, x_bytes, max_tokens, topk)

    assert meta_off >= x_bytes and meta_off % 8 == 0, "int32/fp32 metadata views need 4-byte alignment"
    assert dispatch.row_bytes >= meta_off + topk * 8 and dispatch.row_bytes % 16 == 0
    assert combine.row_bytes >= x_bytes and combine.row_bytes % 16 == 0
    for ex in (dispatch, combine):
        assert ex.slot_bytes == _HEADER + max_tokens * ex.row_bytes
        assert ex.recv_off == ex.send_off + world * ex.slot_bytes
        assert ex.send_off % 16 == 0 and ex.slot_bytes % 16 == 0
    assert dispatch.send_off == 0
    assert combine.send_off == dispatch.recv_off + world * dispatch.slot_bytes
    # Flags follow the slots, one block of world per exchange.
    assert dispatch.flags_off == combine.recv_off + world * combine.slot_bytes
    assert combine.flags_off == dispatch.flags_off + world * 8
    assert dispatch.flags_off % 8 == 0
    assert total >= combine.flags_off + world * 8 and total % 4096 == 0
    # Disjoint signal ranges: dispatch [0, world), combine [world, 2 * world).
    assert (dispatch.signal_off, combine.signal_off) == (0, world)


# ── the provider's address map ───────────────────────────────────────────────


def test_window_address_map(provider):
    """direct matches RCCL's same-node team, and the table honours the Iris
    contract: local entry is the allocation base, direct peers have a base,
    indirect peers have 0."""
    w = provider.allocate_window(1 << 20)
    amap = w.address_map
    me, ws = provider.get_rank(), provider.get_num_ranks()
    bases = amap.peer_bases.tolist()

    assert amap.local_rank == me
    assert amap.allocation_base == w.tensor.data_ptr() == bases[me]
    assert amap.allocation_bytes == w.tensor.numel()
    assert amap.direct == tuple(provider.is_lsa_peer(r) for r in range(ws))
    assert amap.direct[me], "a rank must be able to reach itself"
    # RCCL's LSA team is a consecutive block of world ranks.
    direct_ranks = [r for r in range(ws) if amap.direct[r]]
    assert direct_ranks == list(range(direct_ranks[0], direct_ranks[0] + len(direct_ranks)))
    for r in range(ws):
        assert (bases[r] != 0) == amap.direct[r], f"peer {r}: base {bases[r]:#x}, direct {amap.direct[r]}"


@triton.jit
def _store_to_direct_peers(buf, heap_bases, direct, cur_rank, num_ranks: tl.constexpr, BLOCK: tl.constexpr):
    """Ordinary Iris device code -- unaware the table came from RCCL."""
    offs = tl.arange(0, BLOCK)
    value = tl.full((BLOCK,), cur_rank, tl.int32)
    for dst in range(num_ranks):
        if tl.load(direct + dst) != 0:
            iris.store(buf + cur_rank * BLOCK + offs, value, cur_rank, dst, heap_bases)


def test_iris_store_through_window(provider):
    """Unmodified iris.store writes every direct peer's copy of the window."""
    ws, me = provider.get_num_ranks(), provider.get_rank()
    w = provider.allocate_window(ws * BLOCK * 4)
    buf = w.tensor.view(torch.int32).view(ws, BLOCK)
    buf.fill_(-1)
    direct = torch.tensor(w.address_map.direct, dtype=torch.int32, device=buf.device)
    provider.barrier()

    _store_to_direct_peers[(1,)](buf, w.address_map.peer_bases, direct, me, num_ranks=ws, BLOCK=BLOCK)
    provider.barrier()

    for src in range(ws):
        expect = src if w.address_map.direct[src] else -1
        assert torch.all(buf[src] == expect), f"row from rank {src}"


# ── GIN ──────────────────────────────────────────────────────────────────────


@triton.jit
def _gin_put_to_all(comm, win, rank, BLOCK: tl.constexpr):
    """Program ``peer``: put this rank's send block for ``peer`` into ``peer``'s
    receive slot for this rank, raising signal ``rank`` on ``peer`` by 1."""
    peer = tl.program_id(0)
    world = tl.num_programs(0)
    rccl_gin.put_signal(comm, 0, peer, win, (world + rank) * BLOCK * 4, win, peer * BLOCK * 4, BLOCK * 4, rank, 1)


@triton.jit
def _gin_wait_all(comm, signal_base, seen, max_polls):
    """Program ``src``: bounded wait for signal ``src`` to rise by 1."""
    src = tl.program_id(0)
    base = tl.load(signal_base + src)
    v = rccl_gin.read_signal(comm, 0, src) - base
    polls = 0
    while (v < 1) & (polls < max_polls):
        v = rccl_gin.read_signal(comm, 0, src) - base
        polls += 1
    tl.store(seen + src, v)


def test_gin_put_signal_reaches_every_peer(provider, device_comm):
    """Every rank GIN-puts a tagged block to every rank, itself included. The
    data must be visible once the signal is, with no other synchronisation."""

    ws, me = provider.get_num_ranks(), provider.get_rank()
    w = provider.allocate_window(2 * ws * BLOCK * 4)
    words = w.tensor.view(torch.int32)
    send, recv = words[: ws * BLOCK].view(ws, BLOCK), words[ws * BLOCK :].view(ws, BLOCK)
    recv.fill_(-1)
    send.copy_((me * 1000 + torch.arange(ws, device=send.device, dtype=torch.int32))[:, None].expand(ws, BLOCK))
    base = provider.read_signals(device_comm)
    seen = torch.zeros(ws, dtype=torch.int64, device=send.device)
    provider.barrier()

    lib = rccl_gin.extern_libs()
    _gin_put_to_all[(ws,)](device_comm.ptr, w.handle, me, BLOCK=BLOCK, num_warps=4, extern_libs=lib)
    _gin_wait_all[(ws,)](device_comm.ptr, base, seen, MAX_POLLS, num_warps=4, extern_libs=lib)
    torch.cuda.synchronize()

    assert seen.tolist() == [1] * ws, f"signals seen per source: {seen.tolist()}"
    expect = (torch.arange(ws, device=recv.device, dtype=torch.int32) * 1000 + me)[:, None]
    bad = (recv != expect).any(dim=1).nonzero().flatten().tolist()
    assert not bad, f"wrong blocks from ranks {bad}"
    provider.barrier()


@triton.jit
def _gin_put_then_flush(comm, win, rank, flushed, cycles, BLOCK: tl.constexpr):
    """Program ``peer``: put to ``peer``, then a bounded flush in the same kernel."""
    peer = tl.program_id(0)
    world = tl.num_programs(0)
    rccl_gin.put_signal(comm, 0, peer, win, (world + rank) * BLOCK * 4, win, peer * BLOCK * 4, BLOCK * 4, rank, 1)
    tl.store(flushed + peer, rccl_gin.flush_timeout(comm, 0, cycles))


def test_flush_after_put_in_same_kernel(provider, device_comm):
    """flush must see a put issued earlier in the same kernel. The put's queue
    descriptor has to reach RCCL's CPU proxy while the kernel still runs.
    Without the release in the shim, only the first round in a process
    completed and every later one timed out, so this runs several rounds."""
    ws, me = provider.get_num_ranks(), provider.get_rank()
    w = provider.allocate_window(2 * ws * BLOCK * 4)
    flushed = torch.full((ws,), -1, dtype=torch.int32, device=w.tensor.device)
    seen = torch.zeros(ws, dtype=torch.int64, device=w.tensor.device)
    lib = rccl_gin.extern_libs()

    for rnd in range(FLUSH_ROUNDS):
        base = provider.read_signals(device_comm)
        flushed.fill_(-1)
        provider.barrier()
        _gin_put_then_flush[(ws,)](device_comm.ptr, w.handle, me, flushed, FLUSH_CYCLES, BLOCK=BLOCK, num_warps=4,
                                   extern_libs=lib)  # fmt: skip
        _gin_wait_all[(ws,)](device_comm.ptr, base, seen, MAX_POLLS, num_warps=4, extern_libs=lib)
        torch.cuda.synchronize()
        late = [p for p, f in enumerate(flushed.tolist()) if f]
        assert not late, f"round {rnd}: flush timed out after puts to peers {late}"
        assert seen.tolist() == [1] * ws, f"round {rnd}: signals seen per source: {seen.tolist()}"
    provider.barrier()


def test_read_signals_checks_count(provider):
    """read_signals refuses a count beyond the device comm's signals, which would read
    slots RCCL never allocated. Without GIN there are none, and any count gives zeros."""
    dc = provider.device_comm(signal_count=3)
    try:
        with pytest.raises(ValueError, match="out of range"):
            provider.read_signals(dc, count=-1)
        if provider.gin_available:
            with pytest.raises(ValueError, match="out of range"):
                provider.read_signals(dc, count=4)
            assert provider.read_signals(dc, count=3).shape == (3,)
        else:
            assert torch.equal(provider.read_signals(dc, count=5).cpu(), torch.zeros(5, dtype=torch.int64))
    finally:
        provider.destroy_device_comm(dc)


# ── engines=("rdma",) ────────────────────────────────────────────────────────


@pytest.fixture(scope="module")
def rdma_ctx(provider):
    """An Iris context on the RCCL GIN backend. Depends on ``provider`` only so
    that its skips apply here too (and the shim is built)."""
    ctx = iris.iris(1 << 26, engines=("rdma",), rdma_backend="rccl_gin")
    yield ctx
    ctx.rdma.close()


def test_rdma_heap_is_the_rccl_window(rdma_ctx):
    """The heap is the backend's window: same base, peer bases from RCCL's
    mappings on direct peers and 0 elsewhere, and a fixed size."""
    ctx = rdma_ctx
    assert ctx.rdma.BACKEND_ID == rdma.RCCL_GIN.value, "host and device disagree on the backend id"
    me, ws = ctx.get_rank(), ctx.get_num_ranks()
    win = ctx.rdma.heap_window
    bases = ctx.get_heap_bases().tolist()
    assert bases[me] == win.tensor.data_ptr() == ctx.heap.allocator.get_base_address()
    assert bases == win.address_map.peer_bases.tolist()
    assert ctx.rdma.direct == win.address_map.direct
    for r in range(ws):
        assert (bases[r] != 0) == ctx.rdma.direct[r]
    t = ctx.zeros(1024, dtype=torch.int32)
    assert win.tensor.data_ptr() <= t.data_ptr() < win.tensor.data_ptr() + win.tensor.numel()
    with pytest.raises((MemoryError, RuntimeError)):
        ctx.zeros(1 << 27, dtype=torch.uint8)  # past heap_size: the window cannot grow


def test_rdma_heap_takes_iris_store(rdma_ctx):
    """Unmodified iris.store reaches every direct peer's copy of a heap tensor."""
    ctx = rdma_ctx
    ws, me = ctx.get_num_ranks(), ctx.get_rank()
    buf = ctx.zeros(ws, BLOCK, dtype=torch.int32)
    buf.fill_(-1)
    direct = torch.tensor(ctx.rdma.direct, dtype=torch.int32, device=buf.device)
    ctx.barrier()
    _store_to_direct_peers[(1,)](buf, ctx.get_heap_bases(), direct, me, num_ranks=ws, BLOCK=BLOCK)
    ctx.barrier()
    for src in range(ws):
        expect = src if ctx.rdma.direct[src] else -1
        assert torch.all(buf[src] == expect), f"row from rank {src}"


@triton.jit
def _rdma_put_to_all(send, recv, rank, heap_bases, rdma_ctx, BLOCK: tl.constexpr, BACKEND: tl.constexpr):
    """Program ``peer``: transport-neutral put of this rank's block for ``peer``
    into ``peer``'s ``recv[rank]``, raising signal ``rank`` there by 1."""
    peer = tl.program_id(0)
    rdma.put_signal(send + peer * BLOCK, recv + rank * BLOCK, BLOCK * 4, rank, peer, heap_bases, rdma_ctx, 0, rank, 1,
                    BACKEND)  # fmt: skip


@triton.jit
def _rdma_wait_all(rdma_ctx, signal_base, BACKEND: tl.constexpr):
    """Program ``src``: wait for signal ``src`` to reach its baseline + 1 (absolute target)."""
    src = tl.program_id(0)
    rdma.wait_signal(rdma_ctx, 0, src, tl.load(signal_base + src) + 1, BACKEND)


def test_rdma_put_signal_reaches_every_peer(rdma_ctx):
    """iris.rdma.device put_signal/wait_signal with heap pointers, to every
    rank including itself, over the backend's transport."""
    ctx = rdma_ctx
    if not ctx.rdma.gin_available:
        pytest.skip("no GIN backend (no RDMA NIC?); every peer is load/store reachable")
    ws, me = ctx.get_num_ranks(), ctx.get_rank()
    send = ctx.zeros(ws, BLOCK, dtype=torch.int32)
    recv = ctx.zeros(ws, BLOCK, dtype=torch.int32)
    recv.fill_(-1)
    send.copy_((me * 1000 + torch.arange(ws, device=send.device, dtype=torch.int32))[:, None].expand(ws, BLOCK))
    dc = ctx.rdma.device_comm(signal_count=ws)
    try:
        rctx = ctx.rdma.device_context(dc)
        base = ctx.rdma.read_signals(dc)
        ctx.barrier()
        kw = dict(BACKEND=ctx.rdma.BACKEND_ID, num_warps=4, extern_libs=ctx.rdma.extern_libs())
        _rdma_put_to_all[(ws,)](send, recv, me, ctx.get_heap_bases(), rctx, BLOCK=BLOCK, **kw)
        _rdma_wait_all[(ws,)](rctx, base, **kw)
        torch.cuda.synchronize()
        expect = (torch.arange(ws, device=recv.device, dtype=torch.int32) * 1000 + me)[:, None]
        bad = (recv != expect).any(dim=1).nonzero().flatten().tolist()
        assert not bad, f"wrong blocks from ranks {bad}"
        ctx.barrier()
    finally:
        ctx.rdma.destroy_device_comm(dc)


def test_rdma_device_barrier_needs_direct_peers(rdma_ctx):
    """device_barrier writes flags with iris atomics, which only reach direct
    peers; it must refuse rather than fault when some are not."""
    ctx = rdma_ctx
    if all(ctx.rdma.direct):
        ctx.device_barrier()
    else:
        with pytest.raises(RuntimeError, match="load/store reachable"):
            ctx.device_barrier()


# ── wide EP ──────────────────────────────────────────────────────────────────


@pytest.mark.parametrize("hidden", [256, 7])  # 7: bf16 token of 14 bytes, metadata needs padding
def test_wide_ep_round_trip(rdma_ctx, hidden):
    """dispatch -> local experts -> combine equals the reference computed
    locally. Expert e scales its input by (e + 1), so the reference is exact
    up to bf16 rounding and needs no communication."""
    from iris.experimental.wide_ep import WideEPBuffer

    ctx = rdma_ctx
    ws, me = ctx.get_num_ranks(), ctx.get_rank()
    epr, topk, tokens = 2, 3, 64
    num_experts = epr * ws
    buf = WideEPBuffer(ctx, num_experts, hidden, tokens, topk)
    dev = buf.buffer.device
    gen = torch.Generator(device=dev).manual_seed(1234 + me)

    for step in range(2):  # twice: exercises epoch counting across exchanges
        x = torch.randn(tokens, hidden, generator=gen, device=dev).to(torch.bfloat16)
        idx = torch.rand(tokens, num_experts, generator=gen, device=dev).topk(topk, dim=1).indices
        wts = torch.softmax(torch.rand(tokens, topk, generator=gen, device=dev), dim=1)

        recv_x, recv_expert, recv_weight, handle = buf.dispatch(x, idx, wts)
        assert sum(handle.recv_counts) == recv_x.shape[0]
        assert ((recv_expert == -1) | ((recv_expert >= 0) & (recv_expert < epr))).all()
        scale = torch.where(recv_expert >= 0, (recv_expert + me * epr + 1).float(), 0.0)
        y = (recv_x.float() * (recv_weight * scale).sum(dim=1, keepdim=True)).to(torch.bfloat16)
        out = buf.combine(y, handle)

        ref = x.float() * (wts * (idx + 1).float()).sum(dim=1, keepdim=True)
        err = (out.float() - ref).abs().max().item()
        assert err <= 2e-2 * ref.abs().max().item() + 1e-2, f"step {step}: max err {err}"
    buf.close()
