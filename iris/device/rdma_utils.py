# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.

"""
GPU-initiated RDMA (GDA) device-side utilities for Triton kernels, for NVIDIA/Mellanox mlx5 NICs.

Like the SDMA helpers in ``sdma_utils`` this builds the hardware command in the kernel: a Triton program claims a send
queue slot, writes a 64-byte WQE (all fields big-endian), rings the doorbell (doorbell record, then the BlueFlame
register) and polls the completion queue. The queue itself (QP, CQ, registered memory, connection) is created on the host
with ``xio.rdma_ep.Endpoint``; ``Endpoint.device_context()`` returns the words this module reads, one row per peer.

Context words (``DEVICE_CTX_WORDS`` 64-bit words per peer; see ``rdma-ep-host.h`` in rocm-xio):
    0 SQ buffer, 1 SQ doorbell record, 2 BlueFlame pair, 3 SQ depth, 4 CQ buffer, 5 CQ doorbell record, 6 CQ depth,
    7 QP number, 8 local key (big-endian), 9 remote key (big-endian), 10 state block, 11 RDMA WRITE opcode,
    12 atomic fetch-add opcode, 13 atomic scratch address, 14 its key.

The state block is five device words shared by every program using the QP: ``next`` (next slot to claim), ``committed``
(slots rung to the NIC), ``completed`` (slots the CQ reported done), ``cq_ci`` (CQ consumer index) and ``error``
(1 + the largest syndrome of an error CQE, else 0). Slots are claimed in order, doorbells are rung in claim order, and
every WQE asks for a CQE, so ``completed`` is a plain count. Once a kernel drives a QP through these functions the host
``Endpoint.put`` must not be used on the same endpoint.

Only the RDMA WRITE and atomic fetch-add are implemented, and a WQE always takes its data from registered memory (no
inline data). The functions are meant for a single QP per call site: pass the row of the context for the peer.
"""

import triton
import triton.language as tl
from iris.mem.utils import wait_cnt

# Words per peer in the context tensor (``Endpoint.device_context()``, ``DEVICE_CTX_WORDS`` in rocm-xio)
DEVICE_CTX_WORDS = tl.constexpr(16)

# Context word indices
C_SQ_BUF = tl.constexpr(0)
C_SQ_DBREC = tl.constexpr(1)
C_SQ_BF = tl.constexpr(2)
C_SQ_DEPTH = tl.constexpr(3)
C_CQ_BUF = tl.constexpr(4)
C_CQ_DBREC = tl.constexpr(5)
C_CQ_DEPTH = tl.constexpr(6)
C_QPN = tl.constexpr(7)
C_LKEY = tl.constexpr(8)
C_RKEY = tl.constexpr(9)
C_STATE = tl.constexpr(10)
C_OP_WRITE = tl.constexpr(11)
C_OP_FADD = tl.constexpr(12)
C_SCRATCH = tl.constexpr(13)
C_SCRATCH_LKEY = tl.constexpr(14)

# State word indices
S_NEXT = tl.constexpr(0)
S_COMMITTED = tl.constexpr(1)
S_COMPLETED = tl.constexpr(2)
S_CQ_CI = tl.constexpr(3)
S_ERROR = tl.constexpr(4)

CQE_REQ_ERR = tl.constexpr(13)
CQE_INVALID = tl.constexpr(15)
WQE_CQ_UPDATE = tl.constexpr(0x08)

rdma_ep = None


def bind_rdma_ep():
    """Import rocm-xio's ``rdma_ep`` module (needs a build with an RDMA provider)."""
    global rdma_ep
    if rdma_ep is None:
        try:
            from xio import rdma_ep as imported
        except ImportError:
            try:
                # A build tree's module, put on PYTHONPATH without installing the wheel
                import rdma_ep_py as imported
            except ImportError as exc:
                raise ModuleNotFoundError(
                    "rocm-xio built with an RDMA provider (e.g. -DGDA_MLX5=ON) is required for GPU-initiated RDMA.",
                    name="xio",
                ) from exc
        rdma_ep = imported
    return rdma_ep


@triton.jit
def bswap32(x):
    """Byte-swap the low 32 bits of ``x`` (the NIC reads WQE fields big-endian)."""
    x = tl.cast(x, tl.uint32)
    return ((x & 0xFF) << 24) | ((x & 0xFF00) << 8) | ((x >> 8) & 0xFF00) | (x >> 24)


@triton.jit
def bswap64(x):
    x = tl.cast(x, tl.uint64)
    low = bswap32((x & 0xFFFFFFFF).to(tl.uint32)).to(tl.uint64)
    high = bswap32((x >> 32).to(tl.uint32)).to(tl.uint64)
    return (low << 32) | high


@triton.jit
def _state(ctx):
    return tl.load(ctx + C_STATE).to(tl.pointer_type(tl.uint64))


@triton.jit
def poll_cq(ctx):
    """Consume one CQE if there is one: advance ``cq_ci`` and ``completed`` and acknowledge it to the NIC."""
    state = _state(ctx)
    cq_buf = tl.load(ctx + C_CQ_BUF).to(tl.pointer_type(tl.uint8))
    cq_dbrec = tl.load(ctx + C_CQ_DBREC).to(tl.pointer_type(tl.uint32))
    cq_depth = tl.load(ctx + C_CQ_DEPTH)

    ci = tl.load(state + S_CQ_CI, cache_modifier=".cv", volatile=True)
    cqe = cq_buf + (ci % cq_depth) * 64
    op_own = tl.load(cqe + 63, cache_modifier=".cv", volatile=True).to(tl.uint32)
    owner_now = ((ci // cq_depth) & 1).to(tl.uint32)
    opcode = op_own >> 4
    if ((op_own & 1) == owner_now) & (opcode != CQE_INVALID):
        if tl.atomic_cas(state + S_CQ_CI, ci, ci + 1, sem="acq_rel", scope="gpu") == ci:
            counter = (tl.load(cqe + 60, cache_modifier=".cv", volatile=True).to(tl.uint32) << 8) | tl.load(
                cqe + 61, cache_modifier=".cv", volatile=True
            ).to(tl.uint32)
            if opcode == CQE_REQ_ERR:
                syndrome = tl.load(cqe + 55, cache_modifier=".cv", volatile=True).to(tl.uint64)
                tl.atomic_max(state + S_ERROR, 1 + syndrome, sem="relaxed", scope="gpu")
            tl.store(cq_dbrec, bswap32((ci + 1) & 0xFFFFFF), cache_modifier=".wt")
            # The CQE names the WQE it completes by its 16-bit index: widen it from the running count
            done16 = (counter + 1) & 0xFFFF
            current = tl.load(state + S_COMPLETED, cache_modifier=".cv", volatile=True)
            advanced = current + ((done16.to(tl.uint64) + 0x10000 - (current & 0xFFFF)) & 0xFFFF)
            tl.atomic_max(state + S_COMPLETED, advanced, sem="release", scope="gpu")


@triton.jit
def _claim(ctx):
    """
    Claim the next send-queue slot, waiting (and draining the CQ) while too many WQEs are outstanding. Every WQE
    asks for a CQE, so at most as many WQEs as the CQ has entries may be outstanding, or the CQ would overflow.
    """
    state = _state(ctx)
    limit = tl.minimum(tl.load(ctx + C_SQ_DEPTH), tl.load(ctx + C_CQ_DEPTH))
    index = tl.atomic_add(state + S_NEXT, 1, sem="relaxed", scope="gpu")
    while index - tl.load(state + S_COMPLETED, cache_modifier=".cv", volatile=True) >= limit:
        poll_cq(ctx)
    return index


@triton.jit
def _submit(ctx, index, w0, w1, w2, w3, w4, w5, w6, w7):
    """Write the eight words of the WQE in slot ``index``, then ring the doorbell once earlier slots are rung."""
    state = _state(ctx)
    sq_buf = tl.load(ctx + C_SQ_BUF).to(tl.pointer_type(tl.uint64))
    sq_dbrec = tl.load(ctx + C_SQ_DBREC).to(tl.pointer_type(tl.uint32))
    bf = tl.load(ctx + C_SQ_BF).to(tl.pointer_type(tl.uint64))
    depth = tl.load(ctx + C_SQ_DEPTH)

    slot = sq_buf + (index % depth) * 8
    tl.store(slot + 0, w0, cache_modifier=".wt")
    tl.store(slot + 1, w1, cache_modifier=".wt")
    tl.store(slot + 2, w2, cache_modifier=".wt")
    tl.store(slot + 3, w3, cache_modifier=".wt")
    tl.store(slot + 4, w4, cache_modifier=".wt")
    tl.store(slot + 5, w5, cache_modifier=".wt")
    tl.store(slot + 6, w6, cache_modifier=".wt")
    tl.store(slot + 7, w7, cache_modifier=".wt")
    wait_cnt()
    tl.debug_barrier()

    # Doorbells must reach the NIC in slot order: wait for every earlier slot to be rung. This waits on programs
    # that claimed earlier slots, so every program that posts must be resident: keep the grid within the device.
    while tl.load(state + S_COMMITTED, cache_modifier=".cv", volatile=True) != index:
        pass
    tl.store(sq_dbrec, bswap32((index + 1) & 0xFFFF), cache_modifier=".wt")
    wait_cnt()
    tl.debug_barrier()
    # The BlueFlame register takes the first eight bytes of the newest WQE; the two 256-byte buffers alternate
    tl.store(bf + (index & 1) * 32, w0, cache_modifier=".wt")
    wait_cnt()
    tl.debug_barrier()
    tl.atomic_xchg(state + S_COMMITTED, index + 1, sem="release", scope="gpu")


@triton.jit
def _ctrl_words(ctx, index, opcode, ds):
    """First two words of a WQE: opmod/index/opcode and qpn/ds (big-endian), then fm/ce/se with a CQE requested."""
    qpn = tl.load(ctx + C_QPN)
    w0 = bswap32(((index & 0xFFFF) << 8) | opcode).to(tl.uint64) | (bswap32((qpn << 8) | ds).to(tl.uint64) << 32)
    w1 = (index * 0 + WQE_CQ_UPDATE) << 24  # ``index * 0 + c`` is a constant with the type of ``index``
    return w0, w1


@triton.jit
def put(ctx, laddr, raddr, nbytes):
    """
    Post an RDMA WRITE of ``nbytes`` from local registered memory at ``laddr`` to the peer's ``raddr``.

    Returns the slot index of the WQE: ``quiet(ctx, index)`` waits until it has completed.
    """
    index = _claim(ctx)
    w0, w1 = _ctrl_words(ctx, index, tl.load(ctx + C_OP_WRITE), 3)
    rkey = tl.load(ctx + C_RKEY)
    lkey = tl.load(ctx + C_LKEY)
    w2 = bswap64(raddr)
    w3 = rkey
    w4 = bswap32(nbytes).to(tl.uint64) | (lkey << 32)
    w5 = bswap64(laddr)
    zero = index * 0
    _submit(ctx, index, w0, w1, w2, w3, w4, w5, zero, zero)
    return index


@triton.jit
def atomic_add(ctx, raddr, value):
    """
    Post an RDMA atomic fetch-add of ``value`` to the 8-byte-aligned word at ``raddr`` in the peer's registered memory.
    The fetched value lands in a scratch word and is ignored. Returns the slot index, as ``put`` does.
    """
    index = _claim(ctx)
    w0, w1 = _ctrl_words(ctx, index, tl.load(ctx + C_OP_FADD), 4)
    rkey = tl.load(ctx + C_RKEY)
    scratch = tl.load(ctx + C_SCRATCH)
    scratch_lkey = tl.load(ctx + C_SCRATCH_LKEY)
    w2 = bswap64(raddr)
    w3 = rkey
    w4 = bswap64(value)
    w5 = bswap64(index * 0)  # compare operand, unused by fetch-add
    w6 = bswap32(index * 0 + 8).to(tl.uint64) | (scratch_lkey << 32)
    w7 = bswap64(scratch)
    _submit(ctx, index, w0, w1, w2, w3, w4, w5, w6, w7)
    return index


@triton.jit
def quiet(ctx, index):
    """
    Wait until the WQE in slot ``index`` has completed (and everything before it). Returns the QP's error word: 0 if
    all went well, else 1 + the syndrome of an error CQE (0x04 local protection, 0x13 remote access, 0x15 retries...).
    """
    state = _state(ctx)
    while tl.load(state + S_COMPLETED, cache_modifier=".cv", volatile=True) <= index:
        poll_cq(ctx)
    return tl.load(state + S_ERROR, cache_modifier=".cv", volatile=True)


@triton.jit
def put_signal(ctx, laddr, raddr, nbytes, signal_raddr, signal_value):
    """
    ``put`` followed by an atomic add of ``signal_value`` to ``signal_raddr`` on the same QP. The queue pair delivers
    in order, so a peer that sees the signal also sees the data. Returns the slot of the signal.
    """
    put(ctx, laddr, raddr, nbytes)
    return atomic_add(ctx, signal_raddr, signal_value)


def build_context(endpoints, rank, world_size, device):
    """
    The context tensor a kernel takes: ``world_size`` rows of ``DEVICE_CTX_WORDS`` words, row ``p`` for the endpoint
    connected to peer ``p`` (``endpoints`` maps peer rank -> ``Endpoint``; other rows stay zero).
    """
    import torch

    words = bind_rdma_ep().DEVICE_CTX_WORDS
    assert words == DEVICE_CTX_WORDS.value, "rocm-xio and iris.device.rdma_utils disagree on the context width"
    ctx = torch.zeros((world_size, words), dtype=torch.uint64, device=device)
    for peer, endpoint in endpoints.items():
        ctx[peer] = torch.tensor(endpoint.device_context(), dtype=torch.uint64)
    return ctx
