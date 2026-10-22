# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.

# The parts of GPU-initiated RDMA that need no NIC: the byte swapping the WQE fields use, the context width, the work
# requests a kernel builds and the doorbell it rings (against queues in ordinary GPU memory), and the completion-queue
# bound. Two-host behavior is exercised by hand on a pair of hosts.

import struct

import pytest
import torch
import triton
import triton.language as tl

from iris.device import rdma_utils
from iris.host.memory.symmetric_heap import SymmetricHeap


@triton.jit
def _swap(values, out32, out64, n, BLOCK: tl.constexpr):
    offsets = tl.arange(0, BLOCK)
    mask = offsets < n
    x = tl.load(values + offsets, mask=mask)
    tl.store(out32 + offsets, rdma_utils.bswap32(x), mask=mask)
    tl.store(out64 + offsets, rdma_utils.bswap64(x), mask=mask)


def test_bswap_matches_struct():
    raw = [0, 1, 0x12345678, 0xFFFFFFFF, 0x00FF00FF, 0xDEADBEEF, 0x7FFFFFFF, 0x80000000]
    values = torch.tensor(raw, dtype=torch.int64, device="cuda")
    out32 = torch.zeros(len(raw), dtype=torch.int64, device="cuda")
    out64 = torch.zeros(len(raw), dtype=torch.int64, device="cuda")
    _swap[(1,)](values, out32, out64, len(raw), BLOCK=16)
    want32 = [struct.unpack("<I", struct.pack(">I", v))[0] for v in raw]
    want64 = [struct.unpack("<Q", struct.pack(">Q", v))[0] for v in raw]
    assert out32.cpu().tolist() == want32
    assert [v & 0xFFFFFFFFFFFFFFFF for v in out64.cpu().tolist()] == want64  # int64 storage, read back as 64 bits


def test_a_context_row_is_sixteen_words():
    assert rdma_utils.DEVICE_CTX_WORDS.value == 16
    assert rdma_utils.S_ERROR.value < 8  # the state block has eight words


class FakeQueue:
    """A send and completion queue in ordinary GPU memory, so a kernel can post work with no NIC behind it."""

    SQ_DEPTH, CQ_DEPTH = 8, 4

    def __init__(self):
        dev = "cuda"
        self.sq = torch.zeros(self.SQ_DEPTH * 64, dtype=torch.uint8, device=dev)
        self.cq = torch.zeros(self.CQ_DEPTH * 64, dtype=torch.uint8, device=dev)
        self.cq[63::64] = 0xF1  # invalid, owner 1: what the NIC leaves in a fresh CQ
        self.dbrec = torch.zeros(4, dtype=torch.int32, device=dev)  # [0] CQ consumer index, [1] SQ producer index
        self.bf = torch.zeros(64, dtype=torch.int64, device=dev)
        self.state = torch.zeros(8, dtype=torch.int64, device=dev)
        self.scratch = torch.zeros(1, dtype=torch.int64, device=dev)
        words = [0] * rdma_utils.DEVICE_CTX_WORDS.value
        words[rdma_utils.C_SQ_BUF.value] = self.sq.data_ptr()
        words[rdma_utils.C_SQ_DBREC.value] = self.dbrec.data_ptr() + 4
        words[rdma_utils.C_SQ_BF.value] = self.bf.data_ptr()
        words[rdma_utils.C_SQ_DEPTH.value] = self.SQ_DEPTH
        words[rdma_utils.C_CQ_BUF.value] = self.cq.data_ptr()
        words[rdma_utils.C_CQ_DBREC.value] = self.dbrec.data_ptr()
        words[rdma_utils.C_CQ_DEPTH.value] = self.CQ_DEPTH
        words[rdma_utils.C_QPN.value] = 0x123456
        words[rdma_utils.C_LKEY.value] = 0x11000000  # keys are kept big-endian in the context
        words[rdma_utils.C_RKEY.value] = 0x22000000
        words[rdma_utils.C_STATE.value] = self.state.data_ptr()
        words[rdma_utils.C_OP_WRITE.value] = 0x08
        words[rdma_utils.C_OP_FADD.value] = 0x12
        words[rdma_utils.C_SCRATCH.value] = self.scratch.data_ptr()
        words[rdma_utils.C_SCRATCH_LKEY.value] = 0x33000000
        self.ctx = torch.tensor(words, dtype=torch.uint64, device=dev)  # as build_context makes it

    def wqe(self, slot):
        return self.sq[slot * 64 : (slot + 1) * 64].cpu().numpy().tobytes()

    def complete_through(self, counter):
        """The NIC reports every WQE up to ``counter`` complete in the first CQE."""
        self.cq[63] = 0  # owner 0, opcode 0 (a send completion)
        self.cq[60] = counter >> 8
        self.cq[61] = counter & 0xFF


@triton.jit
def _post(ctx, laddr, raddr, nbytes, count):
    for _ in range(count):
        rdma_utils.put(ctx, laddr, raddr, nbytes)


@triton.jit
def _post_add(ctx, raddr, value):
    rdma_utils.atomic_add(ctx, raddr, value)


def test_put_builds_the_wqe_and_rings_the_doorbell():
    q = FakeQueue()
    _post[(1,)](q.ctx, 0x1000, 0x2000, 256, 2)
    torch.cuda.synchronize()
    for slot in (0, 1):
        wqe = q.wqe(slot)
        opmod_index_opcode, qpn_ds, fm_ce_se = struct.unpack(">III", wqe[:12])
        assert opmod_index_opcode == (slot << 8) | 0x08
        assert qpn_ds == (0x123456 << 8) | 3  # three 16-byte segments: control, remote address, data
        assert fm_ce_se & 0xFF == rdma_utils.WQE_CQ_UPDATE.value  # a CQE is requested
        assert struct.unpack(">QI", wqe[16:28]) == (0x2000, 0x22)
        assert struct.unpack(">IIQ", wqe[32:48]) == (256, 0x11, 0x1000)
    producer_record = q.dbrec.cpu()[1].item() & 0xFFFFFFFF  # as the NIC reads it: big-endian
    assert struct.unpack(">I", producer_record.to_bytes(4, "little"))[0] == 2
    assert q.state.cpu().tolist()[:2] == [2, 2]  # next, committed


def test_atomic_add_builds_a_fetch_add_with_a_scratch_result():
    q = FakeQueue()
    _post_add[(1,)](q.ctx, 0x3000, 7)
    torch.cuda.synchronize()
    wqe = q.wqe(0)
    opmod_index_opcode, qpn_ds = struct.unpack(">II", wqe[:8])
    assert opmod_index_opcode == 0x12
    assert qpn_ds == (0x123456 << 8) | 4  # control, remote address, atomic operands, data
    assert struct.unpack(">QI", wqe[16:28]) == (0x3000, 0x22)
    assert struct.unpack(">QQ", wqe[32:48]) == (7, 0)  # add, compare (unused)
    assert struct.unpack(">IIQ", wqe[48:64]) == (8, 0x33, q.scratch.data_ptr())


def test_a_claim_with_too_many_outstanding_wqes_drains_the_completion_queue():
    # The SQ holds 8 WQEs but the CQ only 4 entries, and every WQE asks for a CQE: with 4 outstanding the next claim must
    # first consume a completion, or the CQ could overflow.
    q = FakeQueue()
    q.state[rdma_utils.S_NEXT.value] = 4
    q.state[rdma_utils.S_COMMITTED.value] = 4
    q.complete_through(3)  # WQEs 0..3 are done
    _post[(1,)](q.ctx, 0x1000, 0x2000, 8, 1)
    torch.cuda.synchronize()
    state = q.state.cpu().tolist()
    assert state[rdma_utils.S_COMPLETED.value] == 4
    assert state[rdma_utils.S_CQ_CI.value] == 1
    assert state[rdma_utils.S_NEXT.value] == 5


def test_rdma_heap_mode_refuses_what_it_cannot_do():
    with pytest.raises(ValueError, match="'torch' and 'vmem_chunked'"):
        SymmetricHeap(1 << 20, 0, 0, 1, "vmem", rdma=True)
    with pytest.raises(ValueError, match="power of two"):
        SymmetricHeap((1 << 20) + 4096, 0, 0, 1, "vmem_chunked", rdma=True)
