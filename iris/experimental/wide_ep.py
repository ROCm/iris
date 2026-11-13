# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.

"""Wide expert parallelism: MoE dispatch and combine across fabric domains.

Prototype. Experts are sharded over every rank; each token is
sent once to each *rank* holding any of its top-k experts, and the partial
results come back the same way. The API follows DeepEP's normal mode:

    buf = WideEPBuffer(provider, num_experts, hidden, max_tokens, topk)
    recv_x, recv_expert, recv_weight, handle = buf.dispatch(x, topk_idx, topk_weights)
    y = <run this rank's experts over recv_x, weighted, summed per row>
    out = buf.combine(y, handle)

Transport, chosen per destination inside one Triton kernel:

- destinations the window's ``SymmetricAddressMap`` marks ``direct`` (RCCL's
  load/store team: the node, or the fabric domain) get plain ``iris.store`` and a release flag (``iris.atomic_xchg``);
- every other destination gets one RCCL GIN put, carrying a weak signal.

The kernels read that ``direct`` mask itself, so routing cannot disagree with
the map about which peers are addressable.

Both write into the same RCCL symmetric window, so a receiver does not care
which path a slot arrived by.

Window layout, per exchange (dispatch, then combine)::

    send: [world][slot]   recv: [world][slot]
    slot = 64-byte header (int64 row count) + capacity rows

A sender fills ``send[dst]`` and copies its first ``64 + count*row_bytes``
bytes into ``recv[src]`` on ``dst``. The header travels in the same put as the
rows, which is what makes a weak signal sufficient.

Completion is counted, not reset, and kept per exchange: dispatch and combine
each have their own flags, signals and round number. A sender raises each
destination's flag to the round (direct peers) or adds 1 to its signal (GIN), and
a receiver waits for all ``world`` sources to reach it. Separate signals
matter: GIN signals are weak and puts are unordered, so a counter shared by
both exchanges could be raised by a rank's next dispatch before its previous
combine landed, releasing the combine wait early.

No exchange needs a barrier: a rank starts dispatch ``n+1`` only after every
combine ``n`` reached it, by which point every receiver has finished reading
what dispatch ``n+1`` overwrites.

Known limits of the prototype: ``dispatch`` synchronises with the host twice
(token layout and received counts), the receive capacity is the full
``max_tokens`` per source, and each GIN destination is a single put from one
workgroup.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
import triton
import triton.language as tl

import iris
from iris.experimental import rccl_gin
from iris.experimental.rccl_gin import RcclGinProvider

__all__ = ["WideEPBuffer", "DispatchHandle"]

HEADER_BYTES = tl.constexpr(64)  # readable from kernels
_HEADER = 64  # the same, for host code


def _align(n: int, a: int) -> int:
    return (n + a - 1) // a * a


@triton.jit
def _is_direct(direct, rank):
    """Whether ``rank`` is load/store reachable, read from the window's
    ``SymmetricAddressMap.direct`` mask. Routing on the map itself, rather than
    on a rank range, keeps iris.store from ever translating against the 0 base
    of a peer the map says is not addressable."""
    return tl.load(direct + rank) != 0


@triton.jit
def _exchange_kernel(
    win_ptr,
    heap_bases,
    comm,
    win,
    send_off,
    recv_off,
    slot_bytes,
    row_bytes,
    rank,
    direct,
    signal,
    CHUNKS: tl.constexpr,
    BLOCK: tl.constexpr,
):
    """Program (dst, chunk): move this rank's send slot for ``dst`` into
    ``dst``'s receive slot for this rank. Same-node slots are copied by all
    CHUNKS programs; a GIN slot is one put, issued by chunk 0."""
    dst = tl.program_id(0)
    chunk = tl.program_id(1)
    src_off = send_off + dst.to(tl.int64) * slot_bytes
    dst_off = recv_off + tl.cast(rank, tl.int64) * slot_bytes
    count = tl.load((win_ptr + src_off).to(tl.pointer_type(tl.int64)))
    nbytes = HEADER_BYTES + count * row_bytes
    if _is_direct(direct, dst):
        src = (win_ptr + src_off).to(tl.pointer_type(tl.int64))
        to = (win_ptr + dst_off).to(tl.pointer_type(tl.int64))
        nwords = nbytes // 8
        for start in range(tl.cast(chunk, tl.int64) * BLOCK, nwords, CHUNKS * BLOCK):
            offs = start + tl.arange(0, BLOCK)
            mask = offs < nwords
            iris.store(to + offs, tl.load(src + offs, mask=mask), rank, dst, heap_bases, mask=mask)
    elif chunk == 0:
        # No flush: a send slot is next rewritten only after its destination's
        # reply has arrived, which already proves this put completed.
        rccl_gin.put_signal(comm, 0, dst, win, dst_off, win, src_off, nbytes, signal, 1)


@triton.jit
def _notify_kernel(win_ptr, heap_bases, flags_off, rank, direct, epoch):
    """Raise this rank's flag on every same-node destination to ``epoch``.
    A separate launch so that every chunk of _exchange_kernel has finished."""
    dst = tl.program_id(0)
    if _is_direct(direct, dst):
        flag = (win_ptr + flags_off).to(tl.pointer_type(tl.int64)) + rank
        iris.atomic_xchg(flag, tl.cast(epoch, tl.int64), rank, dst, heap_bases, sem="release", scope="sys")


@triton.jit
def _wait_kernel(win_ptr, comm, flags_off, signal_off, signal_base, direct, epoch):
    """Program ``src``: wait until ``src``'s slot for this exchange has landed.
    Signal ``signal_off + src`` counts only this exchange's puts from ``src``."""
    src = tl.program_id(0)
    if _is_direct(direct, src):
        flag = (win_ptr + flags_off).to(tl.pointer_type(tl.int64)) + src
        while tl.atomic_add(flag, 0, sem="acquire", scope="sys") < epoch:
            pass
    else:
        sig = signal_off + src
        rccl_gin.wait_signal(comm, 0, sig, tl.load(signal_base + sig) + epoch)


@dataclass(frozen=True)
class _Exchange:
    """One send/receive area of the window, with its own completion state."""

    send_off: int
    recv_off: int
    slot_bytes: int
    row_bytes: int
    flags_off: int  # world int64 flags, one per source
    signal_off: int  # first of world GIN signal ids, one per source


def _plan_window(world: int, x_bytes: int, max_tokens: int, topk: int):
    """Byte layout of a WideEPBuffer window, independent of any device.

    Returns ``(dispatch, combine, meta_off, total_bytes)``. Each exchange holds
    ``world`` send slots then ``world`` receive slots; a slot is the header plus
    ``max_tokens`` rows. Dispatch rows carry the token, then at ``meta_off``
    (the token's size rounded up to 8, so the int32 and fp32 views are
    aligned) the local expert id (int32, -1 if that expert lives elsewhere) and
    weight (fp32) of each top-k choice. Rows are 16-byte aligned so the copy
    kernel's int64 view and any vector access stay aligned. Each exchange's
    ``world`` int64 completion flags follow the slots, and the whole window is
    rounded to a 4 KiB page. Dispatch uses GIN signals ``0..world-1``,
    combine ``world..2*world-1``.
    """
    meta_off = _align(x_bytes, 8)
    rows = (_align(meta_off + topk * 8, 16), _align(x_bytes, 16))
    slots = [_HEADER + max_tokens * row for row in rows]
    flags_off = sum(2 * world * slot for slot in slots)
    regions = []
    off = 0
    for i, (row, slot) in enumerate(zip(rows, slots)):
        regions.append(_Exchange(off, off + world * slot, slot, row, flags_off + i * world * 8, i * world))
        off += 2 * world * slot
    return regions[0], regions[1], meta_off, _align(flags_off + 2 * world * 8, 4096)


@dataclass
class DispatchHandle:
    """What combine needs to route results back: for every (token, dst rank)
    pair sent, the token and its row in the slot; and for every received row,
    its source and its row in that source's slot."""

    token: torch.Tensor  # int64[pairs]
    dst: torch.Tensor  # int64[pairs]
    row: torch.Tensor  # int64[pairs]
    recv_counts: list[int]  # rows received from each source
    recv_src: torch.Tensor  # int64[received]
    recv_row: torch.Tensor  # int64[received]
    recv_headers: torch.Tensor  # int64[world], recv_counts on the device
    num_tokens: int


class WideEPBuffer:
    """Communication buffer for one MoE layer shape. Collective to construct."""

    def __init__(
        self,
        provider: RcclGinProvider,
        num_experts: int,
        hidden: int,
        max_tokens: int,
        topk: int,
        dtype: torch.dtype = torch.bfloat16,
        chunks: int = 16,
    ):
        self.provider = provider
        self.rank = provider.get_rank()
        self.world = provider.get_num_ranks()
        if num_experts % self.world:
            raise ValueError(f"num_experts={num_experts} must divide evenly over {self.world} ranks")
        self.num_experts = num_experts
        self.experts_per_rank = num_experts // self.world
        self.hidden = hidden
        self.max_tokens = max_tokens
        self.topk = topk
        self.dtype = dtype
        self.chunks = chunks
        esize = torch.tensor([], dtype=dtype).element_size()
        self._x_bytes = hidden * esize

        self._dispatch, self._combine, self._meta_off, total = _plan_window(self.world, self._x_bytes, max_tokens, topk)

        self.window = provider.allocate_window(total)
        # The kernels route on this mask: direct peers by iris.store, everyone
        # else by GIN. (allocate_window guarantees every direct peer a base.)
        amap = self.window.address_map
        self._direct = torch.tensor(amap.direct, dtype=torch.int32, device=amap.peer_bases.device)
        self.comm = provider.device_comm(signal_count=2 * self.world)
        self._signal_base = provider.read_signals(self.comm, count=2 * self.world)  # zeros without GIN
        self.window.tensor[self._dispatch.flags_off :].zero_()
        self._rounds = {self._dispatch: 0, self._combine: 0}
        self._lib = rccl_gin.extern_libs()
        provider.barrier()  # every flag is zero before anyone can raise one

        self.window_bytes = total

    # ── views into the window ───────────────────────────────────────────────

    def _slots(self, ex: _Exchange, which: str) -> torch.Tensor:
        start = ex.send_off if which == "send" else ex.recv_off
        return self.window.tensor[start : start + self.world * ex.slot_bytes].view(self.world, ex.slot_bytes)

    def _headers(self, ex: _Exchange, which: str) -> torch.Tensor:
        return self._slots(ex, which)[:, :8].view(torch.int64).view(self.world)

    def _rows(self, ex: _Exchange, which: str) -> torch.Tensor:
        slots = self._slots(ex, which)
        return slots[:, _HEADER:].view(self.world, self.max_tokens, ex.row_bytes)

    def _row_x(self, rows: torch.Tensor) -> torch.Tensor:
        return rows[..., : self._x_bytes].view(self.dtype)

    def _row_expert(self, rows: torch.Tensor) -> torch.Tensor:
        b = self._meta_off
        return rows[..., b : b + self.topk * 4].view(torch.int32)

    def _row_weight(self, rows: torch.Tensor) -> torch.Tensor:
        b = self._meta_off + self.topk * 4
        return rows[..., b : b + self.topk * 4].view(torch.float32)

    # ── transport ───────────────────────────────────────────────────────────

    def _exchange(self, ex: _Exchange) -> torch.Tensor:
        """Send every send slot to its destination and wait for every receive
        slot. Returns the received row counts, per source, on the device."""
        self._rounds[ex] += 1
        rnd = self._rounds[ex]
        w = self.window
        _exchange_kernel[(self.world, self.chunks)](
            w.tensor, w.address_map.peer_bases, self.comm.ptr, w.handle,
            ex.send_off, ex.recv_off, ex.slot_bytes, ex.row_bytes,
            self.rank, self._direct, ex.signal_off + self.rank,
            CHUNKS=self.chunks, BLOCK=1024, num_warps=4, extern_libs=self._lib,
        )  # fmt: skip
        _notify_kernel[(self.world,)](
            w.tensor, w.address_map.peer_bases, ex.flags_off, self.rank, self._direct, rnd,
            num_warps=1,
        )  # fmt: skip
        _wait_kernel[(self.world,)](
            w.tensor, self.comm.ptr, ex.flags_off, ex.signal_off, self._signal_base, self._direct, rnd,
            num_warps=1, extern_libs=self._lib,
        )  # fmt: skip
        return self._headers(ex, "recv")

    # ── MoE API ─────────────────────────────────────────────────────────────

    def dispatch(self, x: torch.Tensor, topk_idx: torch.Tensor, topk_weights: torch.Tensor):
        """Send each token to every rank that owns one of its experts.

        Args:
            x: ``[T, hidden]`` tokens, ``T <= max_tokens``.
            topk_idx: ``[T, topk]`` global expert ids.
            topk_weights: ``[T, topk]`` routing weights.

        Returns ``(recv_x, recv_expert, recv_weight, handle)``. Received rows
        are grouped by source rank in rank order. ``recv_expert[i, j]`` is the
        *local* expert index of the token's j-th choice, or -1 where that
        expert lives on another rank (its ``recv_weight`` is then 0).
        """
        T = x.shape[0]
        if T > self.max_tokens:
            raise ValueError(f"{T} tokens exceed max_tokens={self.max_tokens}")
        epr = self.experts_per_rank
        dst_of = topk_idx // epr  # [T, k]
        sends = torch.zeros(T, self.world, dtype=torch.bool, device=x.device)
        sends.scatter_(1, dst_of, True)
        row_of = torch.cumsum(sends.to(torch.int64), dim=0) - 1  # [T, world]
        token, dst = sends.nonzero(as_tuple=True)
        row = row_of[token, dst]

        ex = self._dispatch
        rows = self._rows(ex, "send")
        self._row_x(rows)[dst, row] = x[token]
        mine = dst_of[token] == dst[:, None]  # [pairs, k]: choice j lives on dst
        local = torch.where(mine, topk_idx[token] - dst[:, None] * epr, -1)
        self._row_expert(rows)[dst, row] = local.to(torch.int32)
        self._row_weight(rows)[dst, row] = torch.where(mine, topk_weights[token].float(), 0.0)
        self._headers(ex, "send").copy_(sends.sum(dim=0))

        headers = self._exchange(ex).clone()
        counts = headers.tolist()  # the one host sync that sizes the outputs

        # (source, row) of every received row, in source order: one gather now,
        # and one scatter in combine, instead of a copy per source.
        n = sum(counts)
        recv_src = torch.repeat_interleave(torch.arange(self.world, device=x.device), headers, output_size=n)
        starts = torch.cumsum(headers, 0) - headers
        recv_row = torch.arange(n, device=x.device) - starts[recv_src]
        recv_rows = self._rows(ex, "recv")[recv_src, recv_row]
        handle = DispatchHandle(
            token=token, dst=dst, row=row, recv_counts=counts, recv_src=recv_src, recv_row=recv_row,
            recv_headers=headers, num_tokens=T,
        )  # fmt: skip
        return (
            self._row_x(recv_rows),
            self._row_expert(recv_rows),
            self._row_weight(recv_rows),
            handle,
        )

    def combine(self, y: torch.Tensor, handle: DispatchHandle) -> torch.Tensor:
        """Return each rank's results to the token's owner and sum them.

        Args:
            y: ``[sum(recv_counts), hidden]``, row ``i`` the weighted sum of this
               rank's experts applied to ``recv_x[i]``.
        Returns ``[T, hidden]`` in ``self.dtype``, accumulated in float32.
        """
        ex = self._combine
        self._row_x(self._rows(ex, "send"))[handle.recv_src, handle.recv_row] = y
        self._headers(ex, "send").copy_(handle.recv_headers)

        self._exchange(ex)  # no host sync: combine already knows what comes back

        back = self._row_x(self._rows(ex, "recv"))[handle.dst, handle.row].float()
        out = torch.zeros(handle.num_tokens, self.hidden, dtype=torch.float32, device=y.device)
        out.index_add_(0, handle.token, back)
        return out.to(self.dtype)

    def close(self):
        """Collectively free the window and device communicator. Idempotent."""
        if self.window is None:
            return
        self.provider.barrier()
        self.provider.destroy_device_comm(self.comm)
        self.provider.free_window(self.window)
        self.window = None
