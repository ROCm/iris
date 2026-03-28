# SPDX-License-Identifier: MIT
# Copyright (c) 2025-2026 Advanced Micro Devices, Inc. All rights reserved.

"""
Point-to-point send/recv for Iris.

Provides torch.distributed-compatible P2P operations using iris.store()
for data movement and epoch-based atomic flags for per-pair synchronization.

Usage:
    >>> ctx = iris.iris()
    >>> p2p = ctx.ccl.init_p2p(max_numel=2**20)
    >>> if rank == 0:
    ...     ctx.ccl.send(tensor, dst=1, p2p_state=p2p)
    >>> elif rank == 1:
    ...     ctx.ccl.recv(tensor, src=0, p2p_state=p2p)
"""

from dataclasses import dataclass
from typing import List, NamedTuple, Optional

import torch
import triton
import triton.language as tl

import iris

from .utils import extract_group_info


# ---------------------------------------------------------------------------
# Config & data structures
# ---------------------------------------------------------------------------


@dataclass
class P2PConfig:
    """Tuning parameters for P2P kernels."""

    block_size: int = 1024
    num_warps: int = 4
    num_stages: int = 2
    waves_per_eu: int = 0


class P2PWork:
    """Handle returned by non-blocking isend/irecv. Call wait() to block."""

    def __init__(self, stream=None):
        if stream is not None:
            self._event = torch.cuda.Event()
            self._event.record(stream)
        else:
            self._event = None

    def wait(self):
        if self._event is not None:
            self._event.synchronize()
        else:
            torch.cuda.synchronize()


class P2POp(NamedTuple):
    """Mirrors torch.distributed.P2POp."""

    op: object  # callable: send, recv, isend, or irecv
    tensor: torch.Tensor
    peer: int
    group: object = None
    tag: int = 0


class P2PState:
    """
    Pre-allocated P2P buffers and flags.  Created collectively by all ranks.

    Layout on each rank's symmetric heap:
        recv_buf : (world_size * max_numel,)  dtype — slot[i] holds data from rank i
        flags    : (world_size,)              int32 — flags[i] = epoch from rank i
    """

    def __init__(self, ctx, max_numel: int = 2**20, dtype=None, config: Optional[P2PConfig] = None):
        if dtype is None:
            dtype = torch.bfloat16
        self._ctx = ctx
        self._dtype = dtype
        self._max_numel = max_numel
        self._config = config or P2PConfig()
        self._world_size = ctx.get_num_ranks()

        # Collective allocations — all ranks must execute in the same order.
        self.recv_buf = ctx.zeros((self._world_size * max_numel,), dtype=dtype)
        self.flags = ctx.zeros((self._world_size,), dtype=torch.int32)

        # Host-side epoch tracking (per peer).
        self._send_epoch = [0] * self._world_size
        self._recv_epoch = [0] * self._world_size


# ---------------------------------------------------------------------------
# Triton kernels
# ---------------------------------------------------------------------------


@triton.jit
def _p2p_store_kernel(
    src_ptr,
    recv_buf_ptr,
    numel,
    slot_offset,  # = sender_group_rank * max_numel
    heap_bases: tl.tensor,
    iris_rank: tl.constexpr,
    dst_rank: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    """Store data from local tensor into dst rank's recv_buf slot via XGMI."""
    pid = tl.program_id(0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    offsets = tl.max_contiguous(tl.multiple_of(offsets, BLOCK_SIZE), BLOCK_SIZE)
    mask = offsets < numel

    data = tl.load(src_ptr + offsets, mask=mask, other=0.0)

    iris.store(
        recv_buf_ptr + slot_offset + offsets,
        data,
        iris_rank,
        dst_rank,
        heap_bases,
        mask=mask,
        cache_modifier=".wt",
    )


@triton.jit
def _p2p_signal_kernel(
    flags_ptr,
    sender_slot,  # index into flags on receiver
    heap_bases: tl.tensor,
    iris_rank: tl.constexpr,
    dst_rank: tl.constexpr,
):
    """Atomically increment flags[sender_slot] on dst_rank with release semantics.

    Stream ordering guarantees all data stores are complete before this runs.
    """
    flag_ptr = flags_ptr + sender_slot
    from_base = tl.load(heap_bases + iris_rank)
    to_base = tl.load(heap_bases + dst_rank)
    ptr_int = tl.cast(flag_ptr, tl.uint64)
    offset = ptr_int - from_base
    remote_ptr = tl.cast(tl.cast(to_base, tl.pointer_type(tl.int8)) + offset, flag_ptr.dtype)
    tl.atomic_add(remote_ptr, 1, sem="release", scope="sys")


@triton.jit
def _p2p_wait_kernel(
    flags_ptr,
    src_slot,
    target_epoch,
    MAX_SPINS: tl.constexpr = 1_000_000_000,
):
    """Spin-wait on local flags[src_slot] until >= target_epoch (acquire)."""
    flag_ptr = flags_ptr + src_slot
    spin = 0
    while tl.atomic_cas(flag_ptr, target_epoch, target_epoch, sem="acquire", scope="sys") < target_epoch:
        spin += 1
        tl.device_assert(spin < MAX_SPINS, "p2p recv: timeout waiting for send")


@triton.jit
def _p2p_copy_kernel(
    dst_ptr,
    recv_buf_ptr,
    numel,
    slot_offset,
    BLOCK_SIZE: tl.constexpr,
):
    """Copy data from local recv_buf slot to user's output tensor."""
    pid = tl.program_id(0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    offsets = tl.max_contiguous(tl.multiple_of(offsets, BLOCK_SIZE), BLOCK_SIZE)
    mask = offsets < numel

    data = tl.load(recv_buf_ptr + slot_offset + offsets, mask=mask, other=0.0)
    tl.store(dst_ptr + offsets, data, mask=mask)


# ---------------------------------------------------------------------------
# Python API
# ---------------------------------------------------------------------------


def init_p2p(ctx, max_numel: int = 2**20, dtype=None, config: Optional[P2PConfig] = None) -> P2PState:
    """Initialize P2P state.  Must be called collectively by all ranks."""
    return P2PState(ctx, max_numel=max_numel, dtype=dtype, config=config)


def isend(ctx, tensor: torch.Tensor, dst: int, p2p_state: P2PState, group=None, tag: int = 0) -> P2PWork:
    """Non-blocking send.  Enqueues store + signal kernels, returns Work."""
    rank_in_group, rank_global, world_size, rank_start, rank_stride = extract_group_info(group, ctx)
    dst_global = rank_start + dst * rank_stride

    tensor = tensor.contiguous()
    numel = tensor.numel()
    if numel > p2p_state._max_numel:
        raise ValueError(f"Tensor has {numel} elements but P2PState max_numel={p2p_state._max_numel}")

    heap_bases = ctx.get_heap_bases()
    cfg = p2p_state._config
    slot_offset = rank_in_group * p2p_state._max_numel
    grid = (triton.cdiv(numel, cfg.block_size),)

    _p2p_store_kernel[grid](
        tensor,
        p2p_state.recv_buf,
        numel,
        slot_offset,
        heap_bases,
        iris_rank=rank_global,
        dst_rank=dst_global,
        BLOCK_SIZE=cfg.block_size,
        num_warps=cfg.num_warps,
        num_stages=cfg.num_stages,
        waves_per_eu=cfg.waves_per_eu,
    )

    _p2p_signal_kernel[(1,)](
        p2p_state.flags,
        rank_in_group,
        heap_bases,
        iris_rank=rank_global,
        dst_rank=dst_global,
    )

    p2p_state._send_epoch[dst] += 1
    return P2PWork()


def irecv(ctx, tensor: torch.Tensor, src: int, p2p_state: P2PState, group=None, tag: int = 0) -> P2PWork:
    """Non-blocking recv.  Enqueues wait + copy kernels, returns Work."""
    if src is None:
        raise ValueError("Wildcard recv (src=None) not supported; specify src.")

    rank_in_group, rank_global, world_size, rank_start, rank_stride = extract_group_info(group, ctx)

    tensor = tensor.contiguous()
    numel = tensor.numel()

    cfg = p2p_state._config
    slot_offset = src * p2p_state._max_numel

    p2p_state._recv_epoch[src] += 1
    target_epoch = p2p_state._recv_epoch[src]

    _p2p_wait_kernel[(1,)](
        p2p_state.flags,
        src,
        target_epoch,
    )

    grid = (triton.cdiv(numel, cfg.block_size),)
    _p2p_copy_kernel[grid](
        tensor,
        p2p_state.recv_buf,
        numel,
        slot_offset,
        BLOCK_SIZE=cfg.block_size,
        num_warps=cfg.num_warps,
        num_stages=cfg.num_stages,
        waves_per_eu=cfg.waves_per_eu,
    )

    return P2PWork()


def send(ctx, tensor: torch.Tensor, dst: int, p2p_state: P2PState, group=None, tag: int = 0) -> None:
    """Blocking send."""
    work = isend(ctx, tensor, dst, p2p_state, group=group, tag=tag)
    work.wait()


def recv(ctx, tensor: torch.Tensor, src: int, p2p_state: P2PState, group=None, tag: int = 0) -> None:
    """Blocking recv."""
    work = irecv(ctx, tensor, src, p2p_state, group=group, tag=tag)
    work.wait()


def batch_isend_irecv(ctx, p2p_op_list: List[P2POp], p2p_state: P2PState) -> List[P2PWork]:
    """Batched P2P ops.  All sends launch before any recvs to prevent deadlock."""
    works = []

    # Phase 1: all sends
    for op in p2p_op_list:
        if op.op.__name__ in ("send", "isend"):
            w = isend(ctx, op.tensor, op.peer, p2p_state, group=op.group, tag=op.tag)
            works.append(w)

    # Phase 2: all recvs
    for op in p2p_op_list:
        if op.op.__name__ in ("recv", "irecv"):
            w = irecv(ctx, op.tensor, op.peer, p2p_state, group=op.group, tag=op.tag)
            works.append(w)

    return works
