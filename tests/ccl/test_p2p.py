# SPDX-License-Identifier: MIT
# Copyright (c) 2025-2026 Advanced Micro Devices, Inc. All rights reserved.

"""Tests for P2P send/recv operations."""

import gc

import pytest
import torch
import torch.distributed as dist

import iris
from iris.ccl.p2p import P2POp, isend, irecv

HEAP_SIZE = 2**33


def _skip_if_not_distributed():
    if not dist.is_initialized():
        pytest.skip("torch.distributed not initialized")


def _skip_if_too_few_ranks(min_ranks=2):
    _skip_if_not_distributed()
    if dist.get_world_size() < min_ranks:
        pytest.skip(f"Need >= {min_ranks} ranks")


# --------------------------------------------------------------------------
# Basic send/recv
# --------------------------------------------------------------------------


@pytest.mark.parametrize("N", [1, 64, 1024, 65536, 2**20])
def test_send_recv_sizes(N):
    """Various message sizes: rank 0 -> rank 1."""
    _skip_if_too_few_ranks(2)
    ctx = iris.iris(HEAP_SIZE)
    p2p = ctx.ccl.init_p2p(max_numel=2**20, dtype=torch.float32)
    ctx.barrier()

    rank = ctx.get_rank()
    device = f"cuda:{rank}"

    if rank == 0:
        t = torch.arange(N, dtype=torch.float32, device=device)
        ctx.ccl.send(t, dst=1, p2p_state=p2p)
    elif rank == 1:
        t = torch.zeros(N, dtype=torch.float32, device=device)
        ctx.ccl.recv(t, src=0, p2p_state=p2p)
        expected = torch.arange(N, dtype=torch.float32, device=device)
        torch.testing.assert_close(t, expected)

    ctx.barrier()
    del p2p, ctx
    gc.collect()


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16, torch.float32])
def test_send_recv_dtypes(dtype):
    """Test multiple dtypes."""
    _skip_if_too_few_ranks(2)
    ctx = iris.iris(HEAP_SIZE)
    p2p = ctx.ccl.init_p2p(max_numel=4096, dtype=dtype)
    ctx.barrier()

    rank = ctx.get_rank()
    device = f"cuda:{rank}"
    N = 4096

    # Broadcast reference from rank 0
    ref = torch.randn(N, dtype=dtype, device=device) if rank == 0 else torch.empty(N, dtype=dtype, device=device)
    dist.broadcast(ref, src=0)

    if rank == 0:
        ctx.ccl.send(ref, dst=1, p2p_state=p2p)
    elif rank == 1:
        out = torch.zeros(N, dtype=dtype, device=device)
        ctx.ccl.recv(out, src=0, p2p_state=p2p)
        torch.testing.assert_close(out, ref)

    ctx.barrier()
    del p2p, ctx
    gc.collect()


# --------------------------------------------------------------------------
# Async isend/irecv
# --------------------------------------------------------------------------


def test_isend_irecv():
    """Non-blocking with explicit wait."""
    _skip_if_too_few_ranks(2)
    ctx = iris.iris(HEAP_SIZE)
    p2p = ctx.ccl.init_p2p(max_numel=4096, dtype=torch.float32)
    ctx.barrier()

    rank = ctx.get_rank()
    device = f"cuda:{rank}"
    N = 2048

    ref = torch.randn(N, dtype=torch.float32, device=device) if rank == 0 else torch.empty(N, dtype=torch.float32, device=device)
    dist.broadcast(ref, src=0)

    if rank == 0:
        work = ctx.ccl.isend(ref, dst=1, p2p_state=p2p)
        work.wait()
    elif rank == 1:
        out = torch.zeros(N, dtype=torch.float32, device=device)
        work = ctx.ccl.irecv(out, src=0, p2p_state=p2p)
        work.wait()
        torch.testing.assert_close(out, ref)

    ctx.barrier()
    del p2p, ctx
    gc.collect()


# --------------------------------------------------------------------------
# Ring pattern (batch_isend_irecv)
# --------------------------------------------------------------------------


def test_ring():
    """Ring: rank i -> (i+1)%W, recv from (i-1+W)%W via batch_isend_irecv."""
    _skip_if_too_few_ranks(2)
    ctx = iris.iris(HEAP_SIZE)
    W = ctx.get_num_ranks()
    p2p = ctx.ccl.init_p2p(max_numel=4096, dtype=torch.float32)
    ctx.barrier()

    rank = ctx.get_rank()
    device = f"cuda:{rank}"
    N = 4096
    dst = (rank + 1) % W
    src = (rank - 1 + W) % W

    send_buf = torch.full((N,), float(rank), dtype=torch.float32, device=device)
    recv_buf = torch.zeros(N, dtype=torch.float32, device=device)

    ops = [
        P2POp(op=isend, tensor=send_buf, peer=dst),
        P2POp(op=irecv, tensor=recv_buf, peer=src),
    ]
    works = ctx.ccl.batch_isend_irecv(ops, p2p)
    for w in works:
        w.wait()

    expected = torch.full((N,), float(src), dtype=torch.float32, device=device)
    torch.testing.assert_close(recv_buf, expected)

    ctx.barrier()
    del p2p, ctx
    gc.collect()


# --------------------------------------------------------------------------
# All-pairs via batch_isend_irecv
# --------------------------------------------------------------------------


def test_all_pairs():
    """Every rank sends to every other rank."""
    _skip_if_too_few_ranks(2)
    ctx = iris.iris(HEAP_SIZE)
    W = ctx.get_num_ranks()
    p2p = ctx.ccl.init_p2p(max_numel=1024, dtype=torch.float32)
    ctx.barrier()

    rank = ctx.get_rank()
    device = f"cuda:{rank}"
    N = 1024

    send_bufs = {}
    recv_bufs = {}
    ops = []

    for peer in range(W):
        if peer != rank:
            s = torch.full((N,), float(rank * 100 + peer), dtype=torch.float32, device=device)
            r = torch.zeros(N, dtype=torch.float32, device=device)
            send_bufs[peer] = s
            recv_bufs[peer] = r
            ops.append(P2POp(op=isend, tensor=s, peer=peer))
            ops.append(P2POp(op=irecv, tensor=r, peer=peer))

    works = ctx.ccl.batch_isend_irecv(ops, p2p)
    for w in works:
        w.wait()

    for peer in range(W):
        if peer != rank:
            expected_val = float(peer * 100 + rank)
            expected = torch.full((N,), expected_val, dtype=torch.float32, device=device)
            torch.testing.assert_close(recv_bufs[peer], expected)

    ctx.barrier()
    del p2p, ctx
    gc.collect()


# --------------------------------------------------------------------------
# Correctness vs torch.distributed
# --------------------------------------------------------------------------


def test_vs_torch_distributed():
    """Correctness check: iris P2P vs RCCL send/recv."""
    _skip_if_too_few_ranks(2)
    ctx = iris.iris(HEAP_SIZE)
    p2p = ctx.ccl.init_p2p(max_numel=8192, dtype=torch.float32)
    ctx.barrier()

    rank = ctx.get_rank()
    device = f"cuda:{rank}"
    N = 8192

    ref = torch.randn(N, dtype=torch.float32, device=device) if rank == 0 else torch.empty(N, dtype=torch.float32, device=device)
    dist.broadcast(ref, src=0)

    if rank == 0:
        ctx.ccl.send(ref, dst=1, p2p_state=p2p)
    elif rank == 1:
        iris_out = torch.zeros(N, dtype=torch.float32, device=device)
        ctx.ccl.recv(iris_out, src=0, p2p_state=p2p)
        torch.testing.assert_close(iris_out, ref)

    ctx.barrier()
    del p2p, ctx
    gc.collect()


# --------------------------------------------------------------------------
# Multiple sequential sends
# --------------------------------------------------------------------------


def test_multiple_sends():
    """Send multiple messages on the same channel (tests epoch tracking)."""
    _skip_if_too_few_ranks(2)
    ctx = iris.iris(HEAP_SIZE)
    p2p = ctx.ccl.init_p2p(max_numel=1024, dtype=torch.float32)
    ctx.barrier()

    rank = ctx.get_rank()
    device = f"cuda:{rank}"
    N = 1024

    for i in range(5):
        if rank == 0:
            t = torch.full((N,), float(i), dtype=torch.float32, device=device)
            ctx.ccl.send(t, dst=1, p2p_state=p2p)
        elif rank == 1:
            t = torch.zeros(N, dtype=torch.float32, device=device)
            ctx.ccl.recv(t, src=0, p2p_state=p2p)
            expected = torch.full((N,), float(i), dtype=torch.float32, device=device)
            torch.testing.assert_close(t, expected)

    ctx.barrier()
    del p2p, ctx
    gc.collect()


# --------------------------------------------------------------------------
# Bidirectional ping-pong
# --------------------------------------------------------------------------


def test_pingpong():
    """Rank 0 sends, rank 1 echoes back (sequential, not batched)."""
    _skip_if_too_few_ranks(2)
    ctx = iris.iris(HEAP_SIZE)
    p2p = ctx.ccl.init_p2p(max_numel=1024, dtype=torch.float32)
    ctx.barrier()

    rank = ctx.get_rank()
    device = f"cuda:{rank}"
    N = 512

    data = torch.randn(N, dtype=torch.float32, device=device) if rank == 0 else torch.empty(N, dtype=torch.float32, device=device)
    dist.broadcast(data, src=0)

    if rank == 0:
        ctx.ccl.send(data, dst=1, p2p_state=p2p)
        echo = torch.zeros(N, dtype=torch.float32, device=device)
        ctx.ccl.recv(echo, src=1, p2p_state=p2p)
        torch.testing.assert_close(echo, data)
    elif rank == 1:
        buf = torch.zeros(N, dtype=torch.float32, device=device)
        ctx.ccl.recv(buf, src=0, p2p_state=p2p)
        ctx.ccl.send(buf, dst=0, p2p_state=p2p)

    ctx.barrier()
    del p2p, ctx
    gc.collect()
