#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
# Copyright (c) 2025-2026 Advanced Micro Devices, Inc. All rights reserved.

"""
Benchmark: iris P2P send/recv vs RCCL dist.send/recv.

Run:
    torchrun --nproc_per_node=8 benchmark/ccl/bench_p2p.py

Measures:
  1. Ping-pong half-RTT latency (rank 0 <-> rank 1)
  2. Unidirectional bandwidth  (rank 0 -> rank 1)
  3. Ring bandwidth             (all ranks, batched)
"""

import gc
import os
import time

import torch
import torch.distributed as dist

import iris
from iris.ccl.p2p import P2POp, isend, irecv


def benchmark(fn, warmup=50, measured=200):
    """Returns median latency in microseconds."""
    torch.cuda.synchronize()
    for _ in range(warmup):
        fn()
        torch.cuda.synchronize()

    times = []
    for _ in range(measured):
        torch.cuda.synchronize()
        t0 = time.perf_counter_ns()
        fn()
        torch.cuda.synchronize()
        t1 = time.perf_counter_ns()
        times.append((t1 - t0) / 1e3)  # ns -> us

    times.sort()
    return times[len(times) // 2]


def _make_pingpong_fn(ctx, t, p2p, rank):
    def fn():
        if rank == 0:
            ctx.ccl.send(t, dst=1, p2p_state=p2p)
            ctx.ccl.recv(t, src=1, p2p_state=p2p)
        elif rank == 1:
            ctx.ccl.recv(t, src=0, p2p_state=p2p)
            ctx.ccl.send(t, dst=0, p2p_state=p2p)

    return fn


def _make_rccl_pingpong_fn(buf, rank):
    def fn():
        if rank == 0:
            dist.send(buf, dst=1)
            dist.recv(buf, src=1)
        elif rank == 1:
            dist.recv(buf, src=0)
            dist.send(buf, dst=0)

    return fn


def _make_uni_fn(ctx, t, p2p, rank):
    def fn():
        if rank == 0:
            ctx.ccl.send(t, dst=1, p2p_state=p2p)
        elif rank == 1:
            ctx.ccl.recv(t, src=0, p2p_state=p2p)

    return fn


def _make_rccl_uni_fn(buf, rank):
    def fn():
        if rank == 0:
            dist.send(buf, dst=1)
        elif rank == 1:
            dist.recv(buf, src=0)

    return fn


def _make_ring_fn(ctx, send_buf, recv_buf, p2p, dst, src):
    def fn():
        ops = [
            P2POp(op=isend, tensor=send_buf, peer=dst),
            P2POp(op=irecv, tensor=recv_buf, peer=src),
        ]
        works = ctx.ccl.batch_isend_irecv(ops, p2p)
        for w in works:
            w.wait()

    return fn


def _make_rccl_ring_fn(send_buf, recv_buf, dst, src):
    def fn():
        ops_rccl = [
            dist.P2POp(dist.isend, send_buf, dst),
            dist.P2POp(dist.irecv, recv_buf, src),
        ]
        reqs = dist.batch_isend_irecv(ops_rccl)
        for r in reqs:
            r.wait()

    return fn


def main():
    rank = int(os.environ.get("LOCAL_RANK", 0))
    torch.cuda.set_device(rank)
    dist.init_process_group(backend="nccl")
    world_size = dist.get_world_size()

    dtype = torch.bfloat16
    elem_bytes = 2  # bf16

    # Message sizes: 1K to 16M elements
    sizes = [1024, 4096, 16384, 65536, 262144, 1 << 20, 4 << 20, 16 << 20]

    ctx = iris.iris(2**33)

    if rank == 0:
        print("# P2P Send/Recv Benchmark — iris vs RCCL")
        print(f"## Hardware: {world_size} GPUs, dtype={dtype}")
        print()

    # ------------------------------------------------------------------
    # 1) Ping-pong (half-RTT): rank 0 sends to 1, rank 1 sends back
    # ------------------------------------------------------------------
    if rank == 0:
        print("### Ping-pong half-RTT latency (us)")
        print("| Elements | Bytes | iris (us) | RCCL (us) | Ratio |")
        print("|----------|-------|-----------|-----------|-------|")

    for N in sizes:
        p2p = ctx.ccl.init_p2p(max_numel=N, dtype=dtype)
        ctx.barrier()

        t = ctx.zeros((N,), dtype=dtype)
        t.fill_(float(rank))

        iris_us = benchmark(_make_pingpong_fn(ctx, t, p2p, rank), warmup=50, measured=200)

        rccl_buf = torch.zeros(N, dtype=dtype, device=f"cuda:{rank}")
        rccl_us = benchmark(_make_rccl_pingpong_fn(rccl_buf, rank), warmup=50, measured=200)

        iris_half = iris_us / 2
        rccl_half = rccl_us / 2

        if rank == 0:
            nbytes = N * elem_bytes
            ratio = iris_half / rccl_half if rccl_half > 0 else float("inf")
            print(f"| {N:>8} | {nbytes:>5} | {iris_half:>9.1f} | {rccl_half:>9.1f} | {ratio:>5.2f}x |")

        ctx.barrier()
        del p2p
        gc.collect()

    if rank == 0:
        print()

    # ------------------------------------------------------------------
    # 2) Unidirectional bandwidth: rank 0 -> rank 1
    # ------------------------------------------------------------------
    if rank == 0:
        print("### Unidirectional bandwidth (rank 0 -> rank 1)")
        print("| Elements | Bytes | iris (us) | RCCL (us) | iris BW (GB/s) | RCCL BW (GB/s) | Ratio |")
        print("|----------|-------|-----------|-----------|----------------|----------------|-------|")

    for N in sizes:
        p2p = ctx.ccl.init_p2p(max_numel=N, dtype=dtype)
        ctx.barrier()

        t = ctx.zeros((N,), dtype=dtype)
        t.fill_(float(rank))

        iris_us = benchmark(_make_uni_fn(ctx, t, p2p, rank), warmup=50, measured=200)

        rccl_buf = torch.zeros(N, dtype=dtype, device=f"cuda:{rank}")
        rccl_us = benchmark(_make_rccl_uni_fn(rccl_buf, rank), warmup=50, measured=200)

        if rank == 0:
            nbytes = N * elem_bytes
            iris_bw = nbytes / (iris_us * 1e-6) / 1e9 if iris_us > 0 else 0
            rccl_bw = nbytes / (rccl_us * 1e-6) / 1e9 if rccl_us > 0 else 0
            ratio = iris_us / rccl_us if rccl_us > 0 else float("inf")
            print(
                f"| {N:>8} | {nbytes:>5} | {iris_us:>9.1f} | {rccl_us:>9.1f} "
                f"| {iris_bw:>14.1f} | {rccl_bw:>14.1f} | {ratio:>5.2f}x |"
            )

        ctx.barrier()
        del p2p
        gc.collect()

    if rank == 0:
        print()

    # ------------------------------------------------------------------
    # 3) Ring bandwidth (all ranks, batch_isend_irecv)
    # ------------------------------------------------------------------
    if rank == 0:
        print("### Ring bandwidth (batch_isend_irecv, all ranks)")
        print("| Elements | Bytes | iris (us) | RCCL (us) | iris BW (GB/s) | RCCL BW (GB/s) | Ratio |")
        print("|----------|-------|-----------|-----------|----------------|----------------|-------|")

    for N in sizes:
        p2p = ctx.ccl.init_p2p(max_numel=N, dtype=dtype)
        ctx.barrier()

        send_buf = ctx.zeros((N,), dtype=dtype)
        recv_buf = ctx.zeros((N,), dtype=dtype)
        send_buf.fill_(float(rank))

        dst = (rank + 1) % world_size
        src = (rank - 1 + world_size) % world_size

        iris_us = benchmark(_make_ring_fn(ctx, send_buf, recv_buf, p2p, dst, src), warmup=50, measured=200)

        rccl_send = torch.zeros(N, dtype=dtype, device=f"cuda:{rank}")
        rccl_recv = torch.zeros(N, dtype=dtype, device=f"cuda:{rank}")
        rccl_us = benchmark(_make_rccl_ring_fn(rccl_send, rccl_recv, dst, src), warmup=50, measured=200)

        if rank == 0:
            nbytes = N * elem_bytes
            iris_bw = nbytes / (iris_us * 1e-6) / 1e9 if iris_us > 0 else 0
            rccl_bw = nbytes / (rccl_us * 1e-6) / 1e9 if rccl_us > 0 else 0
            ratio = iris_us / rccl_us if rccl_us > 0 else float("inf")
            print(
                f"| {N:>8} | {nbytes:>5} | {iris_us:>9.1f} | {rccl_us:>9.1f} "
                f"| {iris_bw:>14.1f} | {rccl_bw:>14.1f} | {ratio:>5.2f}x |"
            )

        ctx.barrier()
        del p2p
        gc.collect()

    if rank == 0:
        print()
        print("## Analysis")
        print("- iris uses direct iris.store() over XGMI + atomic flag signaling")
        print("- RCCL uses ncclSend/ncclRecv with internal buffering")
        print()

    ctx.barrier()
    del ctx
    gc.collect()
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
