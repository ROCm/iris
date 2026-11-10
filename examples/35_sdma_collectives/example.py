#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.

"""
Scatter, gather, broadcast and all-to-all on the SDMA copy engines, triggered from a kernel.

    plan = ctx.ccl.prepare_scatter(buf, root)     # collective: one copy-engine chain per (chunk, receiver)
    handle = plan.start()
    drive[(plan.num_chunks,)](handle, ...)         # mark the chunks this rank sends, wait for the rest
    plan.wait()                                    # every chunk delivered, or TimeoutError
    plan.next_round()

``mark_ready(chunk)`` is the trigger: the chains have waited on the chunk's counter since the plan
was prepared. ``wait(chunk)`` spins until the chunk has arrived, and returns at once for a chunk not
sent to this rank, so the kernel does not need to know the collective's roles. It writes back what
it did, which is what the ``roles:`` lines report.
"""

import argparse
import os

import torch
import torch.distributed as dist
import triton
import triton.language as tl

import iris
from iris.experimental.plan import TriggeredView, tile_chunks

EMPTY = -1
SENT = 1


@triton.jit
def drive(handle, roles):
    """Program c marks chunk c ready if this rank sends it, then waits until it is here."""
    view = TriggeredView.initialize(handle)
    c = tl.program_id(0)
    role = 0
    if view.owns(c):
        view.mark_ready(c)
        role = 1
    view.wait(c)
    tl.store(roles + c, role)


def payload(round, part, rows, cols, device):
    """A part's contents: the round and part above bit 16, the element index below."""
    return (
        torch.arange(rows * cols, dtype=torch.int32, device=device) & 0xFFFF | ((round * 16 + part + 1) << 16)
    ).view(rows, cols)


def layout(collective, root, rank, world):
    """How many parts the tensor has, the parts this rank writes, and the parts that land here."""
    if collective == "scatter":
        return world, (range(world) if rank == root else []), [rank]
    if collective == "gather":
        return world, [rank], (range(world) if rank == root else [])
    if collective == "broadcast":
        return 1, ([0] if rank == root else []), [0]
    # all-to-all: block (r, j) is part r * world + j
    return world * world, [rank * world + j for j in range(world)], [r * world + rank for r in range(world)]


def prepare(ctx, collective, buf, parts, root, tile_rows):
    """The plan, with each part cut into tiles of ``tile_rows`` full rows."""
    rank, world = ctx.get_rank(), ctx.get_num_ranks()
    tile = (tile_rows, buf.shape[1])

    def tiles(part, peers):
        return tile_chunks(parts[part], tile, 1, base=buf, peers=peers)

    if collective == "scatter":
        schedule = [c for r in range(world) if r != root for c in tiles(r, (r,))] if rank == root else None
        return ctx.ccl.prepare_scatter(buf, root, schedule)
    if collective == "gather":
        return ctx.ccl.prepare_gather(buf, root, tiles(rank, (root,)) if rank != root else None)
    if collective == "broadcast":
        others = tuple(p for p in range(world) if p != root)
        return ctx.ccl.prepare_broadcast(buf, root, tiles(0, others) if rank == root else None)
    return ctx.ccl.prepare_all_to_all(
        buf, [c for j in range(world) if j != rank for c in tiles(rank * world + j, (j,))]
    )


def main():
    p = argparse.ArgumentParser(description="Triggered collectives on the SDMA copy engines")
    p.add_argument("--collective", choices=["scatter", "gather", "broadcast", "all_to_all"], default="scatter")
    p.add_argument("--root", type=int, default=0, help="root rank (scatter, gather, broadcast)")
    p.add_argument("--m", type=int, default=64, help="rows per part")
    p.add_argument("--n", type=int, default=128, help="columns")
    p.add_argument("--tile_rows", type=int, default=8, help="rows per chunk")
    p.add_argument("--rounds", type=int, default=3)
    p.add_argument("--heap_size", type=int, default=1 << 28)
    p.add_argument("--rdma_device", default=None, help="reach peers on other hosts over RDMA through this NIC")
    a = p.parse_args()

    torch.cuda.set_device(int(os.environ.get("LOCAL_RANK", 0)))
    dist.init_process_group(backend="nccl")
    try:
        if a.rdma_device is not None:
            # One rank per host: the copy engine still triggers, RDMA carries the data
            ctx = iris.iris(a.heap_size, engines=("sdma", "rdma"), rdma_device=a.rdma_device)
        else:
            ctx = iris.iris(a.heap_size, engines=("sdma",))
        rank, world = ctx.get_rank(), ctx.get_num_ranks()
        n, writes, lands = layout(a.collective, a.root, rank, world)
        buf = ctx.zeros((n * a.m, a.n), dtype=torch.int32)
        parts = [buf[i * a.m : (i + 1) * a.m] for i in range(n)]
        buf.fill_(EMPTY)
        with prepare(ctx, a.collective, buf, parts, a.root, a.tile_rows) as plan:
            roles = torch.zeros(plan.num_chunks, dtype=torch.int32, device=buf.device)
            for round in range(1, a.rounds + 1):
                for part in writes:
                    parts[part].copy_(payload(round, part, a.m, a.n, buf.device))
                handle = plan.start()
                drive[(plan.num_chunks,)](handle, roles)
                plan.wait()
                for part in range(n):
                    want = payload(round, part, a.m, a.n, buf.device) if part in writes or part in lands else None
                    got = parts[part]
                    if want is None and not (got == EMPTY).all():
                        raise RuntimeError(f"round {round}: part {part} should have been left untouched")
                    if want is not None and not torch.equal(got, want):
                        raise RuntimeError(f"round {round}: part {part} differs")
                sent = int((roles == SENT).sum())
                ctx.info(f"round {round} roles: sent {sent} of {plan.num_chunks} chunks, waited for the rest")
                buf.fill_(EMPTY)
                plan.next_round()
        ctx.info(f"{a.collective}: {a.rounds} rounds correct")
        ctx.barrier()
    finally:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
