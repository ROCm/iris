#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.

"""
All-gather on the SDMA copy engines.

1. ``ctx.ccl.all_gather`` with ``Config(engine="sdma")``: the same collective as the default
   compute-unit one, but the copy engines move the data.
2. ``ctx.ccl.prepare_all_gather``: a persistent tiled kernel writes its rows of the output and marks
   each wave of tiles ready; the copy engines send every wave to all peers while the kernel is still
   computing the next one.
"""

import argparse
import os

import torch
import torch.distributed as dist
import triton
import triton.language as tl

import iris
from iris.ccl import Config
from iris.experimental.plan import TriggeredView, tile_chunks


@triton.jit
def scale_tiles(
    x, out, row0, M, N, scale, handle, first_chunk, BM: tl.constexpr, BN: tl.constexpr, NUM_PROGRAMS: tl.constexpr
):
    """Write ``x * scale`` into rows ``row0..row0 + M`` of ``out``, tile by tile, marking each wave ready."""
    view = TriggeredView.initialize(handle)
    tiles_n = tl.cdiv(N, BN)
    num_tiles = tl.cdiv(M, BM) * tiles_n
    for step in range(tl.program_id(0), num_tiles, NUM_PROGRAMS):
        rows = (step // tiles_n) * BM + tl.arange(0, BM)[:, None]
        cols = (step % tiles_n) * BN + tl.arange(0, BN)[None, :]
        mask = (rows < M) & (cols < N)
        tile = tl.load(x + rows * N + cols, mask=mask)
        tl.store(out + (row0 + rows) * N + cols, tile * scale, mask=mask)
        # Wave k (every program's k-th tile) is chunk k: it is sent once all programs have marked it
        view.mark_ready(first_chunk + step // NUM_PROGRAMS)


def check(out, ref, what):
    if not torch.equal(out, ref):
        bad = (out != ref).nonzero()[0].tolist()
        raise RuntimeError(f"{what}: output differs from all_gather_into_tensor, first at {bad}")


def main():
    p = argparse.ArgumentParser(description="All-gather on the SDMA copy engines")
    p.add_argument("--m", type=int, default=1000, help="rows per rank")
    p.add_argument("--n", type=int, default=1024, help="columns")
    p.add_argument("--block", type=int, nargs=2, default=(64, 64), help="tile shape of the producer kernel")
    p.add_argument("--num_programs", type=int, default=32, help="programs of the persistent producer")
    p.add_argument("--rounds", type=int, default=5)
    p.add_argument("--heap_size", type=int, default=1 << 30)
    a = p.parse_args()

    torch.cuda.set_device(int(os.environ.get("LOCAL_RANK", 0)))
    dist.init_process_group(backend="nccl")
    try:
        # engines=("sdma",) sets the copy engines up; each collective call still picks its engine
        ctx = iris.iris(a.heap_size, engines=("sdma",))
        rank, world_size = ctx.get_rank(), ctx.get_num_ranks()
        M, N = a.m, a.n

        torch.manual_seed(rank)
        x = torch.randn(M, N, device=ctx.get_device())
        ref = torch.empty(world_size * M, N, device=x.device)
        dist.all_gather_into_tensor(ref, x)

        # 1. The same all-gather on either engine
        inp = ctx.zeros((M, N), dtype=torch.float32)
        inp.copy_(x)
        out = ctx.zeros((world_size * M, N), dtype=torch.float32)
        for engine in ("cu", "sdma"):
            out.zero_()
            # Peers write into out: every rank must be done wiping it first
            ctx.barrier()
            ctx.ccl.all_gather(out, inp, config=Config(engine=engine))
            check(out, ref, f"all_gather(engine={engine!r})")
        if rank == 0:
            ctx.info("all_gather: cu and sdma engines match")

        # 2. A producer kernel that triggers the all-gather as it goes
        out.zero_()
        mine = out[rank * M : (rank + 1) * M]
        schedule = tile_chunks(mine, a.block, tiles_per_chunk=a.num_programs, base=out)
        with ctx.ccl.prepare_all_gather(out, schedule) as plan:
            for round in range(1, a.rounds + 1):
                handle = plan.start()
                scale_tiles[(a.num_programs,)](
                    x, out, rank * M, M, N, float(round), handle, plan.chunk_id(rank, 0),
                    BM=a.block[0], BN=a.block[1], NUM_PROGRAMS=a.num_programs,
                )  # fmt: skip
                plan.wait()
                check(out, ref * round, f"prepare_all_gather round {round}")
                # Done reading this round: peers may now overwrite out
                plan.next_round()
        if rank == 0:
            ctx.info(f"prepare_all_gather: {a.rounds} rounds of {len(schedule)} chunks per rank match")
        ctx.barrier()
    finally:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
