#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.

"""Transport check for the RCCL GIN provider: an all-to-all of tagged blocks.

Every rank sends one block to every rank from a single Triton kernel. Peers
the window's address map marks ``direct`` (same node) get unmodified
``iris.store``; everyone else gets ``rccl_gin.put_signal``. Each receiver then
checks every block's tag. Run with torchrun; see README.md for the multi-node
launch.

``SMOKE_PATHS=direct|gin|both`` (default ``both``) limits which peers are
exercised, to tell a fault in one path from the other.
"""

import os
import sys

import torch
import torch.distributed as dist
import triton
import triton.language as tl

import iris
from iris.experimental import rccl_gin

BLOCK = 1024  # int32 words per block
MAX_POLLS = 20_000_000  # bounds every wait, so a lost message fails instead of hanging
FLUSH_CYCLES = tl.constexpr(4_000_000_000)  # flush_timeout bound, about 2 s


@triton.jit
def _send_kernel(
    win_ptr, heap_bases, comm, win, flags_off, rank, direct, epoch, flushed,
    BLOCK: tl.constexpr, DIRECT: tl.constexpr, GIN: tl.constexpr,
):  # fmt: skip
    peer = tl.program_id(0)
    # Byte offsets: send block for `peer` at peer*BLOCK*4, receive slot for
    # `rank` at (world + rank)*BLOCK*4 -- both inside the same window.
    world = tl.num_programs(0)
    src_off = peer * BLOCK * 4
    dst_off = (world + rank) * BLOCK * 4
    is_direct = tl.load(direct + peer) != 0
    if DIRECT:
        if is_direct:
            offs = tl.arange(0, BLOCK)
            src = (win_ptr + src_off).to(tl.pointer_type(tl.int32))
            dst = (win_ptr + dst_off).to(tl.pointer_type(tl.int32))
            iris.store(dst + offs, tl.load(src + offs), rank, peer, heap_bases)
            flag = (win_ptr + flags_off).to(tl.pointer_type(tl.int64)) + rank
            iris.atomic_xchg(flag, tl.cast(epoch, tl.int64), rank, peer, heap_bases, sem="release", scope="sys")
    if GIN:
        if not is_direct:
            rccl_gin.put_signal(comm, 0, peer, win, dst_off, win, src_off, BLOCK * 4, rank, 1)
            # Not needed for correctness here (the receiver's signal is the
            # completion); exercises flush seeing a put from the same kernel.
            tl.store(flushed + peer, rccl_gin.flush_timeout(comm, 0, FLUSH_CYCLES))


@triton.jit
def _wait_kernel(win_ptr, comm, flags_off, signal_base, direct, want, seen, epoch, max_polls):
    """Bounded wait. seen[src] gets the last flag or signal value observed for
    ``src`` (relative to the run's start), so success is ``seen[src] >= epoch``."""
    src = tl.program_id(0)
    polls = 0
    if tl.load(want + src) == 0:
        polls = max_polls  # not exercised this run: record the value without waiting
    if tl.load(direct + src) != 0:
        flag = (win_ptr + flags_off).to(tl.pointer_type(tl.int64)) + src
        v = tl.atomic_add(flag, 0, sem="acquire", scope="sys")
        while (v < epoch) & (polls < max_polls):
            v = tl.atomic_add(flag, 0, sem="acquire", scope="sys")
            polls += 1
    else:
        base = tl.load(signal_base + src)
        v = rccl_gin.read_signal(comm, 0, src) - base
        while (v < epoch) & (polls < max_polls):
            v = rccl_gin.read_signal(comm, 0, src) - base
            polls += 1
    tl.store(seen + src, v)


def main():
    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
    dist.init_process_group("gloo")
    rank, world = dist.get_rank(), dist.get_world_size()

    prov = rccl_gin.RcclGinProvider()
    flags_off = 2 * world * BLOCK * 4
    w = prov.allocate_window(flags_off + world * 8)
    dc = prov.device_comm(signal_count=world)
    base = prov.read_signals(dc)

    direct_list = w.address_map.direct
    direct = torch.tensor(direct_list, dtype=torch.int32, device="cuda")
    n_direct = sum(direct_list)
    if rank == 0:
        print(
            f"RCCL {prov.rccl_version}, GIN backend {prov.gin_type_name}, {world} ranks, "
            f"{n_direct} direct + {world - n_direct} GIN peers per rank",
            flush=True,
        )

    paths = os.environ.get("SMOKE_PATHS", "both")
    if paths not in ("both", "direct", "gin"):
        raise ValueError(f"SMOKE_PATHS must be both, direct or gin, not {paths!r}")
    use_direct, use_gin = paths in ("both", "direct"), paths in ("both", "gin")
    exercised = [s for s in range(world) if (use_direct if direct_list[s] else use_gin)]
    want = torch.tensor([int(s in exercised) for s in range(world)], dtype=torch.int32, device="cuda")

    words = w.tensor.view(torch.int32)
    send = words[: world * BLOCK].view(world, BLOCK)
    recv = words[world * BLOCK : 2 * world * BLOCK].view(world, BLOCK)
    w.tensor[flags_off:].zero_()
    seen = torch.zeros(world, dtype=torch.int64, device="cuda")
    flushed = torch.zeros(world, dtype=torch.int32, device="cuda")
    lib = rccl_gin.extern_libs()

    def path(s):
        return "direct" if direct_list[s] else "gin"

    ok = True
    for epoch in (1, 2, 3):
        recv.fill_(-1)
        tags = torch.arange(world, device="cuda", dtype=torch.int32)
        send.copy_((epoch * 1_000_000 + rank * 1000 + tags)[:, None].expand(world, BLOCK))
        prov.barrier()  # nobody may write into recv before its owner cleared it
        _send_kernel[(world,)](
            w.tensor, w.address_map.peer_bases, dc.ptr, w.handle, flags_off, rank, direct, epoch, flushed,
            BLOCK=BLOCK, DIRECT=use_direct, GIN=use_gin, num_warps=4, extern_libs=lib,
        )  # fmt: skip
        _wait_kernel[(world,)](
            w.tensor, dc.ptr, flags_off, base, direct, want, seen, epoch, MAX_POLLS,
            num_warps=4, extern_libs=lib,
        )  # fmt: skip
        torch.cuda.synchronize()

        timed_out = [s for s, f in enumerate(flushed.tolist()) if f]
        if timed_out:
            ok = False
            print(f"[rank {rank}] epoch {epoch}: flush timed out after puts to {timed_out}", flush=True)
        missing = [(s, path(s), v) for s, v in enumerate(seen.tolist()) if v < epoch and s in exercised]
        if missing:
            ok = False
            print(f"[rank {rank}] epoch {epoch}: nothing from (src, path, seen) {missing}", flush=True)
        expect = (epoch * 1_000_000 + torch.arange(world, device="cuda", dtype=torch.int32) * 1000 + rank)[:, None]
        bad = [s for s in (recv != expect).any(dim=1).nonzero().flatten().tolist() if s in exercised]
        if bad:
            ok = False
            print(f"[rank {rank}] epoch {epoch}: wrong blocks from {[(s, path(s)) for s in bad]}", flush=True)

    result = torch.tensor([int(ok)])
    dist.all_reduce(result, op=dist.ReduceOp.MIN)
    if rank == 0:
        verdict = "PASS" if result.item() else "FAIL"
        print(f"GIN_SMOKE_RESULT: {verdict} (paths={paths}, 3 epochs)", flush=True)
    prov.close()
    dist.destroy_process_group()
    sys.exit(0 if result.item() else 1)


if __name__ == "__main__":
    main()
