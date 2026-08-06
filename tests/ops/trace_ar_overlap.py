#!/usr/bin/env python3
"""Measure the overlap in the fused GEMM+AR pipeline instead of inferring it.

End-to-end timings tell us the fused kernel is slow. They do not tell us
whether the comm pools are slow because they are *waiting* (a serialization
problem) or because they are *working* slowly (a bandwidth problem). Those
two have opposite fixes.

The kernel records six timestamps per tile:

    gemm_beg ---- gemm_end          GEMM pool produced the tile
    rs_beg -- rs_ready ---- rs_end  RS pool: spin on gemm flag, then reduce
    ag_beg -- ag_ready ---- ag_end  AG pool: spin on rs flag, then gather

From those:
  * per-pool busy span vs wall span      -> is the pipeline overlapped at all?
  * spin time / total time per pool      -> waiting or working?
  * first-tile-ready latency per pool    -> how deep is the pipeline fill?

Timestamps come from the 100 MHz constant-rate counter, so cycles/100 = us.
"""

import argparse

import numpy as np
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
import triton

import iris

N_GLOBAL = 2880
K_GLOBAL = 4096


def summarize(ts, freq_mhz, label, rank):
    """Turn raw per-tile timestamps into a phase-overlap report."""
    g0 = ts["gemm_beg"].cpu().numpy().astype(np.int64)
    g1 = ts["gemm_end"].cpu().numpy().astype(np.int64)
    r0 = ts["rs_beg"].cpu().numpy().astype(np.int64)
    rr = ts["rs_ready"].cpu().numpy().astype(np.int64)
    r1 = ts["rs_end"].cpu().numpy().astype(np.int64)
    a0 = ts["ag_beg"].cpu().numpy().astype(np.int64)
    ar = ts["ag_ready"].cpu().numpy().astype(np.int64)
    a1 = ts["ag_end"].cpu().numpy().astype(np.int64)

    MAXV = np.iinfo(np.int64).max

    def nz(x):
        return x[(x > 0) & (x != MAXV)]

    allv = np.concatenate([nz(g0), nz(r0), nz(a0)])
    if allv.size == 0:
        print(f"[{label}] no timestamps recorded")
        return
    t0 = allv.min()
    to_us = lambda c: (c - t0) / freq_mhz

    def span(beg, end, name):
        b, e = nz(beg), nz(end)
        if b.size == 0 or e.size == 0:
            return None
        return dict(name=name, start=to_us(b.min()), stop=to_us(e.max()))

    sg = span(g0, g1, "GEMM")
    sr = span(r0, r1, "RS")
    sa = span(a0, a1, "AG")

    print(f"\n[{label}] rank {rank}  ({len(g0)} tiles)")
    print(f"  {'phase':<6} {'start_us':>9} {'stop_us':>9} {'span_us':>9}")
    for s in (sg, sr, sa):
        if s:
            print(f"  {s['name']:<6} {s['start']:9.2f} {s['stop']:9.2f} "
                  f"{s['stop']-s['start']:9.2f}")

    total = max(s["stop"] for s in (sg, sr, sa) if s)

    # Overlap: how much of the comm span sits inside the GEMM span. This is
    # the number the whole fusion argument rests on.
    if sg and sr:
        ov = max(0.0, min(sg["stop"], sr["stop"]) - max(sg["start"], sr["start"]))
        print(f"  GEMM/RS overlap  {ov:8.2f} us "
              f"({100*ov/max(sr['stop']-sr['start'], 1e-9):5.1f}% of RS span)")
    if sr and sa:
        ov = max(0.0, min(sr["stop"], sa["stop"]) - max(sr["start"], sa["start"]))
        print(f"  RS/AG   overlap  {ov:8.2f} us "
              f"({100*ov/max(sa['stop']-sa['start'], 1e-9):5.1f}% of AG span)")

    # Waiting vs working -- the decisive split.
    def split(beg, ready, end, name):
        m = (beg > 0) & (ready > 0) & (end > 0)
        if not m.any():
            return
        wait = (ready[m] - beg[m]) / freq_mhz
        work = (end[m] - ready[m]) / freq_mhz
        frac = wait.sum() / max(wait.sum() + work.sum(), 1e-9)
        print(f"  {name}: spin {wait.mean():7.3f} us/tile   "
              f"work {work.mean():7.3f} us/tile   "
              f"spin is {100*frac:5.1f}% of pool time")

    split(r0, rr, r1, "RS")
    split(a0, ar, a1, "AG")

    # Pipeline fill: when did each pool first have real work to do?
    if nz(rr).size and nz(g1).size:
        print(f"  first GEMM tile done at {to_us(nz(g1).min()):7.2f} us; "
              f"RS first unblocked at {to_us(nz(rr).min()):7.2f} us")
    if nz(ar).size and nz(r1).size:
        print(f"  first RS   tile done at {to_us(nz(r1).min()):7.2f} us; "
              f"AG first unblocked at {to_us(nz(ar).min()):7.2f} us")
    print(f"  kernel wall span {total:7.2f} us")


def _worker(local_rank, world_size, init_url, M, block_m, block_n, tpf, split, dump):
    dist.init_process_group(
        backend="nccl", init_method=init_url, world_size=world_size,
        rank=local_rank, device_id=torch.device(f"cuda:{local_rank}"))
    shmem = iris.iris(1 << 33)
    rank = shmem.get_rank()

    from iris.ops.matmul_all_reduce_hbm_buffer import (
        matmul_all_reduce_hbm_buffer,
        matmul_all_reduce_hbm_buffer_preamble,
    )

    freq_mhz = iris.hip.get_wall_clock_rate(rank) * 1e-3  # kHz -> MHz
    dtype = torch.float16
    K_local = K_GLOBAL // world_size

    A = shmem.zeros((M, K_local), device="cuda", dtype=dtype)
    A.copy_(torch.randn(M, K_local, dtype=dtype, device=f"cuda:{rank}") * 0.1)
    B = torch.randn(K_local, N_GLOBAL, dtype=dtype, device=f"cuda:{rank}") * 0.1
    ref = torch.mm(A, B)
    dist.all_reduce(ref, op=dist.ReduceOp.SUM)

    out = torch.zeros(M, N_GLOBAL, dtype=dtype, device=f"cuda:{rank}")
    g, r_, a_ = split
    ws = matmul_all_reduce_hbm_buffer_preamble(shmem, M, N_GLOBAL, dtype,
                                               block_m, block_n)
    shmem.barrier()

    kw = dict(block_m=block_m, block_n=block_n, block_k=64, num_gemm_sms=g,
              num_rs_sms=r_, num_ag_sms=a_, mfma=32, tiles_per_flag=tpf)

    # warm the JIT and the counters before the traced run
    for _ in range(30):
        out.zero_()
        matmul_all_reduce_hbm_buffer(shmem, out, A, B, workspace=ws, **kw)
    torch.cuda.synchronize()
    shmem.barrier()

    out.zero_()
    matmul_all_reduce_hbm_buffer(shmem, out, A, B, workspace=ws, trace=True, **kw)
    torch.cuda.synchronize()
    d = torch.abs(out - ref).max().item()
    shmem.barrier()

    if dump:
        import numpy as np
        arrs = {k: v.cpu().numpy() for k, v in ws["trace"].items()}
        arrs["freq_mhz"] = np.array(freq_mhz)
        path = dump.replace(".npz", f"_r{rank}.npz")
        np.savez(path, **arrs)
        if rank == 0:
            print(f"dumped raw timestamps -> {dump.replace('.npz', '_r*.npz')}")

    for r in range(world_size):
        if r == rank and rank in (0, world_size - 1):
            print(f"\n{'='*70}")
            print(f"M={M} bm={block_m} bn={block_n} tpf={tpf} "
                  f"G/R/A={g}/{r_}/{a_}  max_diff={d:.4f} "
                  f"{'PASS' if d < 2.0 else 'FAIL'}")
            summarize(ws["trace"], freq_mhz, f"M={M}", rank)
        shmem.barrier()

    dist.destroy_process_group()


def _free_port(explicit=None):
    """Hardcoded TCPStore ports collide when two people share a node.
    Bind :0 and let the OS hand us a free one unless told otherwise."""
    if explicit:
        return explicit
    import socket

    s = socket.socket()
    s.bind(("127.0.0.1", 0))
    port = s.getsockname()[1]
    s.close()
    return port


_PORT = None


def main():
    global _PORT
    p = argparse.ArgumentParser()
    p.add_argument("-r", "--num_ranks", type=int, default=8)
    p.add_argument("--port", type=int, default=None,
                   help="TCPStore port; default picks a free one")
    p.add_argument("-m", type=int, default=2048)
    p.add_argument("--bm", type=int, default=128)
    p.add_argument("--bn", type=int, default=128)
    p.add_argument("--tpf", type=int, default=1)
    p.add_argument("--split", type=str, default="192,32,32")
    p.add_argument("--dump", type=str, default=None,
                   help="write raw per-tile timestamps to this .npz")
    a = p.parse_args()
    _PORT = a.port
    split = tuple(int(x) for x in a.split.split(","))
    mp.spawn(fn=_worker,
             args=(a.num_ranks, f"tcp://127.0.0.1:{_free_port(_PORT)}", a.m, a.bm, a.bn,
                   a.tpf, split, a.dump),
             nprocs=a.num_ranks, join=True)


if __name__ == "__main__":
    main()
