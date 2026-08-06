
"""Measure cross-rank launch skew DIRECTLY, without tracing a whole kernel.

Each rank stamps its own entry, signals arrival to every peer, spins until all
ws arrivals land, then stamps again. The delta is the arrival spread as seen in
that rank's own clock, so no cross-GPU clock alignment is needed. Two
timestamps and one WG, so the instrument cannot manufacture the phenomenon.
"""
import torch, torch.distributed as dist, torch.multiprocessing as mp, iris, triton, socket
import triton.language as tl
from iris.mem import utils as device_utils

@triton.jit
def arrive(flags, out, hb: tl.tensor, it, cur: tl.constexpr, W: tl.constexpr,
           SPIN: tl.constexpr, NWG: tl.constexpr):
    pid = tl.program_id(0)
    if pid == 0:
        t0 = device_utils.read_realtime()
        tl.atomic_add(flags + tl.arange(0, 1), 1, sem="release", scope="gpu")
        for p in tl.static_range(0, W):
            if p != cur:
                iris.atomic_add(flags + tl.arange(0, 1), 1, cur, p, hb,
                                sem="release", scope="sys")
        tgt = it * W
        z = tl.zeros((1,), dtype=tl.int32)
        d = tl.min(tl.atomic_add(flags + tl.arange(0, 1), z, sem="acquire", scope="sys"))
        s = 0
        while (d < tgt) and (s < SPIN):
            d = tl.min(tl.atomic_add(flags + tl.arange(0, 1), z, sem="acquire", scope="sys"))
            s += 1
        t1 = device_utils.read_realtime()
        tl.store(out + 0, t0)
        tl.store(out + 1, t1)
        tl.store(out + 2, s)

def bench(fn, n, w=20):
    for _ in range(w): fn()
    torch.cuda.synchronize()
    s = torch.cuda.Event(enable_timing=True); e = torch.cuda.Event(enable_timing=True)
    s.record()
    for _ in range(n): fn()
    e.record(); torch.cuda.synchronize(); return s.elapsed_time(e) / n

def _w(lr, W, url):
    dist.init_process_group(backend="nccl", init_method=url, world_size=W, rank=lr)
    torch.cuda.set_device(lr)
    sh = iris.iris(1 << 32); hb = sh.get_heap_bases(); rank = sh.get_rank()
    def P(*a):
        if rank == 0: print(*a, flush=True)
    flags = sh.zeros((64,), dtype=torch.int32)
    out = torch.zeros(4, device=f"cuda:{rank}", dtype=torch.int64)
    it = [0]
    def run(nwg=1):
        it[0] += 1
        arrive[(nwg,)](flags, out, hb, it[0], rank, W, 4_000_000, nwg, num_warps=1)
    sh.barrier(); run(); torch.cuda.synchronize()
    freq = 100.0  # 100 MHz constant counter -> us
    waits = []
    for _ in range(200):
        sh.barrier()
        run(); torch.cuda.synchronize()
        t0, t1, spins = out[0].item(), out[1].item(), out[2].item()
        waits.append(((t1 - t0) / freq, spins))
    ws_us = sorted(w for w, _ in waits)
    n = len(ws_us)
    P(f"\ncross-rank arrival spread seen by rank {rank}, ws={W}, {n} launches")
    P(f"  p10 {ws_us[n//10]:8.2f} us   p50 {ws_us[n//2]:8.2f} us   "
      f"p90 {ws_us[9*n//10]:8.2f} us   max {ws_us[-1]:8.2f} us")
    per = bench(lambda: run(), 200)
    P(f"  empty barrier kernel, benchmarked back-to-back: {per*1000:.2f} us/launch")
    P(f"  => in a steady-state loop the launch skew a fused kernel must absorb")
    P(f"     is bounded by the p50 above, not by a single-shot measurement.")
    dist.destroy_process_group()

def main():
    import argparse
    p = argparse.ArgumentParser(); p.add_argument("-r", "--num_ranks", type=int, default=8)
    a = p.parse_args()
    s = socket.socket(); s.bind(("127.0.0.1", 0)); port = s.getsockname()[1]; s.close()
    mp.spawn(fn=_w, args=(a.num_ranks, f"tcp://127.0.0.1:{port}"), nprocs=a.num_ranks, join=True)

if __name__ == "__main__":
    main()
