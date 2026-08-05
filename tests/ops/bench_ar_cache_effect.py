#!/usr/bin/env python3
"""Is our one-shot AllReduce bandwidth number real, or is it Infinity Cache?

Kenji measured one-shot at 419 GB/s (94% of a 448 GB/s line rate) at M=2048.
Yael's cu_scale measured 202 GB/s for the same operation. Both cannot be the
ceiling, and the ceiling is what decides whether a faster two-shot is worth
building.

Prime suspect: methodology. Both benchmarks call the AR repeatedly on the SAME
symmetric buffer. At M=2048 the per-rank read working set is
ws * M * N * 2 = 94.4 MB, which fits in MI355X's 256 MB Infinity Cache. So
iterations 2..N may be served from cache and never touch XGMI at all.

This rotates over enough distinct buffers to blow past the cache and compares:

    hot  : same buffer every iteration      (the number we have been quoting)
    cold : rotate over K buffers, K*94.4MB >> 256MB

If cold is much slower, our quoted line-rate percentages are inflated and the
real headroom for a two-shot is smaller than either of us claimed.
"""

import argparse

import torch
import torch.distributed as dist
import torch.multiprocessing as mp

import iris

N_GLOBAL = 2880
K_GLOBAL = 4096
WARMUP, ITERS = 20, 50


def bench(fn):
    s = torch.cuda.Event(enable_timing=True)
    e = torch.cuda.Event(enable_timing=True)
    for i in range(WARMUP):
        fn(i)
    torch.cuda.synchronize()
    s.record()
    for i in range(ITERS):
        fn(i)
    e.record()
    torch.cuda.synchronize()
    return s.elapsed_time(e) / ITERS


def _worker(local_rank, world_size, init_url, nbuf):
    dist.init_process_group(
        backend="nccl", init_method=init_url, world_size=world_size,
        rank=local_rank, device_id=torch.device(f"cuda:{local_rank}"))
    shmem = iris.iris(1 << 34)
    rank = shmem.get_rank()

    from iris.ops.all_reduce_fast import one_shot_all_reduce

    dtype = torch.float16
    LINE_GBS = 448.0

    if rank == 0:
        print(f"\nOne-shot AR: hot-buffer vs cache-busting   TP={world_size}")
        print(f"  rotating over {nbuf} buffers\n")
        print(f"{'M':>6} {'MB/rank':>9} {'hot ms':>9} {'hot GB/s':>9} "
              f"{'cold ms':>9} {'cold GB/s':>10} {'ratio':>7}")

    for M in [512, 2048]:
        bufs = [shmem.zeros((M, N_GLOBAL), device="cuda", dtype=dtype)
                for _ in range(nbuf)]
        for b in bufs:
            b.copy_(torch.randn(M, N_GLOBAL, dtype=dtype,
                                device=f"cuda:{rank}") * 0.1)
        out = torch.zeros(M, N_GLOBAL, dtype=dtype, device=f"cuda:{rank}")
        shmem.barrier()

        # bytes each rank READS: one full [M,N] from every peer
        mb = world_size * M * N_GLOBAL * 2 / 1e6

        hot = bench(lambda i: one_shot_all_reduce(shmem, out, bufs[0]))
        cold = bench(lambda i: one_shot_all_reduce(shmem, out, bufs[i % nbuf]))

        if rank == 0:
            hg = mb / 1e3 / (hot / 1e3)
            cg = mb / 1e3 / (cold / 1e3)
            print(f"{M:6d} {mb:9.1f} {hot:9.4f} {hg:9.1f} "
                  f"{cold:9.4f} {cg:10.1f} {cold/hot:6.2f}x")
            print(f"       {'':9} {'':9} {100*hg/LINE_GBS:8.0f}% "
                  f"{'':9} {100*cg/LINE_GBS:9.0f}%   of {LINE_GBS:.0f} GB/s")

        del bufs, out
        torch.cuda.empty_cache()
        shmem.barrier()

    shmem.barrier()
    dist.destroy_process_group()


def main():
    p = argparse.ArgumentParser()
    p.add_argument("-r", "--num_ranks", type=int, default=8)
    p.add_argument("-n", "--nbuf", type=int, default=8,
                   help="distinct buffers to rotate over")
    a = p.parse_args()
    mp.spawn(fn=_worker, args=(a.num_ranks, "tcp://127.0.0.1:29519", a.nbuf),
             nprocs=a.num_ranks, join=True)


if __name__ == "__main__":
    main()
