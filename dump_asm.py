#!/usr/bin/env python3
"""Dump generated AMDGCN assembly for Triton and Gluon all-gather kernels."""

import os
import gc
import torch
import torch.distributed as dist
import torch.multiprocessing as mp

os.environ.setdefault("HSA_NO_SCRATCH_RECLAIM", "1")


def _worker(local_rank, world_size, init_url):
    dist.init_process_group(
        backend="nccl",
        init_method=init_url,
        world_size=world_size,
        rank=local_rank,
        device_id=torch.device(f"cuda:{local_rank}"),
    )

    import iris
    import iris.experimental.iris_gluon as iris_gluon
    from iris.ccl import Config
    from iris.ccl.all_gather import all_gather

    M, N = 128, 64
    dtype = torch.float16
    heap_size = 2**33
    rank = local_rank
    ws = world_size

    # Triton kernel
    s = iris.iris(heap_size)
    i = s.zeros((M, N), dtype=dtype)
    i.fill_(float(rank + 1))
    o = s.zeros((ws * M, N), dtype=dtype)
    s.barrier()
    all_gather(o, i, s, config=Config())
    s.barrier()
    del s
    gc.collect()

    # Gluon kernel
    s = iris_gluon.iris(heap_size)
    i = s.zeros((M, N), dtype=dtype)
    i.fill_(float(rank + 1))
    o = s.zeros((ws * M, N), dtype=dtype)
    s.barrier()
    all_gather(o, i, s, config=Config(use_gluon=True))
    s.barrier()
    del s
    gc.collect()

    if rank == 0:
        cache_dir = os.path.expanduser("~/.triton/cache")
        all_files = []
        for root, dirs, files in os.walk(cache_dir):
            for f in files:
                all_files.append(os.path.join(root, f))

        print("=== ALL FILES IN CACHE ===")
        for f in sorted(all_files):
            sz = os.path.getsize(f)
            ext = os.path.splitext(f)[1]
            print(f"  {sz:>10}  {ext:>10}  {f}")

        # Now dump all .amdgcn files
        amdgcn_files = [f for f in all_files if f.endswith(".amdgcn")]
        for f in sorted(amdgcn_files):
            name = os.path.basename(os.path.dirname(f)) + "/" + os.path.basename(f)
            print(f"\n{'=' * 80}")
            print(f"=== {os.path.basename(f)} ===")
            print(f"{'=' * 80}")
            with open(f) as fh:
                print(fh.read())

    dist.barrier()
    dist.destroy_process_group()


def main():
    mp.spawn(fn=_worker, args=(4, "tcp://127.0.0.1:29237"), nprocs=4, join=True)


if __name__ == "__main__":
    main()
