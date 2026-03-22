#!/usr/bin/env python3
"""
Sweep number of CUs (comm_sms) to find how many are needed to saturate XGMI links.
Run with: torchrun --nproc_per_node=N bench_cu_sweep.py
"""

import os
import torch
import torch.distributed as dist

M, N = 8192, 8192
DTYPE = torch.float16
HEAP_SIZE = 2**33


def main():
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    torch.cuda.set_device(local_rank)
    dist.init_process_group(backend="nccl")
    rank = dist.get_rank()
    world_size = dist.get_world_size()

    backend = os.environ.get("BENCH_BACKEND", "triton")
    n_warmup = 10
    n_repeat = 50

    if backend == "gluon":
        import iris.experimental.iris_gluon as iris_mod
        from iris.ccl import Config
        from iris.ccl.all_gather import all_gather
        shmem = iris_mod.iris(HEAP_SIZE)
    else:
        import iris
        from iris.ccl import Config
        from iris.ccl.all_gather import all_gather
        shmem = iris.iris(HEAP_SIZE)

    # CU counts to sweep — must be divisible by world_size for fairness
    # MI355X has 304 CUs per GPU
    cu_values = []
    for c in [1, 2, 4, 8, 16, 32, 48, 64, 80, 96, 128, 152, 192, 256, 304]:
        if c % world_size == 0 or backend == "triton":
            cu_values.append(c)

    # Allocate tensors ONCE before the loop to avoid OOM
    inp = shmem.zeros((M, N), dtype=DTYPE)
    out = shmem.zeros((world_size * M, N), dtype=DTYPE)
    inp.fill_(float(rank + 1))

    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)

    if rank == 0:
        total_bytes = (world_size - 1) * M * N * 2
        print(f"Backend: {backend}, world_size={world_size}, M={M}, N={N}, fp16")
        print(f"Total remote data: {total_bytes / 1e6:.1f} MB")
        print(f"{'CUs':>6s}  {'Time (ms)':>10s}  {'BW (GB/s)':>10s}  {'per-link':>10s}")
        print("-" * 42)

    for num_cus in cu_values:
        if backend == "gluon":
            cfg = Config(use_gluon=True, block_size_m=32, block_size_n=1024, comm_sms=num_cus)
        else:
            cfg = Config(comm_sms=num_cus)

        try:
            for _ in range(n_warmup):
                out.zero_()
                shmem.barrier()
                all_gather(out, inp, shmem, config=cfg)
                shmem.barrier()

            shmem.barrier()
            start.record()
            for _ in range(n_repeat):
                all_gather(out, inp, shmem, config=cfg)
            end.record()
            torch.cuda.synchronize()
            shmem.barrier()

            ms = start.elapsed_time(end) / n_repeat
            total_bytes = (world_size - 1) * M * N * 2
            bw = (total_bytes / 1e9) / (ms / 1e3)
            per_link = (M * N * 2 / 1e9) / (ms / 1e3)

            if rank == 0:
                print(f"{num_cus:>6d}  {ms:>10.3f}  {bw:>10.2f}  {per_link:>10.2f}")

        except Exception as e:
            if rank == 0:
                print(f"{num_cus:>6d}  ERROR: {e}")

    del inp, out, shmem
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
