#!/usr/bin/env python3
"""
Benchmark optimized gluon all-gather vs Triton vs RCCL.
Run with: torchrun --nproc-per-node=4 sweep_layout.py
"""

import os
import torch
import torch.distributed as dist
import iris
import iris.experimental.iris_gluon as iris_gluon
from iris.ccl import Config
from iris.ccl.all_gather import all_gather

M, N = 8192, 8192
DTYPE = torch.float16
HEAP_SIZE = 2**33


def bench(shmem, config, label, n_warmup=10, n_repeat=50):
    rank = shmem.get_rank()
    world_size = shmem.get_num_ranks()

    try:
        inp = shmem.zeros((M, N), dtype=DTYPE)
        out = shmem.zeros((world_size * M, N), dtype=DTYPE)
        inp.fill_(float(rank + 1))

        # Warmup
        for _ in range(n_warmup):
            out.zero_()
            shmem.barrier()
            all_gather(out, inp, shmem, config=config)
            shmem.barrier()

        # Timed runs
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)

        shmem.barrier()
        start.record()
        for _ in range(n_repeat):
            all_gather(out, inp, shmem, config=config)
        end.record()
        torch.cuda.synchronize()
        shmem.barrier()

        ms = start.elapsed_time(end) / n_repeat
        element_size = torch.tensor([], dtype=DTYPE).element_size()
        total_bytes = (world_size - 1) * M * N * element_size
        bw = (total_bytes / 1e9) / (ms / 1e3)

        # Validate
        out.zero_()
        inp.fill_(float(rank + 1))
        shmem.barrier()
        all_gather(out, inp, shmem, config=config)
        shmem.barrier()

        expected = torch.zeros(world_size * M, N, dtype=DTYPE, device=f"cuda:{rank}")
        for r in range(world_size):
            expected[r * M : (r + 1) * M, :] = float(r + 1)
        valid = torch.allclose(out, expected, atol=1e-3)

        if rank == 0:
            status = "PASS" if valid else "FAIL"
            print(f"{label:50s}  {ms:8.3f} ms  {bw:8.2f} GB/s  [{status}]")

        return ms, bw, valid
    except Exception as e:
        if rank == 0:
            err_str = str(e).split("\n")[0][:80]
            print(f"{label:50s}  {'---':>8s}  {'---':>8s}  [ERROR: {err_str}]")
        return None, None, False


def main():
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    torch.cuda.set_device(local_rank)
    dist.init_process_group(backend="nccl")
    rank = dist.get_rank()

    if rank == 0:
        print(f"{'Config':50s}  {'Time':>8s}  {'BW':>8s}  Status")
        print("-" * 85)

    # 1. Triton baseline
    shmem_triton = iris.iris(HEAP_SIZE)
    bench(shmem_triton, Config(), "Triton (persistent, default)")
    del shmem_triton

    # 2. Optimized gluon (load once, write to all ranks)
    shmem_gluon = iris_gluon.iris(HEAP_SIZE)

    # block_size_n must be multiple of 256 (= spt * 64 * 4 warps)
    # spt = bn / 256: bn=256 -> spt=1, bn=512 -> spt=2, bn=1024 -> spt=4
    for bn in [256, 512, 1024, 2048]:
        for bm in [4, 8, 16, 32]:
            cfg = Config(use_gluon=True, block_size_m=bm, block_size_n=bn, num_warps=4)
            bench(shmem_gluon, cfg, f"Gluon opt bm={bm} bn={bn} spt={bn//256}")

    del shmem_gluon

    # 3. RCCL baseline
    pytorch_inp = torch.zeros(M, N, dtype=DTYPE, device=f"cuda:{rank}")
    pytorch_out = torch.zeros(dist.get_world_size() * M, N, dtype=DTYPE, device=f"cuda:{rank}")
    pytorch_inp.fill_(float(rank + 1))

    for _ in range(10):
        dist.all_gather_into_tensor(pytorch_out, pytorch_inp)
    torch.cuda.synchronize()
    dist.barrier()

    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    n_repeat = 50

    dist.barrier()
    start.record()
    for _ in range(n_repeat):
        dist.all_gather_into_tensor(pytorch_out, pytorch_inp)
    end.record()
    torch.cuda.synchronize()
    dist.barrier()

    ms = start.elapsed_time(end) / n_repeat
    element_size = torch.tensor([], dtype=DTYPE).element_size()
    total_bytes = (dist.get_world_size() - 1) * M * N * element_size
    bw = (total_bytes / 1e9) / (ms / 1e3)

    if rank == 0:
        print(f"{'RCCL (all_gather_into_tensor)':50s}  {ms:8.3f} ms  {bw:8.2f} GB/s  [REF]")

    dist.destroy_process_group()


if __name__ == "__main__":
    main()
