#!/usr/bin/env python3
"""
Benchmark all gluon all-gather variants against Triton.
Tests: Triton persistent, Gluon persistent, Gluon hoisted, Gluon partitioned.
"""
import os
import torch
import torch.distributed as dist

M, N = 8192, 8192
DTYPE = torch.float16
HEAP_SIZE = 2**33
n_warmup = 10
n_repeat = 50


def bench_one(label, all_gather_fn, inp, out, shmem, cfg, rank, world_size):
    """Warmup, validate, bench, report."""
    for _ in range(n_warmup):
        out.zero_()
        shmem.barrier()
        all_gather_fn(out, inp, shmem, config=cfg)
        shmem.barrier()

    # Validate
    out.zero_()
    inp.fill_(float(rank + 1))
    shmem.barrier()
    all_gather_fn(out, inp, shmem, config=cfg)
    shmem.barrier()
    expected = torch.zeros(world_size * M, N, dtype=DTYPE, device=f"cuda:{rank}")
    for r in range(world_size):
        expected[r * M : (r + 1) * M, :] = float(r + 1)
    valid = torch.allclose(out, expected, atol=1e-3)

    # Bench
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    shmem.barrier()
    start.record()
    for _ in range(n_repeat):
        all_gather_fn(out, inp, shmem, config=cfg)
    end.record()
    torch.cuda.synchronize()
    shmem.barrier()

    ms = start.elapsed_time(end) / n_repeat
    total_bytes = (world_size - 1) * M * N * 2
    bw = (total_bytes / 1e9) / (ms / 1e3)
    per_link = (M * N * 2 / 1e9) / (ms / 1e3)
    status = "PASS" if valid else "FAIL"

    if rank == 0:
        print(f"{label:<30s}  {ms:>8.3f} ms  {bw:>8.2f} GB/s  {per_link:>8.2f}/link  [{status}]")
    return bw


def main():
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    torch.cuda.set_device(local_rank)
    dist.init_process_group(backend="nccl")
    rank = dist.get_rank()
    world_size = dist.get_world_size()

    from iris.ccl.all_gather import all_gather
    from iris.ccl import Config

    if rank == 0:
        print(f"world_size={world_size}, M={M}, N={N}, fp16")
        print(f"{'Kernel':<30s}  {'Time':>8s}    {'algo_bw':>8s}     {'per-link':>8s}")
        print("-" * 75)

    cu_values = [8, 16, 32, 48, 64, 96]

    for num_cus in cu_values:
        if num_cus % world_size != 0:
            continue
        if rank == 0:
            print(f"\n--- {num_cus} CUs ---")

        # 1. Triton persistent
        import iris
        shmem_t = iris.iris(HEAP_SIZE)
        inp_t = shmem_t.zeros((M, N), dtype=DTYPE)
        out_t = shmem_t.zeros((world_size * M, N), dtype=DTYPE)
        inp_t.fill_(float(rank + 1))
        bench_one("Triton persistent", all_gather, inp_t, out_t, shmem_t,
                  Config(comm_sms=num_cus), rank, world_size)
        del inp_t, out_t, shmem_t
        torch.cuda.empty_cache()
        dist.barrier()

        # 2-4. Gluon variants
        import iris.experimental.iris_gluon as iris_gluon
        shmem_g = iris_gluon.iris(HEAP_SIZE)
        inp_g = shmem_g.zeros((M, N), dtype=DTYPE)
        out_g = shmem_g.zeros((world_size * M, N), dtype=DTYPE)
        inp_g.fill_(float(rank + 1))

        # 2. Gluon persistent (original)
        bench_one("Gluon persistent", all_gather, inp_g, out_g, shmem_g,
                  Config(use_gluon=True, block_size_m=32, block_size_n=1024,
                         comm_sms=num_cus, all_gather_variant="persistent"),
                  rank, world_size)

        # 3. Gluon hoisted
        bench_one("Gluon hoisted", all_gather, inp_g, out_g, shmem_g,
                  Config(use_gluon=True, block_size_m=32, block_size_n=1024,
                         comm_sms=num_cus, all_gather_variant="hoisted"),
                  rank, world_size)

        # 4. Gluon partitioned
        bench_one("Gluon partitioned", all_gather, inp_g, out_g, shmem_g,
                  Config(use_gluon=True, block_size_m=32, block_size_n=1024,
                         comm_sms=num_cus, all_gather_variant="partitioned"),
                  rank, world_size)

        del inp_g, out_g, shmem_g
        torch.cuda.empty_cache()
        dist.barrier()

    dist.destroy_process_group()


if __name__ == "__main__":
    main()
