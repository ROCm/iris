#!/usr/bin/env python3
"""num_warps on the fused HBM-buffer AR -- the one knob never swept.

bench_ar_hbm_buffer.py has zero occurrences of num_warps; it was hardcoded to
8 while everything else was tuned. On a standalone one-shot AR the same knob
measured 4.3x (59.9 -> 256.8 GB/s going 1 -> 8 warps), so the fused kernel is
very likely mistuned on it too.

There is a wrinkle unique to the fused case worth measuring directly:
**num_warps is a kernel-level launch parameter, so all three pools share one
value.** The GEMM pool wants whatever suits an MFMA inner loop; the RS and AG
pools want whatever suits streaming peer loads. If those disagree, fusion pays
a compromise the two-kernel path never does -- and that is a real, previously
untested cost of work-group specialization.

To separate the two effects this sweeps warps against the standalone GEMM and
the standalone one-shot AR as well, so we can see what each pool would have
picked on its own.
"""

import argparse

import torch
import torch.distributed as dist
import torch.multiprocessing as mp
import triton

import iris

N_GLOBAL = 2880
K_GLOBAL = 4096
WARMUP, ITERS = 15, 40

# Best non-warp config per M, from bench_ar_hbm_buffer.py
BEST = {
    128:  dict(block_m=16,  block_n=128, mfma=32, tpf=1),
    512:  dict(block_m=64,  block_n=128, mfma=16, tpf=1),
    2048: dict(block_m=128, block_n=128, mfma=16, tpf=2),
}
SPLITS = [(192, 32, 32), (128, 64, 64), (96, 96, 64), (64, 96, 96)]


def bench(fn):
    s = torch.cuda.Event(enable_timing=True)
    e = torch.cuda.Event(enable_timing=True)
    for _ in range(WARMUP):
        fn()
    torch.cuda.synchronize()
    s.record()
    for _ in range(ITERS):
        fn()
    e.record()
    torch.cuda.synchronize()
    return s.elapsed_time(e) / ITERS


def _worker(local_rank, world_size, init_url):
    dist.init_process_group(
        backend="nccl", init_method=init_url, world_size=world_size,
        rank=local_rank, device_id=torch.device(f"cuda:{local_rank}"))
    shmem = iris.iris(1 << 33)
    rank = shmem.get_rank()
    cu = torch.cuda.get_device_properties(rank).multi_processor_count

    from iris.ops.all_reduce_fast import one_shot_all_reduce
    from iris.ops.matmul_all_reduce_hbm_buffer import (
        matmul_all_reduce_hbm_buffer,
        matmul_all_reduce_hbm_buffer_preamble,
    )

    dtype = torch.float16
    K_local = K_GLOBAL // world_size

    if rank == 0:
        print(f"\nnum_warps sweep, fused HBM-buffer AR   TP={world_size} CUs={cu}")

    for M in [512, 2048, 128]:
        cfg = BEST[M]
        A = shmem.zeros((M, K_local), device="cuda", dtype=dtype)
        A.copy_(torch.randn(M, K_local, dtype=dtype, device=f"cuda:{rank}") * 0.1)
        B = torch.randn(K_local, N_GLOBAL, dtype=dtype, device=f"cuda:{rank}") * 0.1
        ref = torch.mm(A, B)
        dist.all_reduce(ref, op=dist.ReduceOp.SUM)
        torch.cuda.synchronize()

        Ct = torch.empty(M, N_GLOBAL, dtype=dtype, device=f"cuda:{rank}")
        base = bench(lambda: (torch.mm(A, B, out=Ct),
                              dist.all_reduce(Ct, op=dist.ReduceOp.SUM)))

        if rank == 0:
            print(f"\n=== M={M}  ({cfg})  torch={base:.4f}ms ===")

        # what the comm pool would pick on its own
        Cs = shmem.zeros((M, N_GLOBAL), device="cuda", dtype=dtype)
        Co = torch.zeros(M, N_GLOBAL, dtype=dtype, device=f"cuda:{rank}")
        torch.mm(A, B, out=Cs)
        shmem.barrier()
        if rank == 0:
            print("  standalone one-shot AR:")
        for w in [4, 8, 16]:
            try:
                Co.zero_()
                for _ in range(3):
                    one_shot_all_reduce(shmem, Co, Cs, num_warps=w)
                torch.cuda.synchronize()
                if torch.abs(Co - ref).max().item() > 2.0:
                    continue
                ms = bench(lambda w=w: one_shot_all_reduce(shmem, Co, Cs, num_warps=w))
                if rank == 0:
                    gbs = world_size * M * N_GLOBAL * 2 / 1e9 / (ms / 1e3)
                    print(f"    warps={w:2d}  {ms:.4f}ms  {gbs:6.1f} GB/s")
            except Exception:
                continue
        shmem.barrier()

        # the fused kernel: one num_warps shared by all three pools
        if rank == 0:
            print("  fused HBM-buffer (all 3 pools share num_warps):")
        best = (1e9, None)
        wsx = matmul_all_reduce_hbm_buffer_preamble(
            shmem, M, N_GLOBAL, dtype, cfg["block_m"], cfg["block_n"])
        shmem.barrier()
        out = torch.zeros(M, N_GLOBAL, dtype=dtype, device=f"cuda:{rank}")
        for w in [4, 8, 16]:
            for g, r_, a_ in SPLITS:
                if g + r_ + a_ > cu:
                    continue
                try:
                    kw = dict(block_m=cfg["block_m"], block_n=cfg["block_n"],
                              block_k=64, mfma=cfg["mfma"], tiles_per_flag=cfg["tpf"],
                              num_gemm_sms=g, num_rs_sms=r_, num_ag_sms=a_,
                              num_warps=w)
                    ok = True
                    for _ in range(3):
                        out.zero_()
                        matmul_all_reduce_hbm_buffer(shmem, out, A, B,
                                                     workspace=wsx, **kw)
                        torch.cuda.synchronize()
                        if torch.abs(out - ref).max().item() > 2.0:
                            ok = False
                            break
                    shmem.barrier()
                    if not ok:
                        continue
                    ms = bench(lambda kw=kw: matmul_all_reduce_hbm_buffer(
                        shmem, out, A, B, workspace=wsx, **kw))
                    if rank == 0:
                        mark = ""
                        if ms < best[0]:
                            mark = "  ***"
                        print(f"    warps={w:2d} G/R/A={g:3d}/{r_:3d}/{a_:3d}  "
                              f"{ms:.4f}ms  {base/ms:.2f}x{mark}", flush=True)
                    if ms < best[0]:
                        best = (ms, f"warps={w} G/R/A={g}/{r_}/{a_}")
                except Exception:
                    continue

        if rank == 0 and best[1]:
            print(f"  BEST: {best[0]:.4f}ms  {base/best[0]:.2f}x  ({best[1]})")

        del A, B, Cs, Co, out, wsx
        torch.cuda.empty_cache()
        shmem.barrier()

    shmem.barrier()
    dist.destroy_process_group()


def _free_port(explicit=None):
    if explicit:
        return explicit
    import socket

    s = socket.socket()
    s.bind(("127.0.0.1", 0))
    port = s.getsockname()[1]
    s.close()
    return port


def main():
    p = argparse.ArgumentParser()
    p.add_argument("-r", "--num_ranks", type=int, default=8)
    p.add_argument("--port", type=int, default=None)
    a = p.parse_args()
    mp.spawn(fn=_worker,
             args=(a.num_ranks, f"tcp://127.0.0.1:{_free_port(a.port)}"),
             nprocs=a.num_ranks, join=True)


if __name__ == "__main__":
    main()
