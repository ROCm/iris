#!/usr/bin/env python3
"""Does the fused AR silently produce wrong results below num_warps=8?

Two runs of the same kernel disagree. Yael reports num_warps 2 and 4 FAIL with
maxdiff 3.0-3.4 on MI355X. My own warp sweep gated correctness at tol 2.0 over
3 repeated iterations and num_warps=4 PASSED on MI350X -- and was the fastest
M=2048 config I measured. If she is right, that number is garbage.

Differences between the two runs that could explain it, and which this
isolates:
  1. hardware (MI355X vs MI350X -- different CU count changes the CU split)
  2. workspace reuse. My sweep shares ONE workspace across every config in the
     M loop. The flag counters are monotonic, so a config that times out
     leaves them below target and can poison whatever runs next. This gives
     each warp count a FRESH workspace.

Reports maxdiff unconditionally rather than a pass/fail verdict, so a marginal
result is visible instead of being rounded to PASS.
"""

import argparse

import torch
import torch.distributed as dist
import torch.multiprocessing as mp

import iris

N_GLOBAL = 2880
K_GLOBAL = 4096

CASES = [
    # M,   block_m, block_n, mfma, tpf, (G, R, A)
    (128,  16,  128, 32, 1, (192, 32, 32)),
    (512,  64,  128, 16, 1, (96, 96, 64)),
    (2048, 128, 128, 16, 2, (96, 96, 64)),
]


def _worker(local_rank, world_size, init_url):
    dist.init_process_group(
        backend="nccl", init_method=init_url, world_size=world_size,
        rank=local_rank, device_id=torch.device(f"cuda:{local_rank}"))
    shmem = iris.iris(1 << 33)
    rank = shmem.get_rank()
    cu = torch.cuda.get_device_properties(rank).multi_processor_count
    arch = torch.cuda.get_device_properties(rank).gcnArchName.split(":")[0]

    from iris.ops.matmul_all_reduce_hbm_buffer import (
        matmul_all_reduce_hbm_buffer,
        matmul_all_reduce_hbm_buffer_preamble,
    )

    dtype = torch.float16
    K_local = K_GLOBAL // world_size

    if rank == 0:
        print(f"\nfused AR correctness vs num_warps   {arch}  CUs={cu}  "
              f"TP={world_size}")
        print(f"{'M':>6} {'warps':>6} {'G/R/A':>12} {'iter1':>10} {'iter2':>10} "
              f"{'iter3':>10}  verdict")

    for M, bm, bn, mfma, tpf, (g, r_, a_) in CASES:
        if g + r_ + a_ > cu:
            if rank == 0:
                print(f"{M:6d}  split {g}/{r_}/{a_} exceeds {cu} CUs -- skipped")
            continue

        A = shmem.zeros((M, K_local), device="cuda", dtype=dtype)
        A.copy_(torch.randn(M, K_local, dtype=dtype, device=f"cuda:{rank}") * 0.1)
        B = torch.randn(K_local, N_GLOBAL, dtype=dtype, device=f"cuda:{rank}") * 0.1
        ref = torch.mm(A, B)
        dist.all_reduce(ref, op=dist.ReduceOp.SUM)
        torch.cuda.synchronize()

        for w in [2, 4, 8, 16]:
            # fresh workspace per warp count: monotonic counters from a
            # previous config must not carry over
            wsx = matmul_all_reduce_hbm_buffer_preamble(
                shmem, M, N_GLOBAL, dtype, bm, bn)
            shmem.barrier()
            out = torch.zeros(M, N_GLOBAL, dtype=dtype, device=f"cuda:{rank}")
            diffs = []
            try:
                for _ in range(3):
                    out.zero_()
                    matmul_all_reduce_hbm_buffer(
                        shmem, out, A, B, workspace=wsx, block_m=bm, block_n=bn,
                        block_k=64, mfma=mfma, tiles_per_flag=tpf,
                        num_gemm_sms=g, num_rs_sms=r_, num_ag_sms=a_,
                        num_warps=w)
                    torch.cuda.synchronize()
                    diffs.append(torch.abs(out - ref).max().item())
                shmem.barrier()
            except Exception as ex:
                if rank == 0:
                    print(f"{M:6d} {w:6d} {f'{g}/{r_}/{a_}':>12}  EXC "
                          f"{type(ex).__name__}: {str(ex)[:40]}")
                continue

            if rank == 0:
                worst = max(diffs)
                verdict = "PASS" if worst < 0.05 else "WRONG"
                cells = " ".join(f"{d:10.4f}" for d in diffs)
                print(f"{M:6d} {w:6d} {f'{g}/{r_}/{a_}':>12} {cells}  {verdict}",
                      flush=True)
            del wsx, out
            torch.cuda.empty_cache()
            shmem.barrier()

        del A, B
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
