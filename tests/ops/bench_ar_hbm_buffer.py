#!/usr/bin/env python3
"""HBM-buffer fused GEMM + two-shot AllReduce: correctness, then CU-split sweep.

Correctness first (including repeated iterations, which is what validates the
monotonic counters), then sweep the three-way CU split and the GEMM knobs.

Compared against:
  torch.mm + dist.all_reduce   (baseline)
  torch.mm + our one-shot AR   (the current best)
"""

import argparse
import json

import torch
import torch.distributed as dist
import torch.multiprocessing as mp
import triton

import iris

N_GLOBAL = 2880
K_GLOBAL = 4096
WARMUP, ITERS = 20, 50


def bench(fn, pre=None):
    s = torch.cuda.Event(enable_timing=True)
    e = torch.cuda.Event(enable_timing=True)
    for _ in range(WARMUP):
        if pre:
            pre()
        fn()
    torch.cuda.synchronize()
    s.record()
    for _ in range(ITERS):
        if pre:
            pre()
        fn()
    e.record()
    torch.cuda.synchronize()
    return s.elapsed_time(e) / ITERS


def _worker(local_rank, world_size, init_url, outfile):
    dist.init_process_group(
        backend="nccl",
        init_method=init_url,
        world_size=world_size,
        rank=local_rank,
        device_id=torch.device(f"cuda:{local_rank}"),
    )
    shmem = iris.iris(1 << 33)
    rank = shmem.get_rank()
    cu_count = torch.cuda.get_device_properties(rank).multi_processor_count

    from iris.ops.all_reduce_fast import one_shot_all_reduce
    from iris.ops.matmul_all_reduce_hbm_buffer import (
        matmul_all_reduce_hbm_buffer,
        matmul_all_reduce_hbm_buffer_preamble,
    )

    dtype = torch.float16
    K_local = K_GLOBAL // world_size
    results = []

    if rank == 0:
        print(f"\nHBM-buffer fused GEMM + two-shot AR   TP={world_size} CUs={cu_count}")
        print(f"  traffic: one-shot={world_size:.2f}*MN   two-shot="
              f"{2*(world_size-1)/world_size:.2f}*MN  "
              f"({world_size/(2*(world_size-1)/world_size):.1f}x less)\n")

    for M in [512, 2048, 128, 32]:
        A = shmem.zeros((M, K_local), device="cuda", dtype=dtype)
        A.copy_(torch.randn(M, K_local, dtype=dtype, device=f"cuda:{rank}") * 0.1)
        B = torch.randn(K_local, N_GLOBAL, dtype=dtype, device=f"cuda:{rank}") * 0.1

        ref = torch.mm(A, B)
        dist.all_reduce(ref, op=dist.ReduceOp.SUM)
        torch.cuda.synchronize()
        tol = 2.0

        Ct = torch.empty(M, N_GLOBAL, dtype=dtype, device=f"cuda:{rank}")
        base_ms = bench(lambda: (torch.mm(A, B, out=Ct),
                                 dist.all_reduce(Ct, op=dist.ReduceOp.SUM)))

        Cs = shmem.zeros((M, N_GLOBAL), device="cuda", dtype=dtype)
        Co = torch.zeros(M, N_GLOBAL, dtype=dtype, device=f"cuda:{rank}")
        shmem.barrier()
        twok_ms = bench(lambda: (torch.mm(A, B, out=Cs),
                                 one_shot_all_reduce(shmem, Co, Cs)))

        if rank == 0:
            print(f"--- M={M} ---")
            print(f"  torch bulk-sync            {base_ms:.4f}ms  1.00x")
            print(f"  two-kernel one-shot        {twok_ms:.4f}ms  {base_ms/twok_ms:.2f}x")

        best = (1e9, None)
        for block_m in [32, 64, 128]:
            num_m_tiles = triton.cdiv(M, block_m)
            # RS phase shards M-tiles across ranks -- must divide evenly
            if num_m_tiles % world_size != 0:
                continue
            for block_n in [64, 128]:
                for mfma in [16, 32]:
                    for g, r_, a_ in [
                        (128, 64, 64),
                        (160, 48, 48),
                        (192, 32, 32),
                        (96, 80, 80),
                        (64, 96, 96),
                    ]:
                        if g + r_ + a_ > cu_count:
                            continue
                        try:
                            out = torch.zeros(M, N_GLOBAL, dtype=dtype,
                                              device=f"cuda:{rank}")
                            ws = matmul_all_reduce_hbm_buffer_preamble(
                                shmem, M, N_GLOBAL, dtype, block_m, block_n)
                            shmem.barrier()

                            kw = dict(block_m=block_m, block_n=block_n, block_k=64,
                                      num_gemm_sms=g, num_rs_sms=r_, num_ag_sms=a_,
                                      mfma=mfma)
                            # repeated iterations validate the monotonic counters
                            ok = True
                            for _ in range(3):
                                out.zero_()
                                matmul_all_reduce_hbm_buffer(
                                    shmem, out, A, B, workspace=ws, **kw)
                                torch.cuda.synchronize()
                                if torch.abs(out - ref).max().item() > tol:
                                    ok = False
                                    break
                            shmem.barrier()
                            if not ok:
                                continue

                            ms = bench(lambda ws=ws, out=out, kw=kw:
                                       matmul_all_reduce_hbm_buffer(
                                           shmem, out, A, B, workspace=ws, **kw))
                            cfg = (f"bm={block_m} bn={block_n} mfma={mfma} "
                                   f"G/R/A={g}/{r_}/{a_}")
                            results.append(dict(M=M, ms=ms, cfg=cfg,
                                                speedup=base_ms / ms))
                            if ms < best[0]:
                                best = (ms, cfg)
                                if rank == 0:
                                    print(f"    {cfg:<44} {ms:.4f}ms "
                                          f"{base_ms/ms:.2f}x  ***", flush=True)
                        except Exception:
                            continue

        if rank == 0:
            if best[1]:
                print(f"  HBM-buffer two-shot fused  {best[0]:.4f}ms  "
                      f"{base_ms/best[0]:.2f}x  ({best[1]})")
            else:
                print(f"  HBM-buffer two-shot fused  no valid config")
            print(flush=True)

        del A, B, Cs, Co
        torch.cuda.empty_cache()
        shmem.barrier()

    if rank == 0 and outfile:
        with open(outfile, "w") as f:
            json.dump(results, f, indent=2)

    shmem.barrier()
    dist.destroy_process_group()


def main():
    p = argparse.ArgumentParser()
    p.add_argument("-r", "--num_ranks", type=int, default=8)
    p.add_argument("-o", "--output", type=str, default="ar_hbm_buffer.json")
    a = p.parse_args()
    mp.spawn(fn=_worker, args=(a.num_ranks, "tcp://127.0.0.1:29513", a.output),
             nprocs=a.num_ranks, join=True)


if __name__ == "__main__":
    main()
