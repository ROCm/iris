#!/usr/bin/env python3
"""GEMM+AllReduce: FUSED vs torch (the comparison actually asked for).

Three contenders, same shapes:
  1. torch.mm + dist.all_reduce          <- the BSP baseline
  2. iris.ops.matmul_all_reduce (FUSED)  <- single fused kernel, all variants
  3. torch.mm + our one-shot AR          <- two-kernel, for reference

Sweeps the fused variants (atomic / two_shot / one_shot) x tile config,
across decode / hybrid / prefill M.
"""

import os
import torch
import torch.distributed as dist
import iris

torch.cuda.set_device(int(os.environ.get("LOCAL_RANK", 0)))
dist.init_process_group(backend="nccl")

rank = dist.get_rank()
world_size = dist.get_world_size()
shmem = iris.iris(2**33)

from iris.ops import FusedConfig, matmul_all_reduce, matmul_all_reduce_preamble
from iris.ops.all_reduce_fast import one_shot_all_reduce

N, K_global = 2880, 4096
K_local = K_global // world_size
dtype = torch.float16
warmup, iters = 30, 100

s, e = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)

def bench(fn, pre=None):
    for _ in range(warmup):
        if pre: pre()
        fn()
    torch.cuda.synchronize()
    s.record()
    for _ in range(iters):
        if pre: pre()
        fn()
    e.record()
    torch.cuda.synchronize()
    return s.elapsed_time(e) / iters

if rank == 0:
    print(f"GEMM+AllReduce: FUSED vs torch   N={N} K={K_global} TP={world_size} fp16")
    print()

for M in [32, 128, 512, 2048]:
    A = shmem.zeros((M, K_local), dtype=dtype)
    A.copy_(torch.randn(M, K_local, dtype=dtype, device=f"cuda:{rank}") * 0.1)
    B = torch.randn(K_local, N, dtype=dtype, device=f"cuda:{rank}") * 0.1

    # reference
    ref = torch.mm(A, B)
    dist.all_reduce(ref, op=dist.ReduceOp.SUM)
    torch.cuda.synchronize()

    # ---- 1. torch baseline ----
    Ct = torch.empty(M, N, dtype=dtype, device=f"cuda:{rank}")
    torch_ms = bench(lambda: (torch.mm(A, B, out=Ct),
                              dist.all_reduce(Ct, op=dist.ReduceOp.SUM)))

    if rank == 0:
        print(f"M={M}")
        print(f"  torch.mm + dist.all_reduce   {torch_ms:.4f}ms   1.00x  (baseline)")

    # ---- 2. FUSED iris.ops.matmul_all_reduce ----
    best_fused = (999.0, None)
    for variant in ["one_shot", "two_shot", "atomic"]:
        for bm in [32, 64, 128]:
            if bm > M:
                continue
            for bn in [64, 128]:
                try:
                    cfg = FusedConfig(block_size_m=bm, block_size_n=bn,
                                      block_size_k=64, all_reduce_variant=variant)
                    Cf = shmem.zeros((M, N), dtype=dtype)
                    ws = matmul_all_reduce_preamble(shmem, Cf, A, B, config=cfg)
                    shmem.barrier()

                    Cf.zero_()
                    matmul_all_reduce(shmem, Cf, A, B, config=cfg, workspace=ws)
                    torch.cuda.synchronize()
                    d = torch.abs(Cf - ref).max().item()
                    if d > 1.0:
                        continue

                    ms = bench(lambda: matmul_all_reduce(shmem, Cf, A, B,
                                                         config=cfg, workspace=ws),
                               pre=lambda: Cf.zero_())
                    if ms < best_fused[0]:
                        best_fused = (ms, (variant, bm, bn))
                except Exception:
                    continue

    if rank == 0:
        if best_fused[1]:
            v, bm, bn = best_fused[1]
            f = best_fused[0]
            print(f"  FUSED matmul_all_reduce      {f:.4f}ms   "
                  f"{torch_ms/f:.2f}x  ({v}, bm={bm} bn={bn})")
        else:
            print(f"  FUSED matmul_all_reduce      all configs failed")

    # ---- 3. two-kernel (torch.mm + our one-shot) ----
    Cs = shmem.zeros((M, N), dtype=dtype)
    Co = torch.zeros(M, N, dtype=dtype, device=f"cuda:{rank}")
    shmem.barrier()
    torch.mm(A, B, out=Cs)
    one_shot_all_reduce(shmem, Co, Cs)
    torch.cuda.synchronize()
    d2 = torch.abs(Co - ref).max().item()

    if d2 < 1.0:
        twok_ms = bench(lambda: (torch.mm(A, B, out=Cs),
                                 one_shot_all_reduce(shmem, Co, Cs)))
        if rank == 0:
            print(f"  torch.mm + our one-shot AR   {twok_ms:.4f}ms   "
                  f"{torch_ms/twok_ms:.2f}x")
    elif rank == 0:
        print(f"  torch.mm + our one-shot AR   FAIL d={d2:.3f}")

    if rank == 0:
        print()

    del A, B, Cs, Co
    torch.cuda.empty_cache()
    shmem.barrier()

shmem.barrier()
dist.destroy_process_group()
