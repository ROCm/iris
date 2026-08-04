#!/usr/bin/env python3
"""Fast AllReduce vs RCCL — correctness + component breakdown + sweep.

Mirrors the methodology that cracked GEMM+RS open:
  1. component breakdown (GEMM alone / RCCL AR alone / iris.ccl AR alone / fast AR)
  2. correctness before any timing
  3. no barriers in the timed loop (barriers cost 0.07ms on RS)
  4. sweep tile/SMS/warps per variant
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

from iris.ops.all_reduce_fast import (
    one_shot_all_reduce, two_shot_all_reduce, _get_config,
)

M, N, K_global = 2048, 2880, 4096
K_local = K_global // world_size
dtype = torch.float16
warmup, iters = 50, 200

A = shmem.zeros((M, K_local), dtype=dtype)
A.copy_(torch.randn(M, K_local, dtype=dtype, device=f"cuda:{rank}") * 0.1)
B = torch.randn(K_local, N, dtype=dtype, device=f"cuda:{rank}") * 0.1

C_sym = shmem.zeros((M, N), dtype=dtype)
C_out = torch.zeros(M, N, dtype=dtype, device=f"cuda:{rank}")
s, e = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)

def bench(fn):
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    s.record()
    for _ in range(iters):
        fn()
    e.record()
    torch.cuda.synchronize()
    return s.elapsed_time(e) / iters

# ---- reference ----
ref = torch.mm(A, B)
dist.all_reduce(ref, op=dist.ReduceOp.SUM)
torch.cuda.synchronize()

if rank == 0:
    bytes_moved = M * N * 2
    print(f"Fast AllReduce: M={M} N={N} K={K_global} TP={world_size} fp16")
    print(f"tensor = {bytes_moved/1e6:.1f} MB")
    print(f"  one_shot traffic = ws*{bytes_moved/1e6:.1f} = {world_size*bytes_moved/1e6:.1f} MB/rank")
    print(f"  two_shot traffic = 2(ws-1)/ws*{bytes_moved/1e6:.1f} = "
          f"{2*(world_size-1)/world_size*bytes_moved/1e6:.1f} MB/rank")
    print()
    print("=== COMPONENT BREAKDOWN ===")

# GEMM alone
gemm_ms = bench(lambda: torch.mm(A, B, out=C_sym))
if rank == 0:
    print(f"  hipBLASLt GEMM:        {gemm_ms:.4f}ms")

# RCCL AR alone
Cr = torch.mm(A, B)
rccl_ar_ms = bench(lambda: dist.all_reduce(Cr, op=dist.ReduceOp.SUM))
if rank == 0:
    print(f"  RCCL all_reduce:       {rccl_ar_ms:.4f}ms")

# iris.ccl AR alone (expect Python overhead like RS had)
try:
    from iris.ccl.config import Config as CCLConfig
    cc = CCLConfig()
    inp = shmem.zeros((M, N), dtype=dtype)
    inp.copy_(Cr)
    outp = shmem.zeros((M, N), dtype=dtype)
    shmem.barrier()
    ccl_ms = bench(lambda: shmem.ccl.all_reduce(outp, inp, config=cc))
    if rank == 0:
        print(f"  iris.ccl.all_reduce:   {ccl_ms:.4f}ms  ({ccl_ms/rccl_ar_ms:.2f}x vs RCCL)")
except Exception as ex:
    if rank == 0:
        print(f"  iris.ccl.all_reduce:   N/A ({str(ex)[:50]})")

# ---- correctness ----
if rank == 0:
    print()
    print("=== CORRECTNESS ===")

torch.mm(A, B, out=C_sym)
shmem.barrier()
one_shot_all_reduce(shmem, C_out, C_sym)
torch.cuda.synchronize()
d1 = torch.abs(C_out - ref).max().item()
if rank == 0:
    print(f"  one_shot: max_diff={d1:.6f} {'PASS' if d1 < 1.0 else 'FAIL'}")

scratch = None
C_out2 = torch.zeros(M, N, dtype=dtype, device=f"cuda:{rank}")
torch.mm(A, B, out=C_sym)
shmem.barrier()
scratch = two_shot_all_reduce(shmem, C_out2, C_sym, scratch=scratch)
torch.cuda.synchronize()
d2 = torch.abs(C_out2 - ref).max().item()
if rank == 0:
    print(f"  two_shot: max_diff={d2:.6f} {'PASS' if d2 < 1.0 else 'FAIL'}")

if d1 > 1.0 and d2 > 1.0:
    shmem.barrier()
    dist.destroy_process_group()
    exit(1)

# ---- E2E ----
if rank == 0:
    print()
    print("=== END TO END (GEMM + AllReduce) ===")

Cr2 = torch.empty(M, N, dtype=dtype, device=f"cuda:{rank}")
rccl_e2e = bench(lambda: (torch.mm(A, B, out=Cr2),
                          dist.all_reduce(Cr2, op=dist.ReduceOp.SUM)))
if rank == 0:
    print(f"  torch.mm + RCCL AR:    {rccl_e2e:.4f}ms")

shmem.barrier()
if d1 < 1.0:
    os_e2e = bench(lambda: (torch.mm(A, B, out=C_sym),
                            one_shot_all_reduce(shmem, C_out, C_sym)))
    if rank == 0:
        print(f"  torch.mm + one_shot:   {os_e2e:.4f}ms  ({rccl_e2e/os_e2e:.2f}x)")

if d2 < 1.0:
    ts_e2e = bench(lambda: (torch.mm(A, B, out=C_sym),
                            two_shot_all_reduce(shmem, C_out2, C_sym, scratch=scratch)))
    if rank == 0:
        print(f"  torch.mm + two_shot:   {ts_e2e:.4f}ms  ({rccl_e2e/ts_e2e:.2f}x)")

# ---- sweep the standalone AR kernel ----
if rank == 0:
    print()
    print("=== ONE-SHOT AR SWEEP (standalone) ===")
    print(f"{'bm':>4} {'bn':>4} {'sms':>4} {'w':>2} | {'ms':>9} {'vs RCCL':>8}")
    print("-" * 40)

best = (999.0, None)
for bm in [32, 64, 128, 256]:
    if M % bm:
        continue
    for bn in [64, 128]:
        for sms in [32, 64, 128, 196, 304]:
            for w in [2, 4, 8]:
                try:
                    C_out.zero_()
                    for _ in range(10):
                        one_shot_all_reduce(shmem, C_out, C_sym,
                                            block_m=bm, block_n=bn,
                                            num_sms=sms, num_warps=w)
                    torch.cuda.synchronize()
                    if torch.abs(C_out - ref).max().item() > 1.0:
                        continue
                    ms = bench(lambda: one_shot_all_reduce(
                        shmem, C_out, C_sym, block_m=bm, block_n=bn,
                        num_sms=sms, num_warps=w))
                    if ms < best[0]:
                        best = (ms, (bm, bn, sms, w))
                        if rank == 0:
                            print(f"{bm:4d} {bn:4d} {sms:4d} {w:2d} | "
                                  f"{ms:9.4f} {rccl_ar_ms/ms:7.2f}x  ***")
                except Exception:
                    continue

if rank == 0:
    print()
    print(f"RCCL AR:      {rccl_ar_ms:.4f}ms")
    if best[1]:
        bm, bn, sms, w = best[1]
        print(f"Fast one-shot: {best[0]:.4f}ms ({rccl_ar_ms/best[0]:.2f}x)")
        print(f"  bm={bm} bn={bn} sms={sms} warps={w}")

shmem.barrier()
dist.destroy_process_group()
