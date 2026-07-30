#!/usr/bin/env python3
"""Full fusion with batched scatter — no CU split."""

import os
import torch
import torch.distributed as dist
import iris

torch.cuda.set_device(int(os.environ.get("LOCAL_RANK", 0)))
dist.init_process_group(backend="nccl")

rank = dist.get_rank()
world_size = dist.get_world_size()
shmem = iris.iris(2**33)

from iris.ops.matmul_reduce_scatter_batched import matmul_reduce_scatter_batched

M, N, K_global = 2048, 2880, 4096
K_local = K_global // world_size
M_local = M // world_size
dtype = torch.float16
warmup, iters = 30, 100

A = shmem.zeros((M, K_local), dtype=dtype)
A.copy_(torch.randn(M, K_local, dtype=dtype, device=f"cuda:{rank}") * 0.1)
B = torch.randn(K_local, N, dtype=dtype, device=f"cuda:{rank}") * 0.1
C_out = shmem.zeros((M_local, N), dtype=dtype)  # symmetric: peers atomic_add here

# Reference
ref = torch.empty(M_local, N, dtype=dtype, device=f"cuda:{rank}")
C_ref = torch.mm(A, B)
dist.reduce_scatter_tensor(ref, C_ref, op=dist.ReduceOp.SUM)
torch.cuda.synchronize()

s, e = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)

# RCCL baseline
C_r = torch.empty(M, N, dtype=dtype, device=f"cuda:{rank}")
C_ro = torch.empty(M_local, N, dtype=dtype, device=f"cuda:{rank}")
for _ in range(warmup):
    torch.mm(A, B, out=C_r)
    dist.reduce_scatter_tensor(C_ro, C_r, op=dist.ReduceOp.SUM)
torch.cuda.synchronize()
s.record()
for _ in range(iters):
    torch.mm(A, B, out=C_r)
    dist.reduce_scatter_tensor(C_ro, C_r, op=dist.ReduceOp.SUM)
e.record()
torch.cuda.synchronize()
rccl_ms = s.elapsed_time(e) / iters

if rank == 0:
    print(f"Batched full fusion: M={M}, N={N}, K={K_global}, TP={world_size}")
    print(f"RCCL: {rccl_ms:.4f}ms")
    print(f"Two-kernel best (known): 0.130-0.138ms")
    print()
    print(f"{'bm':>4} {'bn':>4} {'batch':>6} {'sms':>4} {'w':>2} | {'reg KB/tile':>11} {'ms':>9} {'vs RCCL':>8} {'ok':>4}")
    print("-" * 66)

best = (999.0, None)
for bm in [16, 32, 64]:
    if M_local % bm != 0:
        continue
    for bn in [32, 64]:
        reg_kb = bm * bn * 4 / 1024   # fp32 accumulator per tile
        for batch in [1, 2, 4, 8]:
            if reg_kb * batch > 96:   # VGPR budget guard
                continue
            for sms in [128, 256, 304]:
                for warps in [4, 8]:
                    try:
                        C_out.zero_()
                        shmem.barrier()
                        for _ in range(10):
                            matmul_reduce_scatter_batched(
                                shmem, C_out, A, B,
                                block_m=bm, block_n=bn, batch=batch,
                                num_sms=sms, num_warps=warps)
                        torch.cuda.synchronize()

                        d = torch.abs(C_out - ref).max().item()
                        ok = d < 1.0
                        if not ok:
                            continue

                        s.record()
                        for _ in range(iters):
                            matmul_reduce_scatter_batched(
                                shmem, C_out, A, B,
                                block_m=bm, block_n=bn, batch=batch,
                                num_sms=sms, num_warps=warps)
                        e.record()
                        torch.cuda.synchronize()
                        ms = s.elapsed_time(e) / iters

                        if ms < best[0]:
                            best = (ms, (bm, bn, batch, sms, warps))
                            if rank == 0:
                                print(f"{bm:4d} {bn:4d} {batch:6d} {sms:4d} {warps:2d} | "
                                      f"{reg_kb*batch:10.1f} {ms:9.4f} {rccl_ms/ms:7.2f}x "
                                      f"{'PASS':>4}  ***")
                    except Exception:
                        continue

if rank == 0:
    print()
    print(f"RCCL:          {rccl_ms:.4f}ms")
    print(f"Two-kernel:    ~0.134ms  ({rccl_ms/0.134:.2f}x)")
    if best[1]:
        bm, bn, batch, sms, w = best[1]
        print(f"Batched fusion: {best[0]:.4f}ms ({rccl_ms/best[0]:.2f}x)")
        print(f"  bm={bm} bn={bn} batch={batch} sms={sms} warps={w}")
        if best[0] < 0.134:
            print(f"  *** BEATS TWO-KERNEL by {(0.134-best[0])*1000:.1f}us ***")
        else:
            print(f"  loses to two-kernel by {(best[0]-0.134)*1000:.1f}us")
    else:
        print("Batched fusion: all configs failed")

shmem.barrier()
dist.destroy_process_group()
