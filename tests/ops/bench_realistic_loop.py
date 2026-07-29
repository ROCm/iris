#!/usr/bin/env python3
"""Realistic inference loop: no barriers between layers.

The 0.092ms RS number was measured with barriers keeping ranks in
lockstep. Real inference has no barriers — ranks drift naturally.

Yael's finding: without barriers, 1000 back-to-back RS calls take
0.147ms each (vs 0.092ms with barriers) because ranks drift and
iris.load hits lagging peers.

This tests both iris and RCCL under realistic drift to see which
degrades more. RCCL's ring has built-in sync (each step waits for
the neighbor), so it may be more drift-tolerant.
"""

import os
import torch
import torch.distributed as dist
import triton
import triton.language as tl
import iris

torch.cuda.set_device(int(os.environ.get("LOCAL_RANK", 0)))
dist.init_process_group(backend="nccl")

rank = dist.get_rank()
world_size = dist.get_world_size()
shmem = iris.iris(2**33)

from iris.ops.reduce_scatter_auto import _one_shot_rs_kernel, _get_config

M, N, K_global = 2048, 2880, 4096
K_local = K_global // world_size
M_local = M // world_size
dtype = torch.float16

A = shmem.zeros((M, K_local), dtype=dtype)
A.copy_(torch.randn(M, K_local, dtype=dtype, device=f"cuda:{rank}"))
B = torch.randn(K_local, N, dtype=dtype, device=f"cuda:{rank}")
C_sym = shmem.zeros((M, N), dtype=dtype)
C_out = torch.zeros(M_local, N, dtype=dtype, device=f"cuda:{rank}")
C_r = torch.empty(M, N, dtype=dtype, device=f"cuda:{rank}")
C_ro = torch.empty(M_local, N, dtype=dtype, device=f"cuda:{rank}")

hb = shmem.get_heap_bases()
cfg = _get_config(world_size, M_local)
s, e = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)

def iris_rs():
    _one_shot_rs_kernel[(cfg["num_sms"],)](
        C_sym, C_out, M, N, M_local,
        C_sym.stride(0), C_sym.stride(1), C_out.stride(0), C_out.stride(1),
        hb, rank, world_size,
        cfg["block_m"], cfg["block_n"], cfg["num_sms"],
        num_warps=cfg["num_warps"])

if rank == 0:
    print(f"Realistic loop test: M={M}, N={N}, K={K_global}, TP={world_size}")
    print("Comparing barriered (lockstep) vs unbarriered (drifting) execution")
    print()

for label, n_iters, use_barrier in [
    ("barriered  (lockstep)", 200, True),
    ("unbarriered (drift)  ", 200, False),
    ("unbarriered (drift)  ", 1000, False),
]:
    # --- iris path ---
    shmem.barrier()
    for _ in range(50):
        torch.mm(A, B, out=C_sym)
        if use_barrier:
            shmem.barrier()
        iris_rs()
    torch.cuda.synchronize()
    shmem.barrier()

    s.record()
    for _ in range(n_iters):
        torch.mm(A, B, out=C_sym)
        if use_barrier:
            shmem.barrier()
        iris_rs()
    e.record()
    torch.cuda.synchronize()
    iris_ms = s.elapsed_time(e) / n_iters

    # --- RCCL path ---
    for _ in range(50):
        torch.mm(A, B, out=C_r)
        dist.reduce_scatter_tensor(C_ro, C_r, op=dist.ReduceOp.SUM)
    torch.cuda.synchronize()

    s.record()
    for _ in range(n_iters):
        torch.mm(A, B, out=C_r)
        dist.reduce_scatter_tensor(C_ro, C_r, op=dist.ReduceOp.SUM)
    e.record()
    torch.cuda.synchronize()
    rccl_ms = s.elapsed_time(e) / n_iters

    if rank == 0:
        print(f"{label} n={n_iters:4d}: iris {iris_ms:.4f}ms  RCCL {rccl_ms:.4f}ms  "
              f"-> {rccl_ms/iris_ms:.2f}x")

# Correctness under drift
ref = torch.empty(M_local, N, dtype=dtype, device=f"cuda:{rank}")
torch.mm(A, B, out=C_r)
dist.reduce_scatter_tensor(ref, C_r, op=dist.ReduceOp.SUM)
torch.cuda.synchronize()

shmem.barrier()
torch.mm(A, B, out=C_sym)
shmem.barrier()
iris_rs()
torch.cuda.synchronize()
d = torch.abs(C_out - ref).max().item()
if rank == 0:
    print()
    print(f"Correctness: max_diff={d:.6f} {'PASS' if d < 1.0 else 'FAIL'}")
    print()
    print("If iris degrades more than RCCL under drift, real inference")
    print("performance will be lower than the barriered benchmark suggests.")

shmem.barrier()
dist.destroy_process_group()
