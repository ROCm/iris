#!/usr/bin/env python3
"""Test + benchmark push-based fused GEMM+RS."""

import os
import torch
import torch.distributed as dist
import iris

torch.cuda.set_device(int(os.environ.get("LOCAL_RANK", 0)))
dist.init_process_group(backend="nccl")

rank = dist.get_rank()
world_size = dist.get_world_size()

heap_size = 2**33
shmem = iris.iris(heap_size)

from iris.ops.matmul_reduce_scatter_push import matmul_reduce_scatter_push

M, N, K_global = 2048, 2880, 4096
K_local = K_global // world_size
M_local = M // world_size
dtype = torch.float16

A = shmem.zeros((M, K_local), dtype=dtype)
A.copy_(torch.randn(M, K_local, dtype=dtype, device=f"cuda:{rank}"))
B = torch.randn(K_local, N, dtype=dtype, device=f"cuda:{rank}")
C_out = torch.zeros(M_local, N, dtype=dtype, device=f"cuda:{rank}")

if rank == 0:
    print(f"Push GEMM+RS: M={M}, N={N}, K={K_global}, TP={world_size}")

# Reference
ref = torch.empty(M_local, N, dtype=dtype, device=f"cuda:{rank}")
C_ref = torch.empty(M, N, dtype=dtype, device=f"cuda:{rank}")
torch.mm(A, B, out=C_ref)
dist.reduce_scatter_tensor(ref, C_ref, op=dist.ReduceOp.SUM)
torch.cuda.synchronize()

# Test
shmem.barrier()
matmul_reduce_scatter_push(shmem, C_out, A, B)
torch.cuda.synchronize()

max_diff = torch.abs(C_out - ref).max().item()
if rank == 0:
    print(f"Correctness: max_diff={max_diff:.6f} {'PASS' if max_diff < 1.0 else 'FAIL'}")
    if max_diff > 1.0:
        print(f"  C_out[0:4,0:4] = {C_out[0:4,0:4]}")
        print(f"  ref[0:4,0:4] = {ref[0:4,0:4]}")
        # Check which rows are wrong
        row_diffs = torch.abs(C_out - ref).max(dim=1).values
        bad_rows = (row_diffs > 1.0).nonzero().squeeze()
        print(f"  bad rows: {bad_rows[:10].tolist()}")

if max_diff > 1.0:
    shmem.barrier()
    dist.destroy_process_group()
    exit(1)

# Benchmark
warmup, iters = 50, 200
s, e = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)

C_rccl = torch.empty(M, N, dtype=dtype, device=f"cuda:{rank}")
C_rccl_out = torch.empty(M_local, N, dtype=dtype, device=f"cuda:{rank}")
for _ in range(warmup):
    torch.mm(A, B, out=C_rccl)
    dist.reduce_scatter_tensor(C_rccl_out, C_rccl, op=dist.ReduceOp.SUM)
torch.cuda.synchronize()
s.record()
for _ in range(iters):
    torch.mm(A, B, out=C_rccl)
    dist.reduce_scatter_tensor(C_rccl_out, C_rccl, op=dist.ReduceOp.SUM)
e.record()
torch.cuda.synchronize()
rccl_ms = s.elapsed_time(e) / iters

shmem.barrier()
for _ in range(warmup):
    C_out.zero_()
    matmul_reduce_scatter_push(shmem, C_out, A, B)
torch.cuda.synchronize()
s.record()
for _ in range(iters):
    C_out.zero_()
    matmul_reduce_scatter_push(shmem, C_out, A, B)
e.record()
torch.cuda.synchronize()
push_ms = s.elapsed_time(e) / iters

if rank == 0:
    print(f"\nRCCL:  {rccl_ms:.3f}ms")
    print(f"Push:  {push_ms:.3f}ms ({rccl_ms/push_ms:.2f}x)")

shmem.barrier()
dist.destroy_process_group()
