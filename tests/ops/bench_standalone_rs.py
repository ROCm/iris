#!/usr/bin/env python3
"""Benchmark standalone RS: iris vs RCCL at GPT-OSS-120B shapes."""

import os
import time
import torch
import torch.distributed as dist

torch.cuda.set_device(int(os.environ.get("LOCAL_RANK", 0)))
dist.init_process_group(backend="nccl")

import iris

rank = dist.get_rank()
world_size = dist.get_world_size()

heap_size = 2**33
shmem = iris.iris(heap_size)

M, N = 2048, 2880
M_local = M // world_size
dtype = torch.float16
warmup, iters = 50, 200

if rank == 0:
    print(f"Standalone RS benchmark: M={M}, N={N}, TP={world_size}, dtype={dtype}")
    print(f"Message size: {M * N * 2 / 1e6:.1f} MB total, {M_local * N * 2 / 1e6:.1f} MB per rank")
    print()

# --- RCCL RS ---
input_rccl = torch.randn(M, N, dtype=dtype, device=f"cuda:{rank}")
output_rccl = torch.empty(M_local, N, dtype=dtype, device=f"cuda:{rank}")

for _ in range(warmup):
    dist.reduce_scatter_tensor(output_rccl, input_rccl, op=dist.ReduceOp.SUM)
torch.cuda.synchronize()

start_event = torch.cuda.Event(enable_timing=True)
end_event = torch.cuda.Event(enable_timing=True)

start_event.record()
for _ in range(iters):
    dist.reduce_scatter_tensor(output_rccl, input_rccl, op=dist.ReduceOp.SUM)
end_event.record()
torch.cuda.synchronize()

rccl_ms = start_event.elapsed_time(end_event) / iters

if rank == 0:
    bw = M * N * 2 * (world_size - 1) / world_size / (rccl_ms / 1000) / 1e9
    print(f"RCCL RS:  {rccl_ms:.3f} ms  ({bw:.1f} GB/s algoBW)")

# --- iris RS ---
from iris.ccl.config import Config as CCLConfig

input_iris = shmem.zeros((M, N), dtype=dtype)
input_iris.copy_(input_rccl)
output_iris = shmem.zeros((M, N), dtype=dtype)

ccl_config = CCLConfig()
shmem.barrier()

for _ in range(warmup):
    shmem.ccl.reduce_scatter(output_iris, input_iris, config=ccl_config)
torch.cuda.synchronize()

start_event = torch.cuda.Event(enable_timing=True)
end_event = torch.cuda.Event(enable_timing=True)

start_event.record()
for _ in range(iters):
    shmem.ccl.reduce_scatter(output_iris, input_iris, config=ccl_config)
end_event.record()
torch.cuda.synchronize()

iris_ms = start_event.elapsed_time(end_event) / iters

if rank == 0:
    bw = M * N * 2 * (world_size - 1) / world_size / (iris_ms / 1000) / 1e9
    print(f"iris RS:  {iris_ms:.3f} ms  ({bw:.1f} GB/s algoBW)")
    print()

    speedup = rccl_ms / iris_ms
    print(f"iris vs RCCL: {speedup:.2f}x {'(faster)' if speedup > 1 else '(slower)'}")

    # Combined projection
    gemm_ms = 0.037  # hipBLASLt from breakdown
    combined_iris = gemm_ms + iris_ms
    combined_rccl = gemm_ms + rccl_ms
    print(f"Projected torch.mm + iris RS: {combined_iris:.3f}ms")
    print(f"Projected torch.mm + RCCL RS: {combined_rccl:.3f}ms")
    print(f"Projected speedup: {combined_rccl / combined_iris:.2f}x")

# Correctness check
ref = torch.empty(M_local, N, dtype=dtype, device=f"cuda:{rank}")
dist.reduce_scatter_tensor(ref, input_rccl, op=dist.ReduceOp.SUM)
torch.cuda.synchronize()

m_start = rank * M_local
m_end = m_start + M_local
max_diff = torch.abs(output_iris[m_start:m_end] - ref).max().item()
if rank == 0:
    print(f"\nCorrectness: max_diff = {max_diff:.6f} {'PASS' if max_diff < 1.0 else 'FAIL'}")

shmem.barrier()
dist.destroy_process_group()
