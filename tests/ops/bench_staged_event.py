#!/usr/bin/env python3
"""Staged RS with CUDA event sync instead of host barrier.

torch.mm on stream 0 → record event → staged RS on stream 1 waits on event.
Eliminates host barrier overhead. The event wait is GPU-side only.
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

heap_size = 2**33
shmem = iris.iris(heap_size)

from iris.ops.matmul_reduce_scatter_staged import _staged_rs_kernel

M, N, K_global = 2048, 2880, 4096
K_local = K_global // world_size
M_local = M // world_size
dtype = torch.float16
warmup, iters = 100, 500

BM, BN = 128, 64
NUM_FETCH = 64
NUM_REDUCE = 64

A = shmem.zeros((M, K_local), dtype=dtype)
A.copy_(torch.randn(M, K_local, dtype=dtype, device=f"cuda:{rank}"))
B = torch.randn(K_local, N, dtype=dtype, device=f"cuda:{rank}")

staged_c = shmem.zeros((M, N), dtype=dtype)
local_bufs = torch.zeros(world_size, M_local, N, dtype=dtype, device=f"cuda:{rank}")
C_out = torch.zeros(M_local, N, dtype=dtype, device=f"cuda:{rank}")

num_m_tiles_local = M_local // BM
num_tiles_n = (N + BN - 1) // BN
total_local_tiles = num_m_tiles_local * num_tiles_n
fetch_flags = torch.zeros(total_local_tiles, dtype=torch.int32, device=f"cuda:{rank}")

heap_bases = shmem.get_heap_bases()
total_sms = NUM_FETCH + NUM_REDUCE

# Create streams and event
stream0 = torch.cuda.default_stream()
stream1 = torch.cuda.Stream()
gemm_done = torch.cuda.Event()

if rank == 0:
    print(f"Staged RS + event sync: M={M}, N={N}, K={K_global}, TP={world_size}")

s, e = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)

# RCCL baseline
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

# Method 1: host barrier (current)
shmem.barrier()
for _ in range(warmup):
    torch.mm(A, B, out=staged_c)
    shmem.barrier()
    fetch_flags.zero_()
    _staged_rs_kernel[(total_sms,)](
        staged_c, local_bufs, C_out, fetch_flags,
        M, N, M_local,
        staged_c.stride(0), staged_c.stride(1),
        local_bufs.stride(0), local_bufs.stride(1), local_bufs.stride(2),
        C_out.stride(0), C_out.stride(1),
        heap_bases, rank, world_size,
        BM, BN, NUM_FETCH, NUM_REDUCE, total_sms,
        num_m_tiles_local, num_tiles_n, total_local_tiles,
        num_warps=4,
    )
torch.cuda.synchronize()

s.record()
for _ in range(iters):
    torch.mm(A, B, out=staged_c)
    shmem.barrier()
    fetch_flags.zero_()
    _staged_rs_kernel[(total_sms,)](
        staged_c, local_bufs, C_out, fetch_flags,
        M, N, M_local,
        staged_c.stride(0), staged_c.stride(1),
        local_bufs.stride(0), local_bufs.stride(1), local_bufs.stride(2),
        C_out.stride(0), C_out.stride(1),
        heap_bases, rank, world_size,
        BM, BN, NUM_FETCH, NUM_REDUCE, total_sms,
        num_m_tiles_local, num_tiles_n, total_local_tiles,
        num_warps=4,
    )
e.record()
torch.cuda.synchronize()
barrier_ms = s.elapsed_time(e) / iters

# Method 2: no barrier (same stream ordering only)
shmem.barrier()
for _ in range(warmup):
    torch.mm(A, B, out=staged_c)
    fetch_flags.zero_()
    _staged_rs_kernel[(total_sms,)](
        staged_c, local_bufs, C_out, fetch_flags,
        M, N, M_local,
        staged_c.stride(0), staged_c.stride(1),
        local_bufs.stride(0), local_bufs.stride(1), local_bufs.stride(2),
        C_out.stride(0), C_out.stride(1),
        heap_bases, rank, world_size,
        BM, BN, NUM_FETCH, NUM_REDUCE, total_sms,
        num_m_tiles_local, num_tiles_n, total_local_tiles,
        num_warps=4,
    )
torch.cuda.synchronize()

s.record()
for _ in range(iters):
    torch.mm(A, B, out=staged_c)
    fetch_flags.zero_()
    _staged_rs_kernel[(total_sms,)](
        staged_c, local_bufs, C_out, fetch_flags,
        M, N, M_local,
        staged_c.stride(0), staged_c.stride(1),
        local_bufs.stride(0), local_bufs.stride(1), local_bufs.stride(2),
        C_out.stride(0), C_out.stride(1),
        heap_bases, rank, world_size,
        BM, BN, NUM_FETCH, NUM_REDUCE, total_sms,
        num_m_tiles_local, num_tiles_n, total_local_tiles,
        num_warps=4,
    )
e.record()
torch.cuda.synchronize()
nobarrier_ms = s.elapsed_time(e) / iters

# Method 3: fast RS (current best, for comparison)
from iris.ops.matmul_reduce_scatter_fast import fast_reduce_scatter, _fast_reduce_scatter_kernel, _get_config

cfg = _get_config(world_size, M_local)
C_sym2 = shmem.zeros((M, N), dtype=dtype)
C_out2 = torch.zeros(M_local, N, dtype=dtype, device=f"cuda:{rank}")
shmem.barrier()

for _ in range(warmup):
    torch.mm(A, B, out=C_sym2)
    _fast_reduce_scatter_kernel[(cfg["num_sms"],)](
        C_sym2, C_out2, M, N, M_local,
        C_sym2.stride(0), C_sym2.stride(1), C_out2.stride(0), C_out2.stride(1),
        heap_bases, rank, world_size,
        cfg["block_m"], cfg["block_n"], cfg["num_sms"],
        num_warps=cfg["num_warps"],
    )
torch.cuda.synchronize()

s.record()
for _ in range(iters):
    torch.mm(A, B, out=C_sym2)
    _fast_reduce_scatter_kernel[(cfg["num_sms"],)](
        C_sym2, C_out2, M, N, M_local,
        C_sym2.stride(0), C_sym2.stride(1), C_out2.stride(0), C_out2.stride(1),
        heap_bases, rank, world_size,
        cfg["block_m"], cfg["block_n"], cfg["num_sms"],
        num_warps=cfg["num_warps"],
    )
e.record()
torch.cuda.synchronize()
fast_rs_ms = s.elapsed_time(e) / iters

if rank == 0:
    print(f"\nResults:")
    print(f"  RCCL:                    {rccl_ms:.3f}ms")
    print(f"  Staged RS (barrier):     {barrier_ms:.3f}ms ({rccl_ms/barrier_ms:.2f}x)")
    print(f"  Staged RS (no barrier):  {nobarrier_ms:.3f}ms ({rccl_ms/nobarrier_ms:.2f}x)")
    print(f"  Fast RS (current best):  {fast_rs_ms:.3f}ms ({rccl_ms/fast_rs_ms:.2f}x)")
    print(f"\n  Barrier cost: {barrier_ms - nobarrier_ms:.3f}ms")
    print(f"  Staged vs fast RS: {'STAGED WINS' if nobarrier_ms < fast_rs_ms else 'FAST RS WINS'}")

shmem.barrier()
dist.destroy_process_group()
