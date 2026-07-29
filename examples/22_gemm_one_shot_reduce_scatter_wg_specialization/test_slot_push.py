#!/usr/bin/env python3
"""Test + benchmark for slot-based push GEMM+RS."""

import os
import sys
import time
import torch
import torch.distributed as dist

torch.cuda.set_device(int(os.environ.get("LOCAL_RANK", 0)))
dist.init_process_group(backend="nccl")

import iris
from gemm_reduce_scatter_slot_push import persistent_gemm_reduce_scatter_slot_push

rank = dist.get_rank()
world_size = dist.get_world_size()

heap_size = 2**33
shmem = iris.iris(heap_size)

M, N, K_local = 2048, 2880, 4096 // world_size
M_local = M // world_size
dtype = torch.float16

# Inputs
A = torch.randn(M, K_local, dtype=dtype, device=f"cuda:{rank}")
B = torch.randn(K_local, N, dtype=dtype, device=f"cuda:{rank}")

# Reference
ref_partial = torch.mm(A, B)
ref_local = torch.empty((M_local, N), dtype=dtype, device=f"cuda:{rank}")
dist.reduce_scatter_tensor(ref_local, ref_partial, op=dist.ReduceOp.SUM)
torch.cuda.synchronize()

# Allocate symmetric buffers
C_slots = shmem.zeros((world_size, M_local, N), dtype=dtype)
C_out = torch.zeros((M_local, N), dtype=dtype, device=f"cuda:{rank}")

num_m_tiles = M // 128
num_n_tiles = (N + 255) // 256
total_tiles = num_m_tiles * num_n_tiles
num_local_m_tiles = M_local // 128
total_local_tiles = num_local_m_tiles * num_n_tiles

gemm_locks = shmem.zeros((total_tiles,), dtype=torch.int32)
scatter_done = shmem.zeros((world_size * total_local_tiles,), dtype=torch.int32)

heap_bases = shmem.get_heap_bases()
NUM_SMS = 304
GEMM_SMS = 256

shmem.barrier()

# Launch
grid = (NUM_SMS,)
persistent_gemm_reduce_scatter_slot_push[grid](
    A, B, C_slots, C_out,
    gemm_locks, scatter_done,
    M, N, K_local,
    A.stride(0), A.stride(1),
    B.stride(0), B.stride(1),
    C_slots.stride(0), C_slots.stride(1), C_slots.stride(2),
    C_out.stride(0), C_out.stride(1),
    128, 256, 64, 4,
    GEMM_SMS, NUM_SMS, 8, K_local % 64 == 0,
    heap_bases, rank, world_size,
    num_warps=8, num_stages=2,
)
torch.cuda.synchronize()
shmem.barrier()

# Correctness
max_diff = torch.abs(C_out - ref_local).max().item()
if rank == 0:
    print(f"Slot push: max_diff = {max_diff:.6f}")
    if max_diff < 1.0:
        print("PASS")
    else:
        print(f"FAIL (diff={max_diff})")
        print(f"C_out[0:4,0:4] = {C_out[0:4,0:4]}")
        print(f"ref[0:4,0:4] = {ref_local[0:4,0:4]}")

# Benchmark
if max_diff < 1.0:
    warmup = 10
    iters = 100

    for _ in range(warmup):
        C_slots.zero_()
        scatter_done.zero_()
        C_out.zero_()
        shmem.barrier()
        persistent_gemm_reduce_scatter_slot_push[grid](
            A, B, C_slots, C_out,
            gemm_locks, scatter_done,
            M, N, K_local,
            A.stride(0), A.stride(1),
            B.stride(0), B.stride(1),
            C_slots.stride(0), C_slots.stride(1), C_slots.stride(2),
            C_out.stride(0), C_out.stride(1),
            128, 256, 64, 4,
            GEMM_SMS, NUM_SMS, 8, K_local % 64 == 0,
            heap_bases, rank, world_size,
            num_warps=8, num_stages=2,
        )
        torch.cuda.synchronize()

    torch.cuda.synchronize()
    start = time.perf_counter()
    for _ in range(iters):
        C_slots.zero_()
        scatter_done.zero_()
        C_out.zero_()
        shmem.barrier()
        persistent_gemm_reduce_scatter_slot_push[grid](
            A, B, C_slots, C_out,
            gemm_locks, scatter_done,
            M, N, K_local,
            A.stride(0), A.stride(1),
            B.stride(0), B.stride(1),
            C_slots.stride(0), C_slots.stride(1), C_slots.stride(2),
            C_out.stride(0), C_out.stride(1),
            128, 256, 64, 4,
            GEMM_SMS, NUM_SMS, 8, K_local % 64 == 0,
            heap_bases, rank, world_size,
            num_warps=8, num_stages=2,
        )
        torch.cuda.synchronize()
    elapsed = (time.perf_counter() - start) / iters * 1000

    if rank == 0:
        print(f"Slot push latency: {elapsed:.3f}ms (target: <0.196ms)")

shmem.barrier()
dist.destroy_process_group()
