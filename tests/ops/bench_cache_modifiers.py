#!/usr/bin/env python3
"""Test cache modifiers for iris.load in RS kernel.

.ca = cache all levels (default) — wastes L2 on remote data
.cg = bypass L1, stream through L2
.cv = bypass all caches, direct from system memory — best for coherence

Also tests vectorization hints.
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


@triton.jit
def rs_ca(input_ptr, output_ptr, M, N, M_local,
          stride_in_m, stride_in_n, stride_out_m, stride_out_n,
          heap_bases: tl.tensor,
          cur_rank: tl.constexpr, world_size: tl.constexpr,
          BM: tl.constexpr, BN: tl.constexpr, NUM_SMS: tl.constexpr):
    pid = tl.program_id(0)
    nm = M_local // BM
    nn = tl.cdiv(N, BN)
    mo = cur_rank * nm
    for t in range(pid, nm * nn, NUM_SMS):
        pm = mo + t // nn
        pn = t % nn
        rm = pm * BM + tl.arange(0, BM)
        rm = tl.max_contiguous(tl.multiple_of(rm, BM), BM)
        rn = pn * BN + tl.arange(0, BN)
        rn = tl.max_contiguous(tl.multiple_of(rn % N, BN), BN)
        off = rm[:, None] * stride_in_m + rn[None, :] * stride_in_n
        bp = input_ptr + off
        sr = pid % world_size
        acc = iris.load(bp, cur_rank, sr, heap_bases, hint=(1, BN), cache_modifier=".ca").to(tl.float32)
        for i in tl.static_range(1, world_size):
            r = (sr + i) % world_size
            acc += iris.load(bp, cur_rank, r, heap_bases, hint=(1, BN), cache_modifier=".ca").to(tl.float32)
        orm = (t // nn) * BM + tl.arange(0, BM)
        orm = tl.max_contiguous(tl.multiple_of(orm, BM), BM)
        tl.store(output_ptr + orm[:, None] * stride_out_m + rn[None, :] * stride_out_n,
                 acc.to(output_ptr.type.element_ty))


@triton.jit
def rs_cg(input_ptr, output_ptr, M, N, M_local,
          stride_in_m, stride_in_n, stride_out_m, stride_out_n,
          heap_bases: tl.tensor,
          cur_rank: tl.constexpr, world_size: tl.constexpr,
          BM: tl.constexpr, BN: tl.constexpr, NUM_SMS: tl.constexpr):
    pid = tl.program_id(0)
    nm = M_local // BM
    nn = tl.cdiv(N, BN)
    mo = cur_rank * nm
    for t in range(pid, nm * nn, NUM_SMS):
        pm = mo + t // nn
        pn = t % nn
        rm = pm * BM + tl.arange(0, BM)
        rm = tl.max_contiguous(tl.multiple_of(rm, BM), BM)
        rn = pn * BN + tl.arange(0, BN)
        rn = tl.max_contiguous(tl.multiple_of(rn % N, BN), BN)
        off = rm[:, None] * stride_in_m + rn[None, :] * stride_in_n
        bp = input_ptr + off
        sr = pid % world_size
        acc = iris.load(bp, cur_rank, sr, heap_bases, hint=(1, BN), cache_modifier=".cg").to(tl.float32)
        for i in tl.static_range(1, world_size):
            r = (sr + i) % world_size
            acc += iris.load(bp, cur_rank, r, heap_bases, hint=(1, BN), cache_modifier=".cg").to(tl.float32)
        orm = (t // nn) * BM + tl.arange(0, BM)
        orm = tl.max_contiguous(tl.multiple_of(orm, BM), BM)
        tl.store(output_ptr + orm[:, None] * stride_out_m + rn[None, :] * stride_out_n,
                 acc.to(output_ptr.type.element_ty))


@triton.jit
def rs_cv(input_ptr, output_ptr, M, N, M_local,
          stride_in_m, stride_in_n, stride_out_m, stride_out_n,
          heap_bases: tl.tensor,
          cur_rank: tl.constexpr, world_size: tl.constexpr,
          BM: tl.constexpr, BN: tl.constexpr, NUM_SMS: tl.constexpr):
    pid = tl.program_id(0)
    nm = M_local // BM
    nn = tl.cdiv(N, BN)
    mo = cur_rank * nm
    for t in range(pid, nm * nn, NUM_SMS):
        pm = mo + t // nn
        pn = t % nn
        rm = pm * BM + tl.arange(0, BM)
        rm = tl.max_contiguous(tl.multiple_of(rm, BM), BM)
        rn = pn * BN + tl.arange(0, BN)
        rn = tl.max_contiguous(tl.multiple_of(rn % N, BN), BN)
        off = rm[:, None] * stride_in_m + rn[None, :] * stride_in_n
        bp = input_ptr + off
        sr = pid % world_size
        acc = iris.load(bp, cur_rank, sr, heap_bases, hint=(1, BN), cache_modifier=".cv").to(tl.float32)
        for i in tl.static_range(1, world_size):
            r = (sr + i) % world_size
            acc += iris.load(bp, cur_rank, r, heap_bases, hint=(1, BN), cache_modifier=".cv").to(tl.float32)
        orm = (t // nn) * BM + tl.arange(0, BM)
        orm = tl.max_contiguous(tl.multiple_of(orm, BM), BM)
        tl.store(output_ptr + orm[:, None] * stride_out_m + rn[None, :] * stride_out_n,
                 acc.to(output_ptr.type.element_ty))


M, N = 2048, 2880
M_local = M // world_size
dtype = torch.float16
warmup, iters = 100, 500
BM, BN, SMS = 128, 64, 128

input_tensor = shmem.zeros((M, N), dtype=dtype)
input_tensor.copy_(torch.randn(M, N, dtype=dtype, device=f"cuda:{rank}"))
heap_bases = shmem.get_heap_bases()
shmem.barrier()

s, e = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)

# RCCL
input_rccl = input_tensor.clone()
output_rccl = torch.empty(M_local, N, dtype=dtype, device=f"cuda:{rank}")
for _ in range(warmup):
    dist.reduce_scatter_tensor(output_rccl, input_rccl, op=dist.ReduceOp.SUM)
torch.cuda.synchronize()
s.record()
for _ in range(iters):
    dist.reduce_scatter_tensor(output_rccl, input_rccl, op=dist.ReduceOp.SUM)
e.record()
torch.cuda.synchronize()
rccl_ms = s.elapsed_time(e) / iters

if rank == 0:
    print(f"Cache modifier test: M={M}, N={N}, TP={world_size}, bm={BM}, bn={BN}, sms={SMS}")
    print(f"RCCL: {rccl_ms:.3f}ms")

for name, kernel in [(".ca", rs_ca), (".cg", rs_cg), (".cv", rs_cv)]:
    out = torch.zeros(M_local, N, dtype=dtype, device=f"cuda:{rank}")
    shmem.barrier()

    for _ in range(warmup):
        kernel[(SMS,)](
            input_tensor, out, M, N, M_local,
            input_tensor.stride(0), input_tensor.stride(1),
            out.stride(0), out.stride(1),
            heap_bases, rank, world_size, BM, BN, SMS,
        )
    torch.cuda.synchronize()

    # Correctness
    ref = torch.empty(M_local, N, dtype=dtype, device=f"cuda:{rank}")
    dist.reduce_scatter_tensor(ref, input_tensor.clone(), op=dist.ReduceOp.SUM)
    torch.cuda.synchronize()
    diff = torch.abs(out - ref).max().item()

    s.record()
    for _ in range(iters):
        kernel[(SMS,)](
            input_tensor, out, M, N, M_local,
            input_tensor.stride(0), input_tensor.stride(1),
            out.stride(0), out.stride(1),
            heap_bases, rank, world_size, BM, BN, SMS,
        )
    e.record()
    torch.cuda.synchronize()
    ms = s.elapsed_time(e) / iters
    bw = M * N * 2 * (world_size - 1) / world_size / (ms / 1000) / 1e9

    if rank == 0:
        print(f"  {name}: {ms:.3f}ms ({bw:.1f} GB/s) diff={diff:.4f} {'PASS' if diff < 1.0 else 'FAIL'}")

shmem.barrier()
dist.destroy_process_group()
