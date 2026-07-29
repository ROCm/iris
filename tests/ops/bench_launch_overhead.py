#!/usr/bin/env python3
"""Attack the Triton launch overhead — last 18% of E2E.

Measured: RS kernel ~0.012ms + GEMM dispatch ~0.013ms = 0.025ms of
the 0.138ms E2E is pure launch overhead.

Tests several ways to reduce it:
1. Baseline: normal kernel[grid](args)
2. Pre-warmed with identical args (specialization cache hit)
3. CompiledKernel direct invocation (bypass JIT wrapper)
4. Fewer kernel args (pack strides into constexprs)
5. CUDA graph of just the RS kernel
"""

import os
import time
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


@triton.jit
def _rs(input_ptr, output_ptr, M, N, M_local,
        stride_in_m, stride_in_n, stride_out_m, stride_out_n,
        heap_bases: tl.tensor,
        cur_rank: tl.constexpr, world_size: tl.constexpr,
        BM: tl.constexpr, BN: tl.constexpr, NUM_SMS: tl.constexpr):
    pid = tl.program_id(0)
    nm = M_local // BM
    nn = tl.cdiv(N, BN)
    m_off = cur_rank * nm
    for t in range(pid, nm * nn, NUM_SMS):
        lm = t // nn
        pn = t % nn
        gm = m_off + lm
        rm = gm * BM + tl.arange(0, BM)
        rm = tl.max_contiguous(tl.multiple_of(rm, BM), BM)
        rn = pn * BN + tl.arange(0, BN)
        rn = tl.max_contiguous(tl.multiple_of(rn % N, BN), BN)
        off = rm[:, None] * stride_in_m + rn[None, :] * stride_in_n
        bp = input_ptr + off
        full = (gm * BM + BM <= M) & (pn * BN + BN <= N)
        if full:
            acc = iris.load(bp, cur_rank, 0, heap_bases, hint=(1, BN)).to(tl.float32)
            for i in tl.static_range(1, world_size):
                acc += iris.load(bp, cur_rank, i, heap_bases, hint=(1, BN)).to(tl.float32)
            om = lm * BM + tl.arange(0, BM)
            om = tl.max_contiguous(tl.multiple_of(om, BM), BM)
            tl.store(output_ptr + om[:, None] * stride_out_m + rn[None, :] * stride_out_n,
                     acc.to(output_ptr.type.element_ty))
        else:
            msk = (rm[:, None] < M) & (rn[None, :] < N)
            acc = iris.load(bp, cur_rank, 0, heap_bases, mask=msk, hint=(1, BN)).to(tl.float32)
            for i in tl.static_range(1, world_size):
                acc += iris.load(bp, cur_rank, i, heap_bases, mask=msk, hint=(1, BN)).to(tl.float32)
            om = lm * BM + tl.arange(0, BM)
            omsk = (om[:, None] < M_local) & (rn[None, :] < N)
            tl.store(output_ptr + om[:, None] * stride_out_m + rn[None, :] * stride_out_n,
                     acc.to(output_ptr.type.element_ty), mask=omsk)


M, N = 2048, 2880
M_local = M // world_size
dtype = torch.float16
warmup, iters = 200, 1000
BM, BN, SMS, W = 128, 64, 196, 4

inp = shmem.zeros((M, N), dtype=dtype)
inp.copy_(torch.randn(M, N, dtype=dtype, device=f"cuda:{rank}"))
out = torch.zeros(M_local, N, dtype=dtype, device=f"cuda:{rank}")
hb = shmem.get_heap_bases()
shmem.barrier()

s, e = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)

args = (inp, out, M, N, M_local,
        inp.stride(0), inp.stride(1), out.stride(0), out.stride(1),
        hb, rank, world_size, BM, BN, SMS)

if rank == 0:
    print(f"Launch overhead test: M={M}, N={N}, TP={world_size}")
    print(f"iters={iters} (high count to expose per-launch cost)")
    print()

# 1. Normal launch
for _ in range(warmup):
    _rs[(SMS,)](*args, num_warps=W)
torch.cuda.synchronize()
s.record()
for _ in range(iters):
    _rs[(SMS,)](*args, num_warps=W)
e.record()
torch.cuda.synchronize()
normal_ms = s.elapsed_time(e) / iters
if rank == 0:
    print(f"1. Normal launch:        {normal_ms:.4f}ms")

# 2. Measure pure Python dispatch (no GPU sync in loop)
torch.cuda.synchronize()
t0 = time.perf_counter()
for _ in range(iters):
    _rs[(SMS,)](*args, num_warps=W)
t1 = time.perf_counter()
torch.cuda.synchronize()
dispatch_ms = (t1 - t0) / iters * 1000
if rank == 0:
    print(f"2. Python dispatch only: {dispatch_ms:.4f}ms  (async, no sync)")
    print(f"   -> launch overhead is {dispatch_ms*1000:.1f}us of the {normal_ms*1000:.1f}us total")

# 3. Pre-resolved compiled kernel (bypass JIT wrapper)
try:
    # Trigger compilation and grab the compiled kernel
    _rs[(SMS,)](*args, num_warps=W)
    torch.cuda.synchronize()

    cache = _rs.cache[_rs.device_caches[0][0] if hasattr(_rs, 'device_caches') else 0]
    ck = list(cache.values())[0] if cache else None

    if ck is not None:
        if rank == 0:
            print(f"3. CompiledKernel found: {type(ck).__name__}")
    else:
        if rank == 0:
            print(f"3. CompiledKernel: cache empty")
except Exception as ex:
    if rank == 0:
        print(f"3. CompiledKernel: N/A ({str(ex)[:60]})")

# 4. CUDA graph of just the RS kernel
try:
    for _ in range(5):
        _rs[(SMS,)](*args, num_warps=W)
    torch.cuda.synchronize()

    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g):
        _rs[(SMS,)](*args, num_warps=W)

    for _ in range(warmup):
        g.replay()
    torch.cuda.synchronize()
    s.record()
    for _ in range(iters):
        g.replay()
    e.record()
    torch.cuda.synchronize()
    graph_ms = s.elapsed_time(e) / iters
    if rank == 0:
        print(f"4. CUDA graph replay:    {graph_ms:.4f}ms ({normal_ms/graph_ms:.2f}x vs normal)")
        if graph_ms < normal_ms:
            print(f"   -> saves {(normal_ms-graph_ms)*1000:.1f}us per call")
except Exception as ex:
    if rank == 0:
        print(f"4. CUDA graph: ERROR ({str(ex)[:60]})")

# 5. Full E2E with graph (GEMM + RS)
try:
    A = shmem.zeros((M, 4096 // world_size), dtype=dtype)
    A.copy_(torch.randn(M, 4096 // world_size, dtype=dtype, device=f"cuda:{rank}"))
    Bm = torch.randn(4096 // world_size, N, dtype=dtype, device=f"cuda:{rank}")

    # Reference E2E without graph
    for _ in range(warmup):
        torch.mm(A, Bm, out=inp)
        _rs[(SMS,)](*args, num_warps=W)
    torch.cuda.synchronize()
    s.record()
    for _ in range(iters):
        torch.mm(A, Bm, out=inp)
        _rs[(SMS,)](*args, num_warps=W)
    e.record()
    torch.cuda.synchronize()
    e2e_normal = s.elapsed_time(e) / iters

    # E2E with graph
    for _ in range(5):
        torch.mm(A, Bm, out=inp)
        _rs[(SMS,)](*args, num_warps=W)
    torch.cuda.synchronize()

    g2 = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g2):
        torch.mm(A, Bm, out=inp)
        _rs[(SMS,)](*args, num_warps=W)

    for _ in range(warmup):
        g2.replay()
    torch.cuda.synchronize()
    s.record()
    for _ in range(iters):
        g2.replay()
    e.record()
    torch.cuda.synchronize()
    e2e_graph = s.elapsed_time(e) / iters

    # RCCL reference
    Cr = torch.empty(M, N, dtype=dtype, device=f"cuda:{rank}")
    Cro = torch.empty(M_local, N, dtype=dtype, device=f"cuda:{rank}")
    for _ in range(warmup):
        torch.mm(A, Bm, out=Cr)
        dist.reduce_scatter_tensor(Cro, Cr, op=dist.ReduceOp.SUM)
    torch.cuda.synchronize()
    s.record()
    for _ in range(iters):
        torch.mm(A, Bm, out=Cr)
        dist.reduce_scatter_tensor(Cro, Cr, op=dist.ReduceOp.SUM)
    e.record()
    torch.cuda.synchronize()
    rccl_ms = s.elapsed_time(e) / iters

    if rank == 0:
        print()
        print(f"5. E2E comparison:")
        print(f"   RCCL:              {rccl_ms:.4f}ms")
        print(f"   normal (2-kernel): {e2e_normal:.4f}ms ({rccl_ms/e2e_normal:.2f}x)")
        print(f"   graph (2-kernel):  {e2e_graph:.4f}ms ({rccl_ms/e2e_graph:.2f}x)")
        if e2e_graph < e2e_normal:
            print(f"   -> graph saves {(e2e_normal-e2e_graph)*1000:.1f}us ({e2e_normal/e2e_graph:.2f}x)")
except Exception as ex:
    if rank == 0:
        print(f"5. E2E graph: ERROR ({str(ex)[:80]})")

shmem.barrier()
dist.destroy_process_group()
