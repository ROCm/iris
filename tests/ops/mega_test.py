#!/usr/bin/env python3
"""Mega test: run ALL experiments in one shot. No wasted node time."""

import os
import torch
import torch.distributed as dist
import triton
import triton.language as tl
import iris
import time

torch.cuda.set_device(int(os.environ.get("LOCAL_RANK", 0)))
dist.init_process_group(backend="nccl")

rank = dist.get_rank()
world_size = dist.get_world_size()

heap_size = 2**33
shmem = iris.iris(heap_size)

M, N, K_global = 2048, 2880, 4096
K_local = K_global // world_size
M_local = M // world_size
dtype = torch.float16
warmup, iters = 50, 200

A_sym = shmem.zeros((M, K_local), dtype=dtype)
A_sym.copy_(torch.randn(M, K_local, dtype=dtype, device=f"cuda:{rank}"))
B = torch.randn(K_local, N, dtype=dtype, device=f"cuda:{rank}")

s, e = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)

def bench(name, fn, setup_fn=None):
    """Benchmark helper."""
    try:
        if setup_fn:
            setup_fn()
        for _ in range(warmup):
            fn()
        torch.cuda.synchronize()
        s.record()
        for _ in range(iters):
            fn()
        e.record()
        torch.cuda.synchronize()
        ms = s.elapsed_time(e) / iters
        return ms
    except Exception as ex:
        if rank == 0:
            print(f"  {name}: ERROR ({str(ex)[:80]})")
        return None

# Reference
ref = torch.empty(M_local, N, dtype=dtype, device=f"cuda:{rank}")
C_ref = torch.empty(M, N, dtype=dtype, device=f"cuda:{rank}")
torch.mm(A_sym, B, out=C_ref)
dist.reduce_scatter_tensor(ref, C_ref, op=dist.ReduceOp.SUM)
torch.cuda.synchronize()

if rank == 0:
    print(f"Mega test: M={M}, N={N}, K={K_global}, TP={world_size}")
    print("=" * 70)

# ======== TEST 1: RCCL baseline ========
C_rccl = torch.empty(M, N, dtype=dtype, device=f"cuda:{rank}")
C_rccl_out = torch.empty(M_local, N, dtype=dtype, device=f"cuda:{rank}")
rccl_ms = bench("RCCL", lambda: (torch.mm(A_sym, B, out=C_rccl), dist.reduce_scatter_tensor(C_rccl_out, C_rccl, op=dist.ReduceOp.SUM)))
if rank == 0:
    print(f"\n1. RCCL baseline: {rccl_ms:.3f}ms")

# ======== TEST 2: Two-kernel fast RS (current best) ========
from iris.ops.matmul_reduce_scatter_fast import _fast_reduce_scatter_kernel, _get_config
cfg = _get_config(world_size, M_local)
C_sym2 = shmem.zeros((M, N), dtype=dtype)
C_out2 = torch.zeros(M_local, N, dtype=dtype, device=f"cuda:{rank}")
heap_bases = shmem.get_heap_bases()
shmem.barrier()

fast_rs_ms = bench("fast RS",
    lambda: (torch.mm(A_sym, B, out=C_sym2),
             _fast_reduce_scatter_kernel[(cfg["num_sms"],)](
                 C_sym2, C_out2, M, N, M_local,
                 C_sym2.stride(0), C_sym2.stride(1), C_out2.stride(0), C_out2.stride(1),
                 heap_bases, rank, world_size, cfg["block_m"], cfg["block_n"], cfg["num_sms"],
                 num_warps=cfg["num_warps"])))
if rank == 0:
    print(f"2. Two-kernel fast RS: {fast_rs_ms:.3f}ms ({rccl_ms/fast_rs_ms:.2f}x)")

# ======== TEST 3: CUDA graph (no barrier) ========
if rank == 0:
    print(f"\n3. CUDA graph (no barrier):")

try:
    C_sym_g = shmem.zeros((M, N), dtype=dtype)
    C_out_g = torch.zeros(M_local, N, dtype=dtype, device=f"cuda:{rank}")
    shmem.barrier()

    # Warmup to compile kernel
    for _ in range(3):
        torch.mm(A_sym, B, out=C_sym_g)
        _fast_reduce_scatter_kernel[(cfg["num_sms"],)](
            C_sym_g, C_out_g, M, N, M_local,
            C_sym_g.stride(0), C_sym_g.stride(1), C_out_g.stride(0), C_out_g.stride(1),
            heap_bases, rank, world_size, cfg["block_m"], cfg["block_n"], cfg["num_sms"],
            num_warps=cfg["num_warps"])
    torch.cuda.synchronize()

    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g):
        torch.mm(A_sym, B, out=C_sym_g)
        _fast_reduce_scatter_kernel[(cfg["num_sms"],)](
            C_sym_g, C_out_g, M, N, M_local,
            C_sym_g.stride(0), C_sym_g.stride(1), C_out_g.stride(0), C_out_g.stride(1),
            heap_bases, rank, world_size, cfg["block_m"], cfg["block_n"], cfg["num_sms"],
            num_warps=cfg["num_warps"])

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
        print(f"  graph replay: {graph_ms:.3f}ms ({rccl_ms/graph_ms:.2f}x)")
except Exception as ex:
    if rank == 0:
        print(f"  graph: ERROR ({str(ex)[:80]})")

# ======== TEST 4: XCD-aware fused ========
if rank == 0:
    print(f"\n4. XCD-aware fused (device-scope flags):")

try:
    from iris.ops.matmul_reduce_scatter_xcd import matmul_reduce_scatter_xcd

    C_out_xcd = torch.zeros(M_local, N, dtype=dtype, device=f"cuda:{rank}")
    shmem.barrier()
    matmul_reduce_scatter_xcd(shmem, C_out_xcd, A_sym, B)
    torch.cuda.synchronize()

    xcd_diff = torch.abs(C_out_xcd - ref).max().item()
    if rank == 0:
        print(f"  correctness: max_diff={xcd_diff:.6f} {'PASS' if xcd_diff < 1.0 else 'FAIL'}")

    if xcd_diff < 1.0:
        for gemm_per_xcd in [16, 24, 30, 34]:
            for bn in [64, 128]:
                ms = bench(f"xcd g={gemm_per_xcd} bn={bn}",
                    lambda: matmul_reduce_scatter_xcd(shmem, C_out_xcd, A_sym, B,
                        block_n=bn, gemm_sms_per_xcd=gemm_per_xcd))
                if ms and rank == 0:
                    print(f"  gemm={gemm_per_xcd}/xcd bn={bn}: {ms:.3f}ms ({rccl_ms/ms:.2f}x)")
except Exception as ex:
    if rank == 0:
        print(f"  XCD: ERROR ({str(ex)[:80]})")

# ======== TEST 5: Scope test (compile check) ========
if rank == 0:
    print(f"\n5. Atomic scope test:")

@triton.jit
def _scope_test_gpu(ptr, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    tl.atomic_add(ptr + pid, 1, sem="release", scope="gpu")
    while tl.atomic_add(ptr + pid, 0, sem="acquire", scope="gpu") < 1:
        pass

@triton.jit
def _scope_test_sys(ptr, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    tl.atomic_add(ptr + pid, 1, sem="release", scope="sys")
    while tl.atomic_add(ptr + pid, 0, sem="acquire", scope="sys") < 1:
        pass

@triton.jit
def _scope_test_default(ptr, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    tl.atomic_add(ptr + pid, 1)
    while tl.atomic_add(ptr + pid, 0) < 1:
        pass

flags = torch.zeros(304, dtype=torch.int32, device=f"cuda:{rank}")
for name, kernel in [("gpu", _scope_test_gpu), ("sys", _scope_test_sys), ("default", _scope_test_default)]:
    flags.zero_()
    try:
        for _ in range(warmup):
            flags.zero_()
            kernel[(304,)](flags, 1)
        torch.cuda.synchronize()
        s.record()
        for _ in range(iters):
            flags.zero_()
            kernel[(304,)](flags, 1)
        e.record()
        torch.cuda.synchronize()
        ms = s.elapsed_time(e) / iters
        if rank == 0:
            print(f"  scope={name}: {ms:.4f}ms per launch")
    except Exception as ex:
        if rank == 0:
            print(f"  scope={name}: ERROR ({str(ex)[:60]})")

# ======== TEST 6: ex22 CU split ========
if rank == 0:
    print(f"\n6. Ex22 CU split sweep:")

import sys as _sys
_sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)),
    '../../examples/22_gemm_one_shot_reduce_scatter_wg_specialization'))
try:
    from gemm_reduce_scatter import persistent_gemm_reduce_scatter_wg_specialized

    NUM_SMS = 304
    for gemm_sms in [76, 152, 228, 266]:
        C_local = shmem.zeros((M, N), dtype=dtype)
        C_global = shmem.zeros((M_local, N), dtype=dtype)
        num_m = M // 128
        num_n = (N + 255) // 256
        locks = shmem.zeros((num_m * num_n,), dtype=torch.int32)

        shmem.barrier()
        try:
            ms = bench(f"ex22 g={gemm_sms}",
                lambda: (C_global.zero_(), locks.zero_(), shmem.barrier(),
                    persistent_gemm_reduce_scatter_wg_specialized[(NUM_SMS,)](
                        A_sym, B, C_local, C_global, locks,
                        M, N, K_local,
                        A_sym.stride(0), A_sym.stride(1), B.stride(0), B.stride(1),
                        C_local.stride(0), C_local.stride(1), C_global.stride(0), C_global.stride(1),
                        128, 256, 64, 4, gemm_sms, NUM_SMS, 8, K_local % 64 == 0,
                        heap_bases, rank, world_size, num_warps=8)))
            if ms and rank == 0:
                print(f"  gemm={gemm_sms} comm={NUM_SMS-gemm_sms}: {ms:.3f}ms ({rccl_ms/ms:.2f}x)")
        except Exception as ex:
            if rank == 0:
                print(f"  gemm={gemm_sms}: ERROR ({str(ex)[:60]})")
except ImportError as ex:
    if rank == 0:
        print(f"  SKIP ({ex})")

# ======== SUMMARY ========
if rank == 0:
    print()
    print("=" * 70)
    print(f"RCCL:          {rccl_ms:.3f}ms")
    print(f"Two-kernel:    {fast_rs_ms:.3f}ms ({rccl_ms/fast_rs_ms:.2f}x)")
    print("=" * 70)

shmem.barrier()
dist.destroy_process_group()
