#!/usr/bin/env python3
"""Comprehensive CU allocation sweep across all fused GEMM+RS variants.

Sweeps GEMM_SMS in XCD increments (38 CUs per XCD on MI355X).
Also sweeps num_warps and tests with/without XCD chiplet transform.
"""

import os
import sys
import torch
import torch.distributed as dist
import iris

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

# RCCL baseline
C_rccl = torch.empty(M, N, dtype=dtype, device=f"cuda:{rank}")
C_rccl_out = torch.empty(M_local, N, dtype=dtype, device=f"cuda:{rank}")
for _ in range(warmup):
    torch.mm(A_sym, B, out=C_rccl)
    dist.reduce_scatter_tensor(C_rccl_out, C_rccl, op=dist.ReduceOp.SUM)
torch.cuda.synchronize()
s.record()
for _ in range(iters):
    torch.mm(A_sym, B, out=C_rccl)
    dist.reduce_scatter_tensor(C_rccl_out, C_rccl, op=dist.ReduceOp.SUM)
e.record()
torch.cuda.synchronize()
rccl_ms = s.elapsed_time(e) / iters

if rank == 0:
    print(f"CU allocation sweep: M={M}, N={N}, K={K_global}, TP={world_size}")
    print(f"RCCL baseline: {rccl_ms:.3f}ms")
    print()

# Two-kernel fast RS baseline
from iris.ops.matmul_reduce_scatter_fast import _fast_reduce_scatter_kernel, _get_config
cfg = _get_config(world_size, M_local)
C_sym = shmem.zeros((M, N), dtype=dtype)
C_out = torch.zeros(M_local, N, dtype=dtype, device=f"cuda:{rank}")
heap_bases = shmem.get_heap_bases()
shmem.barrier()

for _ in range(warmup):
    torch.mm(A_sym, B, out=C_sym)
    _fast_reduce_scatter_kernel[(cfg["num_sms"],)](
        C_sym, C_out, M, N, M_local,
        C_sym.stride(0), C_sym.stride(1), C_out.stride(0), C_out.stride(1),
        heap_bases, rank, world_size,
        cfg["block_m"], cfg["block_n"], cfg["num_sms"],
        num_warps=cfg["num_warps"],
    )
torch.cuda.synchronize()
s.record()
for _ in range(iters):
    torch.mm(A_sym, B, out=C_sym)
    _fast_reduce_scatter_kernel[(cfg["num_sms"],)](
        C_sym, C_out, M, N, M_local,
        C_sym.stride(0), C_sym.stride(1), C_out.stride(0), C_out.stride(1),
        heap_bases, rank, world_size,
        cfg["block_m"], cfg["block_n"], cfg["num_sms"],
        num_warps=cfg["num_warps"],
    )
e.record()
torch.cuda.synchronize()
fast_rs_ms = s.elapsed_time(e) / iters

if rank == 0:
    print(f"Two-kernel fast RS: {fast_rs_ms:.3f}ms ({rccl_ms/fast_rs_ms:.2f}x)")
    print()

# =============================
# Ex22 CU split sweep
# =============================
if rank == 0:
    print("=" * 70)
    print("Ex22 (WG spec + iris.atomic_add push) — CU split sweep")
    print("=" * 70)

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../../examples/22_gemm_one_shot_reduce_scatter_wg_specialization'))
try:
    from gemm_reduce_scatter import persistent_gemm_reduce_scatter_wg_specialized

    NUM_SMS = 304
    # XCD increments: 38, 76, 114, 152, 190, 228, 266
    gemm_sms_values = [38, 76, 114, 152, 190, 228, 266, 280]

    for warps in [4, 8]:
        if rank == 0:
            print(f"\n  num_warps={warps}:")
        for gemm_sms in gemm_sms_values:
            comm_sms = NUM_SMS - gemm_sms
            if comm_sms < 8:
                continue

            C_local = shmem.zeros((M, N), dtype=dtype)
            C_global = shmem.zeros((M_local, N), dtype=dtype)
            num_m = M // 128
            num_n = (N + 255) // 256
            locks = shmem.zeros((num_m * num_n,), dtype=torch.int32)

            shmem.barrier()
            try:
                for _ in range(warmup // 2):
                    C_global.zero_()
                    locks.zero_()
                    shmem.barrier()
                    persistent_gemm_reduce_scatter_wg_specialized[(NUM_SMS,)](
                        A_sym, B, C_local, C_global, locks,
                        M, N, K_local,
                        A_sym.stride(0), A_sym.stride(1),
                        B.stride(0), B.stride(1),
                        C_local.stride(0), C_local.stride(1),
                        C_global.stride(0), C_global.stride(1),
                        128, 256, 64, 4,
                        gemm_sms, NUM_SMS, 8, K_local % 64 == 0,
                        heap_bases, rank, world_size,
                        num_warps=warps,
                    )
                torch.cuda.synchronize()

                s.record()
                for _ in range(iters):
                    C_global.zero_()
                    locks.zero_()
                    shmem.barrier()
                    persistent_gemm_reduce_scatter_wg_specialized[(NUM_SMS,)](
                        A_sym, B, C_local, C_global, locks,
                        M, N, K_local,
                        A_sym.stride(0), A_sym.stride(1),
                        B.stride(0), B.stride(1),
                        C_local.stride(0), C_local.stride(1),
                        C_global.stride(0), C_global.stride(1),
                        128, 256, 64, 4,
                        gemm_sms, NUM_SMS, 8, K_local % 64 == 0,
                        heap_bases, rank, world_size,
                        num_warps=warps,
                    )
                e.record()
                torch.cuda.synchronize()
                ms = s.elapsed_time(e) / iters

                if rank == 0:
                    xcds_gemm = gemm_sms // 38
                    xcds_comm = comm_sms // 38
                    print(f"    gemm={gemm_sms:3d}({xcds_gemm}xcd) comm={comm_sms:3d}({xcds_comm}xcd): {ms:.3f}ms ({rccl_ms/ms:.2f}x)")
            except Exception as ex:
                if rank == 0:
                    print(f"    gemm={gemm_sms:3d} comm={comm_sms:3d}: ERROR ({str(ex)[:60]})")

except ImportError as ex:
    if rank == 0:
        print(f"  SKIP (import error: {ex})")

# =============================
# K-loop atomic push CU sweep
# =============================
if rank == 0:
    print()
    print("=" * 70)
    print("K-loop atomic push (simplest fused) — SMS sweep")
    print("=" * 70)

from iris.ops.matmul_reduce_scatter_kloop import matmul_reduce_scatter_kloop

for warps in [4, 8]:
    if rank == 0:
        print(f"\n  num_warps={warps}:")
    for num_sms in [38, 76, 114, 152, 196, 256, 304]:
        for bm in [64, 128]:
            if M % (world_size * bm) != 0:
                continue
            C_out_k = shmem.zeros((M_local, N), dtype=dtype)
            shmem.barrier()

            try:
                for _ in range(warmup // 2):
                    C_out_k.zero_()
                    shmem.barrier()
                    matmul_reduce_scatter_kloop(shmem, C_out_k, A_sym, B,
                                                block_m=bm, num_sms=num_sms, num_warps=warps)
                torch.cuda.synchronize()

                s.record()
                for _ in range(iters):
                    C_out_k.zero_()
                    shmem.barrier()
                    matmul_reduce_scatter_kloop(shmem, C_out_k, A_sym, B,
                                                block_m=bm, num_sms=num_sms, num_warps=warps)
                e.record()
                torch.cuda.synchronize()
                ms = s.elapsed_time(e) / iters

                if rank == 0:
                    print(f"    bm={bm} sms={num_sms:3d}: {ms:.3f}ms ({rccl_ms/ms:.2f}x)")
            except Exception as ex:
                if rank == 0:
                    print(f"    bm={bm} sms={num_sms:3d}: ERROR ({str(ex)[:60]})")

if rank == 0:
    print()
    print("=" * 70)
    print(f"SUMMARY: RCCL={rccl_ms:.3f}ms, two-kernel={fast_rs_ms:.3f}ms ({rccl_ms/fast_rs_ms:.2f}x)")
    print("=" * 70)

shmem.barrier()
dist.destroy_process_group()
