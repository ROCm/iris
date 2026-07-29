#!/usr/bin/env python3
"""Test + benchmark fused GEMM+RS with monotonic counter flags."""

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

from iris.ops.matmul_reduce_scatter_fused import matmul_reduce_scatter_fused
from iris.ops import FusedConfig

M, N, K_global = 2048, 2880, 4096
K_local = K_global // world_size
M_local = M // world_size
dtype = torch.float16
warmup, iters = 100, 500

A = shmem.zeros((M, K_local), dtype=dtype)
A.copy_(torch.randn(M, K_local, dtype=dtype, device=f"cuda:{rank}"))
B = torch.randn(K_local, N, dtype=dtype, device=f"cuda:{rank}")
C_out = torch.zeros(M_local, N, dtype=dtype, device=f"cuda:{rank}")

config = FusedConfig(block_size_m=128, block_size_n=64, block_size_k=64, group_size_m=4)

if rank == 0:
    print(f"Fused GEMM+RS (monotonic): M={M}, N={N}, K={K_global}, TP={world_size}")

# Correctness
shmem.barrier()
ws = matmul_reduce_scatter_fused(shmem, C_out, A, B, config=config, gemm_sms=240, num_sms=304)
torch.cuda.synchronize()

ref = torch.empty(M_local, N, dtype=dtype, device=f"cuda:{rank}")
C_ref = torch.empty(M, N, dtype=dtype, device=f"cuda:{rank}")
torch.mm(A, B, out=C_ref)
dist.reduce_scatter_tensor(ref, C_ref, op=dist.ReduceOp.SUM)
torch.cuda.synchronize()

max_diff = torch.abs(C_out - ref).max().item()
if rank == 0:
    print(f"Correctness: max_diff={max_diff:.6f} {'PASS' if max_diff < 1.0 else 'FAIL'}")
    if max_diff > 1.0:
        print(f"  C_out[0:4,0:4] = {C_out[0:4,0:4]}")
        print(f"  ref[0:4,0:4] = {ref[0:4,0:4]}")

if max_diff > 1.0:
    shmem.barrier()
    dist.destroy_process_group()
    exit(1)

# Repeated correctness (test monotonic counters across iterations)
if rank == 0:
    print("\nRepeated correctness (5 iterations):")
for i in range(5):
    C_out.zero_()
    matmul_reduce_scatter_fused(shmem, C_out, A, B, config=config, workspace=ws, gemm_sms=240, num_sms=304)
    torch.cuda.synchronize()
    diff = torch.abs(C_out - ref).max().item()
    if rank == 0:
        print(f"  iter {i+1}: max_diff={diff:.6f} {'PASS' if diff < 1.0 else 'FAIL'}")
    if diff > 1.0:
        break

# RCCL baseline
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

# Fused benchmark — sweep GEMM_SMS ratio
if rank == 0:
    print(f"\nRCCL baseline: {rccl_ms:.3f}ms")
    print("Fused GEMM+RS sweep:")

best_ms = 999.0
for gemm_sms in [128, 196, 224, 240, 256, 280]:
    num_sms = 304
    comm_sms = num_sms - gemm_sms
    if comm_sms < 8:
        continue

    shmem.barrier()
    ws2 = None
    for _ in range(warmup):
        C_out.zero_()
        ws2 = matmul_reduce_scatter_fused(
            shmem, C_out, A, B, config=config, workspace=ws2,
            gemm_sms=gemm_sms, num_sms=num_sms,
        )
    torch.cuda.synchronize()

    s.record()
    for _ in range(iters):
        C_out.zero_()
        matmul_reduce_scatter_fused(
            shmem, C_out, A, B, config=config, workspace=ws2,
            gemm_sms=gemm_sms, num_sms=num_sms,
        )
    e.record()
    torch.cuda.synchronize()

    ms = s.elapsed_time(e) / iters
    speedup = rccl_ms / ms
    if rank == 0:
        print(f"  gemm={gemm_sms:3d} comm={comm_sms:3d}: {ms:.3f}ms ({speedup:.2f}x)")
    if ms < best_ms:
        best_ms = ms

# Also benchmark torch.mm + fast iris RS (current best two-kernel)
try:
    from iris.ops.matmul_reduce_scatter_fast import _fast_reduce_scatter_kernel, _get_config
    cfg = _get_config(world_size, M_local)
    C_sym = shmem.zeros((M, N), dtype=dtype)
    C_out3 = torch.zeros(M_local, N, dtype=dtype, device=f"cuda:{rank}")
    heap_bases = shmem.get_heap_bases()
    shmem.barrier()

    for _ in range(warmup):
        torch.mm(A, B, out=C_sym)
        _fast_reduce_scatter_kernel[(cfg["num_sms"],)](
            C_sym, C_out3, M, N, M_local,
            C_sym.stride(0), C_sym.stride(1), C_out3.stride(0), C_out3.stride(1),
            heap_bases, rank, world_size,
            cfg["block_m"], cfg["block_n"], cfg["num_sms"],
            num_warps=cfg["num_warps"],
        )
    torch.cuda.synchronize()

    s.record()
    for _ in range(iters):
        torch.mm(A, B, out=C_sym)
        _fast_reduce_scatter_kernel[(cfg["num_sms"],)](
            C_sym, C_out3, M, N, M_local,
            C_sym.stride(0), C_sym.stride(1), C_out3.stride(0), C_out3.stride(1),
            heap_bases, rank, world_size,
            cfg["block_m"], cfg["block_n"], cfg["num_sms"],
            num_warps=cfg["num_warps"],
        )
    e.record()
    torch.cuda.synchronize()
    two_kernel_ms = s.elapsed_time(e) / iters

    if rank == 0:
        print(f"\nComparison:")
        print(f"  RCCL:                  {rccl_ms:.3f}ms")
        print(f"  Two-kernel (best):     {two_kernel_ms:.3f}ms ({rccl_ms/two_kernel_ms:.2f}x)")
        print(f"  Single-kernel (best):  {best_ms:.3f}ms ({rccl_ms/best_ms:.2f}x)")
except Exception as ex:
    if rank == 0:
        print(f"\nTwo-kernel comparison failed: {ex}")
        print(f"  RCCL:                  {rccl_ms:.3f}ms")
        print(f"  Single-kernel (best):  {best_ms:.3f}ms ({rccl_ms/best_ms:.2f}x)")

shmem.barrier()
dist.destroy_process_group()
