#!/usr/bin/env python3
"""Triton GEMM (.wt stores) + fast RS — test if write-through reduces E2E latency.

Hypothesis: hipBLASLt stores to L2, peers must wait for L2→HBM flush.
Triton GEMM with .wt bypasses L2, data visible to peers immediately.
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
def triton_gemm_wt_kernel(
    A, B, C,
    M, N, K,
    stride_am, stride_ak, stride_bk, stride_bn, stride_cm, stride_cn,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
    GROUP_SIZE_M: tl.constexpr, NUM_SMS: tl.constexpr, EVEN_K: tl.constexpr,
):
    pid = tl.program_id(0)
    num_pid_m = tl.cdiv(M, BLOCK_M)
    num_pid_n = tl.cdiv(N, BLOCK_N)
    total_tiles = num_pid_m * num_pid_n

    for tile_id in range(pid, total_tiles, NUM_SMS):
        num_pid_in_group = GROUP_SIZE_M * num_pid_n
        group_id = tile_id // num_pid_in_group
        first_pid_m = group_id * GROUP_SIZE_M
        group_size_m = min(num_pid_m - first_pid_m, GROUP_SIZE_M)
        pid_m = first_pid_m + ((tile_id % num_pid_in_group) % group_size_m)
        pid_n = (tile_id % num_pid_in_group) // group_size_m

        rm = (pid_m * BLOCK_M + tl.arange(0, BLOCK_M)) % M
        rn = (pid_n * BLOCK_N + tl.arange(0, BLOCK_N)) % N
        rm = tl.max_contiguous(tl.multiple_of(rm, BLOCK_M), BLOCK_M)
        rn = tl.max_contiguous(tl.multiple_of(rn, BLOCK_N), BLOCK_N)

        rk = tl.arange(0, BLOCK_K)
        A_BASE = A + rm[:, None] * stride_am + rk[None, :] * stride_ak
        B_BASE = B + rk[:, None] * stride_bk + rn[None, :] * stride_bn

        acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
        loop_k = tl.cdiv(K, BLOCK_K)
        if not EVEN_K:
            loop_k -= 1

        for k in range(loop_k):
            a = tl.load(tl.multiple_of(A_BASE, (1, 16)))
            b = tl.load(tl.multiple_of(B_BASE, (16, 1)))
            acc += tl.dot(a, b)
            A_BASE += BLOCK_K * stride_ak
            B_BASE += BLOCK_K * stride_bk

        if not EVEN_K:
            rk2 = loop_k * BLOCK_K + tl.arange(0, BLOCK_K)
            A_LAST = A + rm[:, None] * stride_am + rk2[None, :] * stride_ak
            B_LAST = B + rk2[:, None] * stride_bk + rn[None, :] * stride_bn
            a = tl.load(A_LAST, mask=rk2[None, :] < K, other=0.0)
            b = tl.load(B_LAST, mask=rk2[:, None] < K, other=0.0)
            acc += tl.dot(a, b)

        c = acc.to(C.type.element_ty)
        C_ptrs = C + rm[:, None] * stride_cm + rn[None, :] * stride_cn
        tl.store(C_ptrs, c, cache_modifier=".wt")


@triton.jit
def fast_reduce_scatter_kernel(
    input_ptr, output_ptr,
    M, N, M_local,
    stride_in_m, stride_in_n, stride_out_m, stride_out_n,
    heap_bases: tl.tensor,
    cur_rank: tl.constexpr, world_size: tl.constexpr,
    BLOCK_SIZE_M: tl.constexpr, BLOCK_SIZE_N: tl.constexpr,
    NUM_SMS: tl.constexpr,
):
    pid = tl.program_id(0)
    acc_dtype = tl.float32
    num_m_tiles = M_local // BLOCK_SIZE_M
    num_n_tiles = tl.cdiv(N, BLOCK_SIZE_N)
    total_tiles = num_m_tiles * num_n_tiles
    m_offset = cur_rank * num_m_tiles

    for tile_id in range(pid, total_tiles, NUM_SMS):
        local_pid_m = tile_id // num_n_tiles
        pid_n = tile_id % num_n_tiles
        global_pid_m = m_offset + local_pid_m

        rm = global_pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
        rm = tl.max_contiguous(tl.multiple_of(rm, BLOCK_SIZE_M), BLOCK_SIZE_M)
        rn = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
        rn = tl.max_contiguous(tl.multiple_of(rn % N, BLOCK_SIZE_N), BLOCK_SIZE_N)

        in_offset = rm[:, None] * stride_in_m + rn[None, :] * stride_in_n
        base_ptr = input_ptr + in_offset
        is_full = (global_pid_m * BLOCK_SIZE_M + BLOCK_SIZE_M <= M) & (pid_n * BLOCK_SIZE_N + BLOCK_SIZE_N <= N)

        if is_full:
            start_rank = pid % world_size
            acc = iris.load(base_ptr, cur_rank, start_rank, heap_bases, hint=(1, BLOCK_SIZE_N)).to(acc_dtype)
            for i in tl.static_range(1, world_size):
                r = (start_rank + i) % world_size
                acc += iris.load(base_ptr, cur_rank, r, heap_bases, hint=(1, BLOCK_SIZE_N)).to(acc_dtype)
            out_rm = local_pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
            out_rm = tl.max_contiguous(tl.multiple_of(out_rm, BLOCK_SIZE_M), BLOCK_SIZE_M)
            tl.store(output_ptr + out_rm[:, None] * stride_out_m + rn[None, :] * stride_out_n,
                     acc.to(output_ptr.type.element_ty))
        else:
            mask = (rm[:, None] < M) & (rn[None, :] < N)
            start_rank = pid % world_size
            acc = iris.load(base_ptr, cur_rank, start_rank, heap_bases, mask=mask, hint=(1, BLOCK_SIZE_N)).to(acc_dtype)
            for i in tl.static_range(1, world_size):
                r = (start_rank + i) % world_size
                acc += iris.load(base_ptr, cur_rank, r, heap_bases, mask=mask, hint=(1, BLOCK_SIZE_N)).to(acc_dtype)
            out_rm = local_pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
            out_mask = (out_rm[:, None] < M_local) & (rn[None, :] < N)
            tl.store(output_ptr + out_rm[:, None] * stride_out_m + rn[None, :] * stride_out_n,
                     acc.to(output_ptr.type.element_ty), mask=out_mask)


# Setup
M, N, K_global = 2048, 2880, 4096
K_local = K_global // world_size
M_local = M // world_size
dtype = torch.float16
warmup, iters = 100, 500

A = torch.randn(M, K_local, dtype=dtype, device=f"cuda:{rank}")
B = torch.randn(K_local, N, dtype=dtype, device=f"cuda:{rank}")

C_sym = shmem.zeros((M, N), dtype=dtype)
C_out = torch.zeros(M_local, N, dtype=dtype, device=f"cuda:{rank}")
heap_bases = shmem.get_heap_bases()

RS_CONFIG = {2: (128, 64, 128), 4: (64, 64, 32), 8: (32, 64, 32)}
rs_bm, rs_bn, rs_sms = RS_CONFIG.get(world_size, (128, 64, 128))

GEMM_SMS = 304

if rank == 0:
    print(f"Triton GEMM (.wt) + fast RS: M={M}, N={N}, K={K_global}, TP={world_size}")

# Correctness
shmem.barrier()
triton_gemm_wt_kernel[(GEMM_SMS,)](
    A, B, C_sym, M, N, K_local,
    A.stride(0), A.stride(1), B.stride(0), B.stride(1),
    C_sym.stride(0), C_sym.stride(1),
    128, 64, 64, 4, GEMM_SMS, K_local % 64 == 0,
    num_warps=8, num_stages=2,
)
shmem.barrier()
fast_reduce_scatter_kernel[(rs_sms,)](
    C_sym, C_out, M, N, M_local,
    C_sym.stride(0), C_sym.stride(1), C_out.stride(0), C_out.stride(1),
    heap_bases, rank, world_size, rs_bm, rs_bn, rs_sms,
)
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
    shmem.barrier()
    dist.destroy_process_group()
    exit(1)

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

# hipBLASLt GEMM + fast RS (current best)
shmem.barrier()
for _ in range(warmup):
    torch.mm(A, B, out=C_sym)
    fast_reduce_scatter_kernel[(rs_sms,)](
        C_sym, C_out, M, N, M_local,
        C_sym.stride(0), C_sym.stride(1), C_out.stride(0), C_out.stride(1),
        heap_bases, rank, world_size, rs_bm, rs_bn, rs_sms,
    )
torch.cuda.synchronize()
s.record()
for _ in range(iters):
    torch.mm(A, B, out=C_sym)
    fast_reduce_scatter_kernel[(rs_sms,)](
        C_sym, C_out, M, N, M_local,
        C_sym.stride(0), C_sym.stride(1), C_out.stride(0), C_out.stride(1),
        heap_bases, rank, world_size, rs_bm, rs_bn, rs_sms,
    )
e.record()
torch.cuda.synchronize()
hipblas_rs_ms = s.elapsed_time(e) / iters

# Triton GEMM (.wt) + fast RS
shmem.barrier()
for _ in range(warmup):
    triton_gemm_wt_kernel[(GEMM_SMS,)](
        A, B, C_sym, M, N, K_local,
        A.stride(0), A.stride(1), B.stride(0), B.stride(1),
        C_sym.stride(0), C_sym.stride(1),
        128, 64, 64, 4, GEMM_SMS, K_local % 64 == 0,
        num_warps=8, num_stages=2,
    )
    fast_reduce_scatter_kernel[(rs_sms,)](
        C_sym, C_out, M, N, M_local,
        C_sym.stride(0), C_sym.stride(1), C_out.stride(0), C_out.stride(1),
        heap_bases, rank, world_size, rs_bm, rs_bn, rs_sms,
    )
torch.cuda.synchronize()
s.record()
for _ in range(iters):
    triton_gemm_wt_kernel[(GEMM_SMS,)](
        A, B, C_sym, M, N, K_local,
        A.stride(0), A.stride(1), B.stride(0), B.stride(1),
        C_sym.stride(0), C_sym.stride(1),
        128, 64, 64, 4, GEMM_SMS, K_local % 64 == 0,
        num_warps=8, num_stages=2,
    )
    fast_reduce_scatter_kernel[(rs_sms,)](
        C_sym, C_out, M, N, M_local,
        C_sym.stride(0), C_sym.stride(1), C_out.stride(0), C_out.stride(1),
        heap_bases, rank, world_size, rs_bm, rs_bn, rs_sms,
    )
e.record()
torch.cuda.synchronize()
triton_rs_ms = s.elapsed_time(e) / iters

# Standalone GEMM comparison
for _ in range(warmup):
    torch.mm(A, B, out=C_sym)
torch.cuda.synchronize()
s.record()
for _ in range(iters):
    torch.mm(A, B, out=C_sym)
e.record()
torch.cuda.synchronize()
hipblas_gemm = s.elapsed_time(e) / iters

for _ in range(warmup):
    triton_gemm_wt_kernel[(GEMM_SMS,)](
        A, B, C_sym, M, N, K_local,
        A.stride(0), A.stride(1), B.stride(0), B.stride(1),
        C_sym.stride(0), C_sym.stride(1),
        128, 64, 64, 4, GEMM_SMS, K_local % 64 == 0,
        num_warps=8, num_stages=2,
    )
torch.cuda.synchronize()
s.record()
for _ in range(iters):
    triton_gemm_wt_kernel[(GEMM_SMS,)](
        A, B, C_sym, M, N, K_local,
        A.stride(0), A.stride(1), B.stride(0), B.stride(1),
        C_sym.stride(0), C_sym.stride(1),
        128, 64, 64, 4, GEMM_SMS, K_local % 64 == 0,
        num_warps=8, num_stages=2,
    )
e.record()
torch.cuda.synchronize()
triton_gemm = s.elapsed_time(e) / iters

if rank == 0:
    print()
    print(f"Standalone GEMM:")
    print(f"  hipBLASLt:  {hipblas_gemm:.3f}ms")
    print(f"  Triton .wt: {triton_gemm:.3f}ms")
    print()
    print(f"E2E:")
    print(f"  torch.mm + RCCL RS:      {rccl_ms:.3f}ms")
    print(f"  torch.mm + fast iris RS: {hipblas_rs_ms:.3f}ms  ({rccl_ms/hipblas_rs_ms:.2f}x)")
    print(f"  Triton.wt + fast iris RS: {triton_rs_ms:.3f}ms  ({rccl_ms/triton_rs_ms:.2f}x)")

shmem.barrier()
dist.destroy_process_group()
