#!/usr/bin/env python3
"""Two-kernel with tuned Triton GEMM vs hipBLASLt.

Verifies the tuned Triton GEMM (mfma=32, bn=256, stages=3) still hits
0.031ms when writing to SYMMETRIC HEAP (not regular memory).
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
shmem = iris.iris(2**33)


@triton.jit
def _tuned_gemm(
    A, B, C,
    M, N, K,
    stride_am, stride_ak, stride_bk, stride_bn, stride_cm, stride_cn,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
    GROUP_M: tl.constexpr, NUM_SMS: tl.constexpr, EVEN_K: tl.constexpr,
    WRITE_THROUGH: tl.constexpr,
):
    pid = tl.program_id(0)
    num_pid_m = tl.cdiv(M, BLOCK_M)
    num_pid_n = tl.cdiv(N, BLOCK_N)
    total = num_pid_m * num_pid_n

    for tile_id in range(pid, total, NUM_SMS):
        num_pid_in_group = GROUP_M * num_pid_n
        group_id = tile_id // num_pid_in_group
        first_pid_m = group_id * GROUP_M
        gsize = min(num_pid_m - first_pid_m, GROUP_M)
        pid_m = first_pid_m + ((tile_id % num_pid_in_group) % gsize)
        pid_n = (tile_id % num_pid_in_group) // gsize

        rm = (pid_m * BLOCK_M + tl.arange(0, BLOCK_M)) % M
        rn = (pid_n * BLOCK_N + tl.arange(0, BLOCK_N)) % N
        rm = tl.max_contiguous(tl.multiple_of(rm, BLOCK_M), BLOCK_M)
        rn = tl.max_contiguous(tl.multiple_of(rn, BLOCK_N), BLOCK_N)

        rk = tl.arange(0, BLOCK_K)
        A_BASE = A + rm[:, None] * stride_am + rk[None, :] * stride_ak
        B_BASE = B + rk[:, None] * stride_bk + rn[None, :] * stride_bn

        loop_k = tl.cdiv(K, BLOCK_K)
        if not EVEN_K:
            loop_k -= 1

        acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
        for k in range(loop_k):
            a = tl.load(tl.multiple_of(A_BASE, (1, 16)))
            b = tl.load(tl.multiple_of(B_BASE, (16, 1)))
            acc += tl.dot(a, b)
            A_BASE += BLOCK_K * stride_ak
            B_BASE += BLOCK_K * stride_bk

        if not EVEN_K:
            rk2 = loop_k * BLOCK_K + tl.arange(0, BLOCK_K)
            A_L = A + rm[:, None] * stride_am + rk2[None, :] * stride_ak
            B_L = B + rk2[:, None] * stride_bk + rn[None, :] * stride_bn
            a = tl.load(A_L, mask=rk2[None, :] < K, other=0.0)
            b = tl.load(B_L, mask=rk2[:, None] < K, other=0.0)
            acc += tl.dot(a, b)

        c = acc.to(C.type.element_ty)
        cp = C + rm[:, None] * stride_cm + rn[None, :] * stride_cn
        if WRITE_THROUGH:
            tl.store(cp, c, cache_modifier=".wt")
        else:
            tl.store(cp, c)


from iris.ops.matmul_reduce_scatter_fast import _fast_reduce_scatter_kernel, _get_config

M, N, K_global = 2048, 2880, 4096
K_local = K_global // world_size
M_local = M // world_size
dtype = torch.float16
warmup, iters = 100, 300

# Tuned GEMM config from sweep
BM, BN, BK, GM, GSMS, WARPS, STAGES, MFMA = 128, 256, 64, 4, 304, 8, 3, 32

A = shmem.zeros((M, K_local), dtype=dtype)
A.copy_(torch.randn(M, K_local, dtype=dtype, device=f"cuda:{rank}"))
B = torch.randn(K_local, N, dtype=dtype, device=f"cuda:{rank}")

C_regular = torch.zeros(M, N, dtype=dtype, device=f"cuda:{rank}")
C_symmetric = shmem.zeros((M, N), dtype=dtype)
C_out = torch.zeros(M_local, N, dtype=dtype, device=f"cuda:{rank}")

heap_bases = shmem.get_heap_bases()
cfg = _get_config(world_size, M_local)

s, e = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
kw = {"num_warps": WARPS, "num_stages": STAGES}
if getattr(torch.version, "hip", None):
    kw["matrix_instr_nonkdim"] = MFMA

if rank == 0:
    print(f"Two-kernel with tuned GEMM: M={M}, N={N}, K_local={K_local}, TP={world_size}")
    print(f"GEMM config: bm={BM} bn={BN} bk={BK} gm={GM} sms={GSMS} warps={WARPS} stages={STAGES} mfma={MFMA}")
    print()

def bench(fn):
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    s.record()
    for _ in range(iters):
        fn()
    e.record()
    torch.cuda.synchronize()
    return s.elapsed_time(e) / iters

# ===== GEMM only =====
if rank == 0:
    print("GEMM only:")

hipblas_ms = bench(lambda: torch.mm(A, B, out=C_regular))
if rank == 0:
    print(f"  hipBLASLt (regular mem):  {hipblas_ms:.4f}ms")

hipblas_sym_ms = bench(lambda: torch.mm(A, B, out=C_symmetric))
if rank == 0:
    print(f"  hipBLASLt (symmetric):    {hipblas_sym_ms:.4f}ms")

triton_reg_ms = bench(lambda: _tuned_gemm[(GSMS,)](
    A, B, C_regular, M, N, K_local,
    A.stride(0), A.stride(1), B.stride(0), B.stride(1),
    C_regular.stride(0), C_regular.stride(1),
    BM, BN, BK, GM, GSMS, K_local % BK == 0, False, **kw))
if rank == 0:
    print(f"  Triton tuned (regular):   {triton_reg_ms:.4f}ms ({hipblas_ms/triton_reg_ms:.2f}x vs hipBLASLt)")

triton_sym_ms = bench(lambda: _tuned_gemm[(GSMS,)](
    A, B, C_symmetric, M, N, K_local,
    A.stride(0), A.stride(1), B.stride(0), B.stride(1),
    C_symmetric.stride(0), C_symmetric.stride(1),
    BM, BN, BK, GM, GSMS, K_local % BK == 0, False, **kw))
if rank == 0:
    print(f"  Triton tuned (symmetric): {triton_sym_ms:.4f}ms")

triton_wt_ms = bench(lambda: _tuned_gemm[(GSMS,)](
    A, B, C_symmetric, M, N, K_local,
    A.stride(0), A.stride(1), B.stride(0), B.stride(1),
    C_symmetric.stride(0), C_symmetric.stride(1),
    BM, BN, BK, GM, GSMS, K_local % BK == 0, True, **kw))
if rank == 0:
    print(f"  Triton tuned (.wt sym):   {triton_wt_ms:.4f}ms")

# ===== Correctness =====
C_ref_full = torch.mm(A, B)
torch.cuda.synchronize()
_tuned_gemm[(GSMS,)](
    A, B, C_symmetric, M, N, K_local,
    A.stride(0), A.stride(1), B.stride(0), B.stride(1),
    C_symmetric.stride(0), C_symmetric.stride(1),
    BM, BN, BK, GM, GSMS, K_local % BK == 0, False, **kw)
torch.cuda.synchronize()
gemm_diff = torch.abs(C_symmetric - C_ref_full).max().item()
if rank == 0:
    print(f"  GEMM correctness: max_diff={gemm_diff:.6f} {'PASS' if gemm_diff < 1.0 else 'FAIL'}")

# ===== E2E =====
if rank == 0:
    print("\nEnd-to-end (GEMM + fast RS):")

# RCCL baseline
C_r = torch.empty(M, N, dtype=dtype, device=f"cuda:{rank}")
C_ro = torch.empty(M_local, N, dtype=dtype, device=f"cuda:{rank}")
rccl_ms = bench(lambda: (torch.mm(A, B, out=C_r),
                         dist.reduce_scatter_tensor(C_ro, C_r, op=dist.ReduceOp.SUM)))
if rank == 0:
    print(f"  torch.mm + RCCL RS:       {rccl_ms:.4f}ms")

shmem.barrier()

def rs_call(src):
    _fast_reduce_scatter_kernel[(cfg["num_sms"],)](
        src, C_out, M, N, M_local,
        src.stride(0), src.stride(1), C_out.stride(0), C_out.stride(1),
        heap_bases, rank, world_size,
        cfg["block_m"], cfg["block_n"], cfg["num_sms"],
        num_warps=cfg["num_warps"])

hip_rs_ms = bench(lambda: (torch.mm(A, B, out=C_symmetric), rs_call(C_symmetric)))
if rank == 0:
    print(f"  hipBLASLt + fast RS:      {hip_rs_ms:.4f}ms ({rccl_ms/hip_rs_ms:.2f}x)")

tri_rs_ms = bench(lambda: (
    _tuned_gemm[(GSMS,)](
        A, B, C_symmetric, M, N, K_local,
        A.stride(0), A.stride(1), B.stride(0), B.stride(1),
        C_symmetric.stride(0), C_symmetric.stride(1),
        BM, BN, BK, GM, GSMS, K_local % BK == 0, False, **kw),
    rs_call(C_symmetric)))
if rank == 0:
    print(f"  Triton tuned + fast RS:   {tri_rs_ms:.4f}ms ({rccl_ms/tri_rs_ms:.2f}x)")

# Correctness of E2E
ref = torch.empty(M_local, N, dtype=dtype, device=f"cuda:{rank}")
dist.reduce_scatter_tensor(ref, C_ref_full, op=dist.ReduceOp.SUM)
torch.cuda.synchronize()
e2e_diff = torch.abs(C_out - ref).max().item()
if rank == 0:
    print(f"  E2E correctness: max_diff={e2e_diff:.6f} {'PASS' if e2e_diff < 1.0 else 'FAIL'}")
    print()
    print(f"BEST: {min(hip_rs_ms, tri_rs_ms):.4f}ms = {rccl_ms/min(hip_rs_ms, tri_rs_ms):.2f}x over RCCL")

shmem.barrier()
dist.destroy_process_group()
