#!/usr/bin/env python3
"""Tune Triton GEMM to close the gap with hipBLASLt.

Every fused variant pays the Triton GEMM tax (0.072ms vs 0.032ms hipBLASLt).
If we close that gap, fusion becomes viable.

Sweeps: tile sizes, MFMA instruction size (matrix_instr_nonkdim),
num_warps, num_stages, GROUP_SIZE_M.
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
def _tuned_gemm_kernel(
    A, B, C,
    M, N, K,
    stride_am, stride_ak, stride_bk, stride_bn, stride_cm, stride_cn,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
    GROUP_M: tl.constexpr, NUM_SMS: tl.constexpr, EVEN_K: tl.constexpr,
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
        tl.store(C + rm[:, None] * stride_cm + rn[None, :] * stride_cn, c)


M, N, K_global = 2048, 2880, 4096
K_local = K_global // world_size
dtype = torch.float16
warmup, iters = 50, 300

A = torch.randn(M, K_local, dtype=dtype, device=f"cuda:{rank}")
B = torch.randn(K_local, N, dtype=dtype, device=f"cuda:{rank}")
C = torch.zeros(M, N, dtype=dtype, device=f"cuda:{rank}")

s, e = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)

# hipBLASLt baseline
for _ in range(warmup):
    torch.mm(A, B, out=C)
torch.cuda.synchronize()
s.record()
for _ in range(iters):
    torch.mm(A, B, out=C)
e.record()
torch.cuda.synchronize()
hipblas_ms = s.elapsed_time(e) / iters

if rank == 0:
    print(f"Triton GEMM tuning: M={M}, N={N}, K_local={K_local}, TP={world_size}")
    print(f"hipBLASLt: {hipblas_ms:.4f}ms")
    print()
    print(f"{'bm':>4} {'bn':>4} {'bk':>4} {'gm':>3} {'sms':>4} {'w':>2} {'st':>3} {'mfma':>5} | {'ms':>8} {'vs hipBLASLt':>13}")
    print("-" * 75)

C_ref = torch.mm(A, B)
torch.cuda.synchronize()

best_ms = 999.0
best_cfg = None

configs = []
for bm in [64, 128, 256]:
    for bn in [64, 128, 256]:
        for bk in [64, 128]:
            for gm in [1, 4, 8]:
                for sms in [152, 228, 304]:
                    configs.append((bm, bn, bk, gm, sms))

for bm, bn, bk, gm, sms in configs:
    for warps in [4, 8]:
        for stages in [2, 3]:
            for mfma in [16, 32]:
                try:
                    C.zero_()
                    kwargs = {"num_warps": warps, "num_stages": stages}
                    if getattr(torch.version, "hip", None):
                        kwargs["matrix_instr_nonkdim"] = mfma

                    for _ in range(5):
                        _tuned_gemm_kernel[(sms,)](
                            A, B, C, M, N, K_local,
                            A.stride(0), A.stride(1), B.stride(0), B.stride(1),
                            C.stride(0), C.stride(1),
                            bm, bn, bk, gm, sms, K_local % bk == 0,
                            **kwargs)
                    torch.cuda.synchronize()

                    diff = torch.abs(C - C_ref).max().item()
                    if diff > 1.0:
                        continue

                    s.record()
                    for _ in range(iters):
                        _tuned_gemm_kernel[(sms,)](
                            A, B, C, M, N, K_local,
                            A.stride(0), A.stride(1), B.stride(0), B.stride(1),
                            C.stride(0), C.stride(1),
                            bm, bn, bk, gm, sms, K_local % bk == 0,
                            **kwargs)
                    e.record()
                    torch.cuda.synchronize()
                    ms = s.elapsed_time(e) / iters

                    if ms < best_ms:
                        best_ms = ms
                        best_cfg = (bm, bn, bk, gm, sms, warps, stages, mfma)
                        if rank == 0:
                            ratio = ms / hipblas_ms
                            print(f"{bm:4d} {bn:4d} {bk:4d} {gm:3d} {sms:4d} {warps:2d} {stages:3d} {mfma:5d} | {ms:8.4f} {ratio:12.2f}x  ***")
                except Exception:
                    continue

if rank == 0:
    print()
    print(f"hipBLASLt:   {hipblas_ms:.4f}ms")
    print(f"Best Triton: {best_ms:.4f}ms ({best_ms/hipblas_ms:.2f}x slower)")
    if best_cfg:
        print(f"  config: bm={best_cfg[0]} bn={best_cfg[1]} bk={best_cfg[2]} gm={best_cfg[3]} sms={best_cfg[4]} warps={best_cfg[5]} stages={best_cfg[6]} mfma={best_cfg[7]}")
    print()
    print(f"If Triton GEMM matched hipBLASLt, fused would be viable:")
    print(f"  fused best case = max(GEMM, RS) = max({best_ms:.3f}, 0.092) = {max(best_ms, 0.092):.3f}ms")
    print(f"  two-kernel = {hipblas_ms:.3f} + 0.092 + 0.006 = {hipblas_ms + 0.098:.3f}ms")

shmem.barrier()
dist.destroy_process_group()
