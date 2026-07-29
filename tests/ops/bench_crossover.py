#!/usr/bin/env python3
"""Find the M where fused GEMM+RS starts beating two-kernel.

Theory: fusion wins when the overlap window (min(GEMM, RS)) exceeds
the CU-split penalty. At small M, RS dominates (0.092 vs 0.031) and
splitting CUs hurts RS more than hiding GEMM helps.

At larger M, GEMM grows as O(M*N*K) while RS grows as O(M*N).
So GEMM/RS ratio grows linearly with K... but K is fixed here.
Actually both grow linearly with M, so the ratio is constant.

The real variable: at larger M, both phases have more tiles, so
CU-splitting is less harmful (better load balance, less tail effect).
"""

import os
import torch
import torch.distributed as dist
import iris

torch.cuda.set_device(int(os.environ.get("LOCAL_RANK", 0)))
dist.init_process_group(backend="nccl")

rank = dist.get_rank()
world_size = dist.get_world_size()
shmem = iris.iris(2**34)  # bigger heap for larger M

from iris.ops.matmul_reduce_scatter_xcd import matmul_reduce_scatter_xcd
from iris.ops.matmul_reduce_scatter_fast import _fast_reduce_scatter_kernel, _get_config

N, K_global = 2880, 4096
K_local = K_global // world_size
dtype = torch.float16
warmup, iters = 30, 100

heap_bases = shmem.get_heap_bases()
s, e = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)

if rank == 0:
    print(f"Crossover search: N={N}, K={K_global}, TP={world_size}")
    print(f"{'M':>7} | {'RCCL':>8} {'2-kernel':>9} {'fused':>8} | {'2k vs RCCL':>11} {'fused vs RCCL':>14} {'winner':>10}")
    print("-" * 85)

for M in [2048, 4096, 8192, 16384]:
    M_local = M // world_size
    if M_local % 128 != 0:
        continue

    try:
        A = shmem.zeros((M, K_local), dtype=dtype)
        A.copy_(torch.randn(M, K_local, dtype=dtype, device=f"cuda:{rank}") * 0.1)
        B = torch.randn(K_local, N, dtype=dtype, device=f"cuda:{rank}") * 0.1

        C_sym = shmem.zeros((M, N), dtype=dtype)
        C_out = torch.zeros(M_local, N, dtype=dtype, device=f"cuda:{rank}")
        C_r = torch.empty(M, N, dtype=dtype, device=f"cuda:{rank}")
        C_ro = torch.empty(M_local, N, dtype=dtype, device=f"cuda:{rank}")

        cfg = _get_config(world_size, M_local)
        shmem.barrier()

        # RCCL
        for _ in range(warmup):
            torch.mm(A, B, out=C_r)
            dist.reduce_scatter_tensor(C_ro, C_r, op=dist.ReduceOp.SUM)
        torch.cuda.synchronize()
        s.record()
        for _ in range(iters):
            torch.mm(A, B, out=C_r)
            dist.reduce_scatter_tensor(C_ro, C_r, op=dist.ReduceOp.SUM)
        e.record()
        torch.cuda.synchronize()
        rccl_ms = s.elapsed_time(e) / iters

        # Two-kernel
        for _ in range(warmup):
            torch.mm(A, B, out=C_sym)
            _fast_reduce_scatter_kernel[(cfg["num_sms"],)](
                C_sym, C_out, M, N, M_local,
                C_sym.stride(0), C_sym.stride(1), C_out.stride(0), C_out.stride(1),
                heap_bases, rank, world_size,
                cfg["block_m"], cfg["block_n"], cfg["num_sms"],
                num_warps=cfg["num_warps"])
        torch.cuda.synchronize()
        s.record()
        for _ in range(iters):
            torch.mm(A, B, out=C_sym)
            _fast_reduce_scatter_kernel[(cfg["num_sms"],)](
                C_sym, C_out, M, N, M_local,
                C_sym.stride(0), C_sym.stride(1), C_out.stride(0), C_out.stride(1),
                heap_bases, rank, world_size,
                cfg["block_m"], cfg["block_n"], cfg["num_sms"],
                num_warps=cfg["num_warps"])
        e.record()
        torch.cuda.synchronize()
        twok_ms = s.elapsed_time(e) / iters

        # Fused (sweep a few configs, take best)
        best_fused = 999.0
        for gemm_per_xcd in [28, 30, 32]:
            try:
                ws = None
                C_out.zero_()
                shmem.barrier()
                for _ in range(warmup):
                    ws = matmul_reduce_scatter_xcd(shmem, C_out, A, B,
                        gemm_sms_per_xcd=gemm_per_xcd, workspace=ws)
                torch.cuda.synchronize()
                s.record()
                for _ in range(iters):
                    ws = matmul_reduce_scatter_xcd(shmem, C_out, A, B,
                        gemm_sms_per_xcd=gemm_per_xcd, workspace=ws)
                e.record()
                torch.cuda.synchronize()
                ms = s.elapsed_time(e) / iters
                if ms < best_fused:
                    best_fused = ms
            except Exception:
                continue

        if rank == 0:
            twok_sp = rccl_ms / twok_ms
            fused_sp = rccl_ms / best_fused if best_fused < 999 else 0
            winner = "2-kernel" if twok_ms < best_fused else "FUSED"
            print(f"{M:7d} | {rccl_ms:7.3f}ms {twok_ms:8.3f}ms {best_fused:7.3f}ms | "
                  f"{twok_sp:10.2f}x {fused_sp:13.2f}x {winner:>10}")

        del A, B, C_sym, C_out, C_r, C_ro
        torch.cuda.empty_cache()
        shmem.barrier()

    except Exception as ex:
        if rank == 0:
            print(f"{M:7d} | ERROR: {str(ex)[:60]}")

shmem.barrier()
dist.destroy_process_group()
