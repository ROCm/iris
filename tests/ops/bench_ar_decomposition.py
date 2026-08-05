#!/usr/bin/env python3
"""Why does every fused GEMM+AR variant lose? Decompose it.

The HBM-buffer two-shot moves 4.6x fewer bytes than one-shot and is still
slower (0.44x vs 0.48x at M=2048). If cutting traffic 4.6x changes nothing,
traffic was never the bottleneck. This isolates what is.

Measures, at each M:
  1. hipBLASLt GEMM alone                (torch.mm, all CUs)
  2. naive Triton GEMM alone, N CUs      (exactly the GEMM inside our fused
                                          kernel, swept over CU count)
  3. one-shot AR alone                   (the comm we currently pay)
  4. RS+AG two-shot AR alone             (the comm the HBM-buffer version pays)

Then: GEMM tax  = TritonGEMM(gemm_cus) - hipBLASLt(all cus)
      comm gain = one_shot - two_shot
Fusion wins only if comm_gain > GEMM tax. Print both.
"""

import argparse

import torch
import torch.distributed as dist
import torch.multiprocessing as mp
import triton
import triton.language as tl

import iris

N_GLOBAL = 2880
K_GLOBAL = 4096
WARMUP, ITERS = 20, 50


@triton.jit
def _naive_gemm_kernel(
    a_ptr, b_ptr, c_ptr, M, N, K,
    stride_am, stride_ak, stride_bk, stride_bn, stride_cm, stride_cn,
    BLOCK_SIZE_M: tl.constexpr, BLOCK_SIZE_N: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr, NUM_SMS: tl.constexpr,
):
    """Byte-for-byte the GEMM pool of the fused kernel, on its own."""
    pid = tl.program_id(0)
    num_n_tiles = tl.cdiv(N, BLOCK_SIZE_N)
    total_tiles = tl.cdiv(M, BLOCK_SIZE_M) * num_n_tiles

    for tile_id in range(pid, total_tiles, NUM_SMS):
        pid_m = tile_id // num_n_tiles
        pid_n = tile_id % num_n_tiles
        rm = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
        rm = tl.max_contiguous(tl.multiple_of(rm, BLOCK_SIZE_M), BLOCK_SIZE_M)
        rn = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
        rn = tl.max_contiguous(tl.multiple_of(rn % N, BLOCK_SIZE_N), BLOCK_SIZE_N)
        rk = tl.arange(0, BLOCK_SIZE_K)

        a_ptrs = a_ptr + rm[:, None] * stride_am + rk[None, :] * stride_ak
        b_ptrs = b_ptr + rk[:, None] * stride_bk + rn[None, :] * stride_bn
        acc = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)
        for k in range(0, tl.cdiv(K, BLOCK_SIZE_K)):
            k_rem = K - k * BLOCK_SIZE_K
            a = tl.load(a_ptrs, mask=rk[None, :] < k_rem, other=0.0)
            b = tl.load(b_ptrs, mask=rk[:, None] < k_rem, other=0.0)
            acc += tl.dot(a, b)
            a_ptrs += BLOCK_SIZE_K * stride_ak
            b_ptrs += BLOCK_SIZE_K * stride_bk
        off = rm[:, None] * stride_cm + rn[None, :] * stride_cn
        tl.store(c_ptr + off, acc.to(c_ptr.type.element_ty),
                 mask=(rm[:, None] < M) & (rn[None, :] < N))


def bench(fn, pre=None):
    s = torch.cuda.Event(enable_timing=True)
    e = torch.cuda.Event(enable_timing=True)
    for _ in range(WARMUP):
        if pre:
            pre()
        fn()
    torch.cuda.synchronize()
    s.record()
    for _ in range(ITERS):
        if pre:
            pre()
        fn()
    e.record()
    torch.cuda.synchronize()
    return s.elapsed_time(e) / ITERS


def _worker(local_rank, world_size, init_url):
    dist.init_process_group(
        backend="nccl", init_method=init_url, world_size=world_size,
        rank=local_rank, device_id=torch.device(f"cuda:{local_rank}"))
    shmem = iris.iris(1 << 33)
    rank = shmem.get_rank()
    cu_count = torch.cuda.get_device_properties(rank).multi_processor_count

    from iris.ops.all_reduce_fast import one_shot_all_reduce, two_shot_all_reduce

    dtype = torch.float16
    K_local = K_GLOBAL // world_size

    if rank == 0:
        print(f"\nGEMM+AR decomposition   TP={world_size}  CUs={cu_count}  "
              f"N={N_GLOBAL} K={K_GLOBAL} fp16")

    for M in [512, 2048]:
        A = shmem.zeros((M, K_local), device="cuda", dtype=dtype)
        A.copy_(torch.randn(M, K_local, dtype=dtype, device=f"cuda:{rank}") * 0.1)
        B = torch.randn(K_local, N_GLOBAL, dtype=dtype, device=f"cuda:{rank}") * 0.1
        C = shmem.zeros((M, N_GLOBAL), device="cuda", dtype=dtype)
        Cl = torch.zeros(M, N_GLOBAL, dtype=dtype, device=f"cuda:{rank}")

        ref = torch.mm(A, B)
        gemm_ref = ref.clone()
        dist.all_reduce(ref, op=dist.ReduceOp.SUM)
        torch.cuda.synchronize()

        if rank == 0:
            print(f"\n=== M={M} ===")

        hipblas = bench(lambda: torch.mm(A, B, out=C))
        if rank == 0:
            print(f"  hipBLASLt GEMM (all {cu_count} CUs)     {hipblas:.4f}ms")

        # naive Triton GEMM -- the one fusion forces on us
        if rank == 0:
            print(f"  naive Triton GEMM (fused kernel's GEMM pool):")
        best_tri = {}
        for nsms in [128, 160, 192, 224, cu_count]:
            best = 1e9
            for bm in [64, 128]:
                if bm > M:
                    continue
                for bn in [64, 128]:
                    try:
                        Ct = torch.zeros(M, N_GLOBAL, dtype=dtype,
                                         device=f"cuda:{rank}")
                        _naive_gemm_kernel[(nsms,)](
                            A, B, Ct, M, N_GLOBAL, K_local,
                            A.stride(0), A.stride(1), B.stride(0), B.stride(1),
                            Ct.stride(0), Ct.stride(1),
                            BLOCK_SIZE_M=bm, BLOCK_SIZE_N=bn, BLOCK_SIZE_K=64,
                            NUM_SMS=nsms, num_warps=8, matrix_instr_nonkdim=32)
                        torch.cuda.synchronize()
                        if torch.abs(Ct - gemm_ref).max().item() > 2.0:
                            continue
                        ms = bench(lambda bm=bm, bn=bn, nsms=nsms, Ct=Ct:
                                   _naive_gemm_kernel[(nsms,)](
                                       A, B, Ct, M, N_GLOBAL, K_local,
                                       A.stride(0), A.stride(1),
                                       B.stride(0), B.stride(1),
                                       Ct.stride(0), Ct.stride(1),
                                       BLOCK_SIZE_M=bm, BLOCK_SIZE_N=bn,
                                       BLOCK_SIZE_K=64, NUM_SMS=nsms,
                                       num_warps=8, matrix_instr_nonkdim=32))
                        best = min(best, ms)
                    except Exception:
                        continue
            if best < 1e9:
                best_tri[nsms] = best
                if rank == 0:
                    print(f"    {nsms:3d} CUs   {best:.4f}ms   "
                          f"{best/hipblas:.2f}x hipBLASLt")

        # comm alone
        torch.mm(A, B, out=C)
        shmem.barrier()
        os_ms = bench(lambda: one_shot_all_reduce(shmem, Cl, C))
        torch.cuda.synchronize()
        d1 = torch.abs(Cl - ref).max().item()

        Cl2 = torch.zeros(M, N_GLOBAL, dtype=dtype, device=f"cuda:{rank}")
        scratch = None
        torch.mm(A, B, out=C)
        shmem.barrier()
        scratch = two_shot_all_reduce(shmem, Cl2, C, scratch=scratch)
        torch.cuda.synchronize()
        d2 = torch.abs(Cl2 - ref).max().item()
        ts_ms = bench(lambda: two_shot_all_reduce(shmem, Cl2, C, scratch=scratch))

        if rank == 0:
            print(f"  one-shot AR alone  ({world_size:.2f}*MN traffic)   "
                  f"{os_ms:.4f}ms  {'PASS' if d1 < 2.0 else 'FAIL'}")
            print(f"  two-shot AR alone  ({2*(world_size-1)/world_size:.2f}*MN)   "
                  f"{ts_ms:.4f}ms  {'PASS' if d2 < 2.0 else 'FAIL'}")

            gemm_tax = best_tri.get(192, float('nan')) - hipblas
            comm_gain = os_ms - ts_ms
            print()
            print(f"  GEMM tax  (Triton@192CU - hipBLASLt@{cu_count}CU) = "
                  f"{gemm_tax:+.4f}ms")
            print(f"  comm gain (one_shot - two_shot)                = "
                  f"{comm_gain:+.4f}ms")
            print(f"  --> fusion needs comm_gain > GEMM tax: "
                  f"{'YES' if comm_gain > gemm_tax else 'NO'}")

        del A, B, C, Cl, Cl2
        torch.cuda.empty_cache()
        shmem.barrier()

    shmem.barrier()
    dist.destroy_process_group()


def main():
    p = argparse.ArgumentParser()
    p.add_argument("-r", "--num_ranks", type=int, default=8)
    a = p.parse_args()
    mp.spawn(fn=_worker, args=(a.num_ranks, "tcp://127.0.0.1:29515"),
             nprocs=a.num_ranks, join=True)


if __name__ == "__main__":
    main()
