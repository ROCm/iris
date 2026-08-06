#!/usr/bin/env python3
"""GEMM+AllReduce across the full iris fusion taxonomy.

Patterns (docs/conceptual/taxonomy.md):
  P1  unfused bulk synchronous     torch.mm + dist.all_reduce        (baseline)
  P1b unfused bulk synchronous     torch.mm + our one-shot AR        (two-kernel)
  P3a fused sequential             ex08  atomic push
  P3b fused ring                   ex15  ring-based
  P4  fused WG specialization      ex09  one-shot pull, gemm_sms split

Each fused pattern gets its GEMM knobs tuned (BLK_M/N/K, mfma, warps) and its
comm knobs tuned (gemm_sms = the CU split).  mfma=32 was worth 2.3x on the
GEMM+RS study -- the examples hardcode 16, so we sweep it.

Emits a JSON blob per (pattern, M, config) for the tradeoff writeup.
"""

import argparse
import json
import math
import os
import sys
import time

import torch
import torch.distributed as dist
import torch.multiprocessing as mp
import triton

import iris

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
EX = os.path.join(ROOT, "examples")

N_GLOBAL = 2880
K_GLOBAL = 4096
M_LIST = [32, 128, 512, 2048]
WARMUP, ITERS = 20, 50


def _import_wrapper(exdir, alias):
    """Import an example's matmul_wrapper under a private module name."""
    import importlib.util

    path = os.path.join(EX, exdir, "matmul_wrapper.py")
    sys.path.insert(0, os.path.join(EX, exdir))
    spec = importlib.util.spec_from_file_location(alias, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[alias] = mod
    spec.loader.exec_module(mod)
    sys.path.pop(0)
    return mod.matmul


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


def _worker(local_rank, world_size, init_url, outfile):
    dist.init_process_group(
        backend="nccl",
        init_method=init_url,
        world_size=world_size,
        rank=local_rank,
        device_id=torch.device(f"cuda:{local_rank}"),
    )
    shmem = iris.iris(1 << 33)
    rank = shmem.get_rank()
    heap = shmem.get_heap_bases()
    cu_count = torch.cuda.get_device_properties(rank).multi_processor_count
    arch = torch.cuda.get_device_properties(rank).gcnArchName.split(":")[0]

    from iris.ops.all_reduce_fast import one_shot_all_reduce

    mm_atomic = _import_wrapper("08_gemm_all_reduce_atomics", "_ex08")
    mm_oneshot = _import_wrapper("09_gemm_one_shot_all_reduce", "_ex09")
    mm_ring = _import_wrapper("15_gemm_all_reduce_ring_based", "_ex15")

    dtype = torch.float16
    K_local = K_GLOBAL // world_size
    results = []

    def log(**kw):
        kw["world_size"] = world_size
        results.append(kw)
        if rank == 0:
            tag = f"{kw['pattern']:<28} M={kw['M']:<5}"
            if kw.get("ok"):
                print(f"  {tag} {kw['ms']:.4f}ms  {kw['speedup']:.2f}x  {kw.get('cfg','')}", flush=True)
            else:
                print(f"  {tag} {kw.get('why','FAIL')}  {kw.get('cfg','')}", flush=True)

    if rank == 0:
        print(f"\nGEMM+AllReduce taxonomy sweep")
        print(f"  arch={arch} CUs={cu_count} TP={world_size} N={N_GLOBAL} K={K_GLOBAL} fp16\n")

    for M in M_LIST:
        A = shmem.randn(M, K_GLOBAL, device="cuda", dtype=dtype)
        Bfull = shmem.randn(N_GLOBAL, K_GLOBAL, device="cuda", dtype=dtype).T
        start = rank * K_local
        a = A[:, start : start + K_local].contiguous()
        b = Bfull[start : start + K_local, :].contiguous()

        # shmem-resident copies for the fused kernels
        a_s = shmem.zeros((M, K_local), device="cuda", dtype=dtype)
        a_s.copy_(a)
        b_s = shmem.zeros((K_local, N_GLOBAL), device="cuda", dtype=dtype)
        b_s.copy_(b)

        ref = torch.mm(a, b)
        dist.all_reduce(ref, op=dist.ReduceOp.SUM)
        torch.cuda.synchronize()
        tol = 2.0  # fp16 accumulate over K=4096

        if rank == 0:
            print(f"--- M={M} ---", flush=True)

        # ---------- P1: unfused bulk synchronous (torch) ----------
        Ct = torch.empty(M, N_GLOBAL, dtype=dtype, device=f"cuda:{rank}")
        base_ms = bench(lambda: (torch.mm(a, b, out=Ct), dist.all_reduce(Ct, op=dist.ReduceOp.SUM)))
        log(pattern="P1 torch bulk-sync", M=M, ms=base_ms, speedup=1.0, ok=True, cfg="")

        # ---------- P1b: unfused bulk synchronous (ours) ----------
        Cs = shmem.zeros((M, N_GLOBAL), device="cuda", dtype=dtype)
        Co = torch.zeros(M, N_GLOBAL, dtype=dtype, device=f"cuda:{rank}")
        shmem.barrier()
        torch.mm(a, b, out=Cs)
        one_shot_all_reduce(shmem, Co, Cs)
        torch.cuda.synchronize()
        d = torch.abs(Co - ref).max().item()
        if d < tol:
            ms = bench(lambda: (torch.mm(a, b, out=Cs), one_shot_all_reduce(shmem, Co, Cs)))
            log(pattern="P1b two-kernel one-shot", M=M, ms=ms, speedup=base_ms / ms, ok=True, cfg="")
        else:
            log(pattern="P1b two-kernel one-shot", M=M, ok=False, why=f"diff={d:.3f}")

        # ---------- P3a: fused sequential, atomic push (ex08) ----------
        best = (1e9, None)
        for blk_m in [32, 64, 128, 256]:
            if blk_m > M:
                continue
            for blk_n in [64, 128]:
                for mfma in [16, 32]:
                    for nsms in [cu_count, 256, 128]:
                        try:
                            gC = shmem.zeros((M, N_GLOBAL), device="cuda", dtype=dtype)
                            lC = shmem.zeros((M, N_GLOBAL), device="cuda", dtype=dtype)
                            shmem.barrier()
                            gC.zero_()
                            shmem.barrier()
                            mm_atomic.apply(
                                a_s, b_s, lC, gC, None, rank, world_size, nsms,
                                blk_m, blk_n, 64, 4, 1, mfma, 8, heap, arch,
                            )
                            torch.cuda.synchronize()
                            shmem.barrier()
                            dd = torch.abs(gC - ref).max().item()
                            if dd > tol:
                                continue

                            def _pre(gC=gC):
                                gC.zero_()

                            def _run(a_s=a_s, b_s=b_s, lC=lC, gC=gC, blk_m=blk_m,
                                     blk_n=blk_n, mfma=mfma, nsms=nsms):
                                mm_atomic.apply(
                                    a_s, b_s, lC, gC, None, rank, world_size, nsms,
                                    blk_m, blk_n, 64, 4, 1, mfma, 8, heap, arch,
                                )

                            ms = bench(_run, pre=_pre)
                            if ms < best[0]:
                                best = (ms, f"bm={blk_m} bn={blk_n} mfma={mfma} sms={nsms}")
                        except Exception:
                            continue
        if best[1]:
            log(pattern="P3a fused-seq atomic", M=M, ms=best[0], speedup=base_ms / best[0], ok=True, cfg=best[1])
        else:
            log(pattern="P3a fused-seq atomic", M=M, ok=False, why="no valid cfg")

        # ---------- P3c: fused sequential, one-shot pull (ex09) ----------
        best = (1e9, None)
        for blk_m in [32, 64, 128, 256]:
            if blk_m > M:
                continue
            for blk_n in [64, 128]:
                for mfma in [16, 32]:
                    for gemm_sms in [128, 192, 256]:
                        if gemm_sms > cu_count:
                            continue
                        try:
                            tbm = triton.cdiv(M, blk_m)
                            tbn = triton.cdiv(N_GLOBAL, blk_n)
                            ntiles = tbm * tbn
                            gC = shmem.zeros((M, N_GLOBAL), device="cuda", dtype=dtype)
                            lC = shmem.zeros((M, N_GLOBAL), device="cuda", dtype=dtype)
                            tc = shmem.zeros((ntiles,), device="cuda", dtype=torch.int32)
                            lk = shmem.zeros((gemm_sms,), device="cuda", dtype=torch.int32)
                            P = shmem.zeros((gemm_sms, blk_m * blk_n), device="cuda", dtype=torch.float32)
                            shmem.barrier()
                            gC.zero_(); tc.zero_(); lk.zero_()
                            shmem.barrier()
                            mm_oneshot.apply(
                                a_s, b_s, lC, gC, None, P, lk, tc, rank, world_size,
                                gemm_sms, blk_m, blk_n, 64, 1, True, 1, 8, 0, mfma, 1,
                                heap, cu_count,
                            )
                            torch.cuda.synchronize()
                            shmem.barrier()
                            dd = torch.abs(gC - ref).max().item()
                            if dd > tol:
                                continue

                            def _pre(gC=gC):
                                gC.zero_()

                            def _run(blk_m=blk_m, blk_n=blk_n, mfma=mfma, gC=gC, lC=lC,
                                     P=P, lk=lk, tc=tc):
                                mm_oneshot.apply(
                                    a_s, b_s, lC, gC, None, P, lk, tc, rank, world_size,
                                    gemm_sms, blk_m, blk_n, 64, 1, True, 1, 8, 0, mfma, 1,
                                    heap, cu_count,
                                )

                            ms = bench(_run, pre=_pre)
                            if ms < best[0]:
                                best = (ms, f"bm={blk_m} bn={blk_n} mfma={mfma} gemm_sms={gemm_sms}")
                        except Exception:
                            continue
        if best[1]:
            log(pattern="P3c fused-seq one-shot", M=M, ms=best[0], speedup=base_ms / best[0], ok=True, cfg=best[1])
        else:
            log(pattern="P3c fused-seq one-shot", M=M, ok=False, why="no valid cfg")

        # ---------- P3b: fused ring (ex15) ----------
        best = (1e9, None)
        for blk_m in [32, 64, 128, 256]:
            if blk_m > M:
                continue
            for blk_n in [64, 128]:
                for mfma in [16, 32]:
                    for nsms in [cu_count, 256]:
                        try:
                            tbm = triton.cdiv(M, blk_m)
                            tbn = triton.cdiv(N_GLOBAL, blk_n)
                            ntiles = tbm * tbn
                            C = shmem.zeros((M, N_GLOBAL), device="cuda", dtype=dtype)
                            lk = shmem.zeros((ntiles,), device="cuda", dtype=torch.int32)
                            rb = shmem.zeros((M, N_GLOBAL), device="cuda", dtype=torch.float32)
                            shmem.barrier()
                            lk.zero_(); rb.zero_()
                            shmem.barrier()
                            mm_ring.apply(
                                a_s, b_s, rb, None, lk, rank, world_size, nsms,
                                blk_m, blk_n, 64, 1, 1, mfma, 8, heap, arch,
                            )
                            torch.cuda.synchronize()
                            shmem.barrier()
                            dd = torch.abs(rb.to(dtype) - ref).max().item()
                            if dd > tol:
                                continue

                            def _pre(lk=lk, rb=rb):
                                lk.zero_(); rb.zero_()

                            def _run(blk_m=blk_m, blk_n=blk_n, mfma=mfma, nsms=nsms, rb=rb, lk=lk):
                                mm_ring.apply(
                                    a_s, b_s, rb, None, lk, rank, world_size, nsms,
                                    blk_m, blk_n, 64, 1, 1, mfma, 8, heap, arch,
                                )

                            ms = bench(_run, pre=_pre)
                            if ms < best[0]:
                                best = (ms, f"bm={blk_m} bn={blk_n} mfma={mfma} sms={nsms}")
                        except Exception:
                            continue
        if best[1]:
            log(pattern="P3b fused ring", M=M, ms=best[0], speedup=base_ms / best[0], ok=True, cfg=best[1])
        else:
            log(pattern="P3b fused ring", M=M, ok=False, why="no valid cfg")

        if rank == 0:
            print(flush=True)
        del A, Bfull, a, b, a_s, b_s, Cs, Co
        torch.cuda.empty_cache()
        shmem.barrier()

    if rank == 0 and outfile:
        with open(outfile, "w") as f:
            json.dump(results, f, indent=2)
        print(f"wrote {outfile}")

    shmem.barrier()
    dist.destroy_process_group()


def _free_port(explicit=None):
    """Hardcoded TCPStore ports collide when two people share a node.
    Bind :0 and let the OS hand us a free one unless told otherwise."""
    if explicit:
        return explicit
    import socket

    s = socket.socket()
    s.bind(("127.0.0.1", 0))
    port = s.getsockname()[1]
    s.close()
    return port


_PORT = None


def main():
    global _PORT
    p = argparse.ArgumentParser()
    p.add_argument("-r", "--num_ranks", type=int, default=8)
    p.add_argument("--port", type=int, default=None,
                   help="TCPStore port; default picks a free one")
    p.add_argument("-o", "--output", type=str, default="ar_taxonomy.json")
    args = p.parse_args()
    _PORT = args.port
    mp.spawn(
        fn=_worker,
        args=(args.num_ranks, f"tcp://127.0.0.1:{_free_port(_PORT)}", args.output),
        nprocs=args.num_ranks,
        join=True,
    )


if __name__ == "__main__":
    main()
