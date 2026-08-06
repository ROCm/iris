#!/usr/bin/env python3
"""Roofline for GEMM+AllReduce: every algorithm, over the region where it is valid.

Each algorithm moves a DIFFERENT number of bytes for the same AllReduce, so
comparing milliseconds hides why one wins. Plotted as achieved bandwidth
against the XGMI line rate, the picture is:

  one-shot   ws*M*N per rank        runs near line rate, moves 4.6x too much
  two-shot   2(ws-1)/ws*M*N         moves the right bytes, harder to saturate
  RCCL       2(ws-1)/ws*M*N         same bytes as two-shot, reference efficiency

The regions where each is *valid* differ too, and that matters:
  * two-shot needs M/ws to be a sensible tile -- at ws=8 it cannot run at M=32
  * one-shot has no such constraint but its traffic grows with ws

So the sweep records, per M: time, algorithmic bytes, achieved GB/s, and the
fraction of line rate. Everything correctness-gated before it is timed --
a wrong kernel is usually a fast one.
"""

import argparse
import json

import torch
import torch.distributed as dist
import torch.multiprocessing as mp
import triton

import iris

N_GLOBAL = 2880
K_GLOBAL = 4096
LINE_GBS = 448.0
WARMUP, ITERS = 20, 60
TOL = 0.05

M_LIST = [32, 64, 128, 256, 512, 1024, 2048, 4096]


def bench(fn):
    s = torch.cuda.Event(enable_timing=True)
    e = torch.cuda.Event(enable_timing=True)
    for _ in range(WARMUP):
        fn()
    torch.cuda.synchronize()
    s.record()
    for _ in range(ITERS):
        fn()
    e.record()
    torch.cuda.synchronize()
    return s.elapsed_time(e) / ITERS


def _worker(local_rank, world_size, init_url, outfile):
    dist.init_process_group(
        backend="nccl", init_method=init_url, world_size=world_size,
        rank=local_rank, device_id=torch.device(f"cuda:{local_rank}"))
    shmem = iris.iris(1 << 34)
    rank = shmem.get_rank()
    cu = torch.cuda.get_device_properties(rank).multi_processor_count

    from iris.ops.all_reduce_fast import one_shot_all_reduce
    from iris.ops.matmul_all_reduce_hbm_buffer import (
        matmul_all_reduce_hbm_buffer,
        matmul_all_reduce_hbm_buffer_preamble,
    )

    dtype = torch.float16
    K_local = K_GLOBAL // world_size
    rows = []

    if rank == 0:
        print(f"\nAllReduce roofline  ws={world_size} CUs={cu} N={N_GLOBAL} "
              f"K={K_GLOBAL} fp16  line={LINE_GBS:.0f} GB/s")
        print(f"{'M':>6} {'algo':<22} {'ms':>9} {'movedMB':>9} {'GB/s':>8} "
              f"{'%line':>6}  gated")

    def rec(M, algo, ms, mb, ok):
        gbs = (mb / 1e3) / (ms / 1e3) if ms > 0 else 0.0
        rows.append(dict(M=M, algo=algo, ms=ms, moved_mb=mb, gbs=gbs,
                         pct_line=100 * gbs / LINE_GBS, ok=bool(ok),
                         world_size=world_size))
        if rank == 0:
            print(f"{M:6d} {algo:<22} {ms:9.4f} {mb:9.1f} {gbs:8.1f} "
                  f"{100*gbs/LINE_GBS:5.0f}%  {'PASS' if ok else 'FAIL'}",
                  flush=True)

    for M in M_LIST:
        A = shmem.zeros((M, K_local), device="cuda", dtype=dtype)
        A.copy_(torch.randn(M, K_local, dtype=dtype, device=f"cuda:{rank}") * 0.1)
        B = torch.randn(K_local, N_GLOBAL, dtype=dtype, device=f"cuda:{rank}") * 0.1
        ref = torch.mm(A, B)
        dist.all_reduce(ref, op=dist.ReduceOp.SUM)
        torch.cuda.synchronize()

        mn_mb = M * N_GLOBAL * 2 / 1e6
        one_shot_mb = world_size * mn_mb
        two_shot_mb = 2 * (world_size - 1) / world_size * mn_mb

        # ---- RCCL AllReduce alone ----
        Cr = torch.mm(A, B)
        rccl = bench(lambda: dist.all_reduce(Cr, op=dist.ReduceOp.SUM))
        rec(M, "RCCL all_reduce", rccl, two_shot_mb, True)

        # ---- one-shot pull AR alone ----
        Cs = shmem.zeros((M, N_GLOBAL), device="cuda", dtype=dtype)
        Co = torch.zeros(M, N_GLOBAL, dtype=dtype, device=f"cuda:{rank}")
        torch.mm(A, B, out=Cs)
        shmem.barrier()
        one_shot_all_reduce(shmem, Co, Cs)
        torch.cuda.synchronize()
        ok = torch.abs(Co - ref).max().item() < TOL
        os_ms = bench(lambda: one_shot_all_reduce(shmem, Co, Cs))
        rec(M, "iris one-shot", os_ms, one_shot_mb, ok)
        shmem.barrier()

        # ---- GEMM, for the E2E rows ----
        Cg = torch.empty(M, N_GLOBAL, dtype=dtype, device=f"cuda:{rank}")
        gemm = bench(lambda: torch.mm(A, B, out=Cg))
        rec(M, "hipBLASLt GEMM", gemm, 0.0, True)

        # ---- fused two-shot, over its valid region only ----
        best = (1e9, None)
        for bm in [64, 128]:
            nmt = triton.cdiv(M, bm)
            # two-shot shards M-tiles across ranks: needs M/ws to be a tile
            if nmt % world_size != 0:
                continue
            for bn in [128]:
                for tpf in [1, 2]:
                    if (nmt // world_size) * triton.cdiv(N_GLOBAL, bn) % tpf:
                        continue
                    try:
                        wsx = matmul_all_reduce_hbm_buffer_preamble(
                            shmem, M, N_GLOBAL, dtype, bm, bn)
                        shmem.barrier()
                        out = torch.zeros(M, N_GLOBAL, dtype=dtype,
                                          device=f"cuda:{rank}")
                        good = True
                        for _ in range(3):
                            out.zero_()
                            matmul_all_reduce_hbm_buffer(
                                shmem, out, A, B, workspace=wsx, block_m=bm,
                                block_n=bn, block_k=64, mfma=16,
                                tiles_per_flag=tpf, num_gemm_sms=192,
                                num_rs_sms=32, num_ag_sms=32)
                            torch.cuda.synchronize()
                            if torch.abs(out - ref).max().item() > TOL:
                                good = False
                                break
                        shmem.barrier()
                        if not good:
                            continue
                        ms = bench(lambda wsx=wsx, out=out, bm=bm, bn=bn, tpf=tpf:
                                   matmul_all_reduce_hbm_buffer(
                                       shmem, out, A, B, workspace=wsx,
                                       block_m=bm, block_n=bn, block_k=64,
                                       mfma=16, tiles_per_flag=tpf,
                                       num_gemm_sms=192, num_rs_sms=32,
                                       num_ag_sms=32))
                        if ms < best[0]:
                            best = (ms, f"bm{bm}/bn{bn}/tpf{tpf}")
                    except Exception:
                        continue
        if best[1]:
            # comm-only share, backing the GEMM out of the fused time
            rec(M, "fused two-shot (E2E)", best[0], two_shot_mb, True)
            rec(M, "fused two-shot (comm)", max(best[0] - gemm, 1e-6),
                two_shot_mb, True)
        elif rank == 0:
            print(f"{M:6d} {'fused two-shot':<22} {'--':>9}  "
                  f"invalid: M/ws too small to tile", flush=True)

        # ---- E2E references ----
        Ct = torch.empty(M, N_GLOBAL, dtype=dtype, device=f"cuda:{rank}")
        torch_e2e = bench(lambda: (torch.mm(A, B, out=Ct),
                                   dist.all_reduce(Ct, op=dist.ReduceOp.SUM)))
        rec(M, "torch mm+AR (E2E)", torch_e2e, two_shot_mb, True)
        if ok:
            tk = bench(lambda: (torch.mm(A, B, out=Cs),
                                one_shot_all_reduce(shmem, Co, Cs)))
            rec(M, "two-kernel one-shot (E2E)", tk, one_shot_mb, True)

        del A, B, Cs, Co, Cg, Ct
        torch.cuda.empty_cache()
        shmem.barrier()

    if rank == 0 and outfile:
        with open(outfile, "w") as f:
            json.dump(rows, f, indent=2)
        print(f"\nwrote {outfile}")

    shmem.barrier()
    dist.destroy_process_group()


def _free_port(explicit=None):
    if explicit:
        return explicit
    import socket

    s = socket.socket()
    s.bind(("127.0.0.1", 0))
    port = s.getsockname()[1]
    s.close()
    return port


def main():
    p = argparse.ArgumentParser()
    p.add_argument("-r", "--num_ranks", type=int, default=8)
    p.add_argument("--port", type=int, default=None)
    p.add_argument("-o", "--output", default="roofline_ar.json")
    a = p.parse_args()
    mp.spawn(fn=_worker,
             args=(a.num_ranks, f"tcp://127.0.0.1:{_free_port(a.port)}", a.output),
             nprocs=a.num_ranks, join=True)


if __name__ == "__main__":
    main()
