#!/usr/bin/env python3
"""GEMM+AllReduce roofline: one tuned, attributed number per variant.

Every prior table in this study reported wall time and a ratio to torch. That
cannot answer "is this fast because it overlaps, or because the collective got
better, or is it just less bad at something?" This harness separates those.

PER (variant, M, ws) WE EMIT
----------------------------
    T_gemm      tuned torch.mm (hipBLASLt) -- the compute reference
    T_comm      tuned standalone comm for THE SAME ALGORITHM the variant uses
    T_serial    T_gemm + T_comm            -- bulk synchronous, zero overlap
    T_ideal     max(T_gemm, T_comm)        -- comm perfectly hidden behind compute
    T_measured  what the variant actually does

    overlap_ratio = (T_serial - T_measured) / (T_serial - T_ideal)

        1.0  comm fully hidden -- the stated goal
        0.0  no overlap, identical to running them back to back
       <0    fusion is actively costing us
       >1    impossible from overlap alone; the variant also improved a
             component, and the table must say which

Two efficiency axes so we know which side binds:
    comm_pct_line   achieved GB/s over XGMI line rate
    gemm_pct_peak   achieved TFLOP/s over device peak

WHY THE REFERENCES MUST BE TUNED
--------------------------------
overlap_ratio is a ratio of differences against T_gemm and T_comm. An untuned
reference inflates T_serial and makes every variant look good. `num_warps` was
never swept on any variant in this study and is worth up to 4.3x, so the
references are tuned over it here before anything else runs.

ALGORITHM GAIN IS REPORTED SEPARATELY
-------------------------------------
A variant that switches one-shot -> two-shot must not book that as "overlap".
Each variant is scored against the comm reference for its own algorithm, and
the algorithm win is reported on its own axis:

    algorithm_gain = T_comm(one_shot) / T_comm(variant's algorithm)
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
WARMUP, ITERS = 20, 50

# MI355X (gfx950): XGMI line rate and dense FP16 matrix peak. Both are
# denominators only -- every absolute number in the output is measured.
LINE_GBS = 448.0
PEAK_TFLOPS = 2300.0


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


def _worker(local_rank, world_size, init_url, outfile, m_list):
    dist.init_process_group(
        backend="nccl", init_method=init_url, world_size=world_size,
        rank=local_rank, device_id=torch.device(f"cuda:{local_rank}"))
    shmem = iris.iris(1 << 33)
    rank = shmem.get_rank()
    cu = torch.cuda.get_device_properties(rank).multi_processor_count

    from iris.ops.all_reduce_fast import one_shot_all_reduce, two_shot_all_reduce
    from iris.ops.matmul_all_reduce_hbm_buffer import (
        matmul_all_reduce_hbm_buffer,
        matmul_all_reduce_hbm_buffer_preamble,
    )

    dtype = torch.float16
    K_local = K_GLOBAL // world_size
    rows = []

    def emit(**kw):
        kw.update(M=kw["M"], world_size=world_size)
        rows.append(kw)

    if rank == 0:
        print(f"\nGEMM+AR roofline   TP={world_size} CUs={cu} "
              f"N={N_GLOBAL} K={K_GLOBAL} fp16")
        print(f"  line rate {LINE_GBS:.0f} GB/s   peak {PEAK_TFLOPS:.0f} TFLOP/s\n")

    for M in m_list:
        A = shmem.zeros((M, K_local), device="cuda", dtype=dtype)
        A.copy_(torch.randn(M, K_local, dtype=dtype, device=f"cuda:{rank}") * 0.1)
        B = torch.randn(K_local, N_GLOBAL, dtype=dtype, device=f"cuda:{rank}") * 0.1
        ref = torch.mm(A, B)
        dist.all_reduce(ref, op=dist.ReduceOp.SUM)
        torch.cuda.synchronize()
        tol = 2.0

        # traffic each algorithm actually moves, per rank
        mn_bytes = M * N_GLOBAL * 2
        bytes_one = world_size * mn_bytes
        bytes_two = 2 * (world_size - 1) / world_size * mn_bytes
        gemm_flops = 2.0 * M * N_GLOBAL * K_local

        if rank == 0:
            print(f"{'='*78}\nM={M}   MN={mn_bytes/1e6:.1f}MB   "
                  f"one-shot moves {bytes_one/1e6:.1f}MB   "
                  f"two-shot moves {bytes_two/1e6:.1f}MB")

        # ---------------- REFERENCE 1: compute ----------------
        C = shmem.zeros((M, N_GLOBAL), device="cuda", dtype=dtype)
        T_gemm = bench(lambda: torch.mm(A, B, out=C))
        gemm_tflops = gemm_flops / (T_gemm * 1e-3) / 1e12
        if rank == 0:
            print(f"  ref GEMM  hipBLASLt        {T_gemm:.4f}ms  "
                  f"{gemm_tflops:7.1f} TFLOP/s  {100*gemm_tflops/PEAK_TFLOPS:4.1f}% peak")
        emit(kind="ref", name="gemm_hipblaslt", M=M, ms=T_gemm,
             tflops=gemm_tflops, pct_peak=100 * gemm_tflops / PEAK_TFLOPS)

        # ---------------- REFERENCE 2: comm, tuned over num_warps ----------
        # num_warps was never swept in this study and is worth up to 4.3x.
        # The references are tuned over it first; nothing downstream is valid
        # otherwise.
        torch.mm(A, B, out=C)
        shmem.barrier()
        Cl = torch.zeros(M, N_GLOBAL, dtype=dtype, device=f"cuda:{rank}")

        comm_ref = {}
        for algo, fn_ in (("one_shot", one_shot_all_reduce),
                          ("two_shot", two_shot_all_reduce)):
            best = (1e9, None)
            scratch = None
            for bm in [32, 64, 128, 256]:
                if algo == "two_shot" and (M // world_size) % bm:
                    continue
                if M % bm:
                    continue
                for bn in [64, 128]:
                    for sms in [32, 64, 128, 196, cu]:
                        for w in [1, 2, 4, 8, 16]:
                            try:
                                kw = dict(block_m=bm, block_n=bn,
                                          num_sms=sms, num_warps=w)
                                Cl.zero_()
                                if algo == "two_shot":
                                    scratch = fn_(shmem, Cl, C, scratch=scratch, **kw)
                                else:
                                    fn_(shmem, Cl, C, **kw)
                                torch.cuda.synchronize()
                                if torch.abs(Cl - ref).max().item() > tol:
                                    continue
                                if algo == "two_shot":
                                    ms = bench(lambda kw=kw, sc=scratch:
                                               fn_(shmem, Cl, C, scratch=sc, **kw))
                                else:
                                    ms = bench(lambda kw=kw: fn_(shmem, Cl, C, **kw))
                                if ms < best[0]:
                                    best = (ms, f"bm={bm} bn={bn} sms={sms} warps={w}")
                            except Exception:
                                continue
            nbytes = bytes_one if algo == "one_shot" else bytes_two
            if best[1]:
                gbs = nbytes / 1e9 / (best[0] * 1e-3)
                comm_ref[algo] = best[0]
                if rank == 0:
                    print(f"  ref comm  {algo:<10}       {best[0]:.4f}ms  "
                          f"{gbs:7.1f} GB/s  {100*gbs/LINE_GBS:4.1f}% line   {best[1]}")
                emit(kind="ref", name=f"comm_{algo}", M=M, ms=best[0],
                     gbs=gbs, pct_line=100 * gbs / LINE_GBS, cfg=best[1],
                     algo_bytes=nbytes)
            elif rank == 0:
                print(f"  ref comm  {algo:<10}       n/a")

        # RCCL comm reference
        Cr = torch.mm(A, B)
        T_rccl = bench(lambda: dist.all_reduce(Cr, op=dist.ReduceOp.SUM))
        rccl_gbs = bytes_two / 1e9 / (T_rccl * 1e-3)  # ring moves algo bytes
        comm_ref["rccl"] = T_rccl
        if rank == 0:
            print(f"  ref comm  rccl             {T_rccl:.4f}ms  "
                  f"{rccl_gbs:7.1f} GB/s  {100*rccl_gbs/LINE_GBS:4.1f}% line")
        emit(kind="ref", name="comm_rccl", M=M, ms=T_rccl, gbs=rccl_gbs,
             pct_line=100 * rccl_gbs / LINE_GBS, algo_bytes=bytes_two)

        # ---------------- VARIANTS ----------------
        if rank == 0:
            print(f"  {'-'*74}")
            print(f"  {'variant':<26} {'ms':>8} {'vs torch':>9} "
                  f"{'overlap':>8} {'algo gain':>10}  config")

        def score(name, T_meas, algo, cfg=""):
            """Attribute a variant against the comm reference for ITS algorithm."""
            T_c = comm_ref.get(algo)
            if T_c is None:
                return
            T_serial = T_gemm + T_c
            T_ideal = max(T_gemm, T_c)
            denom = T_serial - T_ideal
            ov = (T_serial - T_meas) / denom if denom > 1e-9 else float("nan")
            algo_gain = comm_ref["one_shot"] / T_c if comm_ref.get("one_shot") else float("nan")
            if rank == 0:
                print(f"  {name:<26} {T_meas:8.4f} {torch_ms/T_meas:8.2f}x "
                      f"{ov:8.2f} {algo_gain:9.2f}x  {cfg}")
            emit(kind="variant", name=name, M=M, ms=T_meas, algo=algo,
                 vs_torch=torch_ms / T_meas, overlap_ratio=ov,
                 algorithm_gain=algo_gain, T_serial=T_serial, T_ideal=T_ideal,
                 T_gemm=T_gemm, T_comm=T_c, cfg=cfg)

        # torch bulk-sync -- defines the 1.00x column
        Ct = torch.empty(M, N_GLOBAL, dtype=dtype, device=f"cuda:{rank}")
        torch_ms = bench(lambda: (torch.mm(A, B, out=Ct),
                                  dist.all_reduce(Ct, op=dist.ReduceOp.SUM)))
        if rank == 0:
            print(f"  {'P1 torch bulk-sync':<26} {torch_ms:8.4f} {1.0:8.2f}x "
                  f"{'--':>8} {'--':>10}")
        emit(kind="variant", name="P1 torch bulk-sync", M=M, ms=torch_ms,
             algo="rccl", vs_torch=1.0)

        # two-kernel one-shot, using the tuned comm reference config
        if "one_shot" in comm_ref:
            Cs = shmem.zeros((M, N_GLOBAL), device="cuda", dtype=dtype)
            Co = torch.zeros(M, N_GLOBAL, dtype=dtype, device=f"cuda:{rank}")
            shmem.barrier()
            t = bench(lambda: (torch.mm(A, B, out=Cs),
                               one_shot_all_reduce(shmem, Co, Cs)))
            score("P1b two-kernel one-shot", t, "one_shot", "auto-config")

        # fused HBM-buffer two-shot, tuned over split x warps x tpf
        best = (1e9, None)
        out = torch.zeros(M, N_GLOBAL, dtype=dtype, device=f"cuda:{rank}")
        ws_cache = {}
        for bm in [16, 32, 64, 128]:
            nmt = triton.cdiv(M, bm)
            if nmt % world_size:
                continue
            for bn in [64, 128]:
                for w in [1, 2, 4, 8, 16]:
                    for g, r_, a_ in [(192, 32, 32), (128, 64, 64),
                                      (96, 96, 64), (64, 96, 96)]:
                        for tpf in [1, 2, 4]:
                            try:
                                key = (bm, bn)
                                if key not in ws_cache:
                                    ws_cache[key] = matmul_all_reduce_hbm_buffer_preamble(
                                        shmem, M, N_GLOBAL, dtype, bm, bn)
                                    shmem.barrier()
                                wsp = ws_cache[key]
                                kw = dict(block_m=bm, block_n=bn, block_k=64,
                                          num_gemm_sms=g, num_rs_sms=r_,
                                          num_ag_sms=a_, num_warps=w,
                                          mfma=32, tiles_per_flag=tpf)
                                ok = True
                                for _ in range(2):
                                    out.zero_()
                                    matmul_all_reduce_hbm_buffer(
                                        shmem, out, A, B, workspace=wsp, **kw)
                                    torch.cuda.synchronize()
                                    if torch.abs(out - ref).max().item() > tol:
                                        ok = False
                                        break
                                shmem.barrier()
                                if not ok:
                                    continue
                                ms = bench(lambda wsp=wsp, kw=kw:
                                           matmul_all_reduce_hbm_buffer(
                                               shmem, out, A, B, workspace=wsp, **kw))
                                if ms < best[0]:
                                    best = (ms, f"bm={bm} bn={bn} warps={w} "
                                                f"G/R/A={g}/{r_}/{a_} tpf={tpf}")
                            except Exception:
                                continue
        if best[1]:
            score("P4 fused two-shot HBM", best[0], "two_shot", best[1])
        elif rank == 0:
            print(f"  {'P4 fused two-shot HBM':<26} no valid config")

        del A, B, C, Cl, ws_cache
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
    p.add_argument("-o", "--output", type=str, default="roofline.json")
    p.add_argument("-m", "--m_list", type=str, default="32,128,512,2048")
    a = p.parse_args()
    m_list = [int(x) for x in a.m_list.split(",")]
    mp.spawn(fn=_worker,
             args=(a.num_ranks, f"tcp://127.0.0.1:{_free_port(a.port)}",
                   a.output, m_list),
             nprocs=a.num_ranks, join=True)


if __name__ == "__main__":
    main()
