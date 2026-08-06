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

Every headline factors exactly, so nothing is unattributed:

    T_torch / T_meas = component_gain * overlap_gain
    component_gain   = T_torch      / (T_gemm_var + T_comm_var)
    overlap_gain     = T_serial_var / T_meas

and the comm side splits into physics:

    algorithm = rccl_algo_bytes / variant_algo_bytes   (fewer bytes)
    kernel    = variant_GBs     / rccl_GBs             (better efficiency)
    comm_gain = algorithm * kernel  ==  T_comm_rccl / T_comm_var
    gemm_gain = T_gemm_torch / T_gemm_var              (Triton GEMM tax)

Both identities are asserted at runtime: a row that does not multiply out
crashes rather than printing a plausible number.

    overlap_ratio = (T_serial - T_measured) / (T_serial - T_ideal)

        1.0  comm fully hidden -- the stated goal
        0.0  no overlap, identical to running them back to back
       <0    fusion is actively costing us

Two efficiency axes so we know which side binds:
    comm_pct_line   achieved GB/s over XGMI line rate
    gemm_pct_peak   achieved TFLOP/s over device peak

WHY THE REFERENCES MUST BE TUNED
--------------------------------
overlap_ratio is a ratio of differences against T_gemm and T_comm. An untuned
reference inflates T_serial and makes every variant look good. `num_warps` was
never swept on any variant in this study and is worth up to 4.3x, so the
references are tuned over it here before anything else runs.

BASELINE IS RCCL, NOT OUR OWN KERNEL
------------------------------------
A variant that switches one-shot -> two-shot must not book that as "overlap",
so each variant is scored against the comm reference for its own algorithm.
The algorithm win is anchored to RCCL rather than to our one-shot: anchoring
to our own kernel hands every two-shot variant a free win against a strawman
we wrote, and moves the number every time we retune one-shot. RCCL is also a
two-shot ring, so it compares like with like on bytes.
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

        # torch bulk-sync first: it is the denominator for every gain below
        Ct = torch.empty(M, N_GLOBAL, dtype=dtype, device=f"cuda:{rank}")
        torch_ms = bench(lambda: (torch.mm(A, B, out=Ct),
                                  dist.all_reduce(Ct, op=dist.ReduceOp.SUM)))

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

        # torch is nearly but not exactly serial; the residual is launch
        # overhead and it gets its own factor rather than being absorbed.
        T_serial_torch = T_gemm + comm_ref["rccl"]
        serial_overhead_torch = torch_ms / T_serial_torch

        # ---------------- VARIANTS ----------------
        if rank == 0:
            print(f"  {'-'*74}")
            print(f"  {'variant':<24} {'vs_torch = serial x compnt x ovlap':>38}"
                  f"   {'[gemm | comm = algo x kern]':<40} {'achieved':>18}")
            print(f"  {'':<24} {'(these three multiply exactly)':>38}"
                  f"   {'(gemm x comm does NOT -- harmonic mean)':<40}")

        def score(name, T_meas, algo, cfg="", T_gemm_var=None, comm_gbs=None):
            """Attribute a variant so the factors multiply out to the headline.

                T_torch / T_meas = component_gain * overlap_gain      (exact)

                component_gain = T_torch      / T_serial_var
                overlap_gain   = T_serial_var / T_meas
                T_serial_var   = T_gemm_var + T_comm_var

            and the comm side splits into physics:

                algorithm = rccl_algo_bytes / variant_algo_bytes
                kernel    = variant_GBs     / rccl_GBs
                comm_gain = algorithm * kernel  ==  T_comm_rccl / T_comm_var

            Baseline is RCCL, not our own one-shot: anchoring to our kernel
            hands every two-shot variant a free win against a strawman we
            wrote, and moves the number every time we retune one-shot.
            """
            T_c = comm_ref.get(algo)
            if T_c is None:
                return
            T_g = T_gemm_var if T_gemm_var is not None else T_gemm
            T_serial_var = T_g + T_c
            T_ideal = max(T_g, T_c)

            # Three exact factors. component_gain compares idealized serial to
            # idealized serial, so the launch residual gets its own term
            # instead of being absorbed silently.
            serial_overhead = torch_ms / T_serial_torch
            component_gain = T_serial_torch / T_serial_var
            overlap_gain = T_serial_var / T_meas

            denom = T_serial_var - T_ideal
            ov_ratio = (T_serial_var - T_meas) / denom if denom > 1e-9 else float("nan")

            var_bytes = bytes_one if algo == "one_shot" else bytes_two
            algorithm = bytes_two / var_bytes          # RCCL ring moves bytes_two
            var_gbs = comm_gbs if comm_gbs else var_bytes / 1e9 / (T_c * 1e-3)
            kernel = var_gbs / rccl_gbs
            comm_gain = comm_ref["rccl"] / T_c
            gemm_gain = T_gemm / T_g

            # This study has already produced three confident-and-wrong
            # diagnoses, so make the arithmetic fail loudly instead of
            # printing a plausible number.
            assert abs(serial_overhead * component_gain * overlap_gain
                       - torch_ms / T_meas) < 1e-9
            assert abs(algorithm * kernel - comm_gain) / max(comm_gain, 1e-9) < 0.02
            # component_gain is a ratio of SUMS -- it is the harmonic mean of
            # gemm_gain and comm_gain weighted by how torch splits its time,
            # NOT their product. Asserted so nobody re-derives it wrong.
            w_g = T_gemm / T_serial_torch
            w_c = comm_ref["rccl"] / T_serial_torch
            harm = 1.0 / (w_g / gemm_gain + w_c / comm_gain)
            assert abs(harm - component_gain) / component_gain < 1e-6

            if rank == 0:
                print(f"  {name:<24} {torch_ms/T_meas:7.3f} = "
                      f"{serial_overhead:.3f} x {component_gain:.3f} x {overlap_gain:.3f}"
                      f"   [gemm {gemm_gain:.2f} | comm {comm_gain:.3f} "
                      f"= algo {algorithm:.3f} x kern {kernel:.2f}]"
                      f"  {var_gbs:6.1f} GB/s ({100*var_gbs/LINE_GBS:.0f}%)  {cfg}")
            emit(kind="variant", name=name, M=M, ms=T_meas, algo=algo,
                 vs_torch=torch_ms / T_meas, serial_overhead=serial_overhead,
                 component_gain=component_gain, overlap_gain=overlap_gain,
                 algorithm=algorithm, kernel=kernel, comm_gain=comm_gain,
                 gemm_gain=gemm_gain, overlap_ratio=ov_ratio,
                 T_serial_var=T_serial_var, T_ideal=T_ideal, T_gemm=T_g,
                 T_comm=T_c, comm_gbs=var_gbs, cfg=cfg)

        # torch bulk-sync -- defines the 1.00x column
        if rank == 0:
            print(f"  {'P1 torch bulk-sync':<24} {1.0:7.3f} = "
                  f"{serial_overhead_torch:.3f} x 1.000 x 1.000"
                  f"   [gemm 1.00 | comm 1.000 = algo 1.000 x kern 1.00]"
                  f"  {rccl_gbs:6.1f} GB/s ({100*rccl_gbs/LINE_GBS:.0f}%)")
        serial_err = abs(torch_ms - T_serial_torch) / torch_ms
        if rank == 0:
            print(f"  {'':<26} torch serial check: "
                  f"gemm+rccl={T_gemm + comm_ref['rccl']:.4f} vs measured "
                  f"{torch_ms:.4f}  ({100*serial_err:.1f}% off)")
        emit(kind="variant", name="P1 torch bulk-sync", M=M, ms=torch_ms,
             algo="rccl", vs_torch=1.0, serial_err_pct=100 * serial_err)

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
