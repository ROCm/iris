#!/usr/bin/env python3
"""HBM-buffer fused GEMM + two-shot AllReduce: correctness, then CU-split sweep.

Correctness first (including repeated iterations, which is what validates the
monotonic counters), then sweep the three-way CU split and the GEMM knobs.

Compared against:
  torch.mm + dist.all_reduce   (baseline)
  torch.mm + our one-shot AR   (the current best)
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
    cu_count = torch.cuda.get_device_properties(rank).multi_processor_count

    from iris.ops.all_reduce_fast import one_shot_all_reduce
    from iris.ops.matmul_all_reduce_hbm_buffer import (
        matmul_all_reduce_hbm_buffer,
        matmul_all_reduce_hbm_buffer_preamble,
    )

    dtype = torch.float16
    K_local = K_GLOBAL // world_size
    results = []

    if rank == 0:
        print(f"\nHBM-buffer fused GEMM + two-shot AR   TP={world_size} CUs={cu_count}")
        print(f"  traffic: one-shot={world_size:.2f}*MN   two-shot="
              f"{2*(world_size-1)/world_size:.2f}*MN  "
              f"({world_size/(2*(world_size-1)/world_size):.1f}x less)\n")

    for M in [512, 2048]:
        A = shmem.zeros((M, K_local), device="cuda", dtype=dtype)
        A.copy_(torch.randn(M, K_local, dtype=dtype, device=f"cuda:{rank}") * 0.1)
        B = torch.randn(K_local, N_GLOBAL, dtype=dtype, device=f"cuda:{rank}") * 0.1

        ref = torch.mm(A, B)
        dist.all_reduce(ref, op=dist.ReduceOp.SUM)
        torch.cuda.synchronize()
        tol = 0.05

        Ct = torch.empty(M, N_GLOBAL, dtype=dtype, device=f"cuda:{rank}")
        base_ms = bench(lambda: (torch.mm(A, B, out=Ct),
                                 dist.all_reduce(Ct, op=dist.ReduceOp.SUM)))

        Cs = shmem.zeros((M, N_GLOBAL), device="cuda", dtype=dtype)
        Co = torch.zeros(M, N_GLOBAL, dtype=dtype, device=f"cuda:{rank}")
        shmem.barrier()
        twok_ms = bench(lambda: (torch.mm(A, B, out=Cs),
                                 one_shot_all_reduce(shmem, Co, Cs)))

        if rank == 0:
            print(f"--- M={M} ---")
            print(f"  torch bulk-sync            {base_ms:.4f}ms  1.00x")
            print(f"  two-kernel one-shot        {twok_ms:.4f}ms  {base_ms/twok_ms:.2f}x")

        # KNOWN-GOOD CONTROL. Run one config that is known to pass before the
        # sweep, and say so loudly if it fails. An all-fail sweep reported as
        # "no valid config" cost a full run once -- the kernel was fine and the
        # harness had exhausted the symmetric heap. A sweep where nothing
        # passes is a harness result until this control says otherwise.
        ctl_ok = None
        try:
            cbm, cbn = (64, 128) if M % (64 * world_size) == 0 else (128, 128)
            if triton.cdiv(M, cbm) % world_size == 0:
                cws = matmul_all_reduce_hbm_buffer_preamble(
                    shmem, M, N_GLOBAL, dtype, cbm, cbn)
                shmem.barrier()
                cout = torch.zeros(M, N_GLOBAL, dtype=dtype,
                                   device=f"cuda:{rank}")
                matmul_all_reduce_hbm_buffer(
                    shmem, cout, A, B, workspace=cws, block_m=cbm, block_n=cbn,
                    block_k=64, mfma=16, tiles_per_flag=1, num_gemm_sms=208,
                    num_rs_sms=32, num_ag_sms=16)
                torch.cuda.synchronize()
                ctl_ok = torch.abs(cout - ref).max().item() < tol
                shmem.barrier()
                del cws, cout
                torch.cuda.empty_cache()
        except Exception as ex:
            ctl_ok = False
            if rank == 0:
                print(f"  CONTROL raised {type(ex).__name__}: {str(ex)[:70]}")
        if rank == 0 and ctl_ok is False:
            print("  *** KNOWN-GOOD CONTROL FAILED -- suspect the harness, "
                  "not the sweep ***")

        best = (1e9, None)
        ws_cache = {}
        out = torch.zeros(M, N_GLOBAL, dtype=dtype, device=f"cuda:{rank}")
        first_err = None
        for block_m in [128]:
            num_m_tiles = triton.cdiv(M, block_m)
            # RS phase shards M-tiles across ranks -- must divide evenly
            if num_m_tiles % world_size != 0:
                continue
            for block_n in [128]:
                for mfma in [16]:
                    # The decomposition says AR is 14x the GEMM, so the split
                    # should favour comm heavily. RS pulls from ws peers while
                    # AG pulls from one, so RS also wants more CUs than AG.
                    for tpf, (g, r_, a_) in [
                        (t, sp)
                        for t in (1, 2)
                        for sp in [
                        # Every split swept so far kept RS <= AG. Counting
                        # actual work at M=2048: RS is 46 tiles x 8 peer pulls
                        # = 368 pull-tiles at 64.3us each; AG is 322 tiles x 1
                        # pull at 6.7us. Per CU on an even 32/32 that is RS 92us
                        # vs AG 67us -- RS is the longer pole, so bias toward it.
                        # GEMM still has to be fast because it gates RS.
                        # RS-biased splits (128/96/32, 96/128/32, 160/64/32)
                        # all LOST, and 224/16/16 beat 192/32/32. So the comm
                        # pools are not CU-starved -- the trend runs the other
                        # way and had not bottomed out. Push it further.
                        (224, 16, 16),    # previous best
                        (240, 8, 8),
                        (232, 16, 8),
                        (232, 8, 16),
                        (208, 32, 16),
                        (208, 16, 32),
                        (192, 32, 32),    # control
                        ]
                    ]:
                        if g + r_ + a_ > cu_count:
                            continue
                        try:
                            ws = matmul_all_reduce_hbm_buffer_preamble(
                                shmem, M, N_GLOBAL, dtype, block_m, block_n)
                            shmem.barrier()

                            kw = dict(block_m=block_m, block_n=block_n, block_k=64,
                                      num_gemm_sms=g, num_rs_sms=r_, num_ag_sms=a_,
                                      mfma=mfma, tiles_per_flag=tpf)
                            # repeated iterations validate the monotonic counters
                            ok = True
                            for _ in range(3):
                                out.zero_()
                                matmul_all_reduce_hbm_buffer(
                                    shmem, out, A, B, workspace=ws, **kw)
                                torch.cuda.synchronize()
                                if torch.abs(out - ref).max().item() > tol:
                                    ok = False
                                    break
                            shmem.barrier()
                            if not ok:
                                continue

                            ms = bench(lambda ws=ws, out=out, kw=kw:
                                       matmul_all_reduce_hbm_buffer(
                                           shmem, out, A, B, workspace=ws, **kw))
                            cfg = (f"bm={block_m} bn={block_n} mfma={mfma} "
                                   f"G/R/A={g}/{r_}/{a_} tpf={tpf}")
                            results.append(dict(M=M, ms=ms, cfg=cfg,
                                                speedup=base_ms / ms))
                            if ms < best[0]:
                                best = (ms, cfg)
                                if rank == 0:
                                    print(f"    {cfg:<44} {ms:.4f}ms "
                                          f"{base_ms/ms:.2f}x  ***", flush=True)
                        except Exception as ex:
                            if first_err is None:
                                first_err = f"{type(ex).__name__}: {ex}"
                            continue

        if rank == 0:
            if best[1]:
                print(f"  HBM-buffer two-shot fused  {best[0]:.4f}ms  "
                      f"{base_ms/best[0]:.2f}x  ({best[1]})")
            else:
                print(f"  HBM-buffer two-shot fused  no valid config"
                      f"{'  first error -> ' + first_err if first_err else ''}")
            print(flush=True)

        del A, B, Cs, Co
        torch.cuda.empty_cache()
        shmem.barrier()

    if rank == 0 and outfile:
        with open(outfile, "w") as f:
            json.dump(results, f, indent=2)

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
    p.add_argument("-o", "--output", type=str, default="ar_hbm_buffer.json")
    a = p.parse_args()
    _PORT = a.port
    mp.spawn(fn=_worker, args=(a.num_ranks, f"tcp://127.0.0.1:{_free_port(_PORT)}", a.output),
             nprocs=a.num_ranks, join=True)


if __name__ == "__main__":
    main()
