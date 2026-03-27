#!/usr/bin/env python3
"""Tune iris two_shot all-reduce configs to match RCCL on vLLM shapes."""
import os, sys, torch, torch.distributed as dist

torch.cuda.set_device(int(os.environ.get("LOCAL_RANK", 0)))
dist.init_process_group(backend="nccl")

import iris
from iris.ccl import Config

rank = dist.get_rank()
world_size = dist.get_world_size()

heap_size = 2**33
ctx = iris.iris(heap_size)

N = 2880
ALL_MS = [1, 32, 64, 128, 512, 2048, 4096, 8192]
dtype = torch.bfloat16
element_size = 2

# Config search space - focus on two_shot since it's the best variant
BLOCK_SIZES_M = [16, 32, 64, 128]
BLOCK_SIZES_N = [32, 64, 128, 256]
COMM_SMS_LIST = [32, 64, 96, 128]
DISTRIBUTIONS = [0, 1]  # 0=striding, 1=block
SWIZZLE_SIZES = [2, 4, 8]

def bench_one(inp, out, config, workspace, warmup=5, rep=30):
    """Quick benchmark, return median ms."""
    for _ in range(warmup):
        ctx.ccl.all_reduce_preamble(out, inp, config=config, workspace=workspace)
        ctx.ccl.all_reduce(out, inp, config=config, workspace=workspace)
    torch.cuda.synchronize()
    ctx.barrier()

    start_events = [torch.cuda.Event(enable_timing=True) for _ in range(rep)]
    end_events = [torch.cuda.Event(enable_timing=True) for _ in range(rep)]

    for i in range(rep):
        out.zero_()
        inp.fill_(float(rank + 1))
        ctx.ccl.all_reduce_preamble(out, inp, config=config, workspace=workspace)
        ctx.barrier()
        start_events[i].record()
        ctx.ccl.all_reduce(out, inp, config=config, workspace=workspace)
        end_events[i].record()

    torch.cuda.synchronize()
    times = [s.elapsed_time(e) for s, e in zip(start_events, end_events)]
    times.sort()
    return times[len(times)//2]


# Step 1: Sweep block_size_m x block_size_n for each M value
if rank == 0:
    print("=== Step 1: Block size sweep (comm_sms=64, dist=1, swizzle=4) ===", flush=True)
    print(f"{'M':>6} | {'bm':>4}x{'bn':>4} -> {'ms':>8} | {'bm':>4}x{'bn':>4} -> {'ms':>8} | {'bm':>4}x{'bn':>4} -> {'ms':>8} | best", flush=True)
    print("-" * 100, flush=True)

best_blocks = {}
for M in ALL_MS:
    best_ms = float("inf")
    best_cfg = None
    all_results = []

    for bm in BLOCK_SIZES_M:
        for bn in BLOCK_SIZES_N:
            try:
                config = Config(
                    block_size_m=bm, block_size_n=bn,
                    all_reduce_variant="two_shot",
                    all_reduce_distribution=1,
                    swizzle_size=4, comm_sms=64,
                )
                inp = ctx.zeros((M, N), dtype=dtype)
                out = ctx.zeros((M, N), dtype=dtype)
                inp.fill_(float(rank + 1))
                workspace = ctx.ccl.all_reduce_preamble(out, inp, config=config)
                ms = bench_one(inp, out, config, workspace, warmup=3, rep=20)
                all_results.append((bm, bn, ms))
                if ms < best_ms:
                    best_ms = ms
                    best_cfg = (bm, bn)
            except Exception as e:
                if rank == 0:
                    pass  # skip invalid configs silently

    best_blocks[M] = best_cfg
    if rank == 0:
        # Show top 3
        all_results.sort(key=lambda x: x[2])
        top3 = all_results[:3]
        parts = " | ".join(f"{bm:>4}x{bn:<4} -> {ms:>7.3f}ms" for bm, bn, ms in top3)
        print(f"{M:>6} | {parts} | BEST: {best_cfg[0]}x{best_cfg[1]} = {best_ms:.3f}ms", flush=True)

# Step 2: Sweep comm_sms for best block sizes
if rank == 0:
    print(f"\n=== Step 2: comm_sms sweep ===", flush=True)

best_sms = {}
for M in ALL_MS:
    bm, bn = best_blocks[M]
    best_ms = float("inf")
    best_s = None
    results = []
    for sms in COMM_SMS_LIST:
        try:
            config = Config(
                block_size_m=bm, block_size_n=bn,
                all_reduce_variant="two_shot",
                all_reduce_distribution=1,
                swizzle_size=4, comm_sms=sms,
            )
            inp = ctx.zeros((M, N), dtype=dtype)
            out = ctx.zeros((M, N), dtype=dtype)
            inp.fill_(float(rank + 1))
            workspace = ctx.ccl.all_reduce_preamble(out, inp, config=config)
            ms = bench_one(inp, out, config, workspace, warmup=3, rep=20)
            results.append((sms, ms))
            if ms < best_ms:
                best_ms = ms
                best_s = sms
        except Exception:
            pass
    best_sms[M] = best_s
    if rank == 0:
        parts = " | ".join(f"sms={s:>3}: {ms:.3f}ms" for s, ms in results)
        print(f"  M={M:>5} block={bm}x{bn}: {parts} -> BEST sms={best_s}", flush=True)

# Step 3: Sweep distribution + swizzle for final configs
if rank == 0:
    print(f"\n=== Step 3: distribution + swizzle sweep ===", flush=True)

best_configs = {}
for M in ALL_MS:
    bm, bn = best_blocks[M]
    sms = best_sms[M]
    best_ms = float("inf")
    best_d_sw = None
    results = []
    for dist_mode in DISTRIBUTIONS:
        for sw in SWIZZLE_SIZES:
            try:
                config = Config(
                    block_size_m=bm, block_size_n=bn,
                    all_reduce_variant="two_shot",
                    all_reduce_distribution=dist_mode,
                    swizzle_size=sw, comm_sms=sms,
                )
                inp = ctx.zeros((M, N), dtype=dtype)
                out = ctx.zeros((M, N), dtype=dtype)
                inp.fill_(float(rank + 1))
                workspace = ctx.ccl.all_reduce_preamble(out, inp, config=config)
                ms = bench_one(inp, out, config, workspace, warmup=3, rep=20)
                results.append((dist_mode, sw, ms))
                if ms < best_ms:
                    best_ms = ms
                    best_d_sw = (dist_mode, sw)
            except Exception:
                pass
    best_configs[M] = {
        "block_size_m": bm, "block_size_n": bn,
        "comm_sms": sms,
        "distribution": best_d_sw[0], "swizzle": best_d_sw[1],
        "ms": best_ms,
    }
    if rank == 0:
        parts = " | ".join(f"d={d},sw={s}: {ms:.3f}" for d, s, ms in results)
        print(f"  M={M:>5}: {parts} -> BEST d={best_d_sw[0]},sw={best_d_sw[1]}", flush=True)

# Step 4: Final comparison - RCCL vs best tuned iris
if rank == 0:
    print(f"\n=== Final: RCCL vs Tuned Iris (two_shot) ===", flush=True)
    print(f"{'M':>6} {'N':>6} | {'RCCL(ms)':>10} {'RCCL BW':>10} | {'Iris(ms)':>10} {'Iris BW':>10} {'ratio':>7} | Config", flush=True)
    print("-" * 120, flush=True)

final_results = []
for M in ALL_MS:
    cfg = best_configs[M]

    # RCCL
    tensor_rccl = torch.full((M, N), float(rank + 1), dtype=dtype, device="cuda")
    for _ in range(20):
        dist.all_reduce(tensor_rccl, op=dist.ReduceOp.SUM)
    torch.cuda.synchronize()
    dist.barrier()

    start_events = [torch.cuda.Event(enable_timing=True) for _ in range(50)]
    end_events = [torch.cuda.Event(enable_timing=True) for _ in range(50)]
    for i in range(50):
        tensor_rccl.fill_(float(rank + 1))
        dist.barrier()
        start_events[i].record()
        dist.all_reduce(tensor_rccl, op=dist.ReduceOp.SUM)
        end_events[i].record()
    torch.cuda.synchronize()
    rccl_times = sorted([s.elapsed_time(e) for s, e in zip(start_events, end_events)])
    rccl_ms = rccl_times[len(rccl_times)//2]

    # Tuned iris
    config = Config(
        block_size_m=cfg["block_size_m"], block_size_n=cfg["block_size_n"],
        all_reduce_variant="two_shot",
        all_reduce_distribution=cfg["distribution"],
        swizzle_size=cfg["swizzle"], comm_sms=cfg["comm_sms"],
    )
    inp = ctx.zeros((M, N), dtype=dtype)
    out = ctx.zeros((M, N), dtype=dtype)
    inp.fill_(float(rank + 1))
    workspace = ctx.ccl.all_reduce_preamble(out, inp, config=config)
    iris_ms = bench_one(inp, out, config, workspace, warmup=10, rep=50)

    data_bytes = M * N * element_size * 2 * (world_size - 1) / world_size
    rccl_bw = data_bytes / (rccl_ms / 1000) / 1e9
    iris_bw = data_bytes / (iris_ms / 1000) / 1e9
    ratio = iris_ms / rccl_ms

    final_results.append({
        "M": M, "rccl_ms": rccl_ms, "rccl_bw": rccl_bw,
        "iris_ms": iris_ms, "iris_bw": iris_bw, "ratio": ratio,
        "config": cfg,
    })

    if rank == 0:
        cfg_str = f"bm={cfg['block_size_m']},bn={cfg['block_size_n']},sms={cfg['comm_sms']},d={cfg['distribution']},sw={cfg['swizzle']}"
        print(f"{M:>6} {N:>6} | {rccl_ms:>10.4f} {rccl_bw:>9.2f}G | {iris_ms:>10.4f} {iris_bw:>9.2f}G {ratio:>6.2f}x | {cfg_str}", flush=True)

if rank == 0:
    print("\n--- CSV ---", flush=True)
    print("M,N,rccl_ms,rccl_bw_gbps,iris_tuned_ms,iris_tuned_bw_gbps,ratio,block_size_m,block_size_n,comm_sms,distribution,swizzle_size", flush=True)
    for r in final_results:
        c = r["config"]
        print(f"{r['M']},{N},{r['rccl_ms']:.4f},{r['rccl_bw']:.2f},{r['iris_ms']:.4f},{r['iris_bw']:.2f},{r['ratio']:.3f},"
              f"{c['block_size_m']},{c['block_size_n']},{c['comm_sms']},{c['distribution']},{c['swizzle']}", flush=True)

dist.destroy_process_group()
