#!/usr/bin/env python3
"""Clean vLLM all-reduce benchmark: proper warmup, multiple runs, median."""
import os, sys, time, torch, torch.distributed as dist

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
element_size = 2  # bf16

def do_bench_rccl(tensor, warmup=10, rep=50):
    """Benchmark RCCL all_reduce with proper timing."""
    # Warmup
    for _ in range(warmup):
        dist.all_reduce(tensor, op=dist.ReduceOp.SUM)
    torch.cuda.synchronize()
    dist.barrier()

    # Timed runs
    start_events = [torch.cuda.Event(enable_timing=True) for _ in range(rep)]
    end_events = [torch.cuda.Event(enable_timing=True) for _ in range(rep)]

    for i in range(rep):
        tensor.fill_(float(rank + 1))
        dist.barrier()
        start_events[i].record()
        dist.all_reduce(tensor, op=dist.ReduceOp.SUM)
        end_events[i].record()

    torch.cuda.synchronize()
    times = [s.elapsed_time(e) for s, e in zip(start_events, end_events)]
    times.sort()
    return times[len(times)//2]  # median


def do_bench_iris(inp, out, config, workspace, warmup=10, rep=50):
    """Benchmark iris all_reduce with proper timing."""
    # Warmup
    for _ in range(warmup):
        ctx.ccl.all_reduce_preamble(out, inp, config=config, workspace=workspace)
        ctx.ccl.all_reduce(out, inp, config=config, workspace=workspace)
    torch.cuda.synchronize()
    ctx.barrier()

    # Timed runs
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
    return times[len(times)//2]  # median


# Pre-compile all kernels first
if rank == 0:
    print("Pre-compiling all kernels...", flush=True)

for M in ALL_MS:
    tensor_rccl = torch.full((M, N), float(rank + 1), dtype=dtype, device="cuda")
    dist.all_reduce(tensor_rccl, op=dist.ReduceOp.SUM)

    for variant in ["two_shot", "one_shot", "flat"]:
        inp = ctx.zeros((M, N), dtype=dtype)
        out = ctx.zeros((M, N), dtype=dtype)
        inp.fill_(float(rank + 1))
        config = Config(all_reduce_variant=variant)
        workspace = ctx.ccl.all_reduce_preamble(out, inp, config=config)
        ctx.ccl.all_reduce(out, inp, config=config, workspace=workspace)

torch.cuda.synchronize()
dist.barrier()

if rank == 0:
    print("Compilation done. Starting benchmarks...\n", flush=True)
    print(f"{'M':>6} {'N':>6} | {'RCCL (ms)':>10} {'RCCL BW':>10} | {'two_shot':>10} {'BW':>10} {'ratio':>7} | {'one_shot':>10} {'BW':>10} {'ratio':>7} | {'flat':>10} {'BW':>10} {'ratio':>7} |", flush=True)
    print("-" * 140, flush=True)

results = []

for M in ALL_MS:
    # RCCL
    tensor_rccl = torch.full((M, N), float(rank + 1), dtype=dtype, device="cuda")
    rccl_ms = do_bench_rccl(tensor_rccl, warmup=20, rep=100)

    data_bytes = M * N * element_size * 2 * (world_size - 1) / world_size
    rccl_bw = data_bytes / (rccl_ms / 1000) / 1e9

    row = {"M": M, "rccl_ms": rccl_ms, "rccl_bw": rccl_bw}

    # Iris variants
    for variant in ["two_shot", "one_shot", "flat"]:
        inp = ctx.zeros((M, N), dtype=dtype)
        out = ctx.zeros((M, N), dtype=dtype)
        inp.fill_(float(rank + 1))
        config = Config(all_reduce_variant=variant)
        workspace = ctx.ccl.all_reduce_preamble(out, inp, config=config)

        iris_ms = do_bench_iris(inp, out, config, workspace, warmup=20, rep=100)
        iris_bw = data_bytes / (iris_ms / 1000) / 1e9
        ratio = iris_ms / rccl_ms

        row[f"{variant}_ms"] = iris_ms
        row[f"{variant}_bw"] = iris_bw
        row[f"{variant}_ratio"] = ratio

    results.append(row)

    if rank == 0:
        print(f"{M:>6} {N:>6} | {row['rccl_ms']:>10.3f} {row['rccl_bw']:>9.2f}G | "
              f"{row['two_shot_ms']:>10.3f} {row['two_shot_bw']:>9.2f}G {row['two_shot_ratio']:>6.2f}x | "
              f"{row['one_shot_ms']:>10.3f} {row['one_shot_bw']:>9.2f}G {row['one_shot_ratio']:>6.2f}x | "
              f"{row['flat_ms']:>10.3f} {row['flat_bw']:>9.2f}G {row['flat_ratio']:>6.2f}x |", flush=True)

if rank == 0:
    print("\n--- CSV ---", flush=True)
    print("M,N,rccl_ms,rccl_bw_gbps,two_shot_ms,two_shot_bw_gbps,two_shot_ratio,one_shot_ms,one_shot_bw_gbps,one_shot_ratio,flat_ms,flat_bw_gbps,flat_ratio", flush=True)
    for r in results:
        print(f"{r['M']},{N},{r['rccl_ms']:.4f},{r['rccl_bw']:.2f},"
              f"{r['two_shot_ms']:.4f},{r['two_shot_bw']:.2f},{r['two_shot_ratio']:.3f},"
              f"{r['one_shot_ms']:.4f},{r['one_shot_bw']:.2f},{r['one_shot_ratio']:.3f},"
              f"{r['flat_ms']:.4f},{r['flat_bw']:.2f},{r['flat_ratio']:.3f}", flush=True)

dist.destroy_process_group()
