#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.

"""
Graph-captured all-reduce benchmark: measures true latency floor.

Three-level measurement for each variant:
  1. Wall clock (kernel + launch + sync overhead)
  2. CUDA events (kernel-only, no host overhead)
  3. Graph capture + replay (latency floor, amortized launch)

Compares: one_shot_gluon vs one_shot (triton) vs RCCL (torch.distributed)

Usage:
  PYTHONPATH=/path/to/iris python3 -m torch.distributed.run --nproc_per_node=8 \
    benchmark/ccl/bench_all_reduce_graph.py
"""

import json
import os
import sys
import time

import torch
import torch.distributed as dist

torch.cuda.set_device(int(os.environ.get("LOCAL_RANK", 0)))
dist.init_process_group(backend="nccl")

RANK = dist.get_rank()
WORLD_SIZE = dist.get_world_size()

import iris
from iris.ccl import Config


def bench_wall_clock(fn, warmup=50, iters=200):
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(iters):
        fn()
    torch.cuda.synchronize()
    return (time.perf_counter() - t0) / iters


def bench_cuda_events(fn, warmup=50, iters=200):
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    times = []
    for _ in range(iters):
        start.record()
        fn()
        end.record()
        torch.cuda.synchronize()
        times.append(start.elapsed_time(end))
    return sum(times) / len(times)


def bench_graph_capture(fn, warmup=50, iters=200):
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()

    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        g = torch.cuda.CUDAGraph()
        g.capture_begin()
        fn()
        g.capture_end()
    torch.cuda.current_stream().wait_stream(stream)
    torch.cuda.synchronize()

    for _ in range(warmup):
        g.replay()
    torch.cuda.synchronize()

    t0 = time.perf_counter()
    for _ in range(iters):
        g.replay()
    torch.cuda.synchronize()
    elapsed = (time.perf_counter() - t0) / iters
    del g
    return elapsed


def bench_rccl(numel, dtype=torch.bfloat16, warmup=50, iters=200):
    t = torch.randn(numel, dtype=dtype, device=f"cuda:{RANK}")

    def fn():
        dist.all_reduce(t, op=dist.ReduceOp.SUM)

    dist.barrier()
    wall = bench_wall_clock(fn, warmup, iters)
    dist.barrier()
    events = bench_cuda_events(fn, warmup, iters)
    dist.barrier()
    graph = bench_graph_capture(fn, warmup, iters)
    dist.barrier()
    return wall, events, graph


def bench_iris_variant(numel, ctx, variant, use_gluon, dtype=torch.bfloat16, warmup=50, iters=200):
    inp = ctx.zeros((1, numel), dtype=dtype)
    out = ctx.zeros((1, numel), dtype=dtype)
    inp.fill_(float(ctx.get_rank() + 1))

    config = Config(all_reduce_variant=variant, use_gluon=use_gluon)

    workspace = None
    if variant != "one_shot_gluon":
        workspace = ctx.ccl.all_reduce_preamble(out, inp, config=config)

    def fn():
        ctx.ccl.all_reduce(out, inp, config=config, workspace=workspace, async_op=True)

    def fn_preamble():
        if variant != "one_shot_gluon" and workspace is not None:
            out.zero_()
            ctx.ccl.all_reduce_preamble(out, inp, config=config, workspace=workspace)

    ctx.barrier()

    # Wall clock
    for _ in range(warmup):
        fn_preamble()
        fn()
    torch.cuda.synchronize()
    ctx.barrier()
    t0 = time.perf_counter()
    for _ in range(iters):
        fn()
    torch.cuda.synchronize()
    wall = (time.perf_counter() - t0) / iters

    ctx.barrier()

    # CUDA events
    start_ev = torch.cuda.Event(enable_timing=True)
    end_ev = torch.cuda.Event(enable_timing=True)
    times = []
    for _ in range(iters):
        start_ev.record()
        fn()
        end_ev.record()
        torch.cuda.synchronize()
        times.append(start_ev.elapsed_time(end_ev))
    events = sum(times) / len(times)

    ctx.barrier()

    # Graph capture (single_barrier for gluon)
    # Use async_op=True to skip host barrier, which is not graph-capturable.
    if variant == "one_shot_gluon":
        config_graph = Config(
            all_reduce_variant=variant,
            use_gluon=use_gluon,
            all_reduce_single_barrier=True,
        )
    else:
        config_graph = Config(all_reduce_variant=variant, use_gluon=use_gluon)

    # Pre-create workspace outside capture
    workspace_graph = None
    if variant != "one_shot_gluon":
        workspace_graph = ctx.ccl.all_reduce_preamble(out, inp, config=config_graph)
    else:
        from iris.ccl.gluon.all_reduce import _GluonAllReduceWorkspace
        workspace_graph = _GluonAllReduceWorkspace(ctx, WORLD_SIZE)

    # Pre-extract group info outside capture (calls HIP APIs)
    from iris.ccl.utils import extract_group_info
    _rig, _rg, _ws, _rs, _rst = extract_group_info(None, ctx)

    def fn_graph():
        ctx.ccl.all_reduce(out, inp, config=config_graph, workspace=workspace_graph, async_op=True)

    for _ in range(warmup):
        fn_graph()
    torch.cuda.synchronize()

    try:
        stream = torch.cuda.Stream()
        stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(stream):
            g = torch.cuda.CUDAGraph()
            g.capture_begin()
            fn_graph()
            g.capture_end()
        torch.cuda.current_stream().wait_stream(stream)
        torch.cuda.synchronize()

        ctx.barrier()
        for _ in range(warmup):
            g.replay()
        torch.cuda.synchronize()
        ctx.barrier()

        t0 = time.perf_counter()
        for _ in range(iters):
            g.replay()
        torch.cuda.synchronize()
        graph = (time.perf_counter() - t0) / iters
        del g
    except Exception:
        graph = -1.0

    # Correctness check
    expected = WORLD_SIZE * (WORLD_SIZE + 1) / 2.0
    max_diff = torch.abs(out.view(-1) - expected).max().item()
    correct = max_diff < 1e-2

    ctx.barrier()
    return wall, events, graph, correct, max_diff


def main():
    ctx = iris.iris(2**33)

    sizes = [1024, 2048, 4096, 8192, 16384, 32768, 65536, 131072]
    variants = [
        ("one_shot_gluon", True),
        ("one_shot", False),
        ("two_shot", False),
    ]

    if RANK == 0:
        print(f"\n{'='*130}")
        print(f"Three-level all-reduce benchmark — {WORLD_SIZE}x MI355X, bf16")
        print(f"Levels: wall clock | CUDA events | graph capture")
        print(f"{'='*130}")
        hdr = (
            f"{'numel':>8} {'variant':>18} | "
            f"{'wall(us)':>9} {'events(us)':>10} {'graph(us)':>10} | "
            f"{'vs RCCL wall':>12} {'vs RCCL graph':>13} {'ok':>6}"
        )
        print(hdr)
        print("-" * len(hdr))

    results = []

    for numel in sizes:
        # RCCL baseline
        rccl_wall, rccl_events, rccl_graph = bench_rccl(numel)
        dist.barrier()

        if RANK == 0:
            print(
                f"{numel:>8,} {'RCCL':>18} | "
                f"{rccl_wall*1e6:>8.1f}us {rccl_events*1e3:>9.1f}us {rccl_graph*1e6:>9.1f}us | "
                f"{'baseline':>12} {'baseline':>13} {'--':>6}"
            )

        for variant, use_gluon in variants:
            dist.barrier()
            try:
                wall, events, graph, correct, max_diff = bench_iris_variant(
                    numel, ctx, variant, use_gluon
                )
            except Exception as e:
                if RANK == 0:
                    print(f"{numel:>8,} {variant:>18} | ERROR: {e}")
                continue

            if RANK == 0:
                ok = "OK" if correct else f"FAIL({max_diff:.3f})"
                speedup_wall = rccl_wall / wall if wall > 0 else 0
                speedup_graph = rccl_graph / graph if graph > 0 else 0
                print(
                    f"{numel:>8,} {variant:>18} | "
                    f"{wall*1e6:>8.1f}us {events*1e3:>9.1f}us {graph*1e6:>9.1f}us | "
                    f"{speedup_wall:>11.2f}x {speedup_graph:>12.2f}x {ok:>6}"
                )
                results.append({
                    "numel": numel,
                    "variant": variant,
                    "wall_us": round(wall * 1e6, 2),
                    "events_us": round(events * 1e3, 2),
                    "graph_us": round(graph * 1e6, 2),
                    "rccl_wall_us": round(rccl_wall * 1e6, 2),
                    "rccl_events_us": round(rccl_events * 1e3, 2),
                    "rccl_graph_us": round(rccl_graph * 1e6, 2),
                    "speedup_wall": round(speedup_wall, 3),
                    "speedup_graph": round(speedup_graph, 3),
                    "correct": correct,
                })

    if RANK == 0:
        print(f"{'='*130}")
        out_path = "/tmp/gluon_ar_three_level.json"
        with open(out_path, "w") as f:
            json.dump(results, f, indent=2)
        print(f"\nSaved to {out_path}")

    ctx.barrier()
    del ctx
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
