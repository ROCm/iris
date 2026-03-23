#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
# Copyright (c) 2025 Advanced Micro Devices, Inc. All rights reserved.

"""
Ring Attention profiler: end-to-end timing for the persistent kernel.

Measures total kernel time and wall time for the persistent ring attention
kernel with device-side signal-flag synchronization.

Usage::

    python examples/32_ring_attention/profile_ring_attn.py
    python examples/32_ring_attention/profile_ring_attn.py --num_ranks 4
    python examples/32_ring_attention/profile_ring_attn.py --num_ranks 8 --total_seq_len 16384
"""

import argparse
import json
import os
import sys
import tempfile
import time
from pathlib import Path

import torch
import torch.distributed as dist
import torch.multiprocessing as mp

import iris

sys.path.insert(0, str(Path(__file__).resolve().parent))
from ring_attention_kernels import ring_attn_fwd  # noqa: E402


def _profiled_ring_attn_fwd(q, k, v, shmem, causal=True, scale=None,
                             _ping_pong_bufs=None, _signal_flags=None):
    """
    Instrumented ring_attn_fwd that measures end-to-end kernel time.

    With the persistent kernel, the entire ring loop runs in a single kernel
    launch with device-side synchronization, so per-step host timing is not
    applicable.  This measures total wall time and kernel time instead.

    Returns (output, timing_data).
    """
    # Time the entire persistent kernel launch
    kernel_start = torch.cuda.Event(enable_timing=True)
    kernel_end = torch.cuda.Event(enable_timing=True)

    wall_start = time.perf_counter()
    kernel_start.record()

    output = ring_attn_fwd(
        q, k, v, shmem,
        causal=causal, scale=scale,
        _ping_pong_bufs=_ping_pong_bufs, _signal_flags=_signal_flags,
    )

    kernel_end.record()
    torch.cuda.synchronize()
    wall_end = time.perf_counter()

    kernel_ms = kernel_start.elapsed_time(kernel_end)
    wall_ms = (wall_end - wall_start) * 1000.0

    timing_data = {
        "kernel_ms": kernel_ms,
        "wall_ms": wall_ms,
    }
    return output, timing_data


def _profile_worker(rank, world_size, init_url, cfg, results_file):
    backend = "nccl" if torch.cuda.is_available() else "gloo"
    dist.init_process_group(
        backend=backend,
        init_method=init_url,
        world_size=world_size,
        rank=rank,
        device_id=torch.device(f"cuda:{rank}"),
    )
    torch.cuda.set_device(rank)
    torch.set_default_device(f"cuda:{rank}")

    shmem = iris.iris()

    total_seq = cfg["total_seq"]
    num_heads = cfg["num_heads"]
    head_dim = cfg["head_dim"]
    dtype = getattr(torch, cfg["dtype"])
    causal = cfg["causal"]
    num_warmup = cfg["warmup"]
    num_iters = cfg["iters"]

    seq_local = total_seq // world_size
    scale = head_dim**-0.5

    torch.manual_seed(42 + rank)
    q = torch.randn(seq_local, num_heads, head_dim, dtype=dtype)
    k = torch.randn(seq_local, num_heads, head_dim, dtype=dtype)
    v = torch.randn(seq_local, num_heads, head_dim, dtype=dtype)

    # Pre-allocate ping-pong buffers and signal flags
    k_ping = shmem.empty(k.shape, dtype=k.dtype)
    k_pong = shmem.empty(k.shape, dtype=k.dtype)
    v_ping = shmem.empty(v.shape, dtype=v.dtype)
    v_pong = shmem.empty(v.shape, dtype=v.dtype)
    bufs = (k_ping, k_pong, v_ping, v_pong)
    signal_flags = shmem.zeros((world_size,), dtype=torch.int32)

    shmem.barrier()

    # Warmup
    for _ in range(num_warmup):
        out, _ = _profiled_ring_attn_fwd(
            q, k, v, shmem, causal=causal, scale=scale,
            _ping_pong_bufs=bufs, _signal_flags=signal_flags,
        )
    torch.cuda.synchronize()
    shmem.barrier()

    # Timed iterations
    all_iter_timings = []
    for it in range(num_iters):
        out, timing = _profiled_ring_attn_fwd(
            q, k, v, shmem, causal=causal, scale=scale,
            _ping_pong_bufs=bufs, _signal_flags=signal_flags,
        )
        all_iter_timings.append(timing)
    torch.cuda.synchronize()
    shmem.barrier()

    # Aggregate: average timings across iterations
    avg_kernel_ms = sum(t["kernel_ms"] for t in all_iter_timings) / num_iters
    avg_wall_ms = sum(t["wall_ms"] for t in all_iter_timings) / num_iters

    del shmem
    dist.destroy_process_group()

    if rank == 0:
        result = {
            "config": cfg,
            "world_size": world_size,
            "rank": rank,
            "totals": {
                "kernel_ms": avg_kernel_ms,
                "wall_ms": avg_wall_ms,
            },
        }
        with open(results_file, "w") as f:
            json.dump(result, f, indent=2)


def main():
    parser = argparse.ArgumentParser(description="Ring Attention profiler")
    parser.add_argument("--num_ranks", type=int, default=2)
    parser.add_argument("--total_seq_len", type=int, default=8192)
    parser.add_argument("--num_heads", type=int, default=16)
    parser.add_argument("--head_dim", type=int, default=64)
    parser.add_argument("--no_causal", dest="causal", action="store_false", default=True)
    parser.add_argument("--dtype", default="float16")
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--iters", type=int, default=10)
    args = parser.parse_args()

    cfg = {
        "total_seq": args.total_seq_len,
        "num_heads": args.num_heads,
        "head_dim": args.head_dim,
        "dtype": args.dtype,
        "causal": args.causal,
        "warmup": args.warmup,
        "iters": args.iters,
    }

    with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
        results_file = f.name

    try:
        import socket

        _sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        _sock.bind(("", 0))
        init_url = f"tcp://127.0.0.1:{_sock.getsockname()[1]}"
        _sock.close()

        mp.spawn(
            fn=_profile_worker,
            args=(args.num_ranks, init_url, cfg, results_file),
            nprocs=args.num_ranks,
            join=True,
        )

        with open(results_file) as f:
            result = json.load(f)

        # Print results
        world_size = result["world_size"]
        totals = result["totals"]
        print(f"\n{'=' * 80}")
        print(
            f"Ring Attention Profiling (persistent kernel) — {world_size} GPUs, "
            f"seq={cfg['total_seq']}, H={cfg['num_heads']}, D={cfg['head_dim']}, "
            f"causal={cfg['causal']}"
        )
        print(f"{'=' * 80}")

        print("\n--- Timings (rank 0, averaged) ---")
        print(f"  Kernel (GPU)   : {totals['kernel_ms']:>8.3f} ms")
        print(f"  Wall (end2end) : {totals['wall_ms']:>8.3f} ms")

        # Compute efficiency
        seq_local = cfg["total_seq"] // world_size
        flops = 4 * seq_local * cfg["total_seq"] * cfg["head_dim"] * cfg["num_heads"]
        if cfg["causal"]:
            flops //= 2
        tflops = flops / (totals["kernel_ms"] * 1e-3) / 1e12
        print(f"  TFLOPS         : {tflops:>8.2f}")
        print(f"  MFU (vs 1307)  : {100 * tflops / 1307.4:>7.1f}%")
        print()

    finally:
        os.unlink(results_file)


if __name__ == "__main__":
    main()
