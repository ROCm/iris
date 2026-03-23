#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
# Copyright (c) 2025 Advanced Micro Devices, Inc. All rights reserved.

"""
Ring Attention profiler: per-step timing breakdown.

Instruments the ring attention loop to measure where time is spent:
  - Kernel launch + compute time
  - torch.cuda.synchronize() time
  - shmem.barrier() time

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
import triton

import iris

sys.path.insert(0, str(Path(__file__).resolve().parent))
from ring_attention_kernels import _ring_attn_fwd_kernel  # noqa: E402


def _profiled_ring_attn_fwd(q, k, v, shmem, causal=True, scale=None, _ping_pong_bufs=None):
    """
    Instrumented ring_attn_fwd that collects per-step timing.

    Returns (output, timing_data).
    """
    rank = shmem.get_rank()
    world_size = shmem.get_num_ranks()

    seq_q, num_heads, head_dim = q.shape
    seq_kv = k.shape[0]

    if scale is None:
        scale = head_dim**-0.5

    input_dtype = q.dtype

    O = torch.zeros(seq_q, num_heads, head_dim, dtype=torch.float32, device=q.device)
    M = torch.full((num_heads, seq_q), fill_value=-float("inf"), dtype=torch.float32, device=q.device)
    L = torch.zeros(num_heads, seq_q, dtype=torch.float32, device=q.device)

    BLOCK_Q = 64
    BLOCK_KV = 64
    HEAD_DIM = head_dim

    if _ping_pong_bufs is not None:
        k_ping, k_pong, v_ping, v_pong = _ping_pong_bufs
    else:
        k_ping = shmem.empty(k.shape, dtype=k.dtype)
        k_pong = shmem.empty(k.shape, dtype=k.dtype)
        v_ping = shmem.empty(v.shape, dtype=v.dtype)
        v_pong = shmem.empty(v.shape, dtype=v.dtype)

    k_ping.copy_(k.contiguous())
    v_ping.copy_(v.contiguous())
    shmem.barrier()

    k_cur, k_recv = k_ping, k_pong
    v_cur, v_recv = v_ping, v_pong

    next_rank = (rank + 1) % world_size

    FUSED_PUT_BLOCK = BLOCK_Q * HEAD_DIM
    n_k = k_cur.numel()
    heap_bases = shmem.get_heap_bases()

    step_timings = []

    for step in range(world_size):
        kv_rank = (rank - step) % world_size
        do_put = step < world_size - 1

        # --- Time the kernel launch + execution ---
        kernel_start = torch.cuda.Event(enable_timing=True)
        kernel_end = torch.cuda.Event(enable_timing=True)

        kernel_start.record()

        q_rank_start = rank * seq_q
        kv_rank_start = kv_rank * seq_kv
        grid = (num_heads, triton.cdiv(seq_q, BLOCK_Q))
        _ring_attn_fwd_kernel[grid](
            q,
            k_cur,
            v_cur,
            O,
            M,
            L,
            q.stride(0),
            q.stride(1),
            q.stride(2),
            k_cur.stride(0),
            k_cur.stride(1),
            k_cur.stride(2),
            v_cur.stride(0),
            v_cur.stride(1),
            v_cur.stride(2),
            O.stride(0),
            O.stride(1),
            O.stride(2),
            M.stride(0),
            M.stride(1),
            L.stride(0),
            L.stride(1),
            seq_q,
            seq_kv,
            q_rank_start,
            kv_rank_start,
            scale,
            # fused put params
            k_cur.view(-1),
            k_recv.view(-1),
            v_cur.view(-1),
            v_recv.view(-1),
            n_k,
            put_rank=rank,
            put_next_rank=next_rank,
            heap_bases=heap_bases,
            CAUSAL=causal,
            BLOCK_Q=BLOCK_Q,
            BLOCK_KV=BLOCK_KV,
            HEAD_DIM=HEAD_DIM,
            DO_PUT=do_put,
            PUT_BLOCK=FUSED_PUT_BLOCK,
            num_warps=4,
            num_stages=2,
        )

        kernel_end.record()

        # --- Time sync + barrier ---
        if do_put:
            sync_start = torch.cuda.Event(enable_timing=True)
            sync_end = torch.cuda.Event(enable_timing=True)

            sync_start.record()
            torch.cuda.synchronize()
            sync_end.record()
            torch.cuda.synchronize()  # need to sync to read sync timing

            sync_ms = sync_start.elapsed_time(sync_end)

            barrier_wall_start = time.perf_counter()
            shmem.barrier()
            barrier_wall_end = time.perf_counter()
            barrier_ms = (barrier_wall_end - barrier_wall_start) * 1000.0

            k_cur, k_recv = k_recv, k_cur
            v_cur, v_recv = v_recv, v_cur
        else:
            torch.cuda.synchronize()
            sync_ms = 0.0
            barrier_ms = 0.0

        kernel_ms = kernel_start.elapsed_time(kernel_end)

        step_timings.append(
            {
                "step": step,
                "kv_rank": kv_rank,
                "do_put": do_put,
                "kernel_ms": kernel_ms,
                "sync_ms": sync_ms,
                "barrier_ms": barrier_ms,
                "total_ms": kernel_ms + sync_ms + barrier_ms,
            }
        )

    L_expanded = L.permute(1, 0).unsqueeze(-1)
    output = O / L_expanded

    return output.to(input_dtype), step_timings


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

    # Pre-allocate ping-pong buffers
    k_ping = shmem.empty(k.shape, dtype=k.dtype)
    k_pong = shmem.empty(k.shape, dtype=k.dtype)
    v_ping = shmem.empty(v.shape, dtype=v.dtype)
    v_pong = shmem.empty(v.shape, dtype=v.dtype)
    bufs = (k_ping, k_pong, v_ping, v_pong)

    shmem.barrier()

    # Warmup
    for _ in range(num_warmup):
        out, _ = _profiled_ring_attn_fwd(q, k, v, shmem, causal=causal, scale=scale, _ping_pong_bufs=bufs)
    torch.cuda.synchronize()
    shmem.barrier()

    # Timed iterations — collect per-step timings
    all_iter_timings = []
    for it in range(num_iters):
        out, step_timings = _profiled_ring_attn_fwd(q, k, v, shmem, causal=causal, scale=scale, _ping_pong_bufs=bufs)
        all_iter_timings.append(step_timings)
    torch.cuda.synchronize()
    shmem.barrier()

    # Aggregate: average each step's timings across iterations
    num_steps = world_size
    avg_timings = []
    for s in range(num_steps):
        kernel_vals = [all_iter_timings[it][s]["kernel_ms"] for it in range(num_iters)]
        sync_vals = [all_iter_timings[it][s]["sync_ms"] for it in range(num_iters)]
        barrier_vals = [all_iter_timings[it][s]["barrier_ms"] for it in range(num_iters)]
        total_vals = [all_iter_timings[it][s]["total_ms"] for it in range(num_iters)]
        avg_timings.append(
            {
                "step": s,
                "kv_rank": all_iter_timings[0][s]["kv_rank"],
                "do_put": all_iter_timings[0][s]["do_put"],
                "kernel_ms": sum(kernel_vals) / len(kernel_vals),
                "sync_ms": sum(sync_vals) / len(sync_vals),
                "barrier_ms": sum(barrier_vals) / len(barrier_vals),
                "total_ms": sum(total_vals) / len(total_vals),
            }
        )

    del shmem
    dist.destroy_process_group()

    if rank == 0:
        result = {
            "config": cfg,
            "world_size": world_size,
            "rank": rank,
            "per_step": avg_timings,
            "totals": {
                "kernel_ms": sum(s["kernel_ms"] for s in avg_timings),
                "sync_ms": sum(s["sync_ms"] for s in avg_timings),
                "barrier_ms": sum(s["barrier_ms"] for s in avg_timings),
                "total_ms": sum(s["total_ms"] for s in avg_timings),
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
            f"Ring Attention Profiling — {world_size} GPUs, seq={cfg['total_seq']}, "
            f"H={cfg['num_heads']}, D={cfg['head_dim']}, causal={cfg['causal']}"
        )
        print(f"{'=' * 80}")

        print(
            f"\n{'step':>4} {'kv_rank':>7} {'put':>4} {'kernel':>9} {'sync':>9} {'barrier':>9} {'total':>9}"
        )
        print("-" * 65)
        for s in result["per_step"]:
            print(
                f"{s['step']:>4} {s['kv_rank']:>7} {str(s['do_put']):>4} "
                f"{s['kernel_ms']:>8.3f}ms {s['sync_ms']:>8.3f}ms {s['barrier_ms']:>8.3f}ms {s['total_ms']:>8.3f}ms"
            )

        print("\n--- Totals (rank 0) ---")
        print(
            f"  Kernel compute : {totals['kernel_ms']:>8.3f} ms ({100 * totals['kernel_ms'] / totals['total_ms']:>5.1f}%)"
        )
        print(
            f"  CUDA sync      : {totals['sync_ms']:>8.3f} ms ({100 * totals['sync_ms'] / totals['total_ms']:>5.1f}%)"
        )
        print(
            f"  Barrier        : {totals['barrier_ms']:>8.3f} ms ({100 * totals['barrier_ms'] / totals['total_ms']:>5.1f}%)"
        )
        print(f"  TOTAL          : {totals['total_ms']:>8.3f} ms")

        # Compute efficiency
        seq_local = cfg["total_seq"] // world_size
        flops = 4 * seq_local * cfg["total_seq"] * cfg["head_dim"] * cfg["num_heads"]
        if cfg["causal"]:
            flops //= 2
        tflops = flops / (totals["total_ms"] * 1e-3) / 1e12
        print(f"  TFLOPS         : {tflops:>8.2f}")
        print(f"  MFU (vs 1307)  : {100 * tflops / 1307.4:>7.1f}%")
        print()

    finally:
        os.unlink(results_file)


if __name__ == "__main__":
    main()
