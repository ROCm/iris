#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
# Copyright (c) 2025 Advanced Micro Devices, Inc. All rights reserved.

"""
Ring Attention Scaling Benchmark.

Evaluates strong and weak scaling of ring attention on AMD MI300X GPUs.

**Strong scaling** (fixed total problem size, increasing world_size):
    Total sequence length is held constant; adding more GPUs should reduce
    latency proportionally.  Ideal strong-scaling speedup = world_size.

**Weak scaling** (fixed per-GPU problem size, increasing world_size):
    Each GPU always processes seq_local tokens; adding more GPUs increases
    the total sequence while keeping per-GPU work constant.  Ideal weak-scaling
    efficiency = 100% (flat latency).

The reference is PyTorch ``scaled_dot_product_attention`` running the *full*
sequence on a *single* GPU, which is the baseline both scaling analyses are
measured against.

Usage::

    # Full sweep: world_size in [1, 2, 4, 8], save plots
    python examples/32_ring_attention/scaling_benchmark.py --save_fig scaling.png

    # Quick test with 2 and 4 GPUs only
    python examples/32_ring_attention/scaling_benchmark.py --world_sizes 2 4

    # Show table only (no plotting)
    python examples/32_ring_attention/scaling_benchmark.py --no_plot
"""

import argparse
import json
import os
import sys
import tempfile
from pathlib import Path
from typing import Any

import torch
import torch.distributed as dist
import torch.multiprocessing as mp

import iris

sys.path.insert(0, str(Path(__file__).resolve().parent))

from ring_attention_layer import RingAttention  # noqa: E402


# ---------------------------------------------------------------------------
# Hardware peak specs (MI300X / gfx942)
# ---------------------------------------------------------------------------

_MI300X_FP16_TFLOPS = 1307.4
_MI300X_MEMBW_GBS = 5300.0
_MI300X_CU_COUNT = 304
_FALLBACK_FP16_TFLOPS = 100.0
_FALLBACK_MEMBW_GBS = 500.0
_GB_TO_TB = 1e3


def _get_hw_specs(device: torch.device) -> tuple[float, float]:
    try:
        props = torch.cuda.get_device_properties(device)
        name = props.name.lower()
        if "gfx942" in name or "mi300" in name or props.multi_processor_count == _MI300X_CU_COUNT:
            return _MI300X_FP16_TFLOPS, _MI300X_MEMBW_GBS
    except Exception:
        pass
    return _FALLBACK_FP16_TFLOPS, _FALLBACK_MEMBW_GBS


# ---------------------------------------------------------------------------
# Timing
# ---------------------------------------------------------------------------


def _time_ms(fn, warmup: int = 3, iters: int = 10) -> float:
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    times = []
    for _ in range(iters):
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        fn()
        end.record()
        torch.cuda.synchronize()
        times.append(start.elapsed_time(end))
    times.sort()
    return times[len(times) // 2]


# ---------------------------------------------------------------------------
# FLOPs helpers
# ---------------------------------------------------------------------------


def _attn_flops(seq_q: int, seq_kv: int, num_heads: int, head_dim: int, causal: bool) -> int:
    flops = 4 * seq_q * seq_kv * head_dim * num_heads
    return flops // 2 if causal else flops


# ---------------------------------------------------------------------------
# Worker
# ---------------------------------------------------------------------------


def _scaling_worker(
    rank: int,
    world_size: int,
    init_url: str,
    num_heads: int,
    head_dim: int,
    dtype_str: str,
    causal: bool,
    # Strong scaling: fixed total_seq list
    strong_total_seqs: list[int],
    # Weak scaling: fixed seq_local list
    weak_seq_locals: list[int],
    num_warmup: int,
    num_iters: int,
    results_file: str,
):
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
    device = torch.device(f"cuda:{rank}")
    peak_tflops, peak_bw = _get_hw_specs(device)
    dtype = getattr(torch, dtype_str)
    scale = head_dim**-0.5

    strong_results = []
    weak_results = []

    # ------------------------------------------------------------------
    # Helper: time ring attention for given seq_local
    # ------------------------------------------------------------------
    def _run_ring(seq_local: int, _shmem) -> float:
        torch.manual_seed(42 + rank)
        q = torch.randn(seq_local, num_heads, head_dim, dtype=dtype)
        k = torch.randn_like(q)
        v = torch.randn_like(q)
        layer = RingAttention(_shmem, num_heads=num_heads, head_dim=head_dim, causal=causal, scale=scale)
        _shmem.barrier()
        ms = _time_ms(lambda: layer(q, k, v), warmup=num_warmup, iters=num_iters)
        _shmem.barrier()
        return ms

    # ------------------------------------------------------------------
    # Helper: time single-GPU SDPA (rank 0 only, full sequence)
    # ------------------------------------------------------------------
    def _run_sdpa(total_seq: int) -> float | None:
        if rank != 0:
            return None
        try:
            q_f = torch.randn(num_heads, total_seq, head_dim, dtype=dtype)
            k_f = torch.randn_like(q_f)
            v_f = torch.randn_like(q_f)
            ms = _time_ms(
                lambda: torch.nn.functional.scaled_dot_product_attention(
                    q_f, k_f, v_f, scale=scale, is_causal=causal
                ),
                warmup=num_warmup,
                iters=num_iters,
            )
            return ms
        except torch.OutOfMemoryError:
            print(f"[WARN] SDPA reference OOM at total_seq={total_seq}, skipping")
            torch.cuda.empty_cache()
            return float("nan")

    # ------------------------------------------------------------------
    # STRONG SCALING: fixed total_seq, world_size GPUs
    # ------------------------------------------------------------------
    for total_seq in strong_total_seqs:
        if total_seq % (64 * world_size) != 0:
            continue
        seq_local = total_seq // world_size
        ring_ms = _run_ring(seq_local, shmem)
        ref_ms = _run_sdpa(total_seq)

        if rank == 0:
            ring_flops = _attn_flops(seq_local, total_seq, num_heads, head_dim, causal)
            ring_tflops = ring_flops / (ring_ms * 1e-3) / 1e12

            ref_flops = _attn_flops(total_seq, total_seq, num_heads, head_dim, causal)
            ref_tflops = ref_flops / (ref_ms * 1e-3) / 1e12

            strong_results.append(
                {
                    "total_seq": total_seq,
                    "world_size": world_size,
                    "seq_local": seq_local,
                    "ring_ms": ring_ms,
                    "ref_ms": ref_ms,
                    "speedup": ref_ms / ring_ms,
                    "ideal_speedup": float(world_size),
                    "scaling_efficiency": ref_ms / (ring_ms * world_size),
                    "ring_tflops": ring_tflops,
                    "ref_tflops": ref_tflops,
                    "peak_tflops": peak_tflops,
                    "peak_bw_gbs": peak_bw,
                }
            )

    shmem.barrier()

    # ------------------------------------------------------------------
    # WEAK SCALING: fixed seq_local per GPU, world_size GPUs
    # ------------------------------------------------------------------
    for seq_local in weak_seq_locals:
        if seq_local % 64 != 0:
            continue
        total_seq = seq_local * world_size
        ring_ms = _run_ring(seq_local, shmem)
        # Reference = single-GPU SDPA on the *full* sequence (total_seq)
        ref_ms = _run_sdpa(total_seq)

        if rank == 0:
            ring_flops = _attn_flops(seq_local, total_seq, num_heads, head_dim, causal)
            ring_tflops = ring_flops / (ring_ms * 1e-3) / 1e12

            ref_flops = _attn_flops(total_seq, total_seq, num_heads, head_dim, causal)
            ref_tflops = ref_flops / (ref_ms * 1e-3) / 1e12

            weak_results.append(
                {
                    "seq_local": seq_local,
                    "total_seq": total_seq,
                    "world_size": world_size,
                    "ring_ms": ring_ms,
                    "ref_ms": ref_ms,
                    "speedup": ref_ms / ring_ms,
                    "ring_tflops": ring_tflops,
                    "ref_tflops": ref_tflops,
                    "peak_tflops": peak_tflops,
                    "peak_bw_gbs": peak_bw,
                }
            )

    shmem.barrier()
    del shmem
    dist.destroy_process_group()

    if rank == 0:
        with open(results_file, "w") as f:
            json.dump({"strong": strong_results, "weak": weak_results}, f, indent=2)


# ---------------------------------------------------------------------------
# Print tables
# ---------------------------------------------------------------------------


def _print_strong_table(strong: list[dict[str, Any]]):
    print("\n=== STRONG SCALING (fixed total_seq, increasing world_size) ===")
    hdr = f"{'total_seq':>10} {'GPUs':>5} {'ring ms':>9} {'ref ms':>9} {'speedup':>8} {'ideal':>6} {'eff%':>7} {'ring TF':>9} {'ref TF':>9}"
    print("=" * len(hdr))
    print(hdr)
    print("=" * len(hdr))
    for r in sorted(strong, key=lambda x: (x["total_seq"], x["world_size"])):
        eff = 100.0 * r["scaling_efficiency"]
        print(
            f"{r['total_seq']:>10} {r['world_size']:>5} {r['ring_ms']:>9.3f} {r['ref_ms']:>9.3f} "
            f"{r['speedup']:>8.2f}x {r['ideal_speedup']:>6.1f}x {eff:>6.1f}% "
            f"{r['ring_tflops']:>9.2f} {r['ref_tflops']:>9.2f}"
        )
    print("=" * len(hdr))


def _print_weak_table(weak: list[dict[str, Any]]):
    print("\n=== WEAK SCALING (fixed seq_local per GPU, increasing world_size) ===")
    hdr = f"{'seq_local':>10} {'total_seq':>10} {'GPUs':>5} {'ring ms':>9} {'ref ms':>9} {'speedup':>8} {'ring TF':>9} {'ref TF':>9}"
    print("=" * len(hdr))
    print(hdr)
    print("=" * len(hdr))
    for r in sorted(weak, key=lambda x: (x["seq_local"], x["world_size"])):
        print(
            f"{r['seq_local']:>10} {r['total_seq']:>10} {r['world_size']:>5} "
            f"{r['ring_ms']:>9.3f} {r['ref_ms']:>9.3f} {r['speedup']:>8.2f}x "
            f"{r['ring_tflops']:>9.2f} {r['ref_tflops']:>9.2f}"
        )
    print("=" * len(hdr))


# ---------------------------------------------------------------------------
# Plot
# ---------------------------------------------------------------------------


def _make_scaling_plots(
    strong: list[dict[str, Any]],
    weak: list[dict[str, Any]],
    num_heads: int,
    head_dim: int,
    causal: bool,
    save_fig: str | None,
):
    import matplotlib
    import matplotlib.pyplot as plt
    import numpy as np

    if save_fig:
        matplotlib.use("Agg")

    _print_strong_table(strong)
    _print_weak_table(weak)

    # ---- Layout: 2×2 ----
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle(
        f"Ring Attention Scaling — AMD MI300X (gfx942), FP16, causal={causal}\nH={num_heads}, D={head_dim}",
        fontsize=13,
        fontweight="bold",
    )

    # --- 1. Strong scaling: latency vs world_size per total_seq ---
    ax = axes[0, 0]
    total_seqs_ss = sorted(set(r["total_seq"] for r in strong))
    colors_ss = plt.cm.tab10(np.linspace(0, 0.9, len(total_seqs_ss)))
    for ts, col in zip(total_seqs_ss, colors_ss):
        pts = sorted([r for r in strong if r["total_seq"] == ts], key=lambda x: x["world_size"])
        if not pts:
            continue
        ws_vals = [p["world_size"] for p in pts]
        ring_ms = [p["ring_ms"] for p in pts]
        ref_ms = pts[0]["ref_ms"]  # single-GPU reference is constant

        ax.plot(ws_vals, ring_ms, "o-", color=col, linewidth=2, markersize=8, label=f"Ring S={ts}")
        # Ideal scaling: ref_ms / world_size
        ideal = [ref_ms / ws for ws in ws_vals]
        ax.plot(ws_vals, ideal, "--", color=col, linewidth=1.2, alpha=0.5)
    ax.set_xlabel("Number of GPUs")
    ax.set_ylabel("Latency (ms)")
    ax.set_title("Strong Scaling: Latency vs. GPU Count\n(dashed = ideal 1/N scaling)")
    ax.set_xticks(sorted(set(r["world_size"] for r in strong)))
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    # --- 2. Strong scaling: scaling efficiency % ---
    ax = axes[0, 1]
    for ts, col in zip(total_seqs_ss, colors_ss):
        pts = sorted([r for r in strong if r["total_seq"] == ts], key=lambda x: x["world_size"])
        if not pts:
            continue
        ws_vals = [p["world_size"] for p in pts]
        eff = [100.0 * p["scaling_efficiency"] for p in pts]
        ax.plot(ws_vals, eff, "s-", color=col, linewidth=2, markersize=8, label=f"S={ts}")
    ax.axhline(100, color="gray", linestyle="--", alpha=0.6, linewidth=1.5, label="Ideal (100%)")
    ax.set_xlabel("Number of GPUs")
    ax.set_ylabel("Strong Scaling Efficiency (%)")
    ax.set_title("Strong Scaling Efficiency\n(100% = perfect linear speedup)")
    ax.set_xticks(sorted(set(r["world_size"] for r in strong)))
    ax.set_ylim(0, 130)
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    # --- 3. Weak scaling: latency vs world_size per seq_local ---
    ax = axes[1, 0]
    seq_locals_ws = sorted(set(r["seq_local"] for r in weak))
    colors_ws = plt.cm.tab10(np.linspace(0, 0.9, len(seq_locals_ws)))
    for sl, col in zip(seq_locals_ws, colors_ws):
        pts = sorted([r for r in weak if r["seq_local"] == sl], key=lambda x: x["world_size"])
        if not pts:
            continue
        ws_vals = [p["world_size"] for p in pts]
        ring_ms = [p["ring_ms"] for p in pts]
        # Baseline = single-GPU ring at world_size=1 (first point if available,
        # else we just plot relative to the first measured point)
        ax.plot(ws_vals, ring_ms, "o-", color=col, linewidth=2, markersize=8, label=f"Ring S_local={sl}")
        # Ideal weak scaling = flat (constant latency)
        ax.axhline(ring_ms[0], color=col, linestyle="--", linewidth=1.2, alpha=0.4)
    ax.set_xlabel("Number of GPUs")
    ax.set_ylabel("Latency (ms)")
    ax.set_title("Weak Scaling: Latency vs. GPU Count\n(dashed = ideal flat latency)")
    ax.set_xticks(sorted(set(r["world_size"] for r in weak)))
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    # --- 4. Throughput (TFLOPS per GPU) vs world_size for both strong & weak ---
    ax = axes[1, 1]
    # Strong scaling TFLOPS
    for ts, col in zip(total_seqs_ss, colors_ss):
        pts = sorted([r for r in strong if r["total_seq"] == ts], key=lambda x: x["world_size"])
        if not pts:
            continue
        ws_vals = [p["world_size"] for p in pts]
        tfl = [p["ring_tflops"] for p in pts]
        ax.plot(ws_vals, tfl, "o-", color=col, linewidth=2, markersize=8, label=f"Strong S={ts}")

    ax.set_xlabel("Number of GPUs")
    ax.set_ylabel("TFLOPS (per rank)")
    ax.set_title("Per-Rank Throughput vs. GPU Count\n(strong scaling)")
    ax.set_xticks(sorted(set(r["world_size"] for r in strong)))
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()

    if save_fig:
        plt.savefig(save_fig, dpi=150, bbox_inches="tight")
        print(f"\nSaved scaling figure to: {save_fig}")
    else:
        plt.show()


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def parse_args():
    p = argparse.ArgumentParser(description="Ring Attention strong/weak scaling benchmark")
    p.add_argument(
        "--world_sizes",
        nargs="+",
        type=int,
        default=[1, 2, 4, 8],
        help="GPU counts to benchmark (default: 1 2 4 8)",
    )
    p.add_argument("--num_heads", type=int, default=16)
    p.add_argument("--head_dim", type=int, default=64)
    p.add_argument(
        "--strong_seqs",
        nargs="+",
        type=int,
        default=[4096, 8192, 16384],
        help="Fixed total sequence lengths for strong-scaling sweep",
    )
    p.add_argument(
        "--weak_seq_locals",
        nargs="+",
        type=int,
        default=[1024, 2048, 4096],
        help="Fixed per-GPU sequence lengths for weak-scaling sweep",
    )
    p.add_argument("--no_causal", dest="causal", action="store_false", default=True)
    p.add_argument("--dtype", choices=["float16", "bfloat16"], default="float16")
    p.add_argument("--warmup", type=int, default=5)
    p.add_argument("--iters", type=int, default=20)
    p.add_argument("--save_fig", type=str, default=None)
    p.add_argument("--no_plot", action="store_true")
    return p.parse_args()


def main():
    args = parse_args()

    all_strong: list[dict] = []
    all_weak: list[dict] = []

    # Special case: world_size=1 means single-GPU ring (= SDPA, measured directly)
    if 1 in args.world_sizes:
        print("Measuring world_size=1 (single GPU, ring degenerates to SDPA)...")
        device = torch.device("cuda:0")
        torch.cuda.set_device(0)
        peak_tflops, peak_bw = (
            _MI300X_FP16_TFLOPS
            if torch.cuda.get_device_properties(0).multi_processor_count == _MI300X_CU_COUNT
            else _FALLBACK_FP16_TFLOPS,
            _MI300X_MEMBW_GBS
            if torch.cuda.get_device_properties(0).multi_processor_count == _MI300X_CU_COUNT
            else _FALLBACK_MEMBW_GBS,
        )
        dtype = getattr(torch, args.dtype)
        scale = args.head_dim**-0.5

        for total_seq in args.strong_seqs:
            if total_seq % 64 != 0:
                continue
            q_f = torch.randn(args.num_heads, total_seq, args.head_dim, dtype=dtype, device=device)
            k_f = torch.randn_like(q_f)
            v_f = torch.randn_like(q_f)
            ms = _time_ms(
                lambda: torch.nn.functional.scaled_dot_product_attention(
                    q_f, k_f, v_f, scale=scale, is_causal=args.causal
                ),
                warmup=args.warmup,
                iters=args.iters,
            )
            ref_flops = _attn_flops(total_seq, total_seq, args.num_heads, args.head_dim, args.causal)
            ref_tflops = ref_flops / (ms * 1e-3) / 1e12
            all_strong.append(
                {
                    "total_seq": total_seq,
                    "world_size": 1,
                    "seq_local": total_seq,
                    "ring_ms": ms,
                    "ref_ms": ms,
                    "speedup": 1.0,
                    "ideal_speedup": 1.0,
                    "scaling_efficiency": 1.0,
                    "ring_tflops": ref_tflops,
                    "ref_tflops": ref_tflops,
                    "peak_tflops": peak_tflops,
                    "peak_bw_gbs": peak_bw,
                }
            )

        for seq_local in args.weak_seq_locals:
            if seq_local % 64 != 0:
                continue
            q_f = torch.randn(args.num_heads, seq_local, args.head_dim, dtype=dtype, device=device)
            k_f = torch.randn_like(q_f)
            v_f = torch.randn_like(q_f)
            ms = _time_ms(
                lambda: torch.nn.functional.scaled_dot_product_attention(
                    q_f, k_f, v_f, scale=scale, is_causal=args.causal
                ),
                warmup=args.warmup,
                iters=args.iters,
            )
            ref_flops = _attn_flops(seq_local, seq_local, args.num_heads, args.head_dim, args.causal)
            ref_tflops = ref_flops / (ms * 1e-3) / 1e12
            all_weak.append(
                {
                    "seq_local": seq_local,
                    "total_seq": seq_local,
                    "world_size": 1,
                    "ring_ms": ms,
                    "ref_ms": ms,
                    "speedup": 1.0,
                    "ring_tflops": ref_tflops,
                    "ref_tflops": ref_tflops,
                    "peak_tflops": peak_tflops,
                    "peak_bw_gbs": peak_bw,
                }
            )

        torch.cuda.empty_cache()

    for world_size in sorted(args.world_sizes):
        if world_size == 1:
            continue
        if world_size > torch.cuda.device_count():
            print(f"[skip] world_size={world_size} > available GPUs ({torch.cuda.device_count()})")
            continue

        print(f"\nRunning world_size={world_size}...")

        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            results_file = f.name

        port = 29500 + world_size  # unique port per world_size to avoid conflicts
        init_url = f"tcp://127.0.0.1:{port}"

        try:
            mp.spawn(
                fn=_scaling_worker,
                args=(
                    world_size,
                    init_url,
                    args.num_heads,
                    args.head_dim,
                    args.dtype,
                    args.causal,
                    args.strong_seqs,
                    args.weak_seq_locals,
                    args.warmup,
                    args.iters,
                    results_file,
                ),
                nprocs=world_size,
                join=True,
            )
            with open(results_file) as f:
                data = json.load(f)
            all_strong.extend(data["strong"])
            all_weak.extend(data["weak"])
        finally:
            os.unlink(results_file)

    if not args.no_plot:
        _make_scaling_plots(all_strong, all_weak, args.num_heads, args.head_dim, args.causal, save_fig=args.save_fig)
    else:
        _print_strong_table(all_strong)
        _print_weak_table(all_weak)


if __name__ == "__main__":
    main()
