#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
# Copyright (c) 2025 Advanced Micro Devices, Inc. All rights reserved.

"""
Ring Attention benchmark: performance sweep and roofline analysis.

Measures ring attention throughput across a range of sequence lengths, compares
against a single-device PyTorch ``scaled_dot_product_attention`` reference, and
generates a roofline plot with a performance table.

Usage::

    # 2-GPU sweep (default)
    python examples/32_ring_attention/benchmark.py

    # 4-GPU sweep
    python examples/32_ring_attention/benchmark.py --num_ranks 4

    # Save plots to a file instead of showing interactively
    python examples/32_ring_attention/benchmark.py --save_fig bench.png

Hardware targets (auto-detected from ``rocminfo`` / ``hipGetDeviceProperties``):

    * AMD Instinct MI300X (gfx942): FP16 peak ≈ 1307 TFLOPS, BW ≈ 5300 GB/s
    * Falls back to conservative estimates when hardware info is unavailable.
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

project_root = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(Path(__file__).resolve().parent))

from ring_attention_layer import RingAttention  # noqa: E402


# ---------------------------------------------------------------------------
# Hardware peak specs (MI300X / gfx942 defaults)
# ---------------------------------------------------------------------------

# FP16 matrix peak (TFLOPS) and memory bandwidth (GB/s) for MI300X.
# Source: https://www.amd.com/en/products/accelerators/instinct/mi300/mi300x.html
_MI300X_FP16_TFLOPS = 1307.4
_MI300X_MEMBW_GBS = 5300.0

# MI300X has exactly 304 compute units (used as a fingerprint when the device name
# does not contain an explicit architecture string).
_MI300X_CU_COUNT = 304

# Fallback conservative estimates for unknown hardware
_FALLBACK_FP16_TFLOPS = 100.0
_FALLBACK_MEMBW_GBS = 500.0

# Unit conversion: 1 TB/s = 1000 GB/s
_GB_TO_TB = 1e3


def _get_hw_specs(device: torch.device) -> tuple[float, float]:
    """
    Return (peak_fp16_tflops, peak_membw_gbs) for the given device.

    Detects MI300X by GFX version; falls back to conservative defaults
    for unknown hardware.
    """
    try:
        props = torch.cuda.get_device_properties(device)
        name = props.name.lower()
        # gfx942 = MI300X / MI300A family; 304 CUs is the MI300X fingerprint
        if "gfx942" in name or "mi300" in name or (props.multi_processor_count == _MI300X_CU_COUNT):
            return _MI300X_FP16_TFLOPS, _MI300X_MEMBW_GBS
    except Exception:
        pass
    return _FALLBACK_FP16_TFLOPS, _FALLBACK_MEMBW_GBS


# ---------------------------------------------------------------------------
# FLOPs / bytes helpers
# ---------------------------------------------------------------------------


def _attn_flops(seq_q: int, seq_kv: int, num_heads: int, head_dim: int, causal: bool) -> int:
    """
    Theoretical FLOPs for one attention forward pass (QK^T + softmax + AV).

    Flash-attention FLOPs (no materialised S×S matrix):
        QK^T  :  2 * seq_q * seq_kv * head_dim   per head
        AV    :  2 * seq_q * seq_kv * head_dim   per head
        Total :  4 * seq_q * seq_kv * head_dim * num_heads

    For causal attention roughly half the token-pairs are skipped, so we
    apply a 0.5 factor (exact only for the diagonal block; used as an
    approximation for the whole pass).
    """
    flops = 4 * seq_q * seq_kv * head_dim * num_heads
    if causal:
        flops = flops // 2
    return flops


def _attn_bytes(seq_q: int, seq_kv: int, num_heads: int, head_dim: int, elem_bytes: int = 2) -> int:
    """
    Bytes accessed by a tiled flash-attention kernel (no S×S HBM spill):
        Reads : Q [seq_q × H × D] + K [seq_kv × H × D] + V [seq_kv × H × D]
        Writes: O [seq_q × H × D]
    """
    return elem_bytes * num_heads * head_dim * (2 * seq_q + 2 * seq_kv)


# ---------------------------------------------------------------------------
# Timing utilities
# ---------------------------------------------------------------------------


def _time_ms(fn, warmup: int = 3, iters: int = 10) -> float:
    """Return median latency in ms over *iters* timed calls after *warmup*."""
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
    return times[len(times) // 2]  # median


# ---------------------------------------------------------------------------
# Benchmark worker (runs inside each spawned process)
# ---------------------------------------------------------------------------


def _benchmark_worker(
    rank: int,
    world_size: int,
    init_url: str,
    configs: list[dict[str, Any]],
    results_file: str,
    causal: bool,
    num_warmup: int,
    num_iters: int,
):
    """
    Worker function executed by each GPU rank.

    Rank 0 also runs the single-device SDPA reference and writes results to
    *results_file* as JSON.
    """
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

    results = []

    for cfg in configs:
        total_seq = cfg["total_seq"]
        num_heads = cfg["num_heads"]
        head_dim = cfg["head_dim"]
        dtype = getattr(torch, cfg["dtype"])
        elem_bytes = 2  # fp16 / bf16

        seq_local = total_seq // world_size
        scale = head_dim**-0.5

        torch.manual_seed(42 + rank)
        q = torch.randn(seq_local, num_heads, head_dim, dtype=dtype)
        k = torch.randn(seq_local, num_heads, head_dim, dtype=dtype)
        v = torch.randn(seq_local, num_heads, head_dim, dtype=dtype)

        layer = RingAttention(shmem, num_heads=num_heads, head_dim=head_dim, causal=causal, scale=scale)

        shmem.barrier()

        # ---- Ring attention timing ----
        ring_ms = _time_ms(lambda: layer(q, k, v), warmup=num_warmup, iters=num_iters)

        # All ranks need to sync before SDPA
        shmem.barrier()

        # ---- Reference SDPA on rank 0 (full sequence, single GPU) ----
        ref_ms = None
        if rank == 0:
            try:
                q_full = torch.randn(total_seq, num_heads, head_dim, dtype=dtype)
                k_full = torch.randn_like(q_full)
                v_full = torch.randn_like(q_full)

                # [S, H, D] → [H, S, D] for SDPA
                q_f = q_full.permute(1, 0, 2)
                k_f = k_full.permute(1, 0, 2)
                v_f = v_full.permute(1, 0, 2)

                ref_ms = _time_ms(
                    lambda: torch.nn.functional.scaled_dot_product_attention(
                        q_f, k_f, v_f, scale=scale, is_causal=causal
                    ),
                    warmup=num_warmup,
                    iters=num_iters,
                )
            except torch.OutOfMemoryError:
                print(f"[WARN] SDPA reference OOM at total_seq={total_seq}, skipping")
                ref_ms = float("nan")
                torch.cuda.empty_cache()

            # ---- FLOPs (per rank) ----
            # Ring attention: seq_q × total_seq attention per rank
            ring_flops = _attn_flops(seq_local, total_seq, num_heads, head_dim, causal)
            # Reference: total_seq × total_seq on a single device
            ref_flops = _attn_flops(total_seq, total_seq, num_heads, head_dim, causal)

            # ---- Arithmetic intensity (flash-attn, per rank) ----
            ring_bytes = 0
            for _step in range(world_size):
                ring_bytes += _attn_bytes(seq_local, seq_local, num_heads, head_dim, elem_bytes)
            ring_ai = ring_flops / ring_bytes  # FLOPs/byte

            ref_bytes = _attn_bytes(total_seq, total_seq, num_heads, head_dim, elem_bytes)
            ref_ai = ref_flops / ref_bytes

            ring_tflops = ring_flops / (ring_ms * 1e-3) / 1e12
            ref_tflops = ref_flops / (ref_ms * 1e-3) / 1e12

            results.append(
                {
                    "total_seq": total_seq,
                    "num_heads": num_heads,
                    "head_dim": head_dim,
                    "world_size": world_size,
                    "causal": causal,
                    "dtype": cfg["dtype"],
                    # timings
                    "ring_ms": ring_ms,
                    "ref_ms": ref_ms,
                    "speedup": ref_ms / ring_ms,
                    # TFLOPS
                    "ring_tflops": ring_tflops,
                    "ref_tflops": ref_tflops,
                    # Arithmetic intensity
                    "ring_ai": ring_ai,
                    "ref_ai": ref_ai,
                    # Hardware peaks
                    "peak_tflops": peak_tflops,
                    "peak_bw_gbs": peak_bw,
                }
            )

        shmem.barrier()

    del shmem
    dist.destroy_process_group()

    # Write results from rank 0 to the shared temp file
    if rank == 0:
        with open(results_file, "w") as f:
            json.dump(results, f, indent=2)


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------


def _print_table(results: list[dict[str, Any]]):
    """Print a performance summary table to stdout."""
    if not results:
        print("No results.")
        return
    peak_tflops = results[0]["peak_tflops"]
    hdr = (
        f"{'seq':>8} {'H':>4} {'D':>4} "
        f"{'ring ms':>9} {'ref ms':>9} {'speedup':>8} "
        f"{'ring TFLOPS':>12} {'ref TFLOPS':>12} "
        f"{'ring eff%':>10} {'ref eff%':>10}"
    )
    print()
    print("=" * len(hdr))
    print(hdr)
    print("=" * len(hdr))
    for r in results:
        ring_eff = 100.0 * r["ring_tflops"] / peak_tflops
        ref_eff = 100.0 * r["ref_tflops"] / peak_tflops
        print(
            f"{r['total_seq']:>8} {r['num_heads']:>4} {r['head_dim']:>4} "
            f"{r['ring_ms']:>9.3f} {r['ref_ms']:>9.3f} {r['speedup']:>8.2f}x "
            f"{r['ring_tflops']:>12.2f} {r['ref_tflops']:>12.2f} "
            f"{ring_eff:>9.1f}% {ref_eff:>9.1f}%"
        )
    print("=" * len(hdr))


def _make_plots(results: list[dict[str, Any]], save_fig: str | None):
    """Generate performance table + roofline plot."""
    import matplotlib
    import matplotlib.pyplot as plt

    if save_fig:
        matplotlib.use("Agg")

    if not results:
        print("No results to plot.")
        return

    _print_table(results)

    peak_tflops = results[0]["peak_tflops"]
    peak_bw = results[0]["peak_bw_gbs"]
    world_size = results[0]["world_size"]

    # ---- Roofline plot ----
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))

    # Left: Roofline
    ax = axes[0]
    ai_vals = [r["ring_ai"] for r in results] + [r["ref_ai"] for r in results]
    ai_min = min(ai_vals) * 0.5
    ai_max = max(ai_vals) * 2.0
    ai_range = [ai_min, ai_max]

    # Roofline ceiling: ridge point converts BW from GB/s to TB/s for TFLOPS units
    ridge = peak_tflops / peak_bw * _GB_TO_TB  # ridge point (FLOPs/byte)
    ai_plot = [ai_min, ridge, ai_max]
    roof = [min(peak_tflops, a * peak_bw / _GB_TO_TB) for a in ai_plot]
    ax.loglog(ai_plot, roof, "k--", linewidth=2, label="Roofline (MI300X)")
    ax.axhline(peak_tflops, color="gray", linestyle=":", alpha=0.6, label=f"Peak FP16 ({peak_tflops:.0f} TFLOPS)")
    ax.axvline(ridge, color="gray", linestyle=":", alpha=0.6, label=f"Ridge ({ridge:.1f} FLOP/B)")

    # Ring attention points
    for r in results:
        ax.scatter(r["ring_ai"], r["ring_tflops"], marker="o", s=80, zorder=5)
        ax.annotate(
            f"S={r['total_seq']}",
            (r["ring_ai"], r["ring_tflops"]),
            textcoords="offset points",
            xytext=(4, 4),
            fontsize=7,
        )

    # Reference points
    for r in results:
        ax.scatter(r["ref_ai"], r["ref_tflops"], marker="^", s=80, zorder=5, color="tab:orange")

    import matplotlib.lines as mlines

    ring_handle = mlines.Line2D(
        [], [], color="tab:blue", marker="o", linestyle="None", markersize=8, label="Ring attn (per rank)"
    )
    ref_handle = mlines.Line2D(
        [], [], color="tab:orange", marker="^", linestyle="None", markersize=8, label="SDPA reference (single GPU)"
    )
    ax.legend(handles=[ring_handle, ref_handle] + ax.get_legend_handles_labels()[0][:3], fontsize=8)

    ax.set_xlabel("Arithmetic Intensity (FLOP/Byte)")
    ax.set_ylabel("Performance (TFLOPS)")
    ax.set_title(f"Roofline — AMD MI300X (gfx942)\n{world_size} GPUs, causal={results[0]['causal']}")
    ax.set_xlim(ai_range)
    ax.grid(True, which="both", alpha=0.3)

    # Right: Latency comparison bar chart
    ax2 = axes[1]
    seqs = [r["total_seq"] for r in results]
    ring_ms = [r["ring_ms"] for r in results]
    ref_ms = [r["ref_ms"] for r in results]

    x = range(len(seqs))
    width = 0.35
    bars1 = ax2.bar(
        [i - width / 2 for i in x], ring_ms, width, label=f"Ring attn ({world_size} GPUs)", color="tab:blue", alpha=0.8
    )
    bars2 = ax2.bar([i + width / 2 for i in x], ref_ms, width, label="SDPA ref (1 GPU)", color="tab:orange", alpha=0.8)

    # Add speedup annotations
    for i, r in enumerate(results):
        ax2.text(i, max(ring_ms[i], ref_ms[i]) * 1.05, f"{r['speedup']:.1f}x", ha="center", fontsize=8, color="green")

    ax2.set_xticks(list(x))
    ax2.set_xticklabels([f"S={s}" for s in seqs], rotation=30)
    ax2.set_ylabel("Latency (ms)")
    ax2.set_title(
        f"Latency: Ring Attention vs SDPA Reference\nH={results[0]['num_heads']}, D={results[0]['head_dim']}, causal={results[0]['causal']}"
    )
    ax2.legend()
    ax2.grid(axis="y", alpha=0.3)

    plt.tight_layout()

    if save_fig:
        plt.savefig(save_fig, dpi=150, bbox_inches="tight")
        print(f"\nSaved figure to: {save_fig}")
    else:
        plt.show()


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def parse_args():
    p = argparse.ArgumentParser(description="Ring Attention benchmark + roofline")
    p.add_argument("--num_ranks", type=int, default=2, help="Number of GPUs")
    p.add_argument("--num_heads", type=int, default=16, help="Number of attention heads")
    p.add_argument("--head_dim", type=int, default=64, help="Head dimension")
    p.add_argument(
        "--total_seq_lens",
        nargs="+",
        type=int,
        default=[512, 1024, 2048, 4096, 8192],
        help="Total sequence lengths to sweep",
    )
    p.add_argument(
        "--no_causal", dest="causal", action="store_false", default=True, help="Non-causal (bidirectional) attention"
    )
    p.add_argument("--dtype", choices=["float16", "bfloat16"], default="float16")
    p.add_argument("--warmup", type=int, default=5, help="Warm-up iterations")
    p.add_argument("--iters", type=int, default=20, help="Timed iterations")
    p.add_argument("--save_fig", type=str, default=None, help="Save figure to this path (e.g. bench.png)")
    p.add_argument("--no_plot", action="store_true", help="Skip plotting")
    return p.parse_args()


def main():
    args = parse_args()
    world_size = args.num_ranks

    # Filter configs to ensure seq_len divisible by 64*world_size
    min_seq = 64 * world_size
    configs = []
    for seq in args.total_seq_lens:
        if seq % min_seq != 0:
            print(f"[skip] total_seq={seq} not divisible by {min_seq} (64 * world_size), skipping")
            continue
        if seq % world_size != 0:
            print(f"[skip] total_seq={seq} not divisible by world_size={world_size}, skipping")
            continue
        configs.append(
            {
                "total_seq": seq,
                "num_heads": args.num_heads,
                "head_dim": args.head_dim,
                "dtype": args.dtype,  # string, converted in worker
            }
        )

    if not configs:
        print("No valid configurations to benchmark.")
        return

    print("Ring Attention Benchmark")
    print(f"  GPUs        : {world_size}")
    print(f"  num_heads   : {args.num_heads}")
    print(f"  head_dim    : {args.head_dim}")
    print(f"  causal      : {args.causal}")
    print(f"  dtype       : {args.dtype}")
    print(f"  seq lengths : {[c['total_seq'] for c in configs]}")
    print(f"  warmup/iters: {args.warmup}/{args.iters}")

    # Use a temp file for results (safer than mp.Queue with mp.spawn)
    with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
        results_file = f.name

    try:
        import socket

        _sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        _sock.bind(("", 0))
        init_url = f"tcp://127.0.0.1:{_sock.getsockname()[1]}"
        _sock.close()
        mp.spawn(
            fn=_benchmark_worker,
            args=(world_size, init_url, configs, results_file, args.causal, args.warmup, args.iters),
            nprocs=world_size,
            join=True,
        )

        with open(results_file) as f:
            results = json.load(f)
    finally:
        os.unlink(results_file)

    if not args.no_plot:
        _make_plots(results, save_fig=args.save_fig)
    else:
        _print_table(results)


if __name__ == "__main__":
    main()
