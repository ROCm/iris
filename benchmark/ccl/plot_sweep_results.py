#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
# Copyright (c) 2025-2026 Advanced Micro Devices, Inc. All rights reserved.

"""Plot CCL benchmark results: Iris vs RCCL, per collective.

Consumes the CSV that ``iris.bench`` emits with ``--benchmark_format=csv`` -- one
row per measurement, with the benchmark axes as columns. The ``backend`` axis in
benchmark/ccl/bench_*.py selects Iris or RCCL for the same shapes, so both series
come out of a single sweep:

    python benchmark/ccl/bench_all_reduce.py --benchmark_format=csv \
        --benchmark_out=all_reduce.csv
    python benchmark/ccl/plot_sweep_results.py all_reduce.csv --output ccl.png

Three figures are written, mirroring benchmark/plot_bench.py in triton-shmem:

* ``<output>``                -- semilog-x bus bandwidth (GB/s) vs message size.
* ``<output stem>_latency``   -- log-log latency (ms) vs message size.
* ``<output stem>_speedup``   -- log-log latency ratio, Iris / RCCL. Lower is
                                 better; below the 1.0 line means Iris wins.

Each figure is a grid of one row per collective and one column per rank count,
so a sweep over the ``num_ranks`` axis reads left to right.

A Markdown table goes to stdout or --markdown_out, so CI can drop it into a job
summary.

On bandwidth: the bench scripts already declare bus bytes via ``state.set_bytes``
-- ``(W-1) * bytes`` for all_gather and all_to_all, ``2 * (W-1)/W * bytes`` for
all_reduce -- which is the NCCL/RCCL busBW convention. The framework divides that
by the measured time, so ``bandwidth_gbps`` is already bus bandwidth and is
directly comparable to nccl-tests output. Do not re-apply a bus factor here.

This previously read a wide CSV from comprehensive_sweep.py keyed on comm_sms.
That script was removed when the benchmarks migrated to iris.bench, which left
this reading a format nothing produced; it now reads the framework's own output
and plots against message size.
"""

import argparse
import csv
import os
import statistics
from collections import defaultdict

import matplotlib

matplotlib.use("Agg")  # headless: CI has no display

import matplotlib.pyplot as plt  # noqa: E402

IRIS_COLOR = "#2E86AB"
RCCL_COLOR = "#A23B72"
REFERENCE_BACKEND = "rccl"

# Distinguishes variants within one backend (e.g. all_reduce one_shot vs
# two_shot). Index 0 is the plain case, so a single-variant sweep looks exactly
# as it did before variants were plotted separately.
_MARKERS = ["o", "s", "^", "D", "v", "P", "X", "*"]
_LINESTYLES = ["-", "--", "-.", ":"]


def parse_args():
    parser = argparse.ArgumentParser(
        description="Plot CCL benchmark results, Iris vs RCCL.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("input_csv", nargs="+", help="CSV file(s) from iris.bench --benchmark_format=csv")
    parser.add_argument("--output", default=None, help="Bandwidth image (default: derived from the first input)")
    parser.add_argument("--markdown_out", default=None, help="Write the Markdown table here instead of stdout")
    parser.add_argument("--title", default="CCL Benchmark: Iris vs RCCL", help="Overall plot title")
    parser.add_argument("--caption", default=None, help="Small italic caption under the title (machine, ROCm, dtype)")
    parser.add_argument("--dpi", type=int, default=150, help="DPI for the output images")
    parser.add_argument("--subplot_size", type=float, nargs=2, default=[5.6, 4.4], help="Per-subplot inches (w h)")
    return parser.parse_args()


def _float(row, *keys):
    """First parseable float among ``keys``, else None."""
    for key in keys:
        value = row.get(key)
        if value not in (None, ""):
            try:
                return float(value)
            except ValueError:
                pass
    return None


# Matches _dtype_str in iris/bench/_runner.py, which writes the short name.
_ITEMSIZE = {
    "float16": 2,
    "bfloat16": 2,
    "float32": 4,
    "float64": 8,
    "int8": 1,
    "fp16": 2,
    "bf16": 2,
    "fp32": 4,
}


def _message_bytes(row):
    """Bytes per rank for this point, from the M/N/dtype axes."""
    try:
        elems = int(row["M"]) * int(row["N"])
    except (KeyError, TypeError, ValueError):
        return None
    itemsize = _ITEMSIZE.get((row.get("dtype") or "").strip())
    if itemsize is None:
        return None
    return elems * itemsize


def load_results(paths):
    """``{(op, ranks): {(backend, variant): {size_bytes: (latency_ms, bw)}}}``.

    Points are median-aggregated, so several (M, N) pairs with the same byte
    count collapse to one marker rather than stacking invisibly.
    """
    acc = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))
    for path in paths:
        with open(path, newline="") as f:
            for row in csv.DictReader(f):
                # The framework records skipped combinations with empty timings.
                if (row.get("skipped") or "").strip().lower() == "true":
                    continue
                latency = _float(row, "gpu_time_ms", "time_ms", "mean_ms")
                size = _message_bytes(row)
                if latency is None or size is None:
                    continue
                op = row.get("benchmark") or row.get("name") or os.path.basename(path).replace(".csv", "")
                backend = (row.get("backend") or "iris").strip().lower()
                variant = (row.get("variant") or "").strip()
                try:
                    ranks = int(row.get("num_ranks") or row.get("world_size") or 0)
                except ValueError:
                    ranks = 0
                bw = _float(row, "bandwidth_gbps", "bandwidth", "GB/s")
                acc[(op, ranks)][(backend, variant)][size].append((latency, bw))

    out = {}
    for key, series in acc.items():
        out[key] = {}
        for sk, sizes in series.items():
            out[key][sk] = {}
            for size, points in sizes.items():
                lat = statistics.median(p[0] for p in points)
                bws = [p[1] for p in points if p[1] is not None]
                out[key][sk][size] = (lat, statistics.median(bws) if bws else None)
    return out


def _label(backend, variant, multi_variant):
    """'Iris', or 'Iris (two_shot)' when a backend has several variants."""
    base = "Iris" if backend == "iris" else backend.upper()
    return f"{base} ({variant})" if (multi_variant and variant) else base


def _style(backend, index):
    color = IRIS_COLOR if backend == "iris" else RCCL_COLOR
    # RCCL is the reference, so it defaults to dashed even as variant 0.
    offset = 0 if backend == "iris" else 1
    return color, _MARKERS[index % len(_MARKERS)], _LINESTYLES[(index + offset) % len(_LINESTYLES)]


def _grid(data, args, suptitle):
    """Figure and axes laid out as (rank count x collective).

    Collectives run across and rank counts down, so a single-rank sweep is one
    wide row rather than a tall column, and adding the num_ranks axis grows the
    figure downwards with each collective staying in its own column.
    """
    ops = sorted({op for op, _ in data})
    rank_counts = sorted({ranks for _, ranks in data})
    w, h = args.subplot_size
    fig, axes = plt.subplots(
        len(rank_counts),
        len(ops),
        figsize=(w * len(ops), h * len(rank_counts)),
        squeeze=False,
    )
    # Header offsets are in inches converted to figure fractions. A fixed
    # fraction collides with the title on short figures (one rank count) and
    # floats away from it on tall ones.
    fig_height = h * len(rank_counts)
    fig.suptitle(suptitle, fontsize=15, fontweight="bold", y=1.0 - 0.30 / fig_height)
    if args.caption:
        fig.text(
            0.5,
            1.0 - 0.62 / fig_height,
            args.caption,
            ha="center",
            fontsize=9,
            style="italic",
            alpha=0.75,
        )
    fig._ccl_top = 1.0 - (0.82 if args.caption else 0.55) / fig_height
    return fig, axes, ops, rank_counts


def _finish(fig, args, out_path):
    fig.tight_layout(rect=(0, 0, 1, getattr(fig, "_ccl_top", 0.95)))
    fig.savefig(out_path, dpi=args.dpi, bbox_inches="tight")
    print(f"wrote {out_path}")
    plt.close(fig)


def _series_order(series):
    """Iris first, then RCCL; stable within a backend by variant name."""
    return sorted(series, key=lambda sk: (sk[0] != "iris", sk[1]))


def plot_metric(data, args, out_path, *, index, ylabel, log_y, suptitle):
    """Latency (index 0) or bandwidth (index 1) vs message size."""
    fig, axes, ops, rank_counts = _grid(data, args, suptitle)

    for r, ranks in enumerate(rank_counts):
        for c, op in enumerate(ops):
            ax = axes[r][c]
            series = data.get((op, ranks))
            if not series:
                ax.set_visible(False)
                continue
            multi = {b: 0 for b, _ in series}
            for b, _ in series:
                multi[b] += 1
            for i, sk in enumerate(_series_order(series)):
                backend, variant = sk
                points = {s: v[index] for s, v in series[sk].items() if v[index] is not None}
                if not points:
                    continue
                sizes = sorted(points)
                color, marker, linestyle = _style(backend, i)
                ax.plot(
                    [s / (1024 * 1024) for s in sizes],
                    [points[s] for s in sizes],
                    marker=marker,
                    linestyle=linestyle,
                    color=color,
                    linewidth=1.8,
                    markersize=6,
                    label=_label(backend, variant, multi[backend] > 1),
                )
            ax.set_xscale("log", base=2)
            if log_y:
                ax.set_yscale("log")
            ax.grid(True, which="both", alpha=0.3, linestyle="--")
            ax.set_xlabel("Message size (MiB)", fontsize=10)
            if c == 0:
                ax.set_ylabel(ylabel, fontsize=10)
            ax.set_title(f"{op.replace('_', '-').title()} — {ranks} ranks", fontsize=12, fontweight="bold")
            ax.legend(loc="best", fontsize=9, framealpha=0.85)

    _finish(fig, args, out_path)


def plot_speedup(data, args, out_path):
    """Latency ratio Iris / RCCL. Below 1.0 means Iris is faster."""
    fig, axes, ops, rank_counts = _grid(data, args, f"{args.title} — speedup vs RCCL")
    plotted = False

    for r, ranks in enumerate(rank_counts):
        for c, op in enumerate(ops):
            ax = axes[r][c]
            series = data.get((op, ranks))
            if not series:
                ax.set_visible(False)
                continue
            reference = [sk for sk in series if sk[0] == REFERENCE_BACKEND]
            if not reference:
                ax.set_visible(False)
                continue
            ref = series[sorted(reference, key=lambda sk: sk[1])[0]]

            multi = sum(1 for b, _ in series if b == "iris")
            for i, sk in enumerate(_series_order(series)):
                backend, variant = sk
                if backend == REFERENCE_BACKEND:
                    continue
                shared = sorted(s for s in series[sk] if s in ref and ref[s][0] > 0)
                if not shared:
                    continue
                color, marker, linestyle = _style(backend, i)
                ax.plot(
                    [s / (1024 * 1024) for s in shared],
                    [series[sk][s][0] / ref[s][0] for s in shared],
                    marker=marker,
                    linestyle=linestyle,
                    color=color,
                    linewidth=1.8,
                    markersize=6,
                    label=_label(backend, variant, multi > 1),
                )
                plotted = True
            ax.axhline(1.0, color="black", linewidth=1.0, linestyle="--", alpha=0.7)
            ax.set_xscale("log", base=2)
            ax.set_yscale("log")
            ax.grid(True, which="both", alpha=0.3, linestyle="--")
            ax.set_xlabel("Message size (MiB)", fontsize=10)
            if c == 0:
                ax.set_ylabel("Latency ratio (Iris / RCCL)\nlower is better; <1 means Iris wins", fontsize=10)
            ax.set_title(f"{op.replace('_', '-').title()} — {ranks} ranks", fontsize=12, fontweight="bold")
            ax.legend(loc="best", fontsize=9, framealpha=0.85)

    if not plotted:
        print("[speedup] no Iris/RCCL pairs at matching shapes; skipping the speedup plot")
        plt.close(fig)
        return
    _finish(fig, args, out_path)


def markdown_table(data):
    """Latency and bus bandwidth per operation, rank count and message size."""
    lines = [
        "| Operation | Ranks | Size (MiB) | Iris (ms) | RCCL (ms) | Iris (GB/s) | RCCL (GB/s) | Speedup |",
        "|---|---|---|---|---|---|---|---|",
    ]
    for op, ranks in sorted(data):
        series = data[(op, ranks)]

        def _merge(backend):
            merged = {}
            for (b, _), points in series.items():
                if b == backend:
                    merged.update(points)
            return merged

        iris, rccl = _merge("iris"), _merge(REFERENCE_BACKEND)
        for size in sorted(set(iris) | set(rccl)):
            il, ib = iris.get(size, (None, None))
            rl, rb = rccl.get(size, (None, None))
            # Latency ratio, expressed the way the table reads: >1 = Iris wins.
            speedup = f"{rl / il:.2f}x" if il and rl else "—"
            cells = [
                op,
                str(ranks),
                f"{size / (1024 * 1024):.2f}",
                f"{il:.4f}" if il else "—",
                f"{rl:.4f}" if rl else "—",
                f"{ib:.1f}" if ib else "—",
                f"{rb:.1f}" if rb else "—",
                speedup,
            ]
            lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines)


def main():
    args = parse_args()
    data = load_results(args.input_csv)
    if not data:
        raise SystemExit("no plottable rows found; was the CSV produced with --benchmark_format=csv?")

    bandwidth_path = args.output or os.path.basename(args.input_csv[0]).replace(".csv", "") + "_iris_vs_rccl.png"
    stem, ext = os.path.splitext(bandwidth_path)
    ext = ext or ".png"

    plot_metric(
        data,
        args,
        bandwidth_path,
        index=1,
        ylabel="Bus bandwidth (GB/s, higher is better)",
        log_y=False,
        suptitle=f"{args.title} — bus bandwidth",
    )
    plot_metric(
        data,
        args,
        f"{stem}_latency{ext}",
        index=0,
        ylabel="Latency (ms, lower is better)",
        log_y=True,
        suptitle=f"{args.title} — latency",
    )
    plot_speedup(data, args, f"{stem}_speedup{ext}")

    table = markdown_table(data)
    if args.markdown_out:
        with open(args.markdown_out, "w") as f:
            f.write(table + "\n")
        print(f"wrote {args.markdown_out}")
    else:
        print(table)


if __name__ == "__main__":
    main()
