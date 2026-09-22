#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
# Copyright (c) 2025-2026 Advanced Micro Devices, Inc. All rights reserved.

"""Plot CCL benchmark results: Iris vs RCCL bandwidth per collective.

Consumes the CSV that ``iris.bench`` emits with ``--benchmark_format=csv`` -- one
row per measurement, with the benchmark axes as columns. The ``backend`` axis in
benchmark/ccl/bench_*.py selects Iris or RCCL for the same shapes, so both series
come out of a single sweep:

    python benchmark/ccl/bench_all_reduce.py --benchmark_format=csv \
        --benchmark_out=all_reduce.csv
    python benchmark/ccl/plot_sweep_results.py all_reduce.csv --output ccl.png

Also emits a Markdown table, to stdout or --markdown_out, so CI can drop it into
a job summary.

This previously read a wide CSV from comprehensive_sweep.py keyed on comm_sms.
That script was removed when the benchmarks migrated to iris.bench, which left
this reading a format nothing produced; it now reads the framework's own output
and plots against message size.
"""

import argparse
import csv
import os
from collections import defaultdict

import matplotlib

matplotlib.use("Agg")  # headless: CI has no display

import matplotlib.pyplot as plt  # noqa: E402

IRIS_COLOR = "#2E86AB"
RCCL_COLOR = "#A23B72"


def parse_args():
    parser = argparse.ArgumentParser(
        description="Plot CCL benchmark results, Iris vs RCCL.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("input_csv", nargs="+", help="CSV file(s) from iris.bench --benchmark_format=csv")
    parser.add_argument("--output", default=None, help="Output image (default: derived from the first input)")
    parser.add_argument("--markdown_out", default=None, help="Write the Markdown table here instead of stdout")
    parser.add_argument("--title", default="CCL Benchmark: Iris vs RCCL", help="Overall plot title")
    parser.add_argument("--dpi", type=int, default=150, help="DPI for the output image")
    parser.add_argument("--figsize", type=int, nargs=2, default=[16, 10], help="Figure size in inches")
    return parser.parse_args()


def _bandwidth(row):
    """Bandwidth in GB/s, under whichever column name the framework used."""
    for key in ("bandwidth_gbps", "bandwidth", "GB/s"):
        value = row.get(key)
        if value:
            try:
                return float(value)
            except ValueError:
                pass
    return None


# Matches _dtype_str in iris/bench/_runner.py, which writes the short name.
_ITEMSIZE = {"float16": 2, "bfloat16": 2, "float32": 4, "float64": 8, "int8": 1}


def _message_bytes(row):
    """Bytes per rank for this point, from the M/N/dtype axes."""
    try:
        elems = int(row["M"]) * int(row["N"])
    except (KeyError, TypeError, ValueError):
        return None
    return elems * _ITEMSIZE.get((row.get("dtype") or "").strip(), 2)


def load_results(paths):
    """{operation: {backend: {size_bytes: mean bandwidth}}}."""
    acc = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))
    for path in paths:
        with open(path, newline="") as f:
            for row in csv.DictReader(f):
                # The framework records skipped combinations with empty timings.
                if (row.get("skipped") or "").strip().lower() == "true":
                    continue
                bw = _bandwidth(row)
                size = _message_bytes(row)
                if bw is None or size is None:
                    continue
                op = row.get("name") or row.get("benchmark") or os.path.basename(path).replace(".csv", "")
                backend = (row.get("backend") or "iris").strip().lower()
                acc[op][backend][size].append(bw)

    return {
        op: {be: {sz: sum(v) / len(v) for sz, v in sizes.items()} for be, sizes in backends.items()}
        for op, backends in acc.items()
    }


def markdown_table(data):
    """Iris vs RCCL per operation and message size, with the ratio."""
    lines = [
        "| Operation | Size (MiB) | Iris (GB/s) | RCCL (GB/s) | Iris/RCCL |",
        "|---|---|---|---|---|",
    ]
    for op in sorted(data):
        iris = data[op].get("iris", {})
        rccl = data[op].get("rccl", {})
        for size in sorted(set(iris) | set(rccl)):
            i = iris.get(size)
            r = rccl.get(size)
            ratio = f"{i / r:.2f}x" if i and r else "—"
            lines.append(
                f"| {op} | {size / (1024 * 1024):.2f} "
                f"| {f'{i:.1f}' if i else '—'} "
                f"| {f'{r:.1f}' if r else '—'} | {ratio} |"
            )
    return "\n".join(lines)


def create_plots(data, args):
    ops = sorted(data)
    if not ops:
        raise SystemExit("no plottable rows found; was the CSV produced with --benchmark_format=csv?")

    ncols = 2 if len(ops) > 1 else 1
    nrows = (len(ops) + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=tuple(args.figsize), squeeze=False)
    fig.suptitle(args.title, fontsize=16, fontweight="bold")
    flat = axes.flatten()

    for ax, op in zip(flat, ops):
        for backend, color, style, label in (
            ("iris", IRIS_COLOR, "o-", "Iris"),
            ("rccl", RCCL_COLOR, "s--", "RCCL"),
        ):
            series = data[op].get(backend)
            if not series:
                continue
            sizes = sorted(series)
            ax.plot(
                [s / (1024 * 1024) for s in sizes],
                [series[s] for s in sizes],
                style,
                linewidth=2,
                markersize=7,
                label=label,
                color=color,
            )
        ax.set_xscale("log", base=2)
        ax.set_xlabel("Message size (MiB)", fontsize=11)
        ax.set_ylabel("Bandwidth (GB/s)", fontsize=11)
        ax.set_title(op.replace("_", "-").title(), fontsize=13, fontweight="bold")
        ax.grid(True, alpha=0.3, linestyle="--")
        ax.legend(loc="best", fontsize=10)

    for ax in flat[len(ops) :]:
        ax.set_visible(False)

    fig.tight_layout()
    out = args.output or os.path.basename(args.input_csv[0]).replace(".csv", "") + "_iris_vs_rccl.png"
    fig.savefig(out, dpi=args.dpi, bbox_inches="tight")
    print(f"wrote {out}")


def main():
    args = parse_args()
    data = load_results(args.input_csv)
    create_plots(data, args)

    table = markdown_table(data)
    if args.markdown_out:
        with open(args.markdown_out, "w") as f:
            f.write(table + "\n")
        print(f"wrote {args.markdown_out}")
    else:
        print(table)


if __name__ == "__main__":
    main()
