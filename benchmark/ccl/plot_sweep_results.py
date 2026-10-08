#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
# Copyright (c) 2025-2026 Advanced Micro Devices, Inc. All rights reserved.

"""Plot CCL benchmark results: Iris vs RCCL, one image per collective and dtype.

Consumes the CSV that ``iris.bench`` emits with ``--benchmark_format=csv`` -- one
row per measurement, with the benchmark axes as columns. The ``backend`` axis in
benchmark/ccl/bench_*.py selects Iris or RCCL for the same shapes, so both series
come out of a single sweep:

    python benchmark/ccl/bench_all_reduce.py --benchmark_format=csv \
        --benchmark_out=all_reduce.csv
    python benchmark/ccl/plot_sweep_results.py all_reduce.csv --output_dir plots

Writes ``<output_dir>/<collective>_<dtype>.png`` for every collective and dtype
in the input, e.g. ``all_reduce_bf16.png``. Each image has one row per rank
count and these panels, all against message size:

* bus bandwidth (GB/s), semilog-x
* latency (ms), log-log
* speedup, RCCL latency / Iris latency, semilog-x -- above 1.0 means Iris wins.
  Absent for fp8, which has no RCCL baseline.

A Markdown table goes to stdout or --markdown_out, so CI can drop it into a job
summary. Speedup there uses the same definition as the plots.

On bandwidth: the bench scripts already declare bus bytes via ``state.set_bytes``
-- ``(W-1) * bytes`` for all_gather and all_to_all, ``2 * (W-1)/W * bytes`` for
all_reduce -- which is the NCCL/RCCL busBW convention. The framework divides that
by the measured time, so ``bandwidth_gbps`` is already bus bandwidth and is
directly comparable to nccl-tests output. Do not re-apply a bus factor here.
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
# two_shot). Index 0 is the plain case.
_MARKERS = ["o", "s", "^", "D", "v", "P", "X", "*"]
_LINESTYLES = ["-", "--", "-.", ":"]

# Collectives first in this order, then any others alphabetically.
_OP_ORDER = ["all_reduce", "all_gather", "all_to_all"]
_DTYPE_ORDER = ["fp16", "bf16", "fp8"]


def parse_args():
    parser = argparse.ArgumentParser(
        description="Plot CCL benchmark results, Iris vs RCCL.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("input_csv", nargs="+", help="CSV file(s) from iris.bench --benchmark_format=csv")
    parser.add_argument("--output_dir", default=".", help="Directory for the <collective>_<dtype>.png images")
    parser.add_argument("--markdown_out", default=None, help="Write the Markdown table here instead of stdout")
    parser.add_argument("--title", default="Iris vs RCCL", help="Prefix for each image's title")
    parser.add_argument("--caption", default=None, help="Small italic caption under the title (machine, ROCm)")
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

# Short display tokens. fp8 spellings vary by build (OCP vs fnuz), so they are
# matched by prefix rather than enumerated.
_PRECISION_TOKEN = {"float16": "fp16", "bfloat16": "bf16", "float32": "fp32", "float64": "fp64"}


def _itemsize(name):
    """Bytes per element for a dtype name, or None if unrecognised."""
    if name in _ITEMSIZE:
        return _ITEMSIZE[name]
    if name.startswith("float8") or name == "fp8":
        return 1
    return None


def _precision_token(name):
    """'fp16', 'bf16', 'fp8' -- used in file names, titles and the table."""
    if name.startswith("float8") or name == "fp8":
        return "fp8"
    return _PRECISION_TOKEN.get(name, name)


def _message_bytes(row):
    """Bytes per rank for this point, from the M/N/dtype axes."""
    try:
        elems = int(row["M"]) * int(row["N"])
    except (KeyError, TypeError, ValueError):
        return None
    itemsize = _itemsize((row.get("dtype") or "").strip())
    if itemsize is None:
        return None
    return elems * itemsize


def load_results(paths):
    """``{(op, dtype): {ranks: {(backend, variant): {size: (latency, bw)}}}}``.

    Points are median-aggregated, so several (M, N) pairs with the same byte
    count collapse to one marker rather than stacking invisibly. Each dtype is
    its own group: fp16 and bf16 land on identical x positions, and they are
    plotted separately so a difference between them stays visible.
    """
    acc = defaultdict(lambda: defaultdict(lambda: defaultdict(lambda: defaultdict(list))))
    for path in paths:
        with open(path, newline="") as f:
            for row in csv.DictReader(f):
                # The framework records skipped combinations with empty timings.
                # The rccl arm skips fp8 this way, since RCCL has no fp8 support.
                if (row.get("skipped") or "").strip().lower() == "true":
                    continue
                latency = _float(row, "gpu_time_ms", "time_ms", "mean_ms")
                size = _message_bytes(row)
                if latency is None or size is None:
                    continue
                op = row.get("benchmark") or row.get("name") or os.path.basename(path).replace(".csv", "")
                backend = (row.get("backend") or "iris").strip().lower()
                variant = (row.get("variant") or "").strip()
                dtype = _precision_token((row.get("dtype") or "").strip())
                try:
                    ranks = int(row.get("num_ranks") or row.get("world_size") or 0)
                except ValueError:
                    ranks = 0
                bw = _float(row, "bandwidth_gbps", "bandwidth", "GB/s")
                acc[(op, dtype)][ranks][(backend, variant)][size].append((latency, bw))

    out = {}
    for key, by_ranks in acc.items():
        out[key] = {}
        for ranks, series in by_ranks.items():
            out[key][ranks] = {}
            for sk, sizes in series.items():
                points = {}
                for size, samples in sizes.items():
                    lat = statistics.median(p[0] for p in samples)
                    bws = [p[1] for p in samples if p[1] is not None]
                    points[size] = (lat, statistics.median(bws) if bws else None)
                out[key][ranks][sk] = points
    return out


def _sort_key(order, name):
    return (order.index(name), name) if name in order else (len(order), name)


def _label(series_key, multi_variant):
    """'Iris', or 'Iris (two_shot)' when a backend has several variants."""
    backend, variant = series_key
    base = "Iris" if backend == "iris" else backend.upper()
    return f"{base} ({variant})" if (multi_variant and variant) else base


def _style(backend, index):
    color = IRIS_COLOR if backend == "iris" else RCCL_COLOR
    # RCCL is the reference, so it defaults to dashed even as variant 0.
    offset = 0 if backend == "iris" else 1
    return color, _MARKERS[index % len(_MARKERS)], _LINESTYLES[(index + offset) % len(_LINESTYLES)]


def _series_order(series):
    """Iris first, then RCCL; stable within a backend by variant."""
    return sorted(series, key=lambda sk: (sk[0] != "iris", sk[1]))


def _variants_per_backend(series):
    out = defaultdict(set)
    for backend, variant in series:
        out[backend].add(variant)
    return out


def _reference_for(series, variant):
    """The RCCL series to divide by, or None.

    The reference may not carry the variant label at all (only the Iris path is
    parameterised), so fall back to the sole RCCL series when there is one.
    """
    ref = series.get((REFERENCE_BACKEND, variant))
    if ref is not None:
        return ref
    same = [k for k in series if k[0] == REFERENCE_BACKEND]
    return series[same[0]] if len(same) == 1 else None


def _decorate(ax, ylabel, log_y, title):
    ax.set_xscale("log", base=2)
    if log_y:
        ax.set_yscale("log")
    ax.grid(True, which="both", alpha=0.3, linestyle="--")
    ax.set_xlabel("Message size (MiB)", fontsize=10)
    ax.set_ylabel(ylabel, fontsize=10)
    ax.set_title(title, fontsize=12, fontweight="bold")
    if ax.get_legend_handles_labels()[0]:
        ax.legend(loc="best", fontsize=9, framealpha=0.85)


def _plot_metric(ax, series, index):
    """Bandwidth (index 1) or latency (index 0) for every series."""
    variants = _variants_per_backend(series)
    for i, sk in enumerate(_series_order(series)):
        points = {s: v[index] for s, v in series[sk].items() if v[index] is not None}
        if not points:
            continue
        sizes = sorted(points)
        color, marker, linestyle = _style(sk[0], i)
        ax.plot(
            [s / (1024 * 1024) for s in sizes],
            [points[s] for s in sizes],
            marker=marker,
            linestyle=linestyle,
            color=color,
            linewidth=1.8,
            markersize=6,
            label=_label(sk, len(variants[sk[0]]) > 1),
        )


def _plot_speedup(ax, series):
    """RCCL latency / Iris latency at matching sizes. Returns True if anything was drawn."""
    variants = _variants_per_backend(series)
    drawn = False
    for i, sk in enumerate(_series_order(series)):
        backend, variant = sk
        if backend == REFERENCE_BACKEND:
            continue
        ref = _reference_for(series, variant)
        if ref is None:
            continue
        shared = sorted(s for s in series[sk] if s in ref and series[sk][s][0] > 0)
        if not shared:
            continue
        color, marker, linestyle = _style(backend, i)
        ax.plot(
            [s / (1024 * 1024) for s in shared],
            [ref[s][0] / series[sk][s][0] for s in shared],
            marker=marker,
            linestyle=linestyle,
            color=color,
            linewidth=1.8,
            markersize=6,
            label=_label(sk, len(variants[backend]) > 1),
        )
        drawn = True
    ax.axhline(1.0, color="black", linewidth=1.0, linestyle="--", alpha=0.7)
    return drawn


def plot_one(op, dtype, by_ranks, args):
    """One image: a row per rank count, a column per metric."""
    rank_counts = sorted(by_ranks)
    has_reference = any(sk[0] == REFERENCE_BACKEND for series in by_ranks.values() for sk in series)
    columns = ["bandwidth", "latency"] + (["speedup"] if has_reference else [])

    w, h = args.subplot_size
    fig, axes = plt.subplots(
        len(rank_counts), len(columns), figsize=(w * len(columns), h * len(rank_counts)), squeeze=False
    )
    # Header offsets are in inches converted to figure fractions, so the title
    # sits the same distance above the plots however many rows there are.
    fig_height = h * len(rank_counts)
    name = op.replace("_", "-").title()
    fig.suptitle(f"{args.title}: {name}, {dtype}", fontsize=15, fontweight="bold", y=1.0 - 0.30 / fig_height)
    caption = args.caption
    if not has_reference:
        note = "Iris only: RCCL has no fp8 support" if dtype == "fp8" else "Iris only: no RCCL points"
        caption = f"{caption} — {note}" if caption else note
    if caption:
        fig.text(0.5, 1.0 - 0.62 / fig_height, caption, ha="center", fontsize=9, style="italic", alpha=0.75)

    for r, ranks in enumerate(rank_counts):
        series = by_ranks[ranks]
        for c, column in enumerate(columns):
            ax = axes[r][c]
            title = f"{ranks} ranks"
            if column == "bandwidth":
                _plot_metric(ax, series, 1)
                _decorate(ax, "Bus bandwidth (GB/s, higher is better)", False, f"{title} — bandwidth")
            elif column == "latency":
                _plot_metric(ax, series, 0)
                _decorate(ax, "Latency (ms, lower is better)", True, f"{title} — latency")
            elif not _plot_speedup(ax, series):
                ax.set_visible(False)
            else:
                # Linear: ratios sit within a decade of 1, where a log axis only
                # labels its minor ticks, as 9x10^-1 and the like.
                _decorate(ax, "Speedup (RCCL / Iris latency)\nabove 1 means Iris wins", False, f"{title} — speedup")

    top = 1.0 - (0.82 if caption else 0.55) / fig_height
    fig.tight_layout(rect=(0, 0, 1, top))
    out_path = os.path.join(args.output_dir, f"{op}_{dtype}.png")
    fig.savefig(out_path, dpi=args.dpi, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {out_path}")
    return out_path


def _cell(value, spec):
    """Format a number, or an em dash when it is genuinely absent.

    Checks ``is not None`` rather than truthiness: 0.0 is a real measurement and
    must not be rendered as missing data.
    """
    return format(value, spec) if value is not None else "—"


def markdown_table(data):
    """One section per collective: latency, bus bandwidth and speedup per point.

    Each variant gets its own rows. Flattening them into one series per backend
    would drop points whenever two variants share a message size.
    """
    ops = sorted({op for op, _ in data}, key=lambda o: _sort_key(_OP_ORDER, o))
    show_variant = any(variant for by_ranks in data.values() for s in by_ranks.values() for _, variant in s)
    header = (
        ["Ranks", "Dtype"]
        + (["Variant"] if show_variant else [])
        + ["Size (MiB)", "Iris (ms)", "RCCL (ms)", "Iris (GB/s)", "RCCL (GB/s)", "Speedup"]
    )
    sections = []
    for op in ops:
        lines = [f"### {op}", "", "| " + " | ".join(header) + " |", "|" + "---|" * len(header)]
        dtypes = sorted((d for o, d in data if o == op), key=lambda d: _sort_key(_DTYPE_ORDER, d))
        rows = []
        for dtype in dtypes:
            for ranks, series in data[(op, dtype)].items():
                for variant in sorted({v for b, v in series if b != REFERENCE_BACKEND} or {""}):
                    iris = series.get(("iris", variant), {})
                    reference = _reference_for(series, variant) or {}
                    for size in sorted(set(iris) | set(reference)):
                        rows.append((ranks, _sort_key(_DTYPE_ORDER, dtype), size, dtype, variant, iris, reference))
        for ranks, _, size, dtype, variant, iris, reference in sorted(rows, key=lambda r: r[:3]):
            il, ib = iris.get(size, (None, None))
            rl, rb = reference.get(size, (None, None))
            # Same definition as the plots: RCCL latency / Iris latency, >1 = Iris wins.
            ratio = (rl / il) if (il is not None and rl is not None and il > 0) else None
            cells = (
                [str(ranks), dtype]
                + ([variant or "—"] if show_variant else [])
                + [
                    f"{size / (1024 * 1024):.2f}",
                    _cell(il, ".4f"),
                    _cell(rl, ".4f"),
                    _cell(ib, ".1f"),
                    _cell(rb, ".1f"),
                    _cell(ratio, ".2f") + ("x" if ratio is not None else ""),
                ]
            )
            lines.append("| " + " | ".join(cells) + " |")
        sections.append("\n".join(lines))
    return "\n\n".join(sections)


def main():
    args = parse_args()
    data = load_results(args.input_csv)
    if not data:
        raise SystemExit("no plottable rows found; was the CSV produced with --benchmark_format=csv?")

    os.makedirs(args.output_dir, exist_ok=True)
    for op, dtype in sorted(data, key=lambda k: (_sort_key(_OP_ORDER, k[0]), _sort_key(_DTYPE_ORDER, k[1]))):
        plot_one(op, dtype, data[(op, dtype)], args)

    table = markdown_table(data)
    if args.markdown_out:
        with open(args.markdown_out, "w") as f:
            f.write(table + "\n")
        print(f"wrote {args.markdown_out}")
    else:
        print(table)


if __name__ == "__main__":
    main()
