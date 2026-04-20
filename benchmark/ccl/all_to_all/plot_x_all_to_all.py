#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
# Copyright (c) 2025-2026 Advanced Micro Devices, Inc. All rights reserved.

"""
Scatter-plot: iris.x.all_to_all – Triton vs Gluon bandwidth across problem sizes.

Reads the JSON output produced by benchmark_x.py (--sweep -b) and creates a
scatter plot with two marker series:

  • Blue circles  (◉) → Triton bandwidth (GB/s)
  • Orange squares (■) → Gluon  bandwidth (GB/s)

X-axis: total bytes communicated per rank  (log scale)
Y-axis: achieved bandwidth in GB/s

Usage
-----
    # After running the benchmark sweep:
    python benchmark_x.py -v -b --sweep -r 8 --output_file results.json
    python plot_x_all_to_all.py results.json [--output scatter.png]
"""

import argparse
import json
import sys


def parse_args():
    parser = argparse.ArgumentParser(
        description="Scatter-plot iris.x all_to_all Triton vs Gluon bandwidth",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("input_json", help="JSON results file from benchmark_x.py --sweep")
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Output image file (default: <input_json>.png)",
    )
    parser.add_argument("--title", type=str, default="iris.x all_to_all: Triton vs Gluon bandwidth")
    parser.add_argument("--dpi", type=int, default=150)
    parser.add_argument("--figsize", type=int, nargs=2, default=[10, 6])
    return parser.parse_args()


def load_results(path: str):
    with open(path) as f:
        data = json.load(f)
    # Support both a single-object and a list-of-objects JSON format.
    if isinstance(data, dict):
        data = [data]
    return data


def make_label(row: dict) -> str:
    """Short label for a data point (placed near the marker)."""
    return f"({row['M']}×{row['N']})"


def plot(data, args):
    try:
        import matplotlib.pyplot as plt
        import matplotlib.ticker as ticker
    except ImportError:
        print(
            "ERROR: matplotlib is required for plotting.  Install with:\n    pip install matplotlib\n",
            file=sys.stderr,
        )
        sys.exit(1)

    # Sort by total bytes so the X-axis is monotone.
    data = sorted(data, key=lambda r: r.get("total_bytes", r["M"] * r["N"]))

    triton_x, triton_y, triton_labels = [], [], []
    gluon_x, gluon_y, gluon_labels = [], [], []

    for row in data:
        x = row.get("total_bytes", 0)
        label = make_label(row)

        if "triton_bandwidth_gbps" in row:
            triton_x.append(x)
            triton_y.append(row["triton_bandwidth_gbps"])
            triton_labels.append(label)

        if "gluon_bandwidth_gbps" in row and row["gluon_bandwidth_gbps"] is not None:
            gluon_x.append(x)
            gluon_y.append(row["gluon_bandwidth_gbps"])
            gluon_labels.append(label)

    fig, ax = plt.subplots(figsize=tuple(args.figsize))

    # Two scatter series with distinct markers and colours.
    if triton_x:
        ax.scatter(
            triton_x,
            triton_y,
            marker="o",
            s=80,
            color="#2E86AB",
            label="Triton",
            zorder=3,
        )
        ax.plot(triton_x, triton_y, linestyle="--", color="#2E86AB", linewidth=1, alpha=0.5)
        for x, y, lbl in zip(triton_x, triton_y, triton_labels):
            ax.annotate(lbl, (x, y), textcoords="offset points", xytext=(4, 5), fontsize=6, color="#2E86AB")

    if gluon_x:
        ax.scatter(
            gluon_x,
            gluon_y,
            marker="s",
            s=80,
            color="#E07A5F",
            label="Gluon",
            zorder=3,
        )
        ax.plot(gluon_x, gluon_y, linestyle="--", color="#E07A5F", linewidth=1, alpha=0.5)
        for x, y, lbl in zip(gluon_x, gluon_y, gluon_labels):
            ax.annotate(lbl, (x, y), textcoords="offset points", xytext=(4, -12), fontsize=6, color="#E07A5F")

    # Extract common metadata from first row for subtitle.
    if data:
        r0 = data[0]
        subtitle = (
            f"world_size={r0.get('world_size', '?')}  "
            f"dtype={r0.get('dtype', '?')}  "
            f"BLOCK_M={r0.get('BLOCK_SIZE_M', '?')}  "
            f"BLOCK_N={r0.get('BLOCK_SIZE_N', '?')}"
        )
        ax.set_title(f"{args.title}\n{subtitle}", fontsize=12)
    else:
        ax.set_title(args.title, fontsize=12)

    ax.set_xscale("log", base=2)
    ax.set_xlabel("Total bytes communicated per rank  [log₂ scale]", fontsize=11)
    ax.set_ylabel("Bandwidth (GB/s)", fontsize=11)
    ax.xaxis.set_major_formatter(
        ticker.FuncFormatter(lambda v, _: f"{int(v / 2**20)} MiB" if v >= 2**20 else f"{int(v / 2**10)} KiB")
    )
    ax.grid(True, alpha=0.3, linestyle="--")
    ax.legend(fontsize=11, loc="best")

    plt.tight_layout()

    out = args.output or (args.input_json.rsplit(".", 1)[0] + "_scatter.png")
    plt.savefig(out, dpi=args.dpi, bbox_inches="tight")
    print(f"Scatter plot saved to: {out}")

    try:
        plt.show()
    except Exception:
        pass


def print_table(data):
    """Print a plain-text comparison table to stdout."""
    if not data:
        print("No data to display.")
        return

    header = f"{'M':>7} {'N_per_rank':>10} {'total_bytes':>14} {'Triton (GB/s)':>14} {'Gluon (GB/s)':>13} {'ratio%':>7} {'T_valid':>8} {'G_valid':>8}"
    print("\n" + "=" * len(header))
    print(header)
    print("=" * len(header))

    for r in sorted(data, key=lambda x: x.get("total_bytes", 0)):
        bw_t = r.get("triton_bandwidth_gbps", float("nan"))
        bw_g = r.get("gluon_bandwidth_gbps", float("nan"))
        ratio = r.get("gluon_vs_triton_percent", float("nan"))
        tv = "PASS" if r.get("triton_valid") else ("FAIL" if "triton_valid" in r else "n/a")
        gv = "PASS" if r.get("gluon_valid") else ("FAIL" if "gluon_valid" in r else "n/a")
        print(
            f"{r['M']:>7} {r['N']:>10} {r.get('total_bytes', 0):>14,.0f}"
            f" {bw_t:>14.3f} {bw_g:>13.3f} {ratio:>7.1f} {tv:>8} {gv:>8}"
        )

    print("=" * len(header) + "\n")


def main():
    args = parse_args()
    data = load_results(args.input_json)
    print_table(data)
    plot(data, args)


if __name__ == "__main__":
    main()
