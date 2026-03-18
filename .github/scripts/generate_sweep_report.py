#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
# Copyright (c) 2025 Advanced Micro Devices, Inc. All rights reserved.
#
# Reads one or more sweep_results JSON files produced by sweep_benchmark.py,
# merges them, and writes a Markdown roofline report to stdout (or a file).
#
# Usage:
#   python generate_sweep_report.py sweep_2ranks.json sweep_4ranks.json sweep_8ranks.json \
#       [--output report.md]

import argparse
import json
import sys


def load_results(paths):
    all_results = []
    for p in paths:
        try:
            with open(p) as f:
                data = json.load(f)
            if isinstance(data, list):
                all_results.extend(data)
            else:
                print(f"[WARN] Unexpected format in {p}, skipping", file=sys.stderr)
        except Exception as e:
            print(f"[WARN] Could not read {p}: {e}", file=sys.stderr)
    return all_results


def make_markdown(results):
    if not results:
        return "No benchmark results available.\n"

    # Sort by TFLOPS descending
    results = sorted(results, key=lambda x: -x.get("tflops", 0))

    mi325x_peak_per_gpu = 1307.4  # fp16/bf16 matrix cores per GPU

    lines = []
    lines.append("## Benchmark Sweep Results — Example 10 (GEMM All-Scatter WG Specialization)")
    lines.append("")
    lines.append(
        f"Platform: AMD Instinct MI325X &nbsp;|&nbsp; "
        f"Peak per GPU (fp16/bf16): **{mi325x_peak_per_gpu} TFLOPS** (CDNA3 matrix cores)"
    )
    lines.append("")
    lines.append(
        "| M | N | K | BLK_M | BLK_N | BLK_K | dtype | stages | ranks | "
        "time (ms) | TFLOPS | AI (flop/B) | Peak total (TFLOPS) | Efficiency % |"
    )
    lines.append(
        "|---|---|---|------:|------:|------:|-------|-------:|------:|----------:|-------:|------------:|--------------------:|-------------:|"
    )

    for r in results:
        lines.append(
            f"| {r['M']} | {r['N']} | {r['K']} "
            f"| {r['BLK_M']} | {r['BLK_N']} | {r['BLK_K']} "
            f"| {r['dtype']} | {r['num_stages']} | {r['num_ranks']} "
            f"| {r['total_ms']:.2f} | **{r['tflops']:.1f}** "
            f"| {r['arith_intensity']:.1f} "
            f"| {r['peak_tflops_total']:.0f} "
            f"| {r['efficiency_pct']:.1f}% |"
        )

    lines.append("")

    # Top-5 configs
    lines.append("### Top 5 Configurations by TFLOPS")
    lines.append("")
    lines.append("| Rank | Config | dtype | ranks | TFLOPS | Efficiency % |")
    lines.append("|------|--------|-------|------:|-------:|-------------:|")
    for i, r in enumerate(results[:5], 1):
        cfg = f"{r['M']}×{r['N']}×{r['K']} BLK={r['BLK_M']}×{r['BLK_N']}×{r['BLK_K']} stages={r['num_stages']}"
        lines.append(
            f"| {i} | {cfg} | {r['dtype']} | {r['num_ranks']} | **{r['tflops']:.1f}** | {r['efficiency_pct']:.1f}% |"
        )

    lines.append("")

    # Roofline note
    best = results[0]
    lines.append(
        f"> **Best result**: {best['tflops']:.1f} TFLOPS with "
        f"M={best['M']} N={best['N']} K={best['K']} "
        f"BLK={best['BLK_M']}×{best['BLK_N']}×{best['BLK_K']} "
        f"{best['dtype']} {best['num_stages']} stages {best['num_ranks']} ranks "
        f"— **{best['efficiency_pct']:.1f}% efficiency** vs "
        f"{best['peak_tflops_total']:.0f} TFLOPS theoretical peak "
        f"({best['num_ranks']}× {mi325x_peak_per_gpu} TFLOPS)"
    )
    lines.append("")

    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(description="Generate Markdown roofline report from sweep JSON files")
    parser.add_argument("inputs", nargs="+", help="Input JSON files from sweep_benchmark.py")
    parser.add_argument("--output", "-o", default=None, help="Output Markdown file (default: stdout)")
    args = parser.parse_args()

    results = load_results(args.inputs)
    md = make_markdown(results)

    if args.output:
        with open(args.output, "w") as f:
            f.write(md)
        print(f"Report written to {args.output}", file=sys.stderr)
    else:
        print(md)


if __name__ == "__main__":
    main()
