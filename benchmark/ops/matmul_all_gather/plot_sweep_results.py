#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
# Copyright (c) 2025-2026 Advanced Micro Devices, Inc. All rights reserved.

"""
Plot benchmark sweep results from benchmark_sweep_results.json.

Creates a grouped bar chart comparing TFLOPS across different benchmark variants
and dimension configurations.

Usage:
    python plot_sweep_results.py [--input benchmark_sweep_results.json] [--output plot.png]
"""

import json
import argparse
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np


def extract_tflops(benchmark_data, benchmark_name):
    """Extract TFLOPS value from benchmark result based on benchmark type."""
    if benchmark_data.get("status") == "FAILED":
        return None

    # Try multiple possible TFLOPS keys in order of preference
    possible_keys = [
        "tflops",
        "host_copy_engine_tflops",
        "pytorch_tflops",
        "copy_engine_tflops",
    ]

    for key in possible_keys:
        value = benchmark_data.get(key)
        if value is not None:
            return value

    return None


def plot_sweep_results(input_file, output_file):
    """Create grouped bar chart from sweep results."""

    # Load results
    with open(input_file, "r") as f:
        results = json.load(f)

    if not results:
        print("No results found in input file")
        return

    # Extract dimension configurations and benchmark names
    dim_configs = []
    benchmark_names = set()

    for result in results:
        m, n, k = result["M"], result["N"], result["K"]
        dim_label = f"{m}×{n}×{k}"
        dim_configs.append((m, n, k, dim_label))
        benchmark_names.update(result["benchmarks"].keys())

    benchmark_names = sorted(benchmark_names)

    # Prepare data for plotting
    data = {bench: [] for bench in benchmark_names}

    for result in results:
        for bench in benchmark_names:
            if bench in result["benchmarks"]:
                tflops = extract_tflops(result["benchmarks"][bench], bench)
                data[bench].append(tflops if tflops is not None else 0)
            else:
                data[bench].append(0)

    # Create plot
    fig, ax = plt.subplots(figsize=(14, 8))

    x = np.arange(len(dim_configs))
    width = 0.8 / len(benchmark_names)  # Width of bars

    # Colors for different benchmarks
    colors = {
        "baseline": "#2E86AB",
        "copy_engine": "#A23B72",
        "host_copy_engine": "#F18F01",
        "pytorch": "#C73E1D",
    }

    # Plot bars for each benchmark
    for i, bench in enumerate(benchmark_names):
        offset = width * (i - len(benchmark_names) / 2 + 0.5)
        values = data[bench]
        color = colors.get(bench, f"C{i}")

        bars = ax.bar(
            x + offset,
            values,
            width,
            label=bench.replace("_", " ").title(),
            color=color,
            alpha=0.8,
            edgecolor="black",
            linewidth=0.5,
        )

        # Add value labels on top of bars (only for non-zero values)
        for j, (bar, val) in enumerate(zip(bars, values)):
            if val > 0:
                height = bar.get_height()
                ax.text(
                    bar.get_x() + bar.get_width() / 2,
                    height,
                    f"{val:.1f}",
                    ha="center",
                    va="bottom",
                    fontsize=7,
                    rotation=0,
                )

    # Customize plot
    ax.set_xlabel("Dimension Configuration (M_local×N×K)", fontsize=12, fontweight="bold")
    ax.set_ylabel("TFLOPS", fontsize=12, fontweight="bold")
    ax.set_title("Matmul-All-Gather Benchmark Sweep: TFLOPS Comparison", fontsize=14, fontweight="bold", pad=20)
    ax.set_xticks(x)
    ax.set_xticklabels([label for _, _, _, label in dim_configs], rotation=45, ha="right")
    ax.legend(loc="upper left", fontsize=10)
    ax.grid(axis="y", alpha=0.3, linestyle="--")

    # Set y-axis to start from 0
    ax.set_ylim(bottom=0)

    plt.tight_layout()

    # Save figure
    plt.savefig(output_file, dpi=300, bbox_inches="tight")
    print(f"Plot saved to: {output_file}")

    # Print summary statistics in formatted table
    print("\n" + "=" * 120)
    print("SUMMARY STATISTICS")
    print("=" * 120)
    print()

    # Build table data
    table_data = []
    for dim_config in dim_configs:
        m, n, k, label = dim_config
        idx = dim_configs.index(dim_config)

        row = {
            "M": m,
            "N": n,
            "K": k,
            "Config": label,
        }

        # Add TFLOPS values and validation status for each benchmark
        for bench in benchmark_names:
            val = data[bench][idx]
            if val > 0:
                row[bench] = f"{val:.2f}"
            else:
                row[bench] = "FAILED"

            # Add validation status column
            bench_data = results[idx]["benchmarks"].get(bench, {})
            validation_key = f"{bench}_validation"
            if bench_data.get("success") is True:
                row[validation_key] = "✓"
            elif bench_data.get("success") is False:
                row[validation_key] = "✗"
            else:
                row[validation_key] = "-"

        # Find best performer
        valid_values = [(bench, data[bench][idx]) for bench in benchmark_names if data[bench][idx] > 0]
        if valid_values:
            best_bench, best_val = max(valid_values, key=lambda x: x[1])
            row["Best"] = f"{best_bench} ({best_val:.2f})"
        else:
            row["Best"] = "NONE"

        table_data.append(row)

    # Calculate column widths
    validation_headers = [f"{bench}_validation" for bench in benchmark_names]
    headers = ["M", "N", "K", "Config"] + benchmark_names + validation_headers + ["Best"]
    col_widths = {}
    for header in headers:
        col_widths[header] = len(header)
        for row in table_data:
            col_widths[header] = max(col_widths[header], len(str(row[header])))

    # Print header with cleaner labels
    header_parts = []
    for h in headers:
        # Format validation headers as "bench (✓/✗)"
        if h.endswith("_validation"):
            bench_name = h.replace("_validation", "")
            display_name = f"{bench_name} (val)"
        else:
            display_name = h
        header_parts.append(display_name.ljust(col_widths[h]))
    print("  ".join(header_parts))

    # Print separator
    sep_parts = []
    for h in headers:
        sep_parts.append("-" * col_widths[h])
    print("  ".join(sep_parts))

    # Print rows
    for row in table_data:
        row_parts = []
        for h in headers:
            row_parts.append(str(row[h]).ljust(col_widths[h]))
        print("  ".join(row_parts))

    print("\n" + "=" * 120)


def main():
    parser = argparse.ArgumentParser(
        description="Plot benchmark sweep results", formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument(
        "--input",
        type=str,
        default="benchmark/ops/matmul_all_gather/benchmark_sweep_results.json",
        help="Input JSON file with sweep results",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="benchmark/ops/matmul_all_gather/sweep_results_plot.png",
        help="Output PNG file for plot",
    )

    args = parser.parse_args()

    input_path = Path(args.input)
    if not input_path.exists():
        print(f"Error: Input file not found: {input_path}")
        return 1

    plot_sweep_results(input_path, args.output)
    return 0


if __name__ == "__main__":
    exit(main())
