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
from enum import Enum
import matplotlib.pyplot as plt
import numpy as np


class BenchmarkType(Enum):
    """Canonical benchmark types."""

    IRIS_FUSED = "iris_fused"
    IRIS_OPTIMIZED = "iris_optimized"
    TRITONBLAS_RCCL = "tritonblas_rccl"
    TRITON_DEVICE_SDMA = "triton_device_sdma"
    TRITON_HOST_SDMA = "triton_host_sdma"
    TRITON_GEMM_ONLY = "triton_gemm_only"
    PYTORCH_RCCL = "pytorch_rccl"
    PYTORCH_GEMM_ONLY = "pytorch_gemm_only"


# Normalize benchmark names to canonical types
BENCHMARK_NAME_MAP = {
    "baseline": BenchmarkType.IRIS_FUSED,
    "hbm_buffer": BenchmarkType.IRIS_OPTIMIZED,
    "tritonblas_rccl": BenchmarkType.TRITONBLAS_RCCL,
    "tritonblas_rcclbaseline": BenchmarkType.TRITONBLAS_RCCL,
    "device_copy_engine": BenchmarkType.TRITON_DEVICE_SDMA,
    "copy_engine_device": BenchmarkType.TRITON_DEVICE_SDMA,
    "copy_engine": BenchmarkType.TRITON_DEVICE_SDMA,
    "host_copy_engine": BenchmarkType.TRITON_HOST_SDMA,
    "copy_engine_host": BenchmarkType.TRITON_HOST_SDMA,
    "matmul_only": BenchmarkType.TRITON_GEMM_ONLY,
    "matmulonly": BenchmarkType.TRITON_GEMM_ONLY,
    "pytorchbaseline": BenchmarkType.PYTORCH_RCCL,
    "pytorch_baseline": BenchmarkType.PYTORCH_RCCL,
    "pytorchmatmul_only": BenchmarkType.PYTORCH_GEMM_ONLY,
    "pytorch_matmul_only": BenchmarkType.PYTORCH_GEMM_ONLY,
    "pytorchmatmulonly": BenchmarkType.PYTORCH_GEMM_ONLY,
}

# Display labels for each benchmark type
BENCHMARK_LABELS = {
    BenchmarkType.IRIS_FUSED: "Iris baseline",
    BenchmarkType.IRIS_OPTIMIZED: "Iris optimized fused kernel",
    BenchmarkType.TRITONBLAS_RCCL: "TritonBlas + RCCL",
    BenchmarkType.TRITON_DEVICE_SDMA: "TritonBlas + device-initiated SDMA",
    BenchmarkType.TRITON_HOST_SDMA: "TritonBlas + host-initiated SDMA",
    BenchmarkType.TRITON_GEMM_ONLY: "TritonBlas (GEMM only)",
    BenchmarkType.PYTORCH_RCCL: "Pytorch + RCCL",
    BenchmarkType.PYTORCH_GEMM_ONLY: "Pytorch (GEMM only)",
}

# Colors for each benchmark type
BENCHMARK_COLORS = {
    BenchmarkType.IRIS_FUSED: "#2E7D32",  # Dark Green
    BenchmarkType.IRIS_OPTIMIZED: "#66BB6A",  # Light Green
    BenchmarkType.TRITONBLAS_RCCL: "#26A69A",  # Teal
    BenchmarkType.TRITON_DEVICE_SDMA: "#82E8FF",  # Light Blue
    BenchmarkType.TRITON_HOST_SDMA: "#1976D2",  # Blue
    BenchmarkType.TRITON_GEMM_ONLY: "#7B1FA2",  # Purple
    BenchmarkType.PYTORCH_RCCL: "#F57C00",  # Orange
    BenchmarkType.PYTORCH_GEMM_ONLY: "#FFB74D",  # Light Orange
}

# Preferred display order
BENCHMARK_ORDER = [
    BenchmarkType.IRIS_FUSED,
    BenchmarkType.IRIS_OPTIMIZED,
    BenchmarkType.TRITONBLAS_RCCL,
    BenchmarkType.TRITON_DEVICE_SDMA,
    BenchmarkType.TRITON_HOST_SDMA,
    BenchmarkType.TRITON_GEMM_ONLY,
    BenchmarkType.PYTORCH_RCCL,
    BenchmarkType.PYTORCH_GEMM_ONLY,
]


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


def canonical_operation_name(operation_name):
    """Map variant-specific operation names to the canonical sweep operation."""
    if not operation_name:
        return None

    if "all_gather_matmul" in operation_name:
        return "all_gather_matmul"

    if "matmul_all_gather" in operation_name:
        return "matmul_all_gather"

    return operation_name


def plot_sweep_results(input_file, output_file, device="MI300X"):
    """Create grouped bar chart from sweep results."""

    # Load results
    with open(input_file, "r") as f:
        results = json.load(f)

    if not results:
        print("No results found in input file")
        return

    # Detect operation type from the top-level row first, then fall back to
    # benchmark-specific operation names for older result files.
    operation = "matmul_all_gather"  # default
    for result in results:
        top_level_operation = canonical_operation_name(result.get("operation"))
        if top_level_operation:
            operation = top_level_operation
            break

        if "benchmarks" in result:
            for bench_data in result["benchmarks"].values():
                if "operation" in bench_data:
                    operation = canonical_operation_name(bench_data["operation"])
                    break
            if operation != "matmul_all_gather":
                break

    # Extract dimension configurations and normalize benchmark names
    dim_configs = []
    benchmark_types = set()

    for result in results:
        m, n, k = result["M"], result["N"], result["K"]
        dim_label = f"{m}×{n}×{k}"
        dim_configs.append((m, n, k, dim_label))

        # Normalize benchmark names to canonical types
        for bench_name in result["benchmarks"].keys():
            if bench_name in BENCHMARK_NAME_MAP:
                benchmark_types.add(BENCHMARK_NAME_MAP[bench_name])

    # Sort by preferred order
    def sort_key(bench_type):
        try:
            return BENCHMARK_ORDER.index(bench_type)
        except ValueError:
            return len(BENCHMARK_ORDER)  # Put unknowns at the end

    benchmark_types = sorted(benchmark_types, key=sort_key)

    # Prepare data for plotting - organize by benchmark type
    data = {bench_type: [] for bench_type in benchmark_types}

    for result in results:
        for bench_type in benchmark_types:
            # Find all raw benchmark names that map to this type
            matching_names = [
                name
                for name, btype in BENCHMARK_NAME_MAP.items()
                if btype == bench_type and name in result["benchmarks"]
            ]

            if matching_names:
                # Use the first matching benchmark name
                bench_name = matching_names[0]
                tflops = extract_tflops(result["benchmarks"][bench_name], bench_name)
                data[bench_type].append(tflops if tflops is not None else 0)
            else:
                data[bench_type].append(0)

    # Create plot
    fig, ax = plt.subplots(figsize=(14, 8))

    x = np.arange(len(dim_configs))
    width = 0.8 / len(benchmark_types)  # Width of bars

    # Plot bars for each benchmark type
    for i, bench_type in enumerate(benchmark_types):
        offset = width * (i - len(benchmark_types) / 2 + 0.5)
        values = data[bench_type]
        color = BENCHMARK_COLORS.get(bench_type, f"C{i}")
        display_label = BENCHMARK_LABELS.get(bench_type, str(bench_type))

        bars = ax.bar(
            x + offset,
            values,
            width,
            label=display_label,
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
    ax.set_xlabel("Dimension Configuration (M×N×K)", fontsize=12, fontweight="bold")
    ax.set_ylabel("TFLOPS", fontsize=12, fontweight="bold")
    # Set title based on operation type
    if operation == "all_gather_matmul":
        title = f"All-Gather-Matmul Benchmark Sweep: TFLOPS Comparison ({device})"
    else:
        title = f"Matmul-All-Gather Benchmark Sweep: TFLOPS Comparison ({device})"
    ax.set_title(title, fontsize=14, fontweight="bold", pad=20)
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

        # Add TFLOPS values and validation status for each benchmark type
        for bench_type in benchmark_types:
            val = data[bench_type][idx]
            label = BENCHMARK_LABELS.get(bench_type, str(bench_type))

            if val > 0:
                row[label] = f"{val:.2f}"
            else:
                row[label] = "FAILED"

            # Add validation status column - find the raw benchmark name
            matching_names = [
                name
                for name, btype in BENCHMARK_NAME_MAP.items()
                if btype == bench_type and name in results[idx]["benchmarks"]
            ]

            validation_key = f"{label}_validation"
            if matching_names:
                bench_data = results[idx]["benchmarks"].get(matching_names[0], {})
                if bench_data.get("success") is True:
                    row[validation_key] = "✓"
                elif bench_data.get("success") is False:
                    row[validation_key] = "✗"
                else:
                    row[validation_key] = "-"
            else:
                row[validation_key] = "-"

        # Find best performer
        valid_values = [
            (bench_type, data[bench_type][idx]) for bench_type in benchmark_types if data[bench_type][idx] > 0
        ]
        if valid_values:
            best_type, best_val = max(valid_values, key=lambda x: x[1])
            best_label = BENCHMARK_LABELS.get(best_type, str(best_type))
            row["Best"] = f"{best_label} ({best_val:.2f})"
        else:
            row["Best"] = "NONE"

        table_data.append(row)

    # Calculate column widths
    benchmark_labels = [BENCHMARK_LABELS.get(bt, str(bt)) for bt in benchmark_types]
    validation_headers = [f"{label}_validation" for label in benchmark_labels]
    headers = ["M", "N", "K", "Config"] + benchmark_labels + validation_headers + ["Best"]
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
    parser.add_argument(
        "--device",
        type=str,
        default="MI300X",
        help="Device name to display in plot title",
    )

    args = parser.parse_args()

    input_path = Path(args.input)
    if not input_path.exists():
        print(f"Error: Input file not found: {input_path}")
        return 1

    plot_sweep_results(input_path, args.output, args.device)
    return 0


if __name__ == "__main__":
    exit(main())
