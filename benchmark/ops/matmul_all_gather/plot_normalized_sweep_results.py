#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
# Copyright (c) 2025-2026 Advanced Micro Devices, Inc. All rights reserved.

"""
Plot normalized benchmark sweep results from benchmark_sweep_results.json.

Creates a grouped bar chart comparing performance normalized by TritonBlas GEMM-only baseline.
All values are shown relative to TRITON_GEMM_ONLY (which appears as 1.0).

Usage:
    python plot_normalized_sweep_results.py [--input benchmark_sweep_results.json] [--output plot.png]
"""

import json
import argparse
from pathlib import Path
from enum import Enum
import re
import matplotlib.pyplot as plt
import numpy as np


class BenchmarkType(Enum):
    """Canonical benchmark types."""

    IRIS_FUSED = "iris_fused"
    IRIS_OPTIMIZED = "iris_optimized"
    TRITONBLAS_RCCL = "tritonblas_rccl"
    TRITON_DEVICE_TRIGGERED_SDMA = "triton_device_triggered_sdma"
    TRITON_DEVICE_SDMA = "triton_device_sdma"
    TRITON_HOST_SDMA = "triton_host_sdma"
    TRITON_HIPMEMCPY = "triton_hipmemcpy"
    TRITON_GEMM_ONLY = "triton_gemm_only"
    PYTORCH_RCCL = "pytorch_rccl"
    PYTORCH_GEMM_ONLY = "pytorch_gemm_only"


# Normalize benchmark names to canonical types
BENCHMARK_NAME_MAP = {
    "baseline": BenchmarkType.IRIS_FUSED,
    "hbm_buffer": BenchmarkType.IRIS_OPTIMIZED,
    "tritonblas_rccl": BenchmarkType.TRITONBLAS_RCCL,
    "tritonblas_rcclbaseline": BenchmarkType.TRITONBLAS_RCCL,
    # "device_copy_engine": BenchmarkType.TRITON_DEVICE_TRIGGERED_SDMA,
    "copy_engine_device": BenchmarkType.TRITON_DEVICE_SDMA,
    # "copy_engine": BenchmarkType.TRITON_DEVICE_SDMA,
    "host_copy_engine": BenchmarkType.TRITON_DEVICE_TRIGGERED_SDMA,
    "copy_engine_host": BenchmarkType.TRITON_HOST_SDMA,
    # "copy_engine_host_hip_memcpy": BenchmarkType.TRITON_HIPMEMCPY,
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
    BenchmarkType.TRITONBLAS_RCCL: "TritonBLAS + RCCL",
    BenchmarkType.TRITON_DEVICE_TRIGGERED_SDMA: "TritonBLAS w/ prepopulated device-triggered SDMA",
    BenchmarkType.TRITON_DEVICE_SDMA: "TritonBLAS w/ device-initiated SDMA",
    BenchmarkType.TRITON_HOST_SDMA: "TritonBLAS w/ host-initiated SDMA",
    BenchmarkType.TRITON_HIPMEMCPY: "TritonBlas + host HIP memcpy",
    BenchmarkType.TRITON_GEMM_ONLY: "TritonBLAS (GEMM only)",
    BenchmarkType.PYTORCH_RCCL: "Pytorch + RCCL",
    BenchmarkType.PYTORCH_GEMM_ONLY: "Pytorch (GEMM only)",
}

# Colors for each benchmark type
BENCHMARK_COLORS = {
    BenchmarkType.IRIS_FUSED: "#FC0000",  # Dark Green
    BenchmarkType.IRIS_OPTIMIZED: "#FC0000",  # Light Green
    BenchmarkType.TRITONBLAS_RCCL: "#02AA0A",  # Teal
    BenchmarkType.TRITON_DEVICE_TRIGGERED_SDMA: "#1976D2",  # Light Blue
    BenchmarkType.TRITON_DEVICE_SDMA: "#82E8FF",  # Light Blue
    BenchmarkType.TRITON_HOST_SDMA: "#1976D2",  # Blue
    BenchmarkType.TRITON_HIPMEMCPY: "#5C6BC0",  # Indigo
    BenchmarkType.TRITON_GEMM_ONLY: "#7B1FA2",  # Purple
    BenchmarkType.PYTORCH_RCCL: "#F57C00",  # Orange
    BenchmarkType.PYTORCH_GEMM_ONLY: "#FFFB00",  # Light Orange
}

# Preferred display order
BENCHMARK_ORDER = [
    BenchmarkType.IRIS_FUSED,
    BenchmarkType.IRIS_OPTIMIZED,
    BenchmarkType.TRITONBLAS_RCCL,
    BenchmarkType.TRITON_DEVICE_TRIGGERED_SDMA,
    BenchmarkType.TRITON_HOST_SDMA,
    BenchmarkType.TRITON_HIPMEMCPY,
    BenchmarkType.TRITON_DEVICE_SDMA,
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


def plot_sweep_results(
    input_file, output_file, device="MI300X", g_shapes_only=False, sort_by="none", filter_regex=None
):
    """Create grouped bar chart from sweep results.

    Args:
        input_file: Path to JSON results file
        output_file: Path to output PNG file
        device: Device name for plot title
        g_shapes_only: If True, only plot shapes with labels starting with 'g'
        sort_by: How to sort shapes: 'none', 'narrow-to-wide', 'wide-to-narrow',
                 'compute-intensity', 'memory-bound'
        filter_regex: Optional regex pattern to filter benchmark variants (e.g., 'tritonblas', 'iris|pytorch')
    """

    # Load results
    with open(input_file, "r") as f:
        results = json.load(f)

    if not results:
        print("No results found in input file")
        return

    # Filter for g-shapes if requested
    if g_shapes_only:
        results = [r for r in results if r.get("label", "").startswith("g")]
        if not results:
            print("No g-shapes found in input file")
            return
        print(f"Filtered to {len(results)} g-shapes")

    results = [r for r in results if r.get("label") not in ["g10", "g12", "g13"]]  # Exclude specific shapes

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
        label = result.get("label", f"{m}×{n}×{k}")
        dim_label = f"{label} ({m}×{n}×{k})" if label and not label.startswith(str(m)) else f"{m}×{n}×{k}"
        dim_configs.append((m, n, k, dim_label))

        # Normalize benchmark names to canonical types
        for bench_name in result["benchmarks"].keys():
            if bench_name in BENCHMARK_NAME_MAP:
                benchmark_types.add(BENCHMARK_NAME_MAP[bench_name])

    # Sort results based on sort_by parameter
    if sort_by == "narrow-to-wide":
        # Sort by N (output width) ascending
        results = sorted(results, key=lambda r: r["N"])
    elif sort_by == "wide-to-narrow":
        # Sort by N (output width) descending
        results = sorted(results, key=lambda r: r["N"], reverse=True)
    elif sort_by == "compute-intensity":
        # Sort by compute/memory ratio: (2*M*N*K) / ((M*K + K*N + M*N)*2bytes) descending
        # Higher ratio = more compute bound
        def compute_intensity(r):
            m, n, k = r["M"], r["N"], r["K"]
            flops = 2 * m * n * k
            bytes_moved = (m * k + k * n + m * n) * 2  # fp16
            return flops / bytes_moved if bytes_moved > 0 else 0

        results = sorted(results, key=compute_intensity, reverse=True)
    elif sort_by == "memory-bound":
        # Sort by compute/memory ratio ascending (most memory bound first)
        def compute_intensity(r):
            m, n, k = r["M"], r["N"], r["K"]
            flops = 2 * m * n * k
            bytes_moved = (m * k + k * n + m * n) * 2  # fp16
            return flops / bytes_moved if bytes_moved > 0 else 0

        results = sorted(results, key=compute_intensity)
    elif sort_by == "output-size":
        # Sort by output size (M × N) ascending
        results = sorted(results, key=lambda r: r["M"] * r["N"])
    elif sort_by == "k-size":
        # Sort by K dimension ascending
        results = sorted(results, key=lambda r: r["K"])
    elif sort_by == "m-to-k-ratio":
        # Sort by M/K ratio ascending (using adjusted dimensions for matmul_all_gather)
        def m_to_k_ratio(r):
            m, n, k = r["M"], r["N"], r["K"]
            if operation == "matmul_all_gather":
                # Attention case: n == k
                if n == k:
                    effective_m = m // 8
                    effective_k = k
                else:
                    effective_m = m
                    effective_k = k // 8
            else:
                effective_m = m
                effective_k = k
            return effective_m / effective_k if effective_k > 0 else 0

        results = sorted(results, key=m_to_k_ratio, reverse=True)
    # else: sort_by == "none", keep original order

    # Build dim_configs from the (now sorted or unsorted) results
    dim_configs = []
    for result in results:
        m, n, k = result["M"], result["N"], result["K"]
        label = result.get("label", f"{m}×{n}×{k}")
        # dim_label = f"{label} ({m}×{n}×{k})" if label and not label.startswith(str(m)) else f"{m}×{n}×{k}"
        dim_label = f"{m}×{n}×{k}"
        if operation == "matmul_all_gather":
            # is attention?
            if n == k:
                dim_label = f"{m // 8}×{n // 8}×{k}"
            else:
                dim_label = f"{m}×{k // 8}×{n}"
        dim_configs.append((m, n, k, dim_label))

    # Filter benchmark types by regex if provided
    if filter_regex:
        try:
            pattern = re.compile(filter_regex, re.IGNORECASE)
            filtered_types = []
            for bench_type in benchmark_types:
                # Match against both the enum value and display label
                label = BENCHMARK_LABELS.get(bench_type, str(bench_type))
                enum_value = bench_type.value if hasattr(bench_type, "value") else str(bench_type)
                if pattern.search(label) or pattern.search(enum_value):
                    filtered_types.append(bench_type)
            benchmark_types = set(filtered_types)
            if not benchmark_types:
                print(f"Warning: filter pattern '{filter_regex}' matched no benchmarks")
            else:
                print(f"Filtered to {len(benchmark_types)} benchmark variants matching '{filter_regex}'")
        except re.error as e:
            print(f"Error: Invalid regex pattern '{filter_regex}': {e}")
            return

    # Sort by preferred order
    def sort_key(bench_type):
        try:
            return BENCHMARK_ORDER.index(bench_type)
        except ValueError:
            return len(BENCHMARK_ORDER)  # Put unknowns at the end

    benchmark_types = sorted(benchmark_types, key=sort_key)

    # First pass: extract baseline TFLOPS (TRITON_GEMM_ONLY) for each config
    baseline_tflops = []
    for result in results:
        matching_names = [
            name
            for name, btype in BENCHMARK_NAME_MAP.items()
            if btype == BenchmarkType.TRITONBLAS_RCCL and name in result["benchmarks"]
        ]
        if matching_names:
            baseline = extract_tflops(result["benchmarks"][matching_names[0]], matching_names[0])
            baseline_tflops.append(baseline if baseline is not None else 0)
        else:
            baseline_tflops.append(0)

    # Prepare data for plotting - organize by benchmark type
    data = {bench_type: [] for bench_type in benchmark_types}

    # Second pass: normalize all values by baseline
    for idx, result in enumerate(results):
        baseline = baseline_tflops[idx]
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

                # Normalize: baseline becomes 1.0, faster > 1.0, slower < 1.0
                if tflops is not None and baseline > 0:
                    normalized = tflops / baseline
                    data[bench_type].append(normalized)
                else:
                    data[bench_type].append(0)
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
                    f"{val:.2f}",
                    ha="center",
                    va="bottom",
                    fontsize=7,
                    rotation=0,
                )

    # Customize plot
    ax.set_xlabel("Local GEMM (M×N×K)", fontsize=12, fontweight="bold")
    # ax.set_ylabel("Normalized Performance (rel. to TritonBLAS GEMM)", fontsize=12, fontweight="bold")
    ax.set_ylabel("Normalized Performance (rel. to TritonBLAS + RCCL)", fontsize=12, fontweight="bold")
    # Set title based on operation type
    if operation == "all_gather_matmul":
        title = f"All-Gather-GEMM: Normalized Performance ({device})"
        ax.legend(loc="upper right", fontsize=10, ncols=3)
    else:
        title = f"GEMM-All-Gather: Normalized Performance ({device})"
        ax.legend(loc="upper center", fontsize=10)
    ax.set_title(title, fontsize=14, fontweight="bold", pad=20)
    ax.set_xticks(x)
    ax.set_xticklabels([label for _, _, _, label in dim_configs], rotation=45, ha="right")
    ax.grid(axis="y", alpha=0.3, linestyle="--")

    # Add horizontal line at y=1.0 to mark the baseline (TRITON_GEMM_ONLY)
    ax.axhline(y=1.0, color="red", linestyle="--", linewidth=1.5, alpha=0.7, label="TritonBlas GEMM baseline")

    # Set y-axis to start from 0
    ax.set_ylim(bottom=0)

    plt.tight_layout()

    # Save figure
    plt.savefig(output_file, dpi=300, bbox_inches="tight", transparent=True)
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
                row[label] = f"{val:.3f}x"
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
            row["Best"] = f"{best_label} ({best_val:.3f}x)"
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
        default="benchmark/ops/matmul_all_gather/normalized_sweep_results_plot.png",
        help="Output PNG file for plot",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="MI300X",
        help="Device name to display in plot title",
    )
    parser.add_argument(
        "--g-shapes",
        action="store_true",
        help="Only plot shapes with labels starting with 'g' (e.g., g1, g2, ...)",
    )
    parser.add_argument(
        "--sort",
        type=str,
        choices=[
            "none",
            "narrow-to-wide",
            "wide-to-narrow",
            "compute-intensity",
            "memory-bound",
            "output-size",
            "k-size",
            "m-to-k-ratio",
        ],
        default="none",
        help="Sort shapes by: 'narrow-to-wide' (by N ascending), 'wide-to-narrow' (by N descending), "
        "'compute-intensity' (compute/memory ratio descending), 'memory-bound' (compute/memory ratio ascending), "
        "'output-size' (by M×N output size ascending)",
    )
    parser.add_argument(
        "--filter",
        type=str,
        default=None,
        metavar="REGEX",
        help="Regex pattern to filter benchmark variants (case-insensitive). "
        "Examples: 'tritonblas' (all TritonBlas variants), 'iris|pytorch' (Iris or PyTorch), "
        "'^Iris' (variants starting with 'Iris'), 'RCCL' (all RCCL-based variants). "
        "Matches against benchmark display labels.",
    )

    args = parser.parse_args()

    input_path = Path(args.input)
    if not input_path.exists():
        print(f"Error: Input file not found: {input_path}")
        return 1

    plot_sweep_results(input_path, args.output, args.device, args.g_shapes, args.sort, args.filter)
    return 0


if __name__ == "__main__":
    exit(main())
