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
import math
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
    # GEMM+AllReduce specific benchmarks
    PYTORCH_BASELINE = "pytorchbaseline"
    ONE_SHOT = "one_shot"
    TWO_SHOT = "two_shot"
    COPY_ENGINE_ONE_SHOT = "copy_engine_device_one_shot"
    COPY_ENGINE_TWO_SHOT = "copy_engine_device_two_shot"
    COPY_ENGINE_TWO_SHOT_GPU_INIT = "copy_engine_device_two_shot_gpu_init"
    MATMUL_ONLY = "matmul_only"
    PYTORCH_MATMUL_ONLY = "pytorchmatmul_only"
    COPY_ENGINE_PARTITIONED_GEMM = "copy_engine_partitioned_gemm"


# Normalize benchmark names to canonical types
BENCHMARK_NAME_MAP = {
    # matmul_all_gather benchmarks
    # "baseline": BenchmarkType.IRIS_FUSED,
    "hbm_buffer": BenchmarkType.IRIS_OPTIMIZED,
    "tritonblas_rccl": BenchmarkType.TRITONBLAS_RCCL,
    "tritonblas_rcclbaseline": BenchmarkType.TRITONBLAS_RCCL,
    "device_copy_engine": BenchmarkType.TRITON_DEVICE_TRIGGERED_SDMA,
    "copy_engine_device": BenchmarkType.TRITON_DEVICE_SDMA,
    # "copy_engine": BenchmarkType.TRITON_DEVICE_SDMA,
    # "host_copy_engine": BenchmarkType.TRITON_HOST_SDMA,
    "copy_engine_host": BenchmarkType.TRITON_HOST_SDMA,
    # "copy_engine_host_hip_memcpy": BenchmarkType.TRITON_HIPMEMCPY,
    # matmul_all_reduce benchmarks
    "pytorchbaseline": BenchmarkType.PYTORCH_BASELINE,
    "one_shot": BenchmarkType.ONE_SHOT,
    "two_shot": BenchmarkType.TWO_SHOT,
    "copy_engine_device_one_shot": BenchmarkType.COPY_ENGINE_ONE_SHOT,
    "copy_engine_device_two_shot": BenchmarkType.COPY_ENGINE_TWO_SHOT,
    "copy_engine_device_two_shot_gpu_init": BenchmarkType.COPY_ENGINE_TWO_SHOT_GPU_INIT,
    "matmul_only": BenchmarkType.MATMUL_ONLY,
    "pytorchmatmul_only": BenchmarkType.PYTORCH_MATMUL_ONLY,
    "copy_engine_partitioned_gemm": BenchmarkType.COPY_ENGINE_PARTITIONED_GEMM,
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
    # GEMM+AllReduce
    BenchmarkType.PYTORCH_BASELINE: "PyTorch Baseline",
    BenchmarkType.ONE_SHOT: "One-Shot",
    BenchmarkType.TWO_SHOT: "Two-Shot",
    BenchmarkType.COPY_ENGINE_ONE_SHOT: "Copy Engine One-Shot",
    BenchmarkType.COPY_ENGINE_TWO_SHOT: "Copy Engine Two-Shot",
    BenchmarkType.COPY_ENGINE_TWO_SHOT_GPU_INIT: "Copy Engine Two-Shot (GPU-Init)",
    BenchmarkType.MATMUL_ONLY: "MatMul Only",
    BenchmarkType.PYTORCH_MATMUL_ONLY: "PyTorch MatMul Only",
    BenchmarkType.COPY_ENGINE_PARTITIONED_GEMM: "Copy Engine Partitioned GEMM",
}

# Colors for each benchmark type
BENCHMARK_COLORS = {
    BenchmarkType.IRIS_FUSED: "#FC0000",  # Red
    BenchmarkType.IRIS_OPTIMIZED: "#FC0000",  # Red
    BenchmarkType.TRITONBLAS_RCCL: "#02AA0A",  # Green
    BenchmarkType.TRITON_DEVICE_TRIGGERED_SDMA: "#1976D2",  # Blue
    BenchmarkType.TRITON_DEVICE_SDMA: "#82E8FF",  # Light Blue
    BenchmarkType.TRITON_HOST_SDMA: "#1976D2",  # Blue
    BenchmarkType.TRITON_HIPMEMCPY: "#5C6BC0",  # Indigo
    BenchmarkType.TRITON_GEMM_ONLY: "#7B1FA2",  # Purple
    BenchmarkType.PYTORCH_RCCL: "#F57C00",  # Orange
    BenchmarkType.PYTORCH_GEMM_ONLY: "#FFFB00",  # Yellow
    # GEMM+AllReduce colors
    BenchmarkType.PYTORCH_BASELINE: "#F57C00",  # Orange (similar to PYTORCH_RCCL)
    BenchmarkType.ONE_SHOT: "#1976D2",  # Blue
    BenchmarkType.COPY_ENGINE_ONE_SHOT: "#64B5F6",  # Light Blue (one-shot w/ SDMA)
    BenchmarkType.TWO_SHOT: "#388E3C",  # Green
    BenchmarkType.COPY_ENGINE_TWO_SHOT: "#81C784",  # Light Green (two-shot w/ SDMA)
    BenchmarkType.COPY_ENGINE_TWO_SHOT_GPU_INIT: "#A5D6A7",  # Lighter Green (two-shot w/ GPU-init SDMA)
    BenchmarkType.MATMUL_ONLY: "#7B1FA2",  # Purple
    BenchmarkType.PYTORCH_MATMUL_ONLY: "#FFFB00",  # Yellow
    BenchmarkType.COPY_ENGINE_PARTITIONED_GEMM: "#d62728",  # Red
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
    # GEMM+AllReduce order
    BenchmarkType.PYTORCH_BASELINE,
    BenchmarkType.ONE_SHOT,
    BenchmarkType.COPY_ENGINE_ONE_SHOT,
    BenchmarkType.TWO_SHOT,
    BenchmarkType.COPY_ENGINE_TWO_SHOT,
    BenchmarkType.COPY_ENGINE_TWO_SHOT_GPU_INIT,
    BenchmarkType.MATMUL_ONLY,
    BenchmarkType.PYTORCH_MATMUL_ONLY,
    BenchmarkType.COPY_ENGINE_PARTITIONED_GEMM,
]


def extract_performance_metric(benchmark_data, benchmark_name, operation="matmul_all_gather"):
    """Extract performance metric from benchmark result.

    For matmul_all_gather: returns TFLOPS (higher is better)
    For matmul_all_reduce: returns gpu_time_ms (lower is better, will be inverted during normalization)
    """
    if benchmark_data.get("status") == "FAILED":
        return None

    # For matmul_all_reduce, use gpu_time_ms directly
    if operation == "matmul_all_reduce":
        gpu_time = benchmark_data.get("gpu_time_ms")
        if gpu_time is not None and gpu_time > 0:
            return gpu_time
        return None

    # For matmul_all_gather, use TFLOPS
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

    if "matmul_all_reduce" in operation_name:
        return "matmul_all_reduce"

    return operation_name


def plot_sweep_results(input_file, output_file, device="MI300X", g_shapes_only=False, sort_by="none", filter_regex=None, exclude_regex=None, baseline_type=None, m_filter=None):
    """Create grouped bar chart from sweep results.

    Args:
        input_file: Path to JSON results file
        output_file: Path to output PNG file
        device: Device name for plot title
        g_shapes_only: If True, only plot shapes with labels starting with 'g'
        sort_by: How to sort shapes: 'none', 'narrow-to-wide', 'wide-to-narrow',
                 'compute-intensity', 'memory-bound'
        filter_regex: Optional regex pattern to filter benchmark variants (e.g., 'tritonblas', 'iris|pytorch')
        exclude_regex: Optional regex pattern to exclude specific benchmarks while preserving layout spacing
        baseline_type: Override baseline benchmark type for normalization (auto-detected if None)
        m_filter: Optional list of M values to include (e.g., [2048, 4096, 16384])
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

    # Filter by M values if requested
    if m_filter is not None:
        original_count = len(results)
        results = [r for r in results if r.get("M") in m_filter]
        if not results:
            print(f"No results found for M values: {m_filter}")
            return
        print(f"Filtered to {len(results)} shapes with M in {m_filter} (from {original_count} total)")


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
        # Always use the label from the data (contains model name and operation)
        dim_label = label if label else f"{m}×{n}×{k}"
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
    elif sort_by == "copy-engine-benefit":
        # Sort by intuition: factors that predict when copy engine helps
        # Primary: tiles (low to high) - more tiles = more parallelism to hide CE overhead
        # Secondary: K (low to high) - smaller K = less reduction work
        # Results sorted ascending so predicted benefit increases LEFT→RIGHT
        def ce_sort_key(r):
            k = r["K"]

            # Get tiles from first available benchmark
            benchmarks = r.get("benchmarks", {})
            tiles = 0
            for bench_name in benchmarks.keys():
                bench_data = benchmarks[bench_name]
                tiles = bench_data.get("total_tiles", 0)
                if tiles > 0:
                    break

            # Simple tuple sort: (tiles, K)
            return (tiles, k)

        results = sorted(results, key=ce_sort_key, reverse=False)
    # else: sort_by == "none", keep original order

    # Build dim_configs from the (now sorted or unsorted) results
    dim_configs = []
    for result in results:
        m, n, k = result["M"], result["N"], result["K"]
        label = result.get("label", f"{m}×{n}×{k}")

        # Get tiles from first available benchmark
        benchmarks = result.get("benchmarks", {})
        tiles = 0
        for bench_name in benchmarks.keys():
            bench_data = benchmarks[bench_name]
            tiles = bench_data.get("total_tiles", 0)
            if tiles > 0:
                break

        # For matmul_all_gather, adjust dimension display for local GEMM
        if operation == "matmul_all_gather":
            # is attention?
            if n == k:
                dim_str = f"{m // 8}×{n // 8}×{k}"
            else:
                dim_str = f"{m}×{k // 8}×{n}"
            # Include model label with adjusted dimensions
            dim_label = f"{label}\n{dim_str}" if label else dim_str
        else:
            # For matmul_all_reduce, use the label with tiles and K info
            dim_label = f"{label}\n(tiles={tiles}, K={k})"
        dim_configs.append((m, n, k, dim_label))

    # Filter benchmark types by regex if provided
    if filter_regex:
        try:
            pattern = re.compile(filter_regex, re.IGNORECASE)
            filtered_types = []
            for bench_type in benchmark_types:
                # Match against both the enum value and display label
                label = BENCHMARK_LABELS.get(bench_type, str(bench_type))
                enum_value = bench_type.value if hasattr(bench_type, 'value') else str(bench_type)
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

    # Track which benchmarks to hide (for spacing preservation)
    hidden_benchmarks = set()

    # Always hide the baseline benchmark (it's always 1.0 by definition)
    hidden_benchmarks.add(baseline_type)

    if exclude_regex:
        try:
            exclude_pattern = re.compile(exclude_regex, re.IGNORECASE)
            for bench_type in benchmark_types:
                label = BENCHMARK_LABELS.get(bench_type, str(bench_type))
                enum_value = bench_type.value if hasattr(bench_type, 'value') else str(bench_type)
                if exclude_pattern.search(label) or exclude_pattern.search(enum_value):
                    hidden_benchmarks.add(bench_type)
            if hidden_benchmarks:
                print(f"Hiding {len(hidden_benchmarks)} benchmark variants matching exclude pattern '{exclude_regex}'")
        except re.error as e:
            print(f"Error: Invalid exclude regex pattern '{exclude_regex}': {e}")
            return

    # Sort by preferred order
    def sort_key(bench_type):
        try:
            return BENCHMARK_ORDER.index(bench_type)
        except ValueError:
            return len(BENCHMARK_ORDER)  # Put unknowns at the end

    benchmark_types = sorted(benchmark_types, key=sort_key)

    # Determine baseline type based on operation if not specified
    if baseline_type is None:
        if operation == "matmul_all_reduce":
            baseline_type = BenchmarkType.TRITONBLAS_RCCL
        else:
            baseline_type = BenchmarkType.TRITONBLAS_RCCL

    # First pass: extract baseline performance metric for each config
    baseline_metrics = []
    for result in results:
        matching_names = [
            name
            for name, btype in BENCHMARK_NAME_MAP.items()
            if btype == baseline_type and name in result["benchmarks"]
        ]
        if matching_names:
            baseline = extract_performance_metric(result["benchmarks"][matching_names[0]], matching_names[0], operation)
            baseline_metrics.append(baseline if baseline is not None else 0)
        else:
            baseline_metrics.append(0)

    # Prepare data for plotting - organize by benchmark type (excluding baseline)
    data = {bench_type: [] for bench_type in benchmark_types if bench_type != baseline_type}

    # Second pass: normalize all values by baseline
    for idx, result in enumerate(results):
        baseline = baseline_metrics[idx]
        for bench_type in benchmark_types:
            # Skip the baseline benchmark
            if bench_type == baseline_type:
                continue

            # Find all raw benchmark names that map to this type
            matching_names = [
                name
                for name, btype in BENCHMARK_NAME_MAP.items()
                if btype == bench_type and name in result["benchmarks"]
            ]

            if matching_names:
                # Use the first matching benchmark name
                bench_name = matching_names[0]
                metric = extract_performance_metric(result["benchmarks"][bench_name], bench_name, operation)

                # Normalize: baseline becomes 1.0, faster > 1.0, slower < 1.0
                if metric is not None and baseline > 0:
                    if operation == "matmul_all_reduce":
                        # For runtime: baseline_time / test_time (lower time = higher normalized value)
                        normalized = baseline / metric
                    else:
                        # For TFLOPS: test_tflops / baseline_tflops (higher tflops = higher normalized value)
                        normalized = metric / baseline
                    data[bench_type].append(normalized)
                else:
                    data[bench_type].append(0)
            else:
                data[bench_type].append(0)

    # Create plot with dynamic height based on number of configs
    num_configs = len(dim_configs)
    # Base height of 8, increase for more configurations
    fig_height = max(8, min(12, 8 + (num_configs - 10) * 0.2))
    fig, ax = plt.subplots(figsize=(20, fig_height))

    # Calculate number of visible benchmarks (excluding hidden ones)
    num_visible_benchmarks = len([bt for bt in benchmark_types if bt not in hidden_benchmarks])

    x = np.arange(len(dim_configs))  # Bar group positions (no extra spacing)
    width = 0.8 / num_visible_benchmarks  # Width of bars based on visible benchmarks only

    # Plot bars for each benchmark type
    visible_index = 0  # Track index for visible bars only
    for i, bench_type in enumerate(benchmark_types):
        # Skip plotting if this is the baseline or if this benchmark is hidden
        if bench_type == baseline_type or bench_type in hidden_benchmarks:
            continue

        # Calculate offset based on visible benchmarks only
        offset = width * (visible_index - num_visible_benchmarks / 2 + 0.5)
        values = data[bench_type]
        color = BENCHMARK_COLORS.get(bench_type, f"C{visible_index}")
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

        # Increment visible index for next visible benchmark
        visible_index += 1

        # Add value labels on top of bars (only for non-zero values)
        # for j, (bar, val) in enumerate(zip(bars, values)):
        #     if val > 0:
        #         height = bar.get_height()
        #         ax.text(
        #             bar.get_x() + bar.get_width() / 2,
        #             height,
        #             f"{val:.2f}",
        #             ha="center",
        #             va="bottom",
        #             fontsize=10,
        #             rotation=0,
        #             # fontweight="bold",
        #         )

    # Customize plot
    ax.set_xlabel("Local GEMM (M×N×K)", fontsize=22, fontweight="bold")

    # Set title and labels based on operation type
    baseline_name = BENCHMARK_LABELS.get(baseline_type, "Baseline")

    # Add M filter info to title if applicable
    m_filter_str = ""
    if m_filter:
        m_values_str = ", ".join([f"M={m}" for m in sorted(m_filter)])
        m_filter_str = f" - {m_values_str}"

    if operation == "all_gather_matmul":
        title = f"All-Gather-GEMM: Normalized Performance ({device}){m_filter_str}"
        ylabel = f"Normalized Performance\n(vs {baseline_name})"
        baseline_label = baseline_name
        # Calculate ncols to get 2 rows in legend: ncols = ceil(num_visible_benchmarks / 2)
        num_visible = len([bt for bt in benchmark_types if bt not in hidden_benchmarks])
        ncols = (num_visible + 1) // 2  # ceil division
        ax.legend(loc="upper right", fontsize=16, ncols=2, facecolor='white', framealpha=1.0)
    elif operation == "matmul_all_reduce":
        title = f"GEMM-AllReduce: Speedup vs {baseline_name} ({device}){m_filter_str}"
        ylabel = f"Speedup vs {baseline_name}"
        baseline_label = baseline_name
        ax.legend(loc="upper left", fontsize=16, ncols=2)
    else:
        title = f"GEMM-All-Gather: Normalized Performance ({device}){m_filter_str}"
        ylabel = f"Normalized Performance\n(vs {baseline_name})"
        baseline_label = baseline_name
        ax.legend(loc="upper right", fontsize=16)

    ax.set_ylabel(ylabel, fontsize=22, fontweight="bold")
    ax.set_title(title, fontsize=22, fontweight="bold", pad=20)
    ax.set_xticks(x)
    ax.set_xticklabels([label for _, _, _, label in dim_configs], rotation=45, ha="right", fontsize=14)
    ax.tick_params(axis='y', labelsize=14)
    ax.grid(axis="y", alpha=0.3, linestyle="--")

    # Add horizontal line at y=1.0 to mark the baseline
    ax.axhline(y=1.0, color="red", linestyle="--", linewidth=1.5, alpha=0.7, label=baseline_label)

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
            # Skip baseline benchmark
            if bench_type == baseline_type:
                continue

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
            (bench_type, data[bench_type][idx]) for bench_type in benchmark_types
            if bench_type != baseline_type and data[bench_type][idx] > 0
        ]
        if valid_values:
            best_type, best_val = max(valid_values, key=lambda x: x[1])
            best_label = BENCHMARK_LABELS.get(best_type, str(best_type))
            row["Best"] = f"{best_label} ({best_val:.3f}x)"
        else:
            row["Best"] = "NONE"

        table_data.append(row)

    # Calculate column widths (exclude baseline)
    benchmark_labels = [BENCHMARK_LABELS.get(bt, str(bt)) for bt in benchmark_types if bt != baseline_type]
    validation_headers = [f"{label}_validation" for label in benchmark_labels]
    headers = ["M", "N", "K", "Config"] + benchmark_labels + validation_headers + ["Best"]
    col_widths = {}
    for header in headers:
        col_widths[header] = len(header)
        for row in table_data:
            if header in row:  # Skip missing keys
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

    # Print copy engine speedup analysis for matmul_all_reduce
    if operation == "matmul_all_reduce":
        print("\n" + "=" * 120)
        print("COPY ENGINE SPEEDUP ANALYSIS")
        print("=" * 120)
        print()

        # Calculate geometric mean speedup for Copy Engine vs non-Copy Engine
        one_shot_ratios = []
        two_shot_ratios = []

        for idx in range(len(results)):
            # Get data for one-shot comparison
            if BenchmarkType.ONE_SHOT in data and BenchmarkType.COPY_ENGINE_ONE_SHOT in data:
                one_shot_val = data[BenchmarkType.ONE_SHOT][idx]
                ce_one_shot_val = data[BenchmarkType.COPY_ENGINE_ONE_SHOT][idx]
                if one_shot_val > 0 and ce_one_shot_val > 0:
                    # Ratio > 1.0 means copy engine is faster
                    one_shot_ratios.append(ce_one_shot_val / one_shot_val)

            # Get data for two-shot comparison
            if BenchmarkType.TWO_SHOT in data and BenchmarkType.COPY_ENGINE_TWO_SHOT in data:
                two_shot_val = data[BenchmarkType.TWO_SHOT][idx]
                ce_two_shot_val = data[BenchmarkType.COPY_ENGINE_TWO_SHOT][idx]
                if two_shot_val > 0 and ce_two_shot_val > 0:
                    # Ratio > 1.0 means copy engine is faster
                    two_shot_ratios.append(ce_two_shot_val / two_shot_val)

        if one_shot_ratios:
            geomean_one_shot = math.exp(sum(math.log(r) for r in one_shot_ratios) / len(one_shot_ratios))
            speedup_pct = (geomean_one_shot - 1) * 100
            print("One-Shot with Copy Engine vs One-Shot:")
            print(f"  Geometric Mean Speedup: {geomean_one_shot:.3f}x ({speedup_pct:+.1f}%)")
            print(f"  Configurations: {len(one_shot_ratios)}")
            print()

        if two_shot_ratios:
            geomean_two_shot = math.exp(sum(math.log(r) for r in two_shot_ratios) / len(two_shot_ratios))
            speedup_pct = (geomean_two_shot - 1) * 100
            print("Two-Shot with Copy Engine vs Two-Shot:")
            print(f"  Geometric Mean Speedup: {geomean_two_shot:.3f}x ({speedup_pct:+.1f}%)")
            print(f"  Configurations: {len(two_shot_ratios)}")
            print()

        print("=" * 120)

        # Additional analysis: geomean for cases matching the heuristic
        print("\n" + "=" * 120)
        print("COPY ENGINE SPEEDUP WITH HEURISTIC FILTER")
        print("=" * 120)
        print()
        print("Heuristic: M >= 2048 AND 1 <= M/K <= 32 AND tiles >= 256")
        print()

        one_shot_ratios_heuristic = []
        two_shot_ratios_heuristic = []
        matched_configs = []

        for idx, result in enumerate(results):
            m = result["M"]
            k = result["K"]
            m_to_k = m / k if k > 0 else 0

            # Get tiles
            benchmarks = result.get("benchmarks", {})
            tiles = 0
            for bench_name in benchmarks.keys():
                bench_data = benchmarks[bench_name]
                tiles = bench_data.get("total_tiles", 0)
                if tiles > 0:
                    break

            # Apply heuristic
            matches_heuristic = (m >= 2048 and 1 <= m_to_k <= 32 and tiles >= 256)

            if matches_heuristic:
                label = result.get("label", f"{m}×{result['N']}×{k}")
                matched_configs.append({
                    'label': label,
                    'M': m,
                    'K': k,
                    'tiles': tiles,
                    'm_to_k': m_to_k
                })

                # Get data for one-shot comparison
                if BenchmarkType.ONE_SHOT in data and BenchmarkType.COPY_ENGINE_ONE_SHOT in data:
                    one_shot_val = data[BenchmarkType.ONE_SHOT][idx]
                    ce_one_shot_val = data[BenchmarkType.COPY_ENGINE_ONE_SHOT][idx]
                    if one_shot_val > 0 and ce_one_shot_val > 0:
                        one_shot_ratios_heuristic.append(ce_one_shot_val / one_shot_val)

                # Get data for two-shot comparison
                if BenchmarkType.TWO_SHOT in data and BenchmarkType.COPY_ENGINE_TWO_SHOT in data:
                    two_shot_val = data[BenchmarkType.TWO_SHOT][idx]
                    ce_two_shot_val = data[BenchmarkType.COPY_ENGINE_TWO_SHOT][idx]
                    if two_shot_val > 0 and ce_two_shot_val > 0:
                        two_shot_ratios_heuristic.append(ce_two_shot_val / two_shot_val)

        print(f"Matched Configurations: {len(matched_configs)}/{len(results)}")
        print()

        if matched_configs:
            print("Matched configurations:")
            print(f"{'Label':<45} {'M':>6} {'K':>6} {'M/K':>7} {'Tiles':>6}")
            print("-" * 80)
            for config in matched_configs[:10]:  # Show first 10
                print(f"{config['label']:<45} {config['M']:>6} {config['K']:>6} {config['m_to_k']:>7.2f} {config['tiles']:>6}")
            if len(matched_configs) > 10:
                print(f"... and {len(matched_configs) - 10} more")
            print()

        if one_shot_ratios_heuristic:
            geomean_one_shot_h = math.exp(sum(math.log(r) for r in one_shot_ratios_heuristic) / len(one_shot_ratios_heuristic))
            speedup_pct_h = (geomean_one_shot_h - 1) * 100
            print("One-Shot with Copy Engine vs One-Shot (Heuristic-Matched):")
            print(f"  Geometric Mean Speedup: {geomean_one_shot_h:.3f}x ({speedup_pct_h:+.1f}%)")
            print(f"  Configurations: {len(one_shot_ratios_heuristic)}")
            print()

        if two_shot_ratios_heuristic:
            geomean_two_shot_h = math.exp(sum(math.log(r) for r in two_shot_ratios_heuristic) / len(two_shot_ratios_heuristic))
            speedup_pct_h = (geomean_two_shot_h - 1) * 100
            print("Two-Shot with Copy Engine vs Two-Shot (Heuristic-Matched):")
            print(f"  Geometric Mean Speedup: {geomean_two_shot_h:.3f}x ({speedup_pct_h:+.1f}%)")
            print(f"  Configurations: {len(two_shot_ratios_heuristic)}")
            print()

        print("=" * 120)


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
        choices=["none", "narrow-to-wide", "wide-to-narrow", "compute-intensity", "memory-bound", "output-size", "k-size", "m-to-k-ratio", "copy-engine-benefit"],
        default="none",
        help="Sort shapes by: 'narrow-to-wide' (by N ascending), 'wide-to-narrow' (by N descending), "
             "'compute-intensity' (compute/memory ratio descending), 'memory-bound' (compute/memory ratio ascending), "
             "'output-size' (by M×N output size ascending), 'copy-engine-benefit' (predicted CE benefit, for matmul_all_reduce)",
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
    parser.add_argument(
        "--exclude",
        type=str,
        default=None,
        metavar="REGEX",
        help="Regex pattern to hide specific benchmarks while preserving layout spacing. "
             "Use this to create overlay-compatible plots. "
             "Example: --exclude 'GEMM only' to hide TRITON_GEMM_ONLY while keeping bar positions identical.",
    )
    parser.add_argument(
        "--baseline",
        type=str,
        default=None,
        metavar="BENCHMARK_TYPE",
        help="Override baseline benchmark type for normalization. "
             "Use the BenchmarkType enum value (e.g., 'TRITONBLAS_RCCL', 'MATMUL_ONLY'). "
             "If not specified, uses TRITONBLAS_RCCL for matmul_all_reduce.",
    )
    parser.add_argument(
        "--m-filter",
        type=str,
        default=None,
        metavar="M_VALUES",
        help="Comma-separated list of M (batch size) values to include. "
             "Example: --m-filter '2048,4096,16384' to only plot those batch sizes.",
    )

    args = parser.parse_args()

    input_path = Path(args.input)
    if not input_path.exists():
        print(f"Error: Input file not found: {input_path}")
        return 1

    # Parse baseline type if specified
    baseline_type = None
    if args.baseline:
        try:
            baseline_type = BenchmarkType[args.baseline.upper()]
        except KeyError:
            print(f"Error: Invalid baseline type '{args.baseline}'. Valid options:")
            for bt in BenchmarkType:
                print(f"  - {bt.name}")
            return 1

    # Parse M filter if specified
    m_filter = None
    if args.m_filter:
        try:
            m_filter = [int(m.strip()) for m in args.m_filter.split(',')]
        except ValueError:
            print(f"Error: Invalid M filter '{args.m_filter}'. Must be comma-separated integers.")
            return 1

    plot_sweep_results(input_path, args.output, args.device, args.g_shapes, args.sort, args.filter, args.exclude, baseline_type, m_filter)
    return 0


if __name__ == "__main__":
    exit(main())
