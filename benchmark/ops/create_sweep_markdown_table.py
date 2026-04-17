#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.

"""
Create a markdown table from benchmark_sweep_results.json.

Example:
    python benchmark/ops/create_sweep_markdown_table.py \
        benchmark/ops/matmul_all_gather/benchmark_sweep_results.json \
        --benchmark host_copy_engine
"""

import argparse
import json
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Create a markdown table from benchmark_sweep_results.json",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "results_json",
        type=Path,
        help="Path to benchmark_sweep_results.json",
    )
    parser.add_argument(
        "--benchmark",
        required=True,
        help="Benchmark key to extract, for example host_copy_engine or copy_engine_host",
    )
    parser.add_argument(
        "--pytorch-benchmark",
        default="pytorchbaseline",
        help="PyTorch reference benchmark key used for the speedup calculation",
    )
    parser.add_argument(
        "--sort-by",
        choices=("shape", "label", "speedup", "tflops"),
        default="shape",
        help="How to sort rows in the output table",
    )
    parser.add_argument(
        "--descending",
        action="store_true",
        help="Sort in descending order",
    )
    return parser.parse_args()


def _as_float(value):
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _shape_string(row: dict) -> str:
    return f"{row.get('M')}x{row.get('N')}x{row.get('K')}"


def _label_string(row: dict) -> str:
    label = row.get("label")
    if label:
        return str(label)
    return _shape_string(row)


def _extract_table_rows(data: list[dict], benchmark_key: str, pytorch_key: str) -> list[dict]:
    rows = []
    for row in data:
        benchmarks = row.get("benchmarks") or {}
        bench = benchmarks.get(benchmark_key) or {}
        pytorch = benchmarks.get(pytorch_key) or {}

        bench_tflops = _as_float(bench.get("tflops"))
        pytorch_tflops = _as_float(pytorch.get("tflops"))

        if bench_tflops is None:
            continue

        speedup = None
        if pytorch_tflops not in (None, 0.0):
            speedup = bench_tflops / pytorch_tflops

        rows.append(
            {
                "shape": _shape_string(row),
                "label": _label_string(row),
                "speedup": speedup,
                "tflops": bench_tflops,
            }
        )
    return rows


def _sort_rows(rows: list[dict], sort_by: str, descending: bool) -> list[dict]:
    if sort_by == "shape":
        key_fn = lambda row: tuple(int(dim) for dim in row["shape"].split("x"))
    elif sort_by == "label":
        key_fn = lambda row: row["label"]
    elif sort_by == "speedup":
        key_fn = lambda row: (row["speedup"] is not None, row["speedup"] if row["speedup"] is not None else float("-inf"))
    else:
        key_fn = lambda row: row["tflops"]

    return sorted(rows, key=key_fn, reverse=descending)


def _format_speedup(value) -> str:
    if value is None:
        return "--"
    return f"{value:.2f}x"


def _format_tflops(value) -> str:
    if value is None:
        return "--"
    return f"{value:.1f}"


def main() -> None:
    args = parse_args()

    with open(args.results_json, "r") as f:
        data = json.load(f)

    if not isinstance(data, list):
        raise SystemExit("Expected a top-level JSON list from benchmark_sweep_results.json")

    rows = _extract_table_rows(data, args.benchmark, args.pytorch_benchmark)
    rows = _sort_rows(rows, args.sort_by, args.descending)

    print("| Shape | Label | Speedup vs PyTorch | TFLOPS |")
    print("|---|---|---:|---:|")
    for row in rows:
        print(
            f"| {row['shape']} | {row['label']} | {_format_speedup(row['speedup'])} | {_format_tflops(row['tflops'])} |"
        )


if __name__ == "__main__":
    main()
