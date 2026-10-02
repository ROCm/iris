#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
# Copyright (c) 2025-2026 Advanced Micro Devices, Inc. All rights reserved.

"""
Print a per-shape summary from tune_copy_engine aggregated results.

The input is the top-level ``results.json`` produced by
``benchmark/ops/matmul_all_gather/tune_copy_engine.py``.
"""

import argparse
import json
from collections import defaultdict
from pathlib import Path


def parse_args():
    parser = argparse.ArgumentParser(
        description="Print per-shape summaries from tune_copy_engine results.json",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "results_json",
        type=Path,
        help="Path to the aggregated results.json emitted by tune_copy_engine.py",
    )
    return parser.parse_args()


def _load_results(path: Path):
    with open(path, "r") as f:
        return json.load(f)


def _as_float(value):
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _format_float(value, precision):
    numeric = _as_float(value)
    if numeric is None:
        return "--"
    return f"{numeric:.{precision}f}"


def _sort_key(result):
    tflops = _as_float(result.get("iris_tflops"))
    return (tflops is not None, tflops if tflops is not None else float("-inf"))


def _record_field(result, key):
    benchmark_json = result.get("benchmark_json") or {}
    if key in benchmark_json:
        return benchmark_json.get(key)
    return result.get(key)


def _print_shape_summary(shape_tag, records):
    records = sorted(records, key=_sort_key, reverse=True)
    shape = records[0].get("shape", {})
    print("=" * 170)
    print(f"{shape_tag}  |  M_local={shape.get('m_local')} N={shape.get('n')} K={shape.get('k')}")
    print("=" * 170)
    print(
        f"{'#':>3}  {'m_tiles_per_batch':>17}  {'ms':>9}  {'TFLOPS':>9}  {'Valid':>8}  "
        f"{'batches':>7}  {'group_m':>7}  {'tiles_m':>7}  {'tiles_n':>7}  {'tiles/wave':>10}  "
        f"{'tiles/1st':>10}  {'iters':>7}  "
        f"{'block_m':>7}  {'block_n':>7}  {'block_k':>7}  "
        f"{'tiles/group':>11}  {'gemm_us':>9}  {'scatter_us':>11}  {'ratio':>8}"
    )
    print("-" * 170)

    for idx, record in enumerate(records, start=1):
        print(
            f"{idx:>3}  "
            f"{str(_record_field(record, 'm_tiles_per_batch')):>17}  "
            f"{_format_float(record.get('iris_ms'), 3):>9}  "
            f"{_format_float(record.get('iris_tflops'), 2):>9}  "
            f"{str(record.get('validation') or '--'):>8}  "
            f"{str(_record_field(record, 'num_batches') or '--'):>7}  "
            f"{str(_record_field(record, 'group_size_m') or '--'):>7}  "
            f"{str(_record_field(record, 'num_tiles_m') or '--'):>7}  "
            f"{str(_record_field(record, 'num_tiles_n') or '--'):>7}  "
            f"{str(_record_field(record, 'm_tiles_per_wave') or '--'):>10}  "
            f"{str(_record_field(record, 'm_tiles_first_wave') or '--'):>10}  "
            f"{str(_record_field(record, 'schedule_iterations') or '--'):>7}  "
            f"{str(_record_field(record, 'block_size_m') or _record_field(record, 'output_tile_size_m') or '--'):>7}  "
            f"{str(_record_field(record, 'block_size_n') or _record_field(record, 'output_tile_size_n') or '--'):>7}  "
            f"{str(_record_field(record, 'block_size_k') or _record_field(record, 'output_tile_size_k') or '--'):>7}  "
            f"{str(_record_field(record, 'tiles_per_group') or '--'):>11}  "
            f"{_format_float(_record_field(record, 'gemm_wg_us'), 2):>9}  "
            f"{_format_float(_record_field(record, 'scatter_wg_us'), 2):>11}  "
            f"{_format_float(_record_field(record, 'ratio'), 3):>8}"
        )
    print("")


def main():
    args = parse_args()
    data = _load_results(args.results_json)
    results = data.get("results", [])

    if not results:
        raise SystemExit("No results found in input JSON")

    by_shape = defaultdict(list)
    for result in results:
        by_shape[result.get("shape_tag", "UNKNOWN_SHAPE")].append(result)

    print(f"Input: {args.results_json}")
    print(f"Shapes: {len(by_shape)}")
    print(f"Runs: {len(results)}")
    print("")

    for shape_tag in sorted(by_shape):
        _print_shape_summary(shape_tag, by_shape[shape_tag])


if __name__ == "__main__":
    main()
