#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.

"""Merge benchmark sweep runs, retaining the best-performing benchmark records.

Uses median by default (robust to outliers), with options for mean or trimmed mean.

Example:
    # Default: use median (recommended)
    python benchmark/ops/merge_sweep_results.py \
        benchmark/ops/model_sweep_results_matmul_all_reduce-run1.json \
        benchmark/ops/model_sweep_results_matmul_all_reduce-run2.json \
        benchmark/ops/model_sweep_results_matmul_all_reduce-run3.json \
        -o benchmark/ops/model_sweep_results_matmul_all_reduce-merged.json

    # Use mean (original behavior, sensitive to outliers)
    python benchmark/ops/merge_sweep_results.py ... --aggregate mean

    # Use trimmed mean (drop slowest 10 measurements)
    python benchmark/ops/merge_sweep_results.py ... --aggregate trimmed --trim 10
"""

import argparse
import copy
import json
import statistics
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Merge sweep result files, selecting each benchmark record with the best performance",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("inputs", nargs="+", type=Path, help="Input sweep result JSON files")
    parser.add_argument("-o", "--output", required=True, type=Path, help="Output JSON file")
    parser.add_argument(
        "--aggregate",
        choices=["median", "mean", "trimmed"],
        default="median",
        help="Aggregation method: median (robust, default), mean (original), or trimmed (drop outliers)",
    )
    parser.add_argument(
        "--trim",
        type=int,
        default=10,
        help="Number of slowest measurements to trim when using --aggregate=trimmed",
    )
    return parser.parse_args()


def _record_key(record: dict) -> tuple:
    """Return the fields that identify the same model shape across runs."""
    return (
        record.get("operation"),
        record.get("label"),
        record.get("M"),
        record.get("N"),
        record.get("K"),
    )


def _recompute_metrics(benchmark: dict, aggregate: str, trim_top_n: int) -> dict:
    """Recompute gpu_time_ms, tflops, and bandwidth using specified aggregation method.

    Returns a modified copy of the benchmark with updated metrics.
    """
    all_times_ms = benchmark.get("all_times_ms")
    if not all_times_ms:
        return benchmark

    # Compute new time using specified aggregation
    if aggregate == "median":
        new_mean_ms = statistics.median(all_times_ms)
    elif aggregate == "trimmed":
        if len(all_times_ms) <= trim_top_n:
            new_mean_ms = statistics.mean(all_times_ms)
        else:
            sorted_times = sorted(all_times_ms)
            trimmed_times = sorted_times[:-trim_top_n]
            new_mean_ms = statistics.mean(trimmed_times)
    else:  # mean
        new_mean_ms = statistics.mean(all_times_ms)

    # Create modified benchmark with updated metrics
    result = copy.deepcopy(benchmark)
    old_mean_ms = benchmark["gpu_time_ms"]
    result["gpu_time_ms"] = new_mean_ms

    # Recompute derived metrics (scale by time ratio)
    time_ratio = old_mean_ms / new_mean_ms

    if "tflops" in result and result["tflops"] is not None:
        result["tflops"] = benchmark["tflops"] * time_ratio

    if "bandwidth_gbps" in result and result["bandwidth_gbps"] is not None:
        result["bandwidth_gbps"] = benchmark["bandwidth_gbps"] * time_ratio

    counters = result.get("counters", {})
    if "bandwidth_xgmi_gbps" in counters:
        result["counters"]["bandwidth_xgmi_gbps"] = counters["bandwidth_xgmi_gbps"] * time_ratio

    return result


def _candidate_is_better(current: dict, candidate: dict, context: str, aggregate: str, trim_top_n: int) -> bool:
    """Prefer higher throughput, falling back to lower latency when throughput is absent."""
    # Recompute metrics with specified aggregation
    current_recomputed = _recompute_metrics(current, aggregate, trim_top_n)
    candidate_recomputed = _recompute_metrics(candidate, aggregate, trim_top_n)

    current_tflops = current_recomputed.get("tflops")
    candidate_tflops = candidate_recomputed.get("tflops")

    if current_tflops is not None or candidate_tflops is not None:
        if current_tflops is None:
            return True
        if candidate_tflops is None:
            return False
        return candidate_tflops > current_tflops

    current_total_ms = current_recomputed.get("gpu_time_ms")
    candidate_total_ms = candidate_recomputed.get("gpu_time_ms")

    if current_total_ms is None:
        return candidate_total_ms is not None
    if candidate_total_ms is None:
        return False
    return candidate_total_ms < current_total_ms


def merge_results(result_sets: list[list[dict]], aggregate: str, trim_top_n: int) -> list[dict]:
    """Merge parsed result sets without mutating any of the inputs."""
    merged = []
    records_by_key = {}

    for results in result_sets:
        for record in results:
            if not isinstance(record, dict):
                raise ValueError("Every sweep result must be a JSON object")

            key = _record_key(record)
            if key not in records_by_key:
                copied_record = copy.deepcopy(record)
                # Recompute benchmarks with specified aggregation
                if "benchmarks" in copied_record:
                    for bench_name, bench_data in copied_record["benchmarks"].items():
                        copied_record["benchmarks"][bench_name] = _recompute_metrics(
                            bench_data, aggregate, trim_top_n
                        )
                merged.append(copied_record)
                records_by_key[key] = copied_record
                continue

            target = records_by_key[key]
            target_benchmarks = target.setdefault("benchmarks", {})
            source_benchmarks = record.get("benchmarks") or {}
            if not isinstance(target_benchmarks, dict) or not isinstance(source_benchmarks, dict):
                raise ValueError(f"benchmarks must be an object for result {key!r}")

            for benchmark_name, benchmark in source_benchmarks.items():
                if not isinstance(benchmark, dict):
                    raise ValueError(f"Benchmark {benchmark_name!r} must be a JSON object")
                if benchmark_name not in target_benchmarks:
                    target_benchmarks[benchmark_name] = _recompute_metrics(benchmark, aggregate, trim_top_n)
                    continue
                if not isinstance(target_benchmarks[benchmark_name], dict):
                    raise ValueError(f"Benchmark {benchmark_name!r} must be a JSON object")

                target_benchmark = target_benchmarks[benchmark_name]
                context = f"{key!r}.benchmarks.{benchmark_name}"
                if _candidate_is_better(target_benchmark, benchmark, context, aggregate, trim_top_n):
                    target_benchmarks[benchmark_name] = _recompute_metrics(benchmark, aggregate, trim_top_n)

    return merged


def load_results(path: Path) -> list[dict]:
    with path.open() as file:
        results = json.load(file)
    if not isinstance(results, list):
        raise ValueError(f"Expected a top-level JSON list in {path}")
    return results


def main() -> None:
    args = parse_args()
    try:
        merged = merge_results(
            [load_results(path) for path in args.inputs],
            aggregate=args.aggregate,
            trim_top_n=args.trim,
        )
    except (OSError, json.JSONDecodeError, ValueError) as error:
        raise SystemExit(f"error: {error}") from error

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w") as file:
        json.dump(merged, file, indent=2)
        file.write("\n")

    method_desc = args.aggregate
    if args.aggregate == "trimmed":
        method_desc = f"trimmed mean (removed slowest {args.trim})"
    print(f"Merged {len(args.inputs)} files into {args.output} ({len(merged)} results) using {method_desc}")


if __name__ == "__main__":
    main()
