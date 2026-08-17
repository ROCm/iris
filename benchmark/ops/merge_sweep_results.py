#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.

"""Merge benchmark sweep runs, retaining the best-performing benchmark records.

Example:
    python benchmark/ops/merge_sweep_results.py \
        benchmark/ops/model_sweep_results_matmul_all_reduce-run1.json \
        benchmark/ops/model_sweep_results_matmul_all_reduce-run2.json \
        benchmark/ops/model_sweep_results_matmul_all_reduce-run3.json \
        -o benchmark/ops/model_sweep_results_matmul_all_reduce-merged.json
"""

import argparse
import copy
import json
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Merge sweep result files, selecting each benchmark record with the highest tflops",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("inputs", nargs="+", type=Path, help="Input sweep result JSON files")
    parser.add_argument("-o", "--output", required=True, type=Path, help="Output JSON file")
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


def _optional_number(benchmark: dict, field: str, context: str):
    value = benchmark.get(field)
    if value is not None and (not isinstance(value, (int, float)) or isinstance(value, bool)):
        raise ValueError(f"{context}.{field} must be numeric, got {value!r}")
    return value


def _candidate_is_better(current: dict, candidate: dict, context: str) -> bool:
    """Prefer higher throughput, falling back to lower latency when throughput is absent."""
    current_tflops = _optional_number(current, "tflops", context)
    candidate_tflops = _optional_number(candidate, "tflops", context)
    if current_tflops is not None or candidate_tflops is not None:
        if current_tflops is None:
            return True
        if candidate_tflops is None:
            return False
        return candidate_tflops > current_tflops

    current_total_ms = _optional_number(current, "total_ms", context)
    candidate_total_ms = _optional_number(candidate, "total_ms", context)
    if current_total_ms is None:
        return candidate_total_ms is not None
    if candidate_total_ms is None:
        return False
    return candidate_total_ms < current_total_ms


def merge_results(result_sets: list[list[dict]]) -> list[dict]:
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
                    target_benchmarks[benchmark_name] = copy.deepcopy(benchmark)
                    continue
                if not isinstance(target_benchmarks[benchmark_name], dict):
                    raise ValueError(f"Benchmark {benchmark_name!r} must be a JSON object")

                target_benchmark = target_benchmarks[benchmark_name]
                context = f"{key!r}.benchmarks.{benchmark_name}"
                if _candidate_is_better(target_benchmark, benchmark, context):
                    target_benchmarks[benchmark_name] = copy.deepcopy(benchmark)

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
        merged = merge_results([load_results(path) for path in args.inputs])
    except (OSError, json.JSONDecodeError, ValueError) as error:
        raise SystemExit(f"error: {error}") from error

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w") as file:
        json.dump(merged, file, indent=2)
        file.write("\n")

    print(f"Merged {len(args.inputs)} files into {args.output} ({len(merged)} results)")


if __name__ == "__main__":
    main()
