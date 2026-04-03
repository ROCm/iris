#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
# Copyright (c) 2025-2026 Advanced Micro Devices, Inc. All rights reserved.

"""
Sweep benchmark script for matmul_all_gather variants.

Runs benchmarks across all permutations of M, N, K dimensions for:
- PyTorch baseline (matmul + all_gather_into_tensor)
- Baseline variant
- Copy engine variant
- Host copy engine variant

Outputs results as JSON to stdout and benchmark_sweep_results.json.
"""

import subprocess
import sys
import json
from pathlib import Path
from typing import Optional, Dict, Any


# Project root (3 levels up from this script)
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent.parent

# Dimension configurations to test (M, N, K)
# Each tuple is (M_local, N, K) where M_local is per-rank M dimension
DIMENSION_CONFIGS = [
    (2048, 2048, 16384),
    (2048, 16384, 2048),
    (2048, 16384, 16384),
    (2048, 16384, 65536),
    (2048, 131072, 16384),
    (16384, 2048, 2048),
    (16384, 2048, 16384),
    (16384, 2048, 131072),
    (16384, 16384, 2048),
    (131072, 2048, 16384),
]

# Benchmark configurations
BENCHMARKS = {
    "baseline_and_pytorch": {
        "script": "benchmark/ops/matmul_all_gather/benchmark.py",
        # "extra_args": ["--benchmark_pytorch"],
        "extra_args": [],
        "output_file": "matmul_all_gather_baseline.json",
        "extract_multiple": True,  # Extract both baseline and pytorch from one run
    },
    "host_copy_engine": {
        "script": "benchmark/ops/matmul_all_gather/benchmark_host_copy_engine.py",
        "extra_args": [],
        "output_file": "matmul_all_gather_host_copy_engine.json",
        "extract_multiple": False,
    },
    "matmul_only": {
        "script": "benchmark/ops/matmul_all_gather/benchmark_matmul.py",
        "extra_args": [],
        "output_file": "matmul_only.json",
        "extract_multiple": False,
    },
}

TIMEOUT_SECONDS = 60
NUM_GPUS = 8


def log(msg: str):
    """Log to stderr to keep stdout clean for JSON."""
    print(msg, file=sys.stderr, flush=True)


def run_benchmark(
    benchmark_name: str,
    script: str,
    m: int,
    n: int,
    k: int,
    extra_args: list,
    output_file: str,
) -> Optional[Dict[str, Any]]:
    """
    Run a single benchmark and extract results from JSON output file.

    Returns:
        Dictionary containing benchmark results, or None if failed/timeout.
    """
    cmd = [
        "torchrun",
        f"--nproc_per_node={NUM_GPUS}",
        script,
        "-m",
        str(m),
        "-n",
        str(n),
        "-k",
        str(k),
        "--validate",
        "--benchmark",
        "--output_file",
        output_file,
    ] + extra_args

    log(f"  Running {benchmark_name}: M={m}, N={n}, K={k}")
    log(f"    Command: {' '.join(cmd)}")

    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=TIMEOUT_SECONDS,
            cwd=str(PROJECT_ROOT),
        )

        if result.returncode != 0:
            log("    ✗ Failed: Non-zero return code")
            log(f"    Return code: {result.returncode}")
            # Show last few lines of output for debugging
            output = result.stdout + result.stderr
            lines = output.strip().split("\n")
            log("    Last output lines:")
            for line in lines[-5:]:
                log(f"      {line}")
            return None

        # Read the JSON output file
        json_path = PROJECT_ROOT / output_file
        if not json_path.exists():
            log(f"    ✗ Failed: JSON output file not found: {json_path}")
            return None

        with open(json_path, "r") as f:
            data = json.load(f)

        # Check validation status
        validation_status = ""
        if "success" in data:
            if data["success"]:
                validation_status = " (validation: PASSED)"
            else:
                validation_status = " (validation: FAILED)"

        log(f"    ✓ Success: Loaded JSON results{validation_status}")
        return data

    except subprocess.TimeoutExpired:
        log(f"    ✗ Timeout after {TIMEOUT_SECONDS}s")
        return None
    except json.JSONDecodeError as e:
        log(f"    ✗ Error: Failed to parse JSON: {e}")
        return None
    except Exception as e:
        log(f"    ✗ Error: {e}")
        return None


def main():
    log("=" * 80)
    log("Matmul-All-Gather Benchmark Sweep")
    log("=" * 80)
    log(f"Dimension configurations: {len(DIMENSION_CONFIGS)}")
    for m, n, k in DIMENSION_CONFIGS:
        log(f"  - M={m}, N={n}, K={k}")
    log(f"Benchmarks per configuration: {len(BENCHMARKS)}")
    log(f"Total benchmarks: {len(DIMENSION_CONFIGS) * len(BENCHMARKS)}")
    log(f"Timeout per benchmark: {TIMEOUT_SECONDS}s")
    log(f"GPUs: {NUM_GPUS}")
    log("=" * 80)
    log("")

    # Open output file
    output_file = PROJECT_ROOT / "benchmark/ops/matmul_all_gather/benchmark_sweep_results.json"
    results = []

    log(f"Running {len(DIMENSION_CONFIGS)} dimension configurations...\n")

    for idx, (m, n, k) in enumerate(DIMENSION_CONFIGS, 1):
        log(f"[{idx}/{len(DIMENSION_CONFIGS)}] Testing M={m}, N={n}, K={k}")

        row = {"M": m, "N": n, "K": k, "benchmarks": {}}

        # Run each benchmark variant
        for bench_key, bench_config in BENCHMARKS.items():
            result = run_benchmark(
                benchmark_name=bench_key,
                script=bench_config["script"],
                m=m,
                n=n,
                k=k,
                extra_args=bench_config["extra_args"],
                output_file=bench_config["output_file"],
            )

            if result is not None:
                # Check if this benchmark produces multiple results
                if bench_config.get("extract_multiple", False):
                    # Extract baseline results (tflops, etc.)
                    baseline_result = {k: v for k, v in result.items() if not k.startswith("pytorch_")}
                    row["benchmarks"]["baseline"] = baseline_result

                    # Extract pytorch results (pytorch_tflops, etc.)
                    if "pytorch_tflops" in result:
                        pytorch_result = {
                            "tflops": result.get("pytorch_tflops"),
                            "bandwidth_gbps": result.get("pytorch_bandwidth_gbps"),
                            "total_ms": result.get("pytorch_ms"),
                        }
                        # Copy common fields
                        for field in ["world_size", "operation", "m", "n", "k", "datatype"]:
                            if field in result:
                                pytorch_result[field] = result[field]
                        row["benchmarks"]["pytorch"] = pytorch_result
                    else:
                        row["benchmarks"]["pytorch"] = {"status": "FAILED"}
                else:
                    # Single result benchmark
                    row["benchmarks"][bench_key] = result
            else:
                # Failed benchmark
                if bench_config.get("extract_multiple", False):
                    row["benchmarks"]["baseline"] = {"status": "FAILED"}
                    row["benchmarks"]["pytorch"] = {"status": "FAILED"}
                else:
                    row["benchmarks"][bench_key] = {"status": "FAILED"}

        results.append(row)
        log("")

    # Write JSON file
    log(f"Writing results to {output_file}...")
    with open(output_file, "w") as f:
        json.dump(results, f, indent=2)

    log(f"✓ Results saved to {output_file}\n")

    # Print to stdout (clean JSON)
    print(json.dumps(results, indent=2))

    log("\n" + "=" * 80)
    log("Benchmark sweep complete!")
    log("=" * 80)


if __name__ == "__main__":
    main()
