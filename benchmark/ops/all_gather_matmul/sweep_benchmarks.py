#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
# Copyright (c) 2025-2026 Advanced Micro Devices, Inc. All rights reserved.

"""
Sweep benchmark script for all_gather_matmul variants.

Runs benchmarks across all permutations of M, N, K dimensions for:
- PyTorch baseline (all_gather_into_tensor + matmul)
- HBM buffer variant
- Copy engine (host-initiated)
- Copy engine (device-initiated)

Outputs results as CSV to stdout and benchmark_sweep_results.csv.
"""

import subprocess
import sys
import csv
import re
import itertools
from pathlib import Path
from typing import Optional


# Project root (3 levels up from this script)
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent.parent

# Dimension values to sweep
DIMS = [131072, 2048, 16384]

# Benchmark configurations
BENCHMARKS = {
    "pytorch": {
        "script": "benchmark/ops/all_gather_matmul/benchmark_torchrun.py",
        "extra_args": ["--benchmark_pytorch"],
        "pattern": r"PyTorch all_gather_into_tensor\+matmul.*?(\d+\.\d+)\s+ms",
    },
    "hbm_buffer": {
        "script": "benchmark/ops/all_gather_matmul/benchmark_hbm_buffer.py",
        "extra_args": [],
        "pattern": r"HBM.*?\(M=\d+.*?\):\s+(\d+\.\d+)\s+ms",
    },
    "copy_engine_host": {
        "script": "benchmark/ops/all_gather_matmul/benchmark_copy_engine.py",
        "extra_args": ["--force-host-initiated"],
        "pattern": r"Copy Engine\s+\(M=\d+.*?\):\s+(\d+\.\d+)\s+ms",
    },
    "copy_engine_device": {
        "script": "benchmark/ops/all_gather_matmul/benchmark_copy_engine.py",
        "extra_args": ["--force-device-initiated"],
        "pattern": r"Copy Engine\s+\(M=\d+.*?\):\s+(\d+\.\d+)\s+ms",
    },
}

TIMEOUT_SECONDS = 60
NUM_GPUS = 8


def log(msg: str):
    """Log to stderr to keep stdout clean for CSV."""
    print(msg, file=sys.stderr, flush=True)


def run_benchmark(
    benchmark_name: str,
    script: str,
    m: int,
    n: int,
    k: int,
    extra_args: list,
    pattern: str,
) -> Optional[float]:
    """
    Run a single benchmark and extract the median time.

    Returns:
        Median time in milliseconds, or None if failed/timeout.
    """
    cmd = [
        "torchrun",
        f"--nproc_per_node={NUM_GPUS}",
        script,
        "-m", str(m),
        "-n", str(n),
        "-k", str(k),
        "--benchmark",
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

        # Combine stdout and stderr for searching
        output = result.stdout + result.stderr

        # Search for the timing pattern
        match = re.search(pattern, output, re.IGNORECASE)
        if match:
            time_ms = float(match.group(1))
            log(f"    ✓ Success: {time_ms:.3f} ms")
            return time_ms
        else:
            log("    ✗ Failed: Could not parse timing from output")
            log(f"    Pattern: {pattern}")
            if result.returncode != 0:
                log(f"    Return code: {result.returncode}")
                # Show last few lines of output for debugging
                lines = output.strip().split('\n')
                log("    Last output lines:")
                for line in lines[-5:]:
                    log(f"      {line}")
            return None

    except subprocess.TimeoutExpired:
        log(f"    ✗ Timeout after {TIMEOUT_SECONDS}s")
        return None
    except Exception as e:
        log(f"    ✗ Error: {e}")
        return None


def format_result(value: Optional[float]) -> str:
    """Format a result value for CSV output."""
    if value is None:
        return "FAILED"
    return f"{value:.3f}"


def main():
    log("=" * 80)
    log("All-Gather-Matmul Benchmark Sweep")
    log("=" * 80)
    log(f"Dimensions to sweep: {DIMS}")
    log(f"Total combinations: {len(DIMS) ** 3}")
    log(f"Benchmarks per combination: {len(BENCHMARKS)}")
    log(f"Total benchmarks: {len(DIMS) ** 3 * len(BENCHMARKS)}")
    log(f"Timeout per benchmark: {TIMEOUT_SECONDS}s")
    log(f"GPUs: {NUM_GPUS}")
    log("=" * 80)
    log("")

    # Prepare CSV header
    fieldnames = [
        "M", "N", "K",
        "pytorch_ms",
        "hbm_buffer_ms",
        "copy_engine_host_ms",
        "copy_engine_device_ms",
    ]

    # Open output file
    output_file = PROJECT_ROOT / "benchmark/ops/all_gather_matmul/benchmark_sweep_results.csv"
    results = []

    # Generate all permutations
    permutations = list(itertools.permutations(DIMS, 3))

    log(f"Running {len(permutations)} dimension permutations...\n")

    for idx, (m, n, k) in enumerate(permutations, 1):
        log(f"[{idx}/{len(permutations)}] Testing M={m}, N={n}, K={k}")

        row = {"M": m, "N": n, "K": k}

        # Run each benchmark variant
        for bench_key, bench_config in BENCHMARKS.items():
            result = run_benchmark(
                benchmark_name=bench_key,
                script=bench_config["script"],
                m=m,
                n=n,
                k=k,
                extra_args=bench_config["extra_args"],
                pattern=bench_config["pattern"],
            )

            # Map benchmark key to CSV column name
            col_name = f"{bench_key}_ms"
            row[col_name] = result

        results.append(row)
        log("")

    # Write CSV file
    log(f"Writing results to {output_file}...")
    with open(output_file, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in results:
            writer.writerow(row)

    log(f"✓ Results saved to {output_file}\n")

    # Print to stdout (clean CSV)
    writer = csv.DictWriter(sys.stdout, fieldnames=fieldnames)
    writer.writeheader()
    for row in results:
        # Format numeric values for display
        display_row = {
            "M": row["M"],
            "N": row["N"],
            "K": row["K"],
            "pytorch_ms": format_result(row["pytorch_ms"]),
            "hbm_buffer_ms": format_result(row["hbm_buffer_ms"]),
            "copy_engine_host_ms": format_result(row["copy_engine_host_ms"]),
            "copy_engine_device_ms": format_result(row["copy_engine_device_ms"]),
        }
        writer.writerow(display_row)

    log("\n" + "=" * 80)
    log("Benchmark sweep complete!")
    log("=" * 80)


if __name__ == "__main__":
    main()
