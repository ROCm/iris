#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
# Copyright (c) 2025 Advanced Micro Devices, Inc. All rights reserved.

"""
Comprehensive CCL benchmark with CU sweep across all operations.

This benchmark runs all_gather, all_reduce, all_to_all, and reduce_scatter
with a sweep across different numbers of CUs (comm_sms) and outputs results to CSV.
Runs each benchmark as a separate subprocess to avoid memory accumulation.
"""

import subprocess
import argparse
import csv
import os
from datetime import datetime
import json
import tempfile


def parse_args():
    parser = argparse.ArgumentParser(
        description="Comprehensive CCL benchmark with CU sweep.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # Problem size
    parser.add_argument("-m", type=int, default=16384, help="Number of rows in tensors")
    parser.add_argument("-n", type=int, default=16384, help="Number of columns in tensors")
    parser.add_argument(
        "--datatype",
        type=str,
        default="fp16",
        choices=["fp16", "fp32", "bf16"],
        help="Datatype of tensors",
    )

    # CU sweep parameters
    parser.add_argument(
        "--min_cus",
        type=int,
        default=8,
        help="Minimum number of CUs (comm_sms) to test",
    )
    parser.add_argument(
        "--max_cus",
        type=int,
        default=128,
        help="Maximum number of CUs (comm_sms) to test",
    )
    parser.add_argument(
        "--cu_step",
        type=int,
        default=8,
        help="Step size for CU sweep",
    )

    # Operations to benchmark
    parser.add_argument(
        "--operations",
        type=str,
        nargs="+",
        default=["all_gather", "all_reduce", "all_to_all", "reduce_scatter"],
        choices=["all_gather", "all_reduce", "all_to_all", "reduce_scatter"],
        help="CCL operations to benchmark",
    )

    # All-Gather configuration
    parser.add_argument("--all_gather_block_size_m", type=int, default=32, help="All-Gather: Block size M")
    parser.add_argument("--all_gather_block_size_n", type=int, default=64, help="All-Gather: Block size N")
    parser.add_argument("--all_gather_swizzle_size", type=int, default=4, help="All-Gather: Swizzle size")

    # All-Reduce configuration
    parser.add_argument("--all_reduce_block_size_m", type=int, default=32, help="All-Reduce: Block size M")
    parser.add_argument("--all_reduce_block_size_n", type=int, default=64, help="All-Reduce: Block size N")
    parser.add_argument("--all_reduce_swizzle_size", type=int, default=4, help="All-Reduce: Swizzle size")
    parser.add_argument(
        "--all_reduce_variant",
        type=str,
        default="two_shot",
        choices=["atomic", "spinlock", "ring", "two_shot", "one_shot"],
        help="All-Reduce: Variant to use",
    )
    parser.add_argument(
        "--all_reduce_distribution",
        type=int,
        default=1,
        choices=[0, 1],
        help="All-Reduce: Distribution mode (0=striding, 1=block)",
    )

    # All-to-All configuration
    parser.add_argument("--all_to_all_block_size_m", type=int, default=32, help="All-to-All: Block size M")
    parser.add_argument("--all_to_all_block_size_n", type=int, default=128, help="All-to-All: Block size N")
    parser.add_argument("--all_to_all_swizzle_size", type=int, default=4, help="All-to-All: Swizzle size")

    # Reduce-Scatter configuration
    parser.add_argument("--reduce_scatter_block_size_m", type=int, default=32, help="Reduce-Scatter: Block size M")
    parser.add_argument("--reduce_scatter_block_size_n", type=int, default=64, help="Reduce-Scatter: Block size N")
    parser.add_argument("--reduce_scatter_swizzle_size", type=int, default=4, help="Reduce-Scatter: Swizzle size")
    parser.add_argument(
        "--reduce_scatter_distribution",
        type=int,
        default=1,
        choices=[0, 1],
        help="Reduce-Scatter: Distribution mode (0=striding, 1=block)",
    )

    # General configuration
    parser.add_argument("--num_xcds", type=int, default=None, help="Number of XCDs (auto-detected if not set)")
    parser.add_argument("--heap_size", type=int, default=1 << 34, help="Iris heap size")
    parser.add_argument("-r", "--num_ranks", type=int, default=8, help="Number of ranks/processes")

    # Output
    parser.add_argument(
        "--output_csv",
        type=str,
        default=None,
        help="Output CSV file (default: auto-generated with timestamp)",
    )
    parser.add_argument("--benchmark_rccl", action="store_true", help="Also benchmark RCCL for comparison")
    parser.add_argument("--validate", action="store_false", help="Run validation before benchmarking")
    parser.add_argument("--skip_on_validation_failure", action="store_true", help="Skip benchmark if validation fails")

    return vars(parser.parse_args())


def run_validation(operation, comm_sms, args):
    """Run validation for a single operation."""
    # Get the directory where this script is located
    script_dir = os.path.dirname(os.path.abspath(__file__))
    iris_root = os.path.dirname(os.path.dirname(script_dir))

    script_map = {
        "all_gather": os.path.join(iris_root, "benchmark/ccl/all_gather/benchmark.py"),
        "all_reduce": os.path.join(iris_root, "benchmark/ccl/all_reduce/benchmark.py"),
        "all_to_all": os.path.join(iris_root, "benchmark/ccl/all_to_all/benchmark.py"),
        "reduce_scatter": os.path.join(iris_root, "benchmark/ccl/reduce_scatter/benchmark.py"),
    }

    script_path = script_map[operation]

    with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
        temp_output = f.name

    cmd = [
        "python",
        script_path,
        "-m",
        str(args["m"]),
        "-n",
        str(args["n"]),
        "--datatype",
        args["datatype"],
        "--comm_sms",
        str(comm_sms),
        "-r",
        str(args["num_ranks"]),
        "--heap_size",
        str(args["heap_size"]),
        "--validate",
        "--output_file",
        temp_output,
    ]

    # Add operation-specific parameters (same as benchmark)
    if operation == "all_gather":
        cmd.extend(["--block_size_m", str(args["all_gather_block_size_m"])])
        cmd.extend(["--block_size_n", str(args["all_gather_block_size_n"])])
        cmd.extend(["--swizzle_size", str(args["all_gather_swizzle_size"])])
    elif operation == "all_reduce":
        cmd.extend(["--block_size_m", str(args["all_reduce_block_size_m"])])
        cmd.extend(["--block_size_n", str(args["all_reduce_block_size_n"])])
        cmd.extend(["--swizzle_size", str(args["all_reduce_swizzle_size"])])
        cmd.extend(["--variant", args["all_reduce_variant"]])
        cmd.extend(["--distribution", str(args["all_reduce_distribution"])])
    elif operation == "all_to_all":
        cmd.extend(["--block_size_m", str(args["all_to_all_block_size_m"])])
        cmd.extend(["--block_size_n", str(args["all_to_all_block_size_n"])])
        cmd.extend(["--swizzle_size", str(args["all_to_all_swizzle_size"])])
    elif operation == "reduce_scatter":
        cmd.extend(["--block_size_m", str(args["reduce_scatter_block_size_m"])])
        cmd.extend(["--block_size_n", str(args["reduce_scatter_block_size_n"])])
        cmd.extend(["--swizzle_size", str(args["reduce_scatter_swizzle_size"])])
        cmd.extend(["--all_reduce_distribution", str(args["reduce_scatter_distribution"])])

    if args["num_xcds"] is not None:
        cmd.extend(["--num_xcds", str(args["num_xcds"])])

    print(f"  Validating {operation} with comm_sms={comm_sms}...")
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, check=True)

        with open(temp_output, "r") as f:
            data = json.load(f)

        os.unlink(temp_output)

        success = data.get("success", False)
        return success
    except subprocess.CalledProcessError as e:
        print(f"  Validation failed for {operation}: {e}")
        if os.path.exists(temp_output):
            os.unlink(temp_output)
        return False
    except Exception as e:
        print(f"  Error during validation for {operation}: {e}")
        if os.path.exists(temp_output):
            os.unlink(temp_output)
        return False


def run_benchmark(operation, comm_sms, args):
    """Run a single benchmark as a subprocess and return the results."""
    # Get the directory where this script is located
    script_dir = os.path.dirname(os.path.abspath(__file__))
    # Go up two levels to get to the iris root directory
    iris_root = os.path.dirname(os.path.dirname(script_dir))

    # Map operation to benchmark script (relative to iris root)
    script_map = {
        "all_gather": os.path.join(iris_root, "benchmark/ccl/all_gather/benchmark.py"),
        "all_reduce": os.path.join(iris_root, "benchmark/ccl/all_reduce/benchmark.py"),
        "all_to_all": os.path.join(iris_root, "benchmark/ccl/all_to_all/benchmark.py"),
        "reduce_scatter": os.path.join(iris_root, "benchmark/ccl/reduce_scatter/benchmark.py"),
    }

    script_path = script_map[operation]

    # Create temporary output file
    with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
        temp_output = f.name

    # Build command
    cmd = [
        "python",
        script_path,
        "-m",
        str(args["m"]),
        "-n",
        str(args["n"]),
        "--datatype",
        args["datatype"],
        "--comm_sms",
        str(comm_sms),
        "-r",
        str(args["num_ranks"]),
        "--heap_size",
        str(args["heap_size"]),
        "--benchmark",
        "--output_file",
        temp_output,
    ]

    # Add operation-specific parameters
    if operation == "all_gather":
        cmd.extend(["--block_size_m", str(args["all_gather_block_size_m"])])
        cmd.extend(["--block_size_n", str(args["all_gather_block_size_n"])])
        cmd.extend(["--swizzle_size", str(args["all_gather_swizzle_size"])])
    elif operation == "all_reduce":
        cmd.extend(["--block_size_m", str(args["all_reduce_block_size_m"])])
        cmd.extend(["--block_size_n", str(args["all_reduce_block_size_n"])])
        cmd.extend(["--swizzle_size", str(args["all_reduce_swizzle_size"])])
        cmd.extend(["--variant", args["all_reduce_variant"]])
        cmd.extend(["--distribution", str(args["all_reduce_distribution"])])
    elif operation == "all_to_all":
        cmd.extend(["--block_size_m", str(args["all_to_all_block_size_m"])])
        cmd.extend(["--block_size_n", str(args["all_to_all_block_size_n"])])
        cmd.extend(["--swizzle_size", str(args["all_to_all_swizzle_size"])])
    elif operation == "reduce_scatter":
        cmd.extend(["--block_size_m", str(args["reduce_scatter_block_size_m"])])
        cmd.extend(["--block_size_n", str(args["reduce_scatter_block_size_n"])])
        cmd.extend(["--swizzle_size", str(args["reduce_scatter_swizzle_size"])])
        cmd.extend(["--all_reduce_distribution", str(args["reduce_scatter_distribution"])])

    if args["num_xcds"] is not None:
        cmd.extend(["--num_xcds", str(args["num_xcds"])])

    # Add --benchmark_rccl flag if requested
    if args.get("benchmark_rccl", False):
        cmd.append("--benchmark_rccl")

    # Set NCCL environment variables to control number of channels (CUs)
    env = os.environ.copy()
    if args.get("benchmark_rccl", False):
        env["NCCL_MIN_NCHANNELS"] = str(comm_sms)
        env["NCCL_MAX_NCHANNELS"] = str(comm_sms)

    # Run benchmark
    print(f"\nRunning {operation} with comm_sms={comm_sms}...")
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, check=True, env=env)

        # Read results from JSON file
        with open(temp_output, "r") as f:
            data = json.load(f)

        # Clean up temp file
        os.unlink(temp_output)

        return data
    except subprocess.CalledProcessError as e:
        print(f"Error running {operation}: {e}")
        print(f"stdout: {e.stdout}")
        print(f"stderr: {e.stderr}")
        if os.path.exists(temp_output):
            os.unlink(temp_output)
        return None
    except Exception as e:
        print(f"Error processing results for {operation}: {e}")
        if os.path.exists(temp_output):
            os.unlink(temp_output)
        return None


def main():
    args = parse_args()

    # Generate CU sweep range
    cu_values = list(range(args["min_cus"], args["max_cus"] + 1, args["cu_step"]))

    results = []

    print(f"{'=' * 80}")
    print("Comprehensive CCL Benchmark Sweep")
    print(f"Operations: {', '.join(args['operations'])}")
    print(f"CU range: {args['min_cus']} to {args['max_cus']} (step {args['cu_step']})")
    print(f"Problem size: {args['m']}x{args['n']}")
    print(f"Datatype: {args['datatype']}")
    print(f"Ranks: {args['num_ranks']}")
    print(f"{'=' * 80}")

    for comm_sms in cu_values:
        print(f"\n{'=' * 80}")
        print(f"Testing with comm_sms={comm_sms}")
        print(f"{'=' * 80}")

        for operation in args["operations"]:
            # Run validation if requested
            validation_passed = True
            if args.get("validate", False):
                validation_passed = run_validation(operation, comm_sms, args)
                if validation_passed:
                    print(f"  ✓ Validation passed for {operation}")
                else:
                    print(f"  ✗ Validation FAILED for {operation}")
                    if args.get("skip_on_validation_failure", False):
                        print(f"  Skipping benchmark for {operation} due to validation failure")
                        continue

            # Run benchmark
            data = run_benchmark(operation, comm_sms, args)

            if data is not None:
                # Add validation status to result
                if args.get("validate", False):
                    validation_status = "passed" if validation_passed else "failed"
                else:
                    validation_status = "not_run"
                # Extract relevant fields and add to results
                result = {
                    "operation": operation,
                    "comm_sms": comm_sms,
                    "m": args["m"],
                    "n": args["n"],
                    "world_size": args["num_ranks"],
                    "datatype": args["datatype"],
                    "block_size_m": data.get("block_size_m"),
                    "block_size_n": data.get("block_size_n"),
                    "swizzle_size": data.get("swizzle_size"),
                    "num_xcds": data.get("num_xcds"),
                    "iris_latency_ms": data.get(f"{operation}_ms"),
                    "iris_bandwidth_gbps": data.get("bandwidth_gbps"),
                }

                # Add operation-specific fields
                if operation == "all_reduce":
                    result["variant"] = args["all_reduce_variant"]
                    result["distribution"] = args["all_reduce_distribution"]
                elif operation == "reduce_scatter":
                    result["distribution"] = args["reduce_scatter_distribution"]

                # Add RCCL results if available
                if args.get("benchmark_rccl", False):
                    result["rccl_latency_ms"] = data.get("rccl_ms")
                    result["rccl_bandwidth_gbps"] = data.get("rccl_bandwidth_gbps")
                    result["iris_vs_rccl_ratio"] = data.get("rccl_ratio_percent", 0) / 100.0

                results.append(result)

                print(f"  Iris: {result['iris_latency_ms']:.3f} ms, {result['iris_bandwidth_gbps']:.3f} GB/s")
                if args.get("benchmark_rccl", False) and result.get("rccl_bandwidth_gbps"):
                    print(f"  RCCL: {result['rccl_latency_ms']:.3f} ms, {result['rccl_bandwidth_gbps']:.3f} GB/s")
                    print(f"  Ratio: {result['iris_vs_rccl_ratio']:.2f}x")

    # Generate output filename if not provided
    if args["output_csv"] is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        args["output_csv"] = f"ccl_sweep_{timestamp}.csv"

    # Write results to CSV
    if results:
        # Collect all unique fieldnames from all results
        all_fieldnames = set()
        for result in results:
            all_fieldnames.update(result.keys())

        # Sort fieldnames for consistent column order
        # Put common fields first, then operation-specific fields
        common_fields = [
            "operation",
            "comm_sms",
            "m",
            "n",
            "world_size",
            "datatype",
            "block_size_m",
            "block_size_n",
            "swizzle_size",
            "num_xcds",
            "iris_latency_ms",
            "iris_bandwidth_gbps",
        ]
        optional_fields = sorted(all_fieldnames - set(common_fields))
        fieldnames = [f for f in common_fields if f in all_fieldnames] + optional_fields

        with open(args["output_csv"], "w", newline="") as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(results)

        print(f"\n{'=' * 80}")
        print(f"Results written to: {args['output_csv']}")
        print(f"Total benchmarks run: {len(results)}")
        print(f"{'=' * 80}\n")
    else:
        print("\nNo results collected!")


if __name__ == "__main__":
    main()
