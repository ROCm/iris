#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
# Copyright (c) 2025 Advanced Micro Devices, Inc. All rights reserved.

"""
Plot comprehensive CCL benchmark sweep results.

This script reads the CSV output from comprehensive_sweep.py and creates
subplots comparing Iris vs RCCL bandwidth for each collective operation.
"""

import argparse
import csv
import matplotlib.pyplot as plt
import numpy as np
from collections import defaultdict
import os


def parse_args():
    parser = argparse.ArgumentParser(
        description="Plot CCL benchmark sweep results.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    parser.add_argument(
        "input_csv",
        type=str,
        help="Input CSV file from comprehensive_sweep.py",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Output plot file (default: auto-generated from input filename)",
    )
    parser.add_argument(
        "--title",
        type=str,
        default="CCL Benchmark: Iris vs RCCL",
        help="Overall plot title",
    )
    parser.add_argument(
        "--dpi",
        type=int,
        default=150,
        help="DPI for output image",
    )
    parser.add_argument(
        "--figsize",
        type=int,
        nargs=2,
        default=[16, 10],
        help="Figure size in inches (width height)",
    )

    return parser.parse_args()


def load_results(csv_file):
    """Load results from CSV file and organize by operation."""
    data = defaultdict(lambda: {"comm_sms": [], "iris_bw": [], "rccl_bw": []})

    with open(csv_file, "r") as f:
        reader = csv.DictReader(f)
        for row in reader:
            operation = row["operation"]
            comm_sms = int(row["comm_sms"])
            iris_bw = float(row["iris_bandwidth_gbps"])

            data[operation]["comm_sms"].append(comm_sms)
            data[operation]["iris_bw"].append(iris_bw)

            # RCCL data may not be present for all operations
            if "rccl_bandwidth_gbps" in row and row["rccl_bandwidth_gbps"]:
                rccl_bw = float(row["rccl_bandwidth_gbps"])
                data[operation]["rccl_bw"].append(rccl_bw)
            else:
                data[operation]["rccl_bw"].append(None)

    return data


def plot_results(data, args):
    """Create subplots comparing Iris vs RCCL for each operation."""
    operations = sorted(data.keys())
    num_ops = len(operations)

    # Create subplots - 2x2 grid for up to 4 operations
    if num_ops <= 2:
        nrows, ncols = 1, num_ops
    elif num_ops <= 4:
        nrows, ncols = 2, 2
    else:
        nrows = (num_ops + 1) // 2
        ncols = 2

    fig, axes = plt.subplots(nrows, ncols, figsize=tuple(args.figsize))
    fig.suptitle(args.title, fontsize=16, fontweight="bold")

    # Flatten axes for easier iteration
    if num_ops == 1:
        axes = [axes]
    else:
        axes = axes.flatten() if num_ops > 1 else [axes]

    for idx, operation in enumerate(operations):
        ax = axes[idx]
        op_data = data[operation]

        comm_sms = np.array(op_data["comm_sms"])
        iris_bw = np.array(op_data["iris_bw"])
        rccl_bw = np.array(op_data["rccl_bw"])

        # Plot Iris bandwidth
        ax.plot(comm_sms, iris_bw, "o-", linewidth=2, markersize=8, label="Iris", color="#2E86AB")

        # Plot RCCL bandwidth if available
        if not all(x is None for x in rccl_bw):
            # Filter out None values
            valid_indices = [i for i, x in enumerate(rccl_bw) if x is not None]
            if valid_indices:
                rccl_comm_sms = comm_sms[valid_indices]
                rccl_bw_valid = rccl_bw[valid_indices]
                ax.plot(rccl_comm_sms, rccl_bw_valid, "s--", linewidth=2, markersize=8, label="RCCL", color="#A23B72")

        # Formatting
        ax.set_xlabel("Number of CUs (comm_sms)", fontsize=11)
        ax.set_ylabel("Bandwidth (GB/s)", fontsize=11)
        ax.set_title(f"{operation.replace('_', '-').title()}", fontsize=13, fontweight="bold")
        ax.grid(True, alpha=0.3, linestyle="--")
        ax.legend(loc="best", fontsize=10)

        # Set x-axis to show all CU values
        ax.set_xticks(comm_sms)

        # Add some padding to y-axis
        y_min = min(
            iris_bw.min(), rccl_bw[rccl_bw is not None].min() if any(x is not None for x in rccl_bw) else iris_bw.min()
        )
        y_max = max(
            iris_bw.max(), rccl_bw[rccl_bw is not None].max() if any(x is not None for x in rccl_bw) else iris_bw.max()
        )
        y_range = y_max - y_min
        ax.set_ylim(y_min - 0.1 * y_range, y_max + 0.1 * y_range)

    # Hide unused subplots
    for idx in range(num_ops, len(axes)):
        axes[idx].set_visible(False)

    plt.tight_layout()

    # Generate output filename if not provided
    if args.output is None:
        base_name = os.path.splitext(args.input_csv)[0]
        args.output = f"{base_name}_plot.png"

    plt.savefig(args.output, dpi=args.dpi, bbox_inches="tight")
    print(f"\nPlot saved to: {args.output}")

    # Also display if running interactively
    try:
        plt.show()
    except Exception:
        pass


def main():
    args = parse_args()

    print(f"Loading results from: {args.input_csv}")
    data = load_results(args.input_csv)

    print(f"Found {len(data)} operations:")
    for op in sorted(data.keys()):
        num_points = len(data[op]["comm_sms"])
        print(f"  - {op}: {num_points} data points")

    print("\nCreating plots...")
    plot_results(data, args)


if __name__ == "__main__":
    main()
