#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
# Copyright (c) 2025 Advanced Micro Devices, Inc. All rights reserved.

import json
import os
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# ==============================================================================
# Configuration
# ==============================================================================
IRIS_RESULTS_DIR = "results/flash_decode_iris"
RCCL_RESULTS_DIR = "results/fd_rccl"
OUTPUT_DIR = "plots"

# ==============================================================================
# Helper Functions
# ==============================================================================


def load_data_from_directory(directory_path: str, backend_name: str) -> pd.DataFrame:
    """Walks a directory, loads all .json files, and returns a pandas DataFrame."""
    all_results = []

    if not os.path.isdir(directory_path):
        print(f"Warning: Directory not found: '{directory_path}'. Skipping this backend.")
        return None

    print(f"Loading results for '{backend_name}' from '{directory_path}'...")
    for filename in os.listdir(directory_path):
        if filename.endswith(".json"):
            file_path = os.path.join(directory_path, filename)
            try:
                with open(file_path, "r") as f:
                    data = json.load(f)
                    all_results.append(data)
            except (json.JSONDecodeError, KeyError) as e:
                print(f"Warning: Could not parse '{file_path}'. Error: {e}. Skipping file.")

    if not all_results:
        print(f"Warning: No valid .json files found in '{directory_path}'.")
        return None

    df = pd.DataFrame(all_results)
    df["backend"] = backend_name
    print(f"Successfully loaded {len(df)} results for backend '{backend_name}'.")
    return df


def plot_config_group(group_df: pd.DataFrame, config_key: tuple):
    """Creates and saves a bar chart for a specific configuration group."""
    num_heads, head_dim, num_seqs = config_key

    pivot_df = group_df.pivot(index="kv_len", columns="backend", values="avg_time_ms")
    pivot_df.sort_index(inplace=True)

    fig, ax = plt.subplots(figsize=(12, 7))
    x = np.arange(len(pivot_df.index))
    width = 0.35

    if "Iris" in pivot_df.columns:
        ax.bar(x - width / 2, pivot_df["Iris"], width, label="Iris", color="skyblue", zorder=3)
    if "RCCL" in pivot_df.columns:
        ax.bar(x + width / 2, pivot_df["RCCL"], width, label="RCCL", color="coral", zorder=3)

    title = f"Heads: {num_heads}, Head Dim: {head_dim}, Batch Size: {num_seqs}"
    ax.set_title(title, fontsize=16, pad=15)
    ax.set_ylabel("Average Time (ms)", fontsize=12)
    ax.set_xlabel("Local KV Cache Length (per Rank)", fontsize=12)

    ax.grid(axis="y", linestyle="--", alpha=0.7, zorder=0)

    ax.set_xticks(x)
    ax.set_xticklabels(pivot_df.index, rotation=45, ha="right")

    ax.legend(fontsize=12)
    fig.tight_layout()

    filename = f"h{num_heads}_d{head_dim}_s{num_seqs}.png"
    output_path = os.path.join(OUTPUT_DIR, filename)
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved plot to '{output_path}'")


# ==============================================================================
# Main Execution Block
# ==============================================================================

if __name__ == "__main__":
    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)
        print(f"Created directory: '{OUTPUT_DIR}'")

    iris_df = load_data_from_directory(IRIS_RESULTS_DIR, "Iris")
    rccl_df = load_data_from_directory(RCCL_RESULTS_DIR, "RCCL")

    data_frames = [df for df in [iris_df, rccl_df] if df is not None]
    if not data_frames:
        print("Error: No valid data loaded from any directory. Exiting.")
        exit()

    combined_df = pd.concat(data_frames, ignore_index=True)

    config_groups = combined_df.groupby(["num_heads", "head_dim", "num_seqs"])

    print(f"\nFound {len(config_groups)} unique configurations to plot.")

    for config_key, group_df in config_groups:
        plot_config_group(group_df, config_key)

    print("\nPlotting complete.")
