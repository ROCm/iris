#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
# Copyright (c) 2025 Advanced Micro Devices, Inc. All rights reserved.

import sys
import json
import itertools
import os
from pathlib import Path

import torch
import torch.distributed as dist

import iris
from fd_layer_rccl import FDLayerRCCL

# ==============================================================================
# Benchmark Configuration Sweep
# ==============================================================================

KV_LEN_SWEEP = [8192, 16384, 32768, 65536, 131072, 262144, 524288]
NUM_HEADS_SWEEP = [96]
HEAD_DIM_SWEEP = [128]
NUM_SEQS_SWEEP = [1, 4, 8, 16]


# --- Generate configurations (this is a cartesian product to get all configs) ---
CONFIG_SWEEP = []
param_product = itertools.product(KV_LEN_SWEEP, NUM_HEADS_SWEEP, HEAD_DIM_SWEEP, NUM_SEQS_SWEEP)
for kv_len, num_heads, head_dim, num_seqs in param_product:
    CONFIG_SWEEP.append(
        {
            "kv_len": kv_len,
            "num_heads": num_heads,
            "head_dim": head_dim,
            "num_seqs": num_seqs,
        }
    )

OUTPUT_DIR = "results/fd_rccl"
DATATYPE = torch.float16
N_WARMUP = 100
N_REPEAT = 1000
PAGE_SIZE = 1

# ==============================================================================
# Helper Functions
# ==============================================================================


def prepare_perf_data(config, num_query_heads, num_kv_heads):
    """Prepares local data for the performance test on the current rank."""
    num_blocks_per_rank = (config["kv_len"] + PAGE_SIZE - 1) // PAGE_SIZE

    query = torch.randn(config["num_seqs"], num_query_heads, config["head_dim"], dtype=DATATYPE).cuda()
    key_cache_this_rank = torch.randn(
        num_blocks_per_rank, PAGE_SIZE, num_kv_heads, config["head_dim"], dtype=DATATYPE
    ).cuda()
    value_cache_this_rank = torch.randn(
        num_blocks_per_rank, PAGE_SIZE, num_kv_heads, config["head_dim"], dtype=DATATYPE
    ).cuda()
    block_tables_this_rank = torch.arange(num_blocks_per_rank, dtype=torch.int32).repeat(config["num_seqs"], 1).cuda()

    return {
        "query": query,
        "key_cache_this_rank": key_cache_this_rank,
        "value_cache_this_rank": value_cache_this_rank,
        "block_tables_this_rank": block_tables_this_rank,
    }


# ==============================================================================
# Main Execution Block
# ==============================================================================


def main():
    dist.init_process_group(backend="nccl")
    rank = int(os.environ["RANK"])
    world_size = int(os.environ["WORLD_SIZE"])
    torch.cuda.set_device(int(os.environ["LOCAL_RANK"]))

    if rank == 0:
        if not os.path.exists(OUTPUT_DIR):
            os.makedirs(OUTPUT_DIR)
            print(f"Created output directory: '{OUTPUT_DIR}'")

    tp_group = dist.new_group(ranks=range(world_size))

    torch.manual_seed(42)
    torch.set_default_device("cuda")
    all_results = []

    # Loop through configs
    for i, config in enumerate(CONFIG_SWEEP):
        if rank == 0:
            print(f"\n--- Running Config {i + 1}/{len(CONFIG_SWEEP)}: {config} ---")

        num_query_heads = config["num_heads"]
        num_kv_heads = num_query_heads // 8 if num_query_heads >= 8 else 1
        scale = config["head_dim"] ** -0.5

        keyword_params = {
            "page_size": PAGE_SIZE,
            "scale": scale,
            "soft_cap": 0.0,
            "max_allowed_batch": config["num_seqs"],
        }

        fd_layer = FDLayerRCCL(
            rank,
            world_size,
            num_query_heads,
            num_kv_heads,
            config["head_dim"],  # q_head_dim
            config["head_dim"],  # v_head_dim
            tp_group,
            **keyword_params,
        )

        tensor_data = prepare_perf_data(config, num_query_heads, num_kv_heads)

        kv_lens_per_rank = [config["kv_len"]] * config["num_seqs"]
        kv_lens_tensor = torch.tensor(kv_lens_per_rank, dtype=torch.int32).cuda()
        global_kv_lens_tensor = kv_lens_tensor.unsqueeze(0).repeat(world_size, 1)

        def run_experiment():
            return fd_layer(
                tensor_data["query"],
                tensor_data["key_cache_this_rank"],
                tensor_data["value_cache_this_rank"],
                global_kv_lens_tensor,
                tensor_data["block_tables_this_rank"],
            )

        time_ms = iris.do_bench(
            fn=run_experiment, barrier_fn=dist.barrier, n_warmup=N_WARMUP, n_repeat=N_REPEAT, return_mode="mean"
        )
        dist.barrier()

        if rank == 0:
            global_kv_len = config["kv_len"] * world_size
            print(f"Result -> Global KV Length: {global_kv_len}, Avg. Time: {time_ms:.3f} ms")

            result_entry = config.copy()
            result_entry["global_kv_len"] = global_kv_len
            result_entry["avg_time_ms"] = time_ms

            filename = f"h{config['num_heads']}_d{config['head_dim']}_s{config['num_seqs']}_kv{config['kv_len']}.json"
            output_path = os.path.join(OUTPUT_DIR, filename)

            with open(output_path, "w") as f:
                json.dump(result_entry, f, indent=4)
            print(f"Saved result to '{output_path}'")
    print("\nBenchmark sweep complete.")

    dist.barrier()
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
