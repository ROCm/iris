#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
# Copyright (c) 2025 Advanced Micro Devices, Inc. All rights reserved.

import sys
import json
import itertools
from pathlib import Path

import torch
import iris

# ==============================================================================
# Path and Module Setup
# ==============================================================================

project_root = Path(__file__).resolve()
while not (project_root / 'tests').is_dir() or not (project_root / 'examples').is_dir():
    if project_root == project_root.parent:
        raise FileNotFoundError(
            "Could not find project root. Make sure your 'tests' and 'examples' "
            "directories are siblings in the project structure."
        )
    project_root = project_root.parent

module_dir = project_root / "examples" / "13_flash_decode"
if module_dir.is_dir():
    sys.path.insert(0, str(module_dir))
else:
    raise FileNotFoundError(f"Target directory not found: {module_dir}")
from fd_fused_layer import FDFusedLayer

# ==============================================================================
# Benchmark Configuration Sweep
# ==============================================================================

KV_LEN_SWEEP = [8192, 16384, 32768, 65536, 131072, 262144, 524288, 1048576]
NUM_HEADS_SWEEP = [96]
HEAD_DIM_SWEEP = [128]
NUM_SEQS_SWEEP = [1]


# --- Generate configurations (this is a cartesian product to get all configs) ---
CONFIG_SWEEP = []
param_product = itertools.product(KV_LEN_SWEEP, NUM_HEADS_SWEEP, HEAD_DIM_SWEEP, NUM_SEQS_SWEEP)
for kv_len, num_heads, head_dim, num_seqs in param_product:
    CONFIG_SWEEP.append({
        "kv_len": kv_len,
        "num_heads": num_heads,
        "head_dim": head_dim,
        "num_seqs": num_seqs,
    })

OUTPUT_FILENAME = "benchmark_results.json"
DATATYPE = torch.float16
N_WARMUP = 100
N_REPEAT = 1000

# ==============================================================================
# Helper Functions
# ==============================================================================

def prepare_perf_data(cfg, num_query_heads, num_kv_heads):
    """Prepares local data for the performance test on the current rank."""
    num_blocks_per_rank = (cfg['kv_len'] + cfg['block_size'] - 1) // cfg['block_size']
    
    query = torch.randn(cfg['num_seqs'], num_query_heads, cfg['head_dim'], dtype=cfg['dtype']).cuda()
    key_cache_this_rank = torch.randn(num_blocks_per_rank, cfg['block_size'], num_kv_heads, cfg['head_dim'], dtype=cfg['dtype']).cuda()
    value_cache_this_rank = torch.randn(num_blocks_per_rank, cfg['block_size'], num_kv_heads, cfg['head_dim'], dtype=cfg['dtype']).cuda()
    block_tables_this_rank = torch.arange(num_blocks_per_rank, dtype=torch.int32).repeat(cfg['num_seqs'], 1).cuda()

    return {
        "query": query, "key_cache_this_rank": key_cache_this_rank,
        "value_cache_this_rank": value_cache_this_rank, "block_tables_this_rank": block_tables_this_rank
    }

# ==============================================================================
# Main Execution Block
# ==============================================================================

def main():
    _iris = iris.iris()
    rank = _iris.get_rank()
    world_size = _iris.get_num_ranks()

    torch.manual_seed(42)
    torch.set_default_device("cuda")
    all_results = []

    # Loop through configs
    for i, config in enumerate(CONFIG_SWEEP):
        if rank == 0:
            print(f"\n--- Running Config {i+1}/{len(CONFIG_SWEEP)}: {config} ---")

        cfg = { "block_size": 1, "soft_cap": 0.0, "dtype": DATATYPE, **config }
        num_query_heads = cfg['num_heads']
        num_kv_heads = num_query_heads // 8 if num_query_heads >= 8 else 1
        scale = cfg['head_dim']**-0.5
        
        common_params = {
            "num_q_heads": num_query_heads, "num_kv_heads": num_kv_heads,
            "q_head_dim": cfg['head_dim'], "v_head_dim": cfg['head_dim'],
            "page_size": cfg['block_size'], "scale": scale,
            "soft_cap": cfg['soft_cap'], "max_allowed_batch": cfg['num_seqs']
        }
        fd_layer = FDFusedLayer(_iris, rank, rank, world_size, world_size, **common_params)
        
        tensor_data = prepare_perf_data(cfg, num_query_heads, num_kv_heads)
        kv_lens_tensor = torch.tensor([cfg['kv_len']], dtype=torch.int32).cuda()
        global_kv_lens_tensor = torch.cat([kv_lens_tensor.view(1, -1) for _ in range(world_size)], dim=0)
        
        def run_experiment():
            return fd_layer(
                tensor_data['query'], tensor_data['key_cache_this_rank'], 
                tensor_data['value_cache_this_rank'], global_kv_lens_tensor, 
                tensor_data['block_tables_this_rank']
            )
        
        time_ms = iris.do_bench(
            fn=run_experiment, barrier_fn=_iris.barrier,
            preamble_fn=getattr(fd_layer, 'clear_flags', None),
            n_warmup=N_WARMUP, n_repeat=N_REPEAT, return_mode="mean"
        )
        _iris.barrier()

        if rank == 0:
            global_kv_len = cfg['kv_len'] * world_size
            print(f"Result -> Global KV Length: {global_kv_len}, Avg. Time: {time_ms:.3f} ms")
            
            result_entry = config.copy()
            result_entry['global_kv_len'] = global_kv_len
            result_entry['avg_time_ms'] = time_ms
            all_results.append(result_entry)

            # Overwrite the file
            with open(OUTPUT_FILENAME, 'w') as f:
                json.dump(all_results, f, indent=4)
            print(f"Updated '{OUTPUT_FILENAME}' with {len(all_results)} total result(s).")
            
    if rank == 0:
        print(f"\nBenchmark sweep complete.")

    _iris.barrier()

if __name__ == "__main__":
    main()