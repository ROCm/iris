#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
# Copyright (c) 2025 Advanced Micro Devices, Inc. All rights reserved.

import json
import itertools
import os
import argparse

import torch
import torch.distributed as dist

import iris
from examples.benchmark.reference.flash_decode_rccl.flash_decode_layer_rccl import flash_decode_layer_rccl


def parse_args():
    """
    Arguments for the benchmark
    The default parameters are in dataset/flash_decode_config_rccl.json
    A different config file can be set with the --config flag
    """
    parser = argparse.ArgumentParser(
        description="Run Flash Decode RCCL benchmark with parameters from a config file.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    parser.add_argument(
        "-c",
        "--config",
        type=str,
        default="dataset/flash_decode_config_rccl.json",
        help="Path to the JSON configuration file",
    )

    config_args, _ = parser.parse_known_args()

    config_defaults = {}
    if os.path.exists(config_args.config):
        try:
            with open(config_args.config, "r") as f:
                config_from_file = json.load(f)
            if config_from_file:
                print(f"Configuration successfully loaded from '{config_args.config}'")
                config_defaults = {**config_from_file, **config_from_file.get("sweep_parameters", {})}
                if "sweep_parameters" in config_defaults:
                    del config_defaults["sweep_parameters"]
        except json.JSONDecodeError:
            print(f"Error: Config file '{config_args.config}' is not valid JSON.")
    else:
        print(f"Warning: Config file '{config_args.config}' not found.")

    parser.set_defaults(**config_defaults)

    parser.add_argument("--output_dir", type=str, help="Directory to save results")
    parser.add_argument("--data_type", type=str, choices=["float16", "bfloat16", "float32"], help="PyTorch data type")
    parser.add_argument("--warmup_iterations", type=int, help="Number of warmup iterations")
    parser.add_argument("--repeat_iterations", type=int, help="Number of benchmark iterations")
    parser.add_argument("--page_size", type=int, help="Page size for KV cache", default=1)

    parser.add_argument("--kv_len", type=int, nargs="+", help="Override KV_LEN_SWEEP")
    parser.add_argument("--num_heads", type=int, nargs="+", help="Override NUM_HEADS_SWEEP")
    parser.add_argument("--head_dim", type=int, nargs="+", help="Override HEAD_DIM_SWEEP")
    parser.add_argument("--num_seqs", type=int, nargs="+", help="Override NUM_SEQS_SWEEP")

    final_args = parser.parse_args()
    return final_args


def prepare_perf_data(config, num_query_heads, num_kv_heads, page_size, datatype):
    """Prepares local data for the performance test on the current rank."""
    num_blocks_per_rank = (config["kv_len"] + page_size - 1) // page_size

    query = torch.randn(config["num_seqs"], num_query_heads, config["head_dim"], dtype=datatype).cuda()
    key_cache_this_rank = torch.randn(
        num_blocks_per_rank, page_size, num_kv_heads, config["head_dim"], dtype=datatype
    ).cuda()
    value_cache_this_rank = torch.randn(
        num_blocks_per_rank, page_size, num_kv_heads, config["head_dim"], dtype=datatype
    ).cuda()
    block_tables_this_rank = torch.arange(num_blocks_per_rank, dtype=torch.int32).repeat(config["num_seqs"], 1).cuda()

    return {
        "query": query,
        "key_cache_this_rank": key_cache_this_rank,
        "value_cache_this_rank": value_cache_this_rank,
        "block_tables_this_rank": block_tables_this_rank,
    }


def run_benchmark(args):
    local_rank = int(os.environ["LOCAL_RANK"])
    dist.init_process_group(backend="nccl", device_id=torch.device(f"cuda:{local_rank}"))
    rank = int(os.environ["RANK"])
    world_size = int(os.environ["WORLD_SIZE"])
    torch.cuda.set_device(int(os.environ["LOCAL_RANK"]))

    output_dir = args.output_dir
    datatype = getattr(torch, args.data_type)
    page_size = args.page_size

    if rank == 0:
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
            print(f"Created output directory: '{output_dir}'")

    tp_group = dist.new_group(ranks=range(world_size))
    torch.manual_seed(42)

    config_sweep = []
    param_product = itertools.product(args.kv_len, args.num_heads, args.head_dim, args.num_seqs)
    for kv_len, num_heads, head_dim, num_seqs in param_product:
        config_sweep.append(
            {
                "kv_len": kv_len,
                "num_heads": num_heads,
                "head_dim": head_dim,
                "num_seqs": num_seqs,
            }
        )

    # Loop through configs
    for i, config in enumerate(config_sweep):
        if rank == 0:
            print(f"\n--- Running Config {i + 1}/{len(config_sweep)}: {config} ---")

        num_query_heads = config["num_heads"]
        num_kv_heads = num_query_heads // 8 if num_query_heads >= 8 else 1
        scale = config["head_dim"] ** -0.5

        keyword_params = {
            "page_size": page_size,
            "scale": scale,
            "soft_cap": 0.0,
            "max_allowed_batch": config["num_seqs"],
        }

        fd_layer = flash_decode_layer_rccl(
            rank,
            world_size,
            num_query_heads,
            num_kv_heads,
            config["head_dim"],
            config["head_dim"],
            tp_group,
            **keyword_params,
        )

        tensor_data = prepare_perf_data(config, num_query_heads, num_kv_heads, page_size, datatype)

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
            fn=run_experiment,
            barrier_fn=dist.barrier,
            n_warmup=args.warmup_iterations,
            n_repeat=args.repeat_iterations,
            return_mode="mean",
        )
        dist.barrier()

        if rank == 0:
            global_kv_len = config["kv_len"] * world_size
            print(f"Result -> Global KV Length: {global_kv_len}, Avg. Time: {time_ms:.3f} ms")

            result_entry = config.copy()
            result_entry["global_kv_len"] = global_kv_len
            result_entry["avg_time_ms"] = time_ms

            filename = f"h{config['num_heads']}_d{config['head_dim']}_s{config['num_seqs']}_kv{config['kv_len']}.json"
            output_path = os.path.join(output_dir, filename)

            with open(output_path, "w") as f:
                json.dump(result_entry, f, indent=4)
            print(f"Saved result to '{output_path}'")

    if rank == 0:
        print("\nBenchmark sweep complete.")

    dist.barrier()
    dist.destroy_process_group()


if __name__ == "__main__":
    args = parse_args()
    run_benchmark(args)
