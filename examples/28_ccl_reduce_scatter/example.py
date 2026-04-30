#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
# Copyright (c) 2025 Advanced Micro Devices, Inc. All rights reserved.

"""
Example: iris.ccl.reduce_scatter

Each rank has input (M, N); each rank reduces its assigned tiles from all ranks
and stores the result only to its own output (same shape (M, N)).

Run with:
    torchrun --nproc_per_node=<num_gpus> --standalone example.py [--validate]
"""

import argparse
import os

import torch
import torch.distributed as dist

import iris
from iris.ccl import Config


def parse_args():
    parser = argparse.ArgumentParser(
        description="CCL reduce-scatter example",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("-m", type=int, default=1024, help="Number of rows")
    parser.add_argument("-n", type=int, default=512, help="Number of columns")
    parser.add_argument("--heap_size", type=int, default=1 << 31, help="Iris heap size")
    parser.add_argument("--block_size_m", type=int, default=32, help="Block size for M dimension tiling")
    parser.add_argument("--block_size_n", type=int, default=64, help="Block size for N dimension tiling")
    parser.add_argument("--comm_sms", type=int, default=64, help="Number of SMs for reduce-scatter kernel")
    parser.add_argument("--num_stages", type=int, default=1, help="Number of stages")
    parser.add_argument("--num_warps", type=int, default=4, help="Number of warps")
    parser.add_argument("--waves_per_eu", type=int, default=0, help="Number of waves per EU")
    parser.add_argument("--datatype", type=str, default="fp16", choices=["fp16", "fp32", "bf16"], help="Data type")
    parser.add_argument("-v", "--validate", action="store_true", help="Validate output against reference")
    return vars(parser.parse_args())


def main():
    args = parse_args()

    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    torch.cuda.set_device(local_rank)
    dist.init_process_group(backend="gloo")

    ctx = iris.iris(heap_size=args["heap_size"])
    rank = ctx.get_rank()
    world_size = ctx.get_num_ranks()

    dtype_map = {"fp16": torch.float16, "fp32": torch.float32, "bf16": torch.bfloat16}
    dtype = dtype_map[args["datatype"]]
    M, N = args["m"], args["n"]

    # Each rank fills its input with (rank + 1)
    input_tensor = ctx.zeros((M, N), dtype=dtype)
    input_tensor.fill_(float(rank + 1))
    output_tensor = ctx.zeros((M, N), dtype=dtype)

    config_kwargs = {
        "block_size_m": args["block_size_m"],
        "block_size_n": args["block_size_n"],
        "comm_sms": args["comm_sms"],
        "num_stages": args["num_stages"],
        "num_warps": args["num_warps"],
        "waves_per_eu": args["waves_per_eu"],
        "all_reduce_distribution": 1,
    }
    config = Config(**config_kwargs)

    ctx.barrier()
    ctx.ccl.reduce_scatter(output_tensor, input_tensor, config=config)
    torch.cuda.synchronize()

    if rank == 0:
        ctx.info(f"reduce_scatter: world_size={world_size}, shape=({M},{N}), dtype={dtype}")

    if args["validate"]:
        # Reference: gather all inputs, sum, then each rank checks its assigned tiles
        ref_list = [torch.empty(M, N, dtype=dtype, device=input_tensor.device) for _ in range(world_size)]
        dist.all_gather(ref_list, input_tensor)
        full_reduced = sum(ref_list).float()

        block_size_m = args["block_size_m"]
        block_size_n = args["block_size_n"]
        num_pid_m = (M + block_size_m - 1) // block_size_m
        num_pid_n = (N + block_size_n - 1) // block_size_n
        total_tiles = num_pid_m * num_pid_n
        tiles_per_rank = (total_tiles + world_size - 1) // world_size
        start_tile = rank * tiles_per_rank

        # Build mask of (i,j) belonging to this rank's tiles (block distribution)
        pid_m = torch.arange(M, device=output_tensor.device) // block_size_m
        pid_n = torch.arange(N, device=output_tensor.device) // block_size_n
        tile_id = pid_m[:, None] * num_pid_n + pid_n[None, :]
        mask = (tile_id >= start_tile) & (tile_id < start_tile + tiles_per_rank)

        out_float = output_tensor.float()
        expected_where = full_reduced[mask]
        actual_where = out_float[mask]
        assert torch.allclose(actual_where, expected_where, atol=0.6), f"Rank {rank}: output mismatch on assigned tiles"
        if rank == 0:
            ctx.info("Validation passed: output matches reference")

    ctx.barrier()
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
