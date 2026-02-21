#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
# Copyright (c) 2025 Advanced Micro Devices, Inc. All rights reserved.

"""
Example: iris.ops.matmul_reduce_scatter

Fused GEMM + reduce-scatter: each rank computes A @ B; results are summed and
each rank stores only its assigned tiles.

Run with:
    torchrun --nproc_per_node=<num_gpus> --standalone example.py [--validate]
"""

import argparse
import os

import torch
import torch.distributed as dist

import iris
from iris.ops import FusedConfig


def parse_args():
    parser = argparse.ArgumentParser(
        description="Fused matmul + reduce-scatter example",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("-m", type=int, default=512, help="Rows of A")
    parser.add_argument("-n", type=int, default=512, help="Columns of B")
    parser.add_argument("-k", type=int, default=256, help="Inner dimension")
    parser.add_argument("--heap_size", type=int, default=1 << 31, help="Iris heap size")
    parser.add_argument("--datatype", type=str, default="fp16", choices=["fp16", "fp32", "bf16"], help="Data type")
    parser.add_argument("-v", "--validate", action="store_true", help="Validate output against reference")
    return vars(parser.parse_args())


def main():
    args = parse_args()

    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    torch.cuda.set_device(local_rank)
    dist.init_process_group(backend="nccl")

    ctx = iris.iris(heap_size=args["heap_size"])
    rank = ctx.get_rank()
    world_size = ctx.get_num_ranks()

    dtype_map = {"fp16": torch.float16, "fp32": torch.float32, "bf16": torch.bfloat16}
    dtype = dtype_map[args["datatype"]]
    M, K, N = args["m"], args["k"], args["n"]

    torch.manual_seed(42)
    A = ctx.randn((M, K), dtype=dtype)
    B = ctx.randn((K, N), dtype=dtype)
    output = ctx.zeros((M, N), dtype=dtype)

    config = FusedConfig(block_size_m=64, block_size_n=64, block_size_k=32)

    ctx.barrier()
    ctx.ops.matmul_reduce_scatter(output, A, B, config=config)
    torch.cuda.synchronize()

    if rank == 0:
        ctx.info(f"matmul_reduce_scatter: world_size={world_size}, A=({M},{K}), B=({K},{N}), dtype={dtype}")

    if args["validate"]:
        # Reference: matmul + all_reduce; verify only this rank's assigned tiles
        C_ref = torch.matmul(A.clone().float(), B.clone().float()).to(dtype)
        dist.all_reduce(C_ref, op=dist.ReduceOp.SUM)
        torch.cuda.synchronize()

        num_pid_m = (M + config.block_size_m - 1) // config.block_size_m
        num_pid_n = (N + config.block_size_n - 1) // config.block_size_n
        total_tiles = num_pid_m * num_pid_n
        tiles_per_rank = (total_tiles + world_size - 1) // world_size
        start_tile = rank * tiles_per_rank
        end_tile = min(start_tile + tiles_per_rank, total_tiles)
        for tile_id in range(start_tile, end_tile):
            pid_m, pid_n = tile_id // num_pid_n, tile_id % num_pid_n
            m0, m1 = pid_m * config.block_size_m, min((pid_m + 1) * config.block_size_m, M)
            n0, n1 = pid_n * config.block_size_n, min((pid_n + 1) * config.block_size_n, N)
            iris_tile = output[m0:m1, n0:n1]
            ref_tile = C_ref[m0:m1, n0:n1]
            assert torch.allclose(iris_tile.float(), ref_tile.float(), atol=1.0, rtol=0.05), (
                f"Rank {rank}: tile ({pid_m},{pid_n}) mismatch. "
                f"Max diff: {(iris_tile.float() - ref_tile.float()).abs().max().item():.4f}"
            )
        if rank == 0:
            ctx.info(f"Validation passed: rank 0 verified tiles {start_tile}..{end_tile - 1}")

    ctx.barrier()
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
