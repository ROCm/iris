#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
# Copyright (c) 2025 Advanced Micro Devices, Inc. All rights reserved.

"""
Example: iris.ccl.all_to_all

Input and output are both (M, N*world_size): input[:, r*N:(r+1)*N] is sent to rank r.

Run with:
    torchrun --nproc_per_node=<num_gpus> --standalone example.py [--validate]
"""

import argparse
import os

import torch
import torch.distributed as dist

import iris


def parse_args():
    parser = argparse.ArgumentParser(
        description="CCL all-to-all example",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("-m", type=int, default=512, help="Number of rows")
    parser.add_argument("-n", type=int, default=128, help="Number of columns per rank slice")
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
    M, N = args["m"], args["n"]

    # input[:, r*N:(r+1)*N] is the slice sent to rank r; fill with unique values
    input_tensor = ctx.zeros((M, N * world_size), dtype=dtype)
    for target_rank in range(world_size):
        input_tensor[:, target_rank * N : (target_rank + 1) * N] = float(rank * 10 + target_rank + 1)
    output_tensor = ctx.zeros((M, N * world_size), dtype=dtype)

    ctx.barrier()
    ctx.ccl.all_to_all(output_tensor, input_tensor)
    torch.cuda.synchronize()

    if rank == 0:
        ctx.info(f"all_to_all: world_size={world_size}, shape=({M},{N * world_size}), dtype={dtype}")

    if args["validate"]:
        for src_rank in range(world_size):
            expected = float(src_rank * 10 + rank + 1)
            chunk = output_tensor[:, src_rank * N : (src_rank + 1) * N]
            assert torch.allclose(chunk, torch.full_like(chunk, expected), atol=0.5), (
                f"Rank {rank}: chunk from rank {src_rank} mismatch. "
                f"Got {chunk[0, 0].item():.1f}, expected {expected:.1f}"
            )
        if rank == 0:
            ctx.info(f"Validation passed: output[0,0] = {output_tensor[0, 0].item():.1f}")

    ctx.barrier()
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
