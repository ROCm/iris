#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
# Copyright (c) 2025 Advanced Micro Devices, Inc. All rights reserved.

"""
Example: iris.ccl.all_gather

Each rank contributes an (M, N) tensor; every rank receives the concatenated (world_size*M, N) result.

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
        description="CCL all-gather example",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("-m", type=int, default=512, help="Number of rows per rank")
    parser.add_argument("-n", type=int, default=256, help="Number of columns")
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

    # Each rank fills its input with (rank + 1)
    input_tensor = ctx.zeros((M, N), dtype=dtype)
    input_tensor.fill_(float(rank + 1))
    output_tensor = ctx.zeros((world_size * M, N), dtype=dtype)

    ctx.barrier()
    ctx.ccl.all_gather(output_tensor, input_tensor)
    torch.cuda.synchronize()

    if rank == 0:
        ctx.info(f"all_gather: world_size={world_size}, input=({M},{N}), output=({world_size * M},{N}), dtype={dtype}")

    if args["validate"]:
        for r in range(world_size):
            expected = float(r + 1)
            chunk = output_tensor[r * M : (r + 1) * M]
            assert torch.allclose(chunk, torch.full_like(chunk, expected), atol=0.5), (
                f"Rank {rank}: chunk {r} mismatch. Got {chunk[0, 0].item():.1f}, expected {expected:.1f}"
            )
        if rank == 0:
            ctx.info(f"Validation passed: output[0,0] = {output_tensor[0, 0].item():.1f}")

    ctx.barrier()
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
