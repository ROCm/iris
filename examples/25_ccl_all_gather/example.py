#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
# Copyright (c) 2025 Advanced Micro Devices, Inc. All rights reserved.

"""
Example: iris.ccl.all_gather

Each rank contributes an (M, N) tensor; every rank receives the concatenated (world_size*M, N) result.

Run with:
    torchrun --nproc_per_node=<num_gpus> --standalone example.py [--validate] [--use_gluon]
"""

import argparse
import os

import torch
import torch.distributed as dist

import iris
from iris.ccl import Config


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
    parser.add_argument("--block_size_m", type=int, default=32, help="Block size for M dimension tiling")
    parser.add_argument("--block_size_n", type=int, default=64, help="Block size for N dimension tiling")
    parser.add_argument("--comm_sms", type=int, default=64, help="Number of SMs for all-gather kernel")
    parser.add_argument("--num_stages", type=int, default=1, help="Number of stages")
    parser.add_argument("--num_warps", type=int, default=4, help="Number of warps")
    parser.add_argument("--waves_per_eu", type=int, default=0, help="Number of waves per EU")
    parser.add_argument("--use_gluon", action="store_true", help="Use Gluon kernel backend")
    return vars(parser.parse_args())


def main():
    args = parse_args()

    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    torch.cuda.set_device(local_rank)
    dist.init_process_group(backend="gloo")

    if args["use_gluon"]:
        import iris.experimental.iris_gluon as iris_gluon

        ctx = iris_gluon.iris(heap_size=args["heap_size"])
    else:
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

    config_kwargs = {
        "block_size_m": args["block_size_m"],
        "block_size_n": args["block_size_n"],
        "comm_sms": args["comm_sms"],
        "num_stages": args["num_stages"],
        "num_warps": args["num_warps"],
        "waves_per_eu": args["waves_per_eu"],
        "use_gluon": args["use_gluon"],
    }
    config = Config(**config_kwargs)

    ctx.barrier()
    ctx.ccl.all_gather(output_tensor, input_tensor, config=config)
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
