#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
# Copyright (c) 2025 Advanced Micro Devices, Inc. All rights reserved.

"""
Example: broadcast

Demonstrates: ctx.broadcast(value, src=0)

Rank 0 picks a random scalar, broadcasts it to all ranks, and every rank
validates it received the correct value.

Run with:
    torchrun --nproc_per_node=<num_gpus> --standalone example.py [--validate]
"""

import argparse
import os
import random

import torch
import torch.distributed as dist

import iris


def parse_args():
    parser = argparse.ArgumentParser(
        description="Broadcast example",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--heap_size", type=int, default=1 << 30, help="Iris heap size")
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

    # Rank 0 picks a random value; other ranks pass None.
    value = random.randint(1, 10000) if rank == 0 else None
    result = ctx.broadcast(value, src=0)

    if rank == 0:
        ctx.info(f"broadcast: world_size={world_size}, src=0, value={value}")

    if args["validate"]:
        # The source rank's original value should equal the broadcast result.
        if rank == 0:
            assert result == value, f"Rank 0: broadcast returned {result}, expected {value}"
        # All ranks must have received the same value.
        # Gather results to rank 0 for a cross-rank check.
        results = [None] * world_size
        dist.all_gather_object(results, result)
        if rank == 0:
            for r, v in enumerate(results):
                assert v == value, f"Rank {r}: got {v}, expected {value}"
            ctx.info(f"Validation passed: all {world_size} ranks received {value}")

    ctx.barrier()
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
