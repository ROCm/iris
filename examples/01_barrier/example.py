#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
# Copyright (c) 2025 Advanced Micro Devices, Inc. All rights reserved.

"""
Example: barrier synchronization

Demonstrates: ctx.barrier(), ctx.device_barrier()

Each rank writes its identity to a symmetric buffer, synchronizes with a host
barrier, then rank 0 validates.  Also shows device_barrier usage.

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
        description="Barrier synchronization example",
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

    # --- Host barrier ---
    # Each rank writes (rank + 1) into its own 1-element symmetric buffer.
    buf = ctx.zeros(1, dtype=torch.float32)
    buf.fill_(float(rank + 1))

    # Barrier: after this, every rank's write is visible to all others.
    ctx.barrier()

    # Rank 0 gathers all values via torch.distributed and validates.
    gathered = [torch.zeros(1, device="cuda", dtype=torch.float32) for _ in range(world_size)]
    dist.all_gather(gathered, buf)

    if args["validate"]:
        for r in range(world_size):
            expected = float(r + 1)
            actual = gathered[r].item()
            assert actual == expected, f"Host barrier: rank {r} expected {expected}, got {actual}"
        if rank == 0:
            ctx.info(f"Host barrier validated: {[g.item() for g in gathered]}")

    # --- Device barrier ---
    ctx.device_barrier()

    if rank == 0:
        ctx.info(f"barrier: world_size={world_size}, host and device barriers completed.")

    ctx.barrier()
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
