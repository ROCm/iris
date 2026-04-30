#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
# Copyright (c) 2025 Advanced Micro Devices, Inc. All rights reserved.

"""
Example: symmetric heap basics

Demonstrates: ctx.zeros(), ctx.ones(), ctx.randn(), ctx.is_symmetric(), ctx.as_symmetric()

Allocates tensors on the iris symmetric heap, checks symmetry, and imports an
external CUDA tensor into the heap.  No communication or Triton kernels.

Run with:
    torchrun --nproc_per_node=<num_gpus> --standalone example.py
"""

import argparse
import os

import torch
import torch.distributed as dist

import iris


def parse_args():
    parser = argparse.ArgumentParser(
        description="Symmetric-heap basics example",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--heap_size", type=int, default=1 << 30, help="Iris heap size")
    return vars(parser.parse_args())


def main():
    args = parse_args()

    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    torch.cuda.set_device(local_rank)
    dist.init_process_group(backend="gloo")

    ctx = iris.iris(heap_size=args["heap_size"])
    rank = ctx.get_rank()
    world_size = ctx.get_num_ranks()

    # --- Allocate tensors on the symmetric heap ---
    z = ctx.zeros(4, 4, dtype=torch.float32)
    o = ctx.ones(4, 4, dtype=torch.float32)
    r = ctx.randn(4, 4, dtype=torch.float32)

    # --- Verify symmetric-heap membership ---
    assert ctx.is_symmetric(z), "zeros tensor should be symmetric"
    assert ctx.is_symmetric(o), "ones tensor should be symmetric"
    assert ctx.is_symmetric(r), "randn tensor should be symmetric"

    # --- Regular CUDA tensor is NOT symmetric ---
    external = torch.randn(4, 4, device="cuda", dtype=torch.float32)
    assert not ctx.is_symmetric(external), "external tensor should not be symmetric"

    # --- Import external tensor into the heap ---
    imported = ctx.as_symmetric(external)
    assert ctx.is_symmetric(imported), "imported tensor should be symmetric"

    if rank == 0:
        ctx.info(f"heap_basics: world_size={world_size}")
        ctx.info(f"  zeros  -> symmetric={ctx.is_symmetric(z)}, shape={list(z.shape)}")
        ctx.info(f"  ones   -> symmetric={ctx.is_symmetric(o)}, shape={list(o.shape)}")
        ctx.info(f"  randn  -> symmetric={ctx.is_symmetric(r)}, shape={list(r.shape)}")
        ctx.info(f"  external CUDA tensor -> symmetric={ctx.is_symmetric(external)}")
        ctx.info(f"  as_symmetric(external) -> symmetric={ctx.is_symmetric(imported)}")
        ctx.info("All checks passed.")

    ctx.barrier()
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
