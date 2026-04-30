#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
# Copyright (c) 2025 Advanced Micro Devices, Inc. All rights reserved.
"""
Example: Remote Load and Store
Demonstrates: iris.load(), iris.store() via Triton kernels

This example shows how to use iris.store() to push data from one GPU's
symmetric heap into another GPU's symmetric heap, and iris.load() to pull
data from a remote GPU's symmetric heap into a local buffer.

Rank 0 fills a buffer with arange(N).  A Triton kernel on rank 0 then
stores that data into rank 1's buffer using iris.store().  After a barrier,
a second Triton kernel on rank 1 loads data from rank 0's buffer using
iris.load().  Both results are validated.

Run with:
    torchrun --nproc_per_node=2 --standalone examples/03_mem_load_store/example.py [--validate]
"""

import argparse
import os

import torch
import torch.distributed as dist
import triton
import triton.language as tl

import iris


# ---------------------------------------------------------------------------
# Triton kernels
# ---------------------------------------------------------------------------

@triton.jit
def store_kernel(
    src_ptr,
    dst_ptr,
    N,
    from_rank: tl.constexpr,
    to_rank: tl.constexpr,
    heap_bases: tl.tensor,
    BLOCK_SIZE: tl.constexpr,
):
    """Store local data into a remote rank's buffer via iris.store()."""
    pid = tl.program_id(0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < N
    data = tl.load(src_ptr + offsets, mask=mask)
    iris.store(dst_ptr + offsets, data, from_rank, to_rank, heap_bases, mask=mask)


@triton.jit
def load_kernel(
    src_ptr,
    dst_ptr,
    N,
    to_rank: tl.constexpr,
    from_rank: tl.constexpr,
    heap_bases: tl.tensor,
    BLOCK_SIZE: tl.constexpr,
):
    """Load data from a remote rank's buffer into a local buffer via iris.load()."""
    pid = tl.program_id(0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < N
    data = iris.load(src_ptr + offsets, to_rank, from_rank, heap_bases, mask=mask)
    tl.store(dst_ptr + offsets, data, mask=mask)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args():
    parser = argparse.ArgumentParser(
        description="iris.load / iris.store example",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--validate", action="store_true", help="Run correctness check")
    parser.add_argument("--heap_size", type=int, default=1 << 30, help="Symmetric heap size in bytes")
    parser.add_argument("-m", type=int, default=1024, help="Number of float32 elements to transfer")
    return vars(parser.parse_args())


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    args = parse_args()
    N = args["m"]
    BLOCK_SIZE = 1024

    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    torch.cuda.set_device(local_rank)
    dist.init_process_group(backend="gloo")

    ctx = iris.iris(heap_size=args["heap_size"])
    rank = ctx.get_rank()
    world_size = ctx.get_num_ranks()
    heap_bases = ctx.get_heap_bases()

    assert world_size >= 2, "This example requires at least 2 ranks."

    # Allocate symmetric-heap buffers visible to all ranks.
    source = ctx.zeros(N, dtype=torch.float32)   # rank 0 fills this
    store_dst = ctx.zeros(N, dtype=torch.float32)  # target for iris.store
    load_dst = ctx.zeros(N, dtype=torch.float32)   # target for iris.load result

    # Rank 0 initialises its source buffer.
    if rank == 0:
        source.copy_(torch.arange(N, dtype=torch.float32, device="cuda"))

    ctx.barrier()

    grid = (triton.cdiv(N, BLOCK_SIZE),)

    # --- Step 1: rank 0 stores data into rank 1's store_dst buffer ----------
    if rank == 0:
        store_kernel[grid](source, store_dst, N, 0, 1, heap_bases, BLOCK_SIZE)

    ctx.barrier()

    # --- Step 2: rank 1 loads data from rank 0's source buffer --------------
    if rank == 1:
        load_kernel[grid](source, load_dst, N, 1, 0, heap_bases, BLOCK_SIZE)

    ctx.barrier()

    # --- Validation ---------------------------------------------------------
    if args["validate"]:
        expected = torch.arange(N, dtype=torch.float32, device="cuda")

        if rank == 1:
            # store_dst on rank 1 should contain rank 0's arange data
            torch.testing.assert_close(store_dst, expected, rtol=0, atol=0)
            print(f"[rank {rank}] iris.store validation PASSED")

            # load_dst on rank 1 should also contain rank 0's arange data
            torch.testing.assert_close(load_dst, expected, rtol=0, atol=0)
            print(f"[rank {rank}] iris.load  validation PASSED")
    else:
        if rank == 0:
            print("Run with --validate to check correctness.")

    ctx.barrier()
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
