#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
# Copyright (c) 2025 Advanced Micro Devices, Inc. All rights reserved.
"""
Example: Remote Put and Get
Demonstrates: iris.put(), iris.get() via Triton kernels

iris.put() reads from a local buffer and writes to a remote buffer in a
single kernel call (local load + remote store).  iris.get() reads from a
remote buffer and writes to a local buffer (remote load + local store).

Rank 0 fills a source buffer with arange(N).  A put kernel on rank 0 pushes
that data into rank 1's put_dst buffer.  After a barrier, a get kernel on
rank 1 pulls data from rank 0's source buffer into rank 1's get_dst buffer.
Both results are validated.

Run with:
    torchrun --nproc_per_node=2 --standalone examples/04_mem_put_get/example.py [--validate]
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
def put_kernel(
    local_ptr,
    remote_ptr,
    N,
    from_rank: tl.constexpr,
    to_rank: tl.constexpr,
    heap_bases: tl.tensor,
    BLOCK_SIZE: tl.constexpr,
):
    """Put: read local from_ptr, write remote to_ptr."""
    pid = tl.program_id(0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < N
    iris.put(local_ptr + offsets, remote_ptr + offsets, from_rank, to_rank, heap_bases, mask=mask)


@triton.jit
def get_kernel(
    remote_ptr,
    local_ptr,
    N,
    from_rank: tl.constexpr,
    to_rank: tl.constexpr,
    heap_bases: tl.tensor,
    BLOCK_SIZE: tl.constexpr,
):
    """Get: read remote from_ptr, write local to_ptr."""
    pid = tl.program_id(0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < N
    iris.get(remote_ptr + offsets, local_ptr + offsets, from_rank, to_rank, heap_bases, mask=mask)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def parse_args():
    parser = argparse.ArgumentParser(
        description="iris.put / iris.get example",
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

    # Symmetric-heap buffers.
    source = ctx.zeros(N, dtype=torch.float32)  # rank 0 fills this
    put_dst = ctx.zeros(N, dtype=torch.float32)  # target for iris.put
    get_dst = ctx.zeros(N, dtype=torch.float32)  # target for iris.get

    # Rank 0 initialises its source buffer.
    if rank == 0:
        source.copy_(torch.arange(N, dtype=torch.float32, device="cuda"))

    ctx.barrier()

    grid = (triton.cdiv(N, BLOCK_SIZE),)

    # --- Step 1: rank 0 puts data into rank 1's put_dst buffer --------------
    if rank == 0:
        put_kernel[grid](source, put_dst, N, 0, 1, heap_bases, BLOCK_SIZE)

    ctx.barrier()

    # --- Step 2: rank 1 gets data from rank 0's source buffer ---------------
    if rank == 1:
        get_kernel[grid](source, get_dst, N, 0, 1, heap_bases, BLOCK_SIZE)

    ctx.barrier()

    # --- Validation ---------------------------------------------------------
    if args["validate"]:
        expected = torch.arange(N, dtype=torch.float32, device="cuda")

        if rank == 1:
            # put_dst on rank 1 should contain rank 0's arange data
            torch.testing.assert_close(put_dst, expected, rtol=0, atol=0)
            print(f"[rank {rank}] iris.put validation PASSED")

            # get_dst on rank 1 should also contain rank 0's arange data
            torch.testing.assert_close(get_dst, expected, rtol=0, atol=0)
            print(f"[rank {rank}] iris.get validation PASSED")
    else:
        if rank == 0:
            print("Run with --validate to check correctness.")

    ctx.barrier()
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
