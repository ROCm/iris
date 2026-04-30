#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
# Copyright (c) 2025 Advanced Micro Devices, Inc. All rights reserved.
"""
Example: Device Context OO API (Producer-Consumer)
Demonstrates: iris.mem.triton.Context (object-oriented wrapper)

The Context API bundles rank and heap_bases into a device-side object so
that every iris operation only needs the target rank -- no heap_bases
parameter.  This example implements the same producer-consumer pattern as
06_mem_message_passing but uses the cleaner Context API.

Host side:
    device_ctx = shmem.get_device_context()   # returns a uint64 tensor
    # pass device_ctx as a kernel argument

Device side:
    ctx = Context.initialize(device_ctx, rank, world_size)
    ctx.store(ptr, value, to_rank=1, mask=mask)
    ctx.load(ptr, from_rank=0, mask=mask)
    ctx.atomic_xchg(ptr, val, to_rank=1, sem="release", scope="sys")

Run with:
    torchrun --nproc_per_node=2 --standalone examples/07_mem_context/example.py [--validate]
"""

import argparse
import os

import torch
import torch.distributed as dist
import triton
import triton.language as tl

import iris
from iris.mem.triton.context import Context


# ---------------------------------------------------------------------------
# Triton kernels
# ---------------------------------------------------------------------------

@triton.jit
def producer_kernel(
    device_ctx,
    src_ptr,
    remote_data_ptr,
    remote_flag_ptr,
    N,
    rank: tl.constexpr,
    world_size: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    """
    Producer (rank 0): store data into rank 1's buffer via ctx.store(),
    then signal flag=1 via ctx.atomic_xchg() with release semantics.
    """
    ctx = Context.initialize(device_ctx, rank, world_size)

    pid = tl.program_id(0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < N

    # Load local source data (local load, no translation needed)
    data = tl.load(src_ptr + offsets, mask=mask)

    # Remote store into rank 1's data buffer
    ctx.store(remote_data_ptr + offsets, data, to_rank=1, mask=mask)

    # Only the first program sets the flag.
    if pid == 0:
        one = tl.full([], 1, dtype=tl.int32)
        ctx.atomic_xchg(remote_flag_ptr, one, to_rank=1, sem="release", scope="sys")


@triton.jit
def consumer_kernel(
    device_ctx,
    data_ptr,
    result_ptr,
    flag_ptr,
    N,
    rank: tl.constexpr,
    world_size: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    """
    Consumer (rank 1): spin on local flag until producer signals, then
    read data from local buffer and double it.
    """
    # Initialize context (even though this kernel only does local ops after
    # the spin, initialising the context is required for ctx.load if you
    # wanted to pull from a remote rank instead).
    _ctx = Context.initialize(device_ctx, rank, world_size)  # noqa: F841

    pid = tl.program_id(0)

    # Spin until the flag is set to 1 by the producer.
    if pid == 0:
        while tl.load(flag_ptr, volatile=True) != 1:
            pass

    # Data was placed in our local buffer by the producer's ctx.store().
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < N
    data = tl.load(data_ptr + offsets, mask=mask)
    result = data * 2
    tl.store(result_ptr + offsets, result, mask=mask)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args():
    parser = argparse.ArgumentParser(
        description="Context OO API message-passing example",
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

    shmem = iris.iris(heap_size=args["heap_size"])
    rank = shmem.get_rank()
    world_size = shmem.get_num_ranks()

    assert world_size == 2, "This example requires exactly 2 ranks."

    # Get the encoded device context tensor (uint64 array containing rank,
    # world_size, and heap base pointers for every rank).
    device_ctx = shmem.get_device_context()

    # Symmetric-heap allocations.
    source = shmem.zeros(N, dtype=torch.float32)
    data_buf = shmem.zeros(N, dtype=torch.float32)
    result = shmem.zeros(N, dtype=torch.float32)
    flag = shmem.zeros(1, dtype=torch.int32)

    # Producer fills source.
    if rank == 0:
        source.copy_(torch.arange(N, dtype=torch.float32, device="cuda") + 1.0)

    shmem.barrier()

    grid = (triton.cdiv(N, BLOCK_SIZE),)

    if rank == 0:
        producer_kernel[grid](
            device_ctx, source, data_buf, flag, N, rank, world_size, BLOCK_SIZE,
        )
    else:
        consumer_kernel[grid](
            device_ctx, data_buf, result, flag, N, rank, world_size, BLOCK_SIZE,
        )

    shmem.barrier()

    # --- Validation ---------------------------------------------------------
    if args["validate"]:
        if rank == 1:
            expected = (torch.arange(N, dtype=torch.float32, device="cuda") + 1.0) * 2
            torch.testing.assert_close(result, expected, rtol=0, atol=0)
            print(f"[rank {rank}] Context API message-passing validation PASSED")
    else:
        if rank == 0:
            print("Run with --validate to check correctness.")

    shmem.barrier()
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
