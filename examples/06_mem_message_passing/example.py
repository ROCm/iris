#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
# Copyright (c) 2025 Advanced Micro Devices, Inc. All rights reserved.
"""
Example: Flag-Based Message Passing (Producer-Consumer)
Demonstrates: iris.store() + iris.atomic_xchg() for signalling,
              iris.load() for data transfer and flag polling

Rank 0 (producer) stores a data payload into rank 1's buffer via
iris.store(), then signals readiness by setting a flag on rank 1 to 1
via iris.atomic_xchg() with release semantics.

Rank 1 (consumer) spins on the flag using iris.load() with volatile=True
until it reads 1, then reads the data from its own local buffer (the data
was already placed there by the producer), doubles it, and stores the
result locally.

This pattern is the building block for pipelined producer-consumer overlap
in fused GEMM + communication kernels.

Run with:
    torchrun --nproc_per_node=2 --standalone examples/06_mem_message_passing/example.py [--validate]
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
def producer_kernel(
    src_ptr,
    remote_data_ptr,
    remote_flag_ptr,
    N,
    heap_bases: tl.tensor,
    BLOCK_SIZE: tl.constexpr,
):
    """
    Producer (runs on rank 0):
      1. Store data into rank 1's buffer via iris.store().
      2. Signal flag=1 on rank 1 via iris.atomic_xchg() with release semantics.
    """
    pid = tl.program_id(0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < N

    # Load local source data
    data = tl.load(src_ptr + offsets, mask=mask)

    # Remote store: push data into rank 1's data buffer
    iris.store(remote_data_ptr + offsets, data, 0, 1, heap_bases, mask=mask)

    # Only the first program sets the flag (avoid redundant atomics).
    if pid == 0:
        one = tl.full([], 1, dtype=tl.int32)
        iris.atomic_xchg(remote_flag_ptr, one, 0, 1, heap_bases, sem="release", scope="sys")


@triton.jit
def consumer_kernel(
    data_ptr,
    result_ptr,
    flag_ptr,
    N,
    BLOCK_SIZE: tl.constexpr,
):
    """
    Consumer (runs on rank 1):
      1. Spin on local flag until it becomes 1.
      2. Read the data (already in local memory, deposited by producer).
      3. Multiply by 2 and store to result buffer.
    """
    pid = tl.program_id(0)

    # Only the first program spins on the flag; all programs wait for it.
    if pid == 0:
        # Spin-wait until flag is set to 1 by the producer.
        while tl.load(flag_ptr, volatile=True) != 1:
            pass

    # After the flag is visible the data is guaranteed to be visible
    # (the producer used release semantics on the flag write).
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
        description="Flag-based message passing example",
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

    assert world_size == 2, "This example requires exactly 2 ranks."

    # Symmetric-heap allocations.
    source = ctx.zeros(N, dtype=torch.float32)   # producer's data
    data_buf = ctx.zeros(N, dtype=torch.float32)   # consumer receives data here
    result = ctx.zeros(N, dtype=torch.float32)     # consumer writes 2*data here
    flag = ctx.zeros(1, dtype=torch.int32)         # signalling flag

    # Producer fills source.
    if rank == 0:
        source.copy_(torch.arange(N, dtype=torch.float32, device="cuda") + 1.0)

    ctx.barrier()

    grid = (triton.cdiv(N, BLOCK_SIZE),)

    if rank == 0:
        # Producer: store data remotely, then signal flag.
        producer_kernel[grid](source, data_buf, flag, N, heap_bases, BLOCK_SIZE)
    else:
        # Consumer: spin on flag, read data, compute result.
        consumer_kernel[grid](data_buf, result, flag, N, BLOCK_SIZE)

    ctx.barrier()

    # --- Validation ---------------------------------------------------------
    if args["validate"]:
        if rank == 1:
            expected = (torch.arange(N, dtype=torch.float32, device="cuda") + 1.0) * 2
            torch.testing.assert_close(result, expected, rtol=0, atol=0)
            print(f"[rank {rank}] message-passing validation PASSED")
    else:
        if rank == 0:
            print("Run with --validate to check correctness.")

    ctx.barrier()
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
