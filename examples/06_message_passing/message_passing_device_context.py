#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.

"""
Message Passing with DeviceContext API

This example demonstrates the DeviceContext API - an object-oriented interface
for Iris operations that follows the gluon pattern.

"""

import argparse
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
import triton
import triton.language as tl

import iris
from iris import DeviceContext


@triton.jit
def device_context_producer_kernel(
    context_tensor,  # Encoded context from iris.get_device_context()
    source_buffer,
    target_buffer,
    flag,
    buffer_size,
    rank: tl.constexpr,
    world_size: tl.constexpr,
    consumer_rank: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    """
    Producer kernel using DeviceContext API.

    Note how we don't need to pass heap_bases - it's encapsulated in DeviceContext.
    """
    # Initialize device context from encoded tensor
    ctx = DeviceContext.initialize(context_tensor, rank, world_size)

    pid = tl.program_id(0)

    # Compute start index of this block
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)

    # Guard for out-of-bounds accesses
    mask = offsets < buffer_size

    # Load from local buffer (no translation needed, so we just use tl.load)
    values = tl.load(source_buffer + offsets, mask=mask)

    # Store to remote buffer using DeviceContext (much cleaner API!)
    ctx.store(target_buffer + offsets, values, to_rank=consumer_rank, mask=mask)

    # Signal completion with atomic CAS
    ctx.atomic_cas(flag + pid, 0, 1, to_rank=consumer_rank, sem="release", scope="sys")


@triton.jit
def device_context_consumer_kernel(
    context_tensor,
    buffer,
    flag,
    buffer_size,
    rank: tl.constexpr,
    world_size: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    """Consumer kernel using DeviceContext API."""
    # Initialize device context from encoded tensor
    ctx = DeviceContext.initialize(context_tensor, rank, world_size)

    pid = tl.program_id(0)

    # Spin-wait on flag
    while ctx.atomic_cas(flag + pid, 1, 1, to_rank=rank, sem="acquire", scope="sys") != 1:
        pass

    # Process the data (just read and verify it exists)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < buffer_size

    # Load the received data
    data = tl.load(buffer + offsets, mask=mask)


def _worker(local_rank: int, world_size: int, init_url: str, args: dict):
    """Worker function for PyTorch distributed execution."""
    backend = "nccl" if torch.cuda.is_available() else "gloo"
    dist.init_process_group(
        backend=backend,
        init_method=init_url,
        world_size=world_size,
        rank=local_rank,
        device_id=torch.device(f"cuda:{local_rank}"),
    )

    # Initialize Iris
    ctx = iris.iris(heap_size=args["heap_size"])
    rank = ctx.get_rank()
    world_size = ctx.get_num_ranks()

    # Get device context tensor for kernels
    context_tensor = ctx.get_device_context()

    # Allocate buffers
    buffer_size = args["buffer_size"]
    block_size = args["block_size"]
    source_buffer = ctx.zeros(buffer_size, dtype=torch.float32)
    target_buffer = ctx.zeros(buffer_size, dtype=torch.float32)
    num_blocks = (buffer_size + block_size - 1) // block_size
    flag = ctx.zeros(num_blocks, dtype=torch.int32)

    # Initialize source buffer with data
    source_buffer.copy_(torch.arange(buffer_size, dtype=torch.float32))

    # Determine producer/consumer
    producer_rank = 0
    consumer_rank = 1 if world_size > 1 else 0

    ctx.barrier()

    if rank == producer_rank:
        ctx.info(f"Producer: Sending {buffer_size} elements to rank {consumer_rank}")

        # Launch producer kernel with DeviceContext
        device_context_producer_kernel[(num_blocks,)](
            context_tensor,
            source_buffer,
            target_buffer,
            flag,
            buffer_size,
            rank,
            world_size,
            consumer_rank,
            block_size,
        )

        ctx.info("Producer: Data sent successfully using DeviceContext API")

    if rank == consumer_rank:
        ctx.info(f"Consumer: Waiting for data from rank {producer_rank}")

        # Launch consumer kernel with DeviceContext
        device_context_consumer_kernel[(num_blocks,)](
            context_tensor,
            target_buffer,
            flag,
            buffer_size,
            rank,
            world_size,
            block_size,
        )

        # Verify the data
        expected = torch.arange(buffer_size, dtype=torch.float32, device=target_buffer.device)
        if torch.allclose(target_buffer, expected):
            ctx.info("Consumer: Data received and verified successfully using DeviceContext API!")
        else:
            ctx.error("Consumer: Data verification failed!")

    ctx.barrier()
    dist.destroy_process_group()


def main():
    parser = argparse.ArgumentParser(description="DeviceContext Message Passing Example")
    parser.add_argument("--buffer_size", type=int, default=1024, help="Buffer size")
    parser.add_argument("--block_size", type=int, default=256, help="Block size")
    parser.add_argument("--heap_size", type=int, default=1 << 30, help="Iris heap size (default: 1GB)")
    parser.add_argument("--num_ranks", type=int, default=2, help="Number of ranks/processes")
    args = vars(parser.parse_args())

    world_size = args["num_ranks"]
    init_url = "tcp://127.0.0.1:23456"

    print(f"Spawning {world_size} processes for DeviceContext example...")
    mp.spawn(_worker, args=(world_size, init_url, args), nprocs=world_size, join=True)
    print("DeviceContext example completed!")


if __name__ == "__main__":
    main()
