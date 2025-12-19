# SPDX-License-Identifier: MIT
# Copyright (c) 2025 Advanced Micro Devices, Inc. All rights reserved.

import argparse

import torch
import torch.distributed as dist
import torch.multiprocessing as mp
import triton
import triton.language as tl
import random

import iris


@triton.jit
def producer_kernel(
    source_buffer,  # tl.tensor: pointer to source data
    target_buffer,  # tl.tensor: pointer to target data
    flag,  # tl.tensor: pointer to flags
    buffer_size,  # int32: total number of elements
    producer_rank: tl.constexpr,
    consumer_rank: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
    heap_bases_ptr: tl.tensor,  # tl.tensor: pointer to heap bases pointers
):
    pid = tl.program_id(0)

    # Compute start index of this block
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)

    # Guard for out-of-bounds accesses
    mask = offsets < buffer_size

    # Put chunk into remote buffer
    iris.put(source_buffer + offsets, target_buffer + offsets, producer_rank, consumer_rank, heap_bases_ptr, mask=mask)

    # Set flag to signal completion
    iris.atomic_cas(flag + pid, 0, 1, producer_rank, consumer_rank, heap_bases_ptr, sem="release", scope="sys")


@triton.jit
def consumer_kernel(
    buffer,  # tl.tensor: pointer to shared buffer (read from target_rank)
    flag,  # tl.tensor: sync flag per block
    buffer_size,  # int32: total number of elements
    consumer_rank: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
    heap_bases_ptr: tl.tensor,  # tl.tensor: pointer to heap bases pointers
):
    pid = tl.program_id(0)

    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < buffer_size

    # Spin-wait until writer sets flag[pid] = 1
    done = 0
    while done == 0:
        done = iris.atomic_cas(
            flag + pid, 1, 0, consumer_rank, consumer_rank, heap_bases_ptr, sem="acquire", scope="sys"
        )

    # Read from the target buffer (written by producer)
    values = tl.load(buffer + offsets, mask=mask)

    # Do something with values...
    # (Here you might write to output, do computation, etc.)
    values = values * 2

    # Store chunk to target buffer
    tl.store(
        buffer + offsets,
        values,
        mask=mask,
    )

    # Optionally reset the flag for next iteration
    tl.store(flag + pid, 0)


torch.manual_seed(123)
random.seed(123)


def torch_dtype_from_str(datatype: str) -> torch.dtype:
    dtype_map = {
        "fp16": torch.float16,
        "fp32": torch.float32,
        "int8": torch.int8,
        "bf16": torch.bfloat16,
    }
    try:
        return dtype_map[datatype]
    except KeyError:
        print(f"Unknown datatype: {datatype}")
        exit(1)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Parse Message Passing configuration.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "-t",
        "--datatype",
        type=str,
        default="fp32",
        choices=["fp16", "fp32", "int8", "bf16"],
        help="Datatype of computation",
    )
    parser.add_argument("-s", "--buffer_size", type=int, default=4096, help="Buffer Size")
    parser.add_argument("-b", "--block_size", type=int, default=512, help="Block Size")

    parser.add_argument("-p", "--heap_size", type=int, default=1 << 33, help="Iris heap size")
    parser.add_argument("-r", "--num_ranks", type=int, default=2, help="Number of ranks/processes")

    return vars(parser.parse_args())


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

    # Main benchmark logic
    shmem = iris.iris(args["heap_size"])
    dtype = torch_dtype_from_str(args["datatype"])
    cur_rank = shmem.get_rank()
    world_size = shmem.get_num_ranks()

    # Allocate source and destination buffers on the symmetric heap
    source_buffer = shmem.zeros(args["buffer_size"], device="cuda", dtype=dtype)
    if dtype.is_floating_point:
        destination_buffer = shmem.randn(args["buffer_size"], device="cuda", dtype=dtype)
    else:
        ii = torch.iinfo(dtype)
        destination_buffer = shmem.randint(ii.min, ii.max, (args["buffer_size"],), device="cuda", dtype=dtype)

    if world_size != 2:
        raise ValueError("This example requires exactly two processes.")

    producer_rank = 0
    consumer_rank = 1

    n_elements = source_buffer.numel()
    grid = lambda meta: (triton.cdiv(n_elements, meta["BLOCK_SIZE"]),)
    num_blocks = triton.cdiv(n_elements, args["block_size"])

    # Allocate flags on the symmetric heap
    flags = shmem.zeros((num_blocks,), device="cuda", dtype=torch.int32)

    if cur_rank == producer_rank:
        shmem.info(f"Rank {cur_rank} is sending data to rank {consumer_rank}.")
        kk = producer_kernel[grid](
            source_buffer,
            destination_buffer,
            flags,
            n_elements,
            producer_rank,
            consumer_rank,
            args["block_size"],
            shmem.get_heap_bases(),
        )
    else:
        shmem.info(f"Rank {cur_rank} is receiving data from rank {producer_rank}.")
        kk = consumer_kernel[grid](
            destination_buffer, flags, n_elements, consumer_rank, args["block_size"], shmem.get_heap_bases()
        )
    shmem.barrier()
    shmem.info(f"Rank {cur_rank} has finished sending/receiving data.")
    shmem.info("Validating output...")

    success = True
    if cur_rank == consumer_rank:
        expected = source_buffer * 2
        diff_mask = ~torch.isclose(destination_buffer, expected, atol=1)
        breaking_indices = torch.nonzero(diff_mask, as_tuple=False)

        if not torch.allclose(destination_buffer, expected, atol=1):
            max_diff = (destination_buffer - expected).abs().max().item()
            shmem.info(f"Max absolute difference: {max_diff}")
            for idx in breaking_indices:
                idx = tuple(idx.tolist())
                computed_val = destination_buffer[idx]
                expected_val = expected[idx]
                shmem.info(f"Mismatch at index {idx}: C={computed_val}, expected={expected_val}")
                success = False
                break

        if success:
            shmem.info("Validation successful.")
        else:
            shmem.info("Validation failed.")

    shmem.barrier()

    dist.barrier()
    dist.destroy_process_group()


def main():
    args = parse_args()

    num_ranks = args["num_ranks"]

    init_url = "tcp://127.0.0.1:29500"
    mp.spawn(
        fn=_worker,
        args=(num_ranks, init_url, args),
        nprocs=num_ranks,
        join=True,
    )


if __name__ == "__main__":
    main()
