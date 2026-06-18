#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.

"""
Example: producer-consumer message passing with a single unified kernel.

Same semantics as example 31 (message_passing) but uses one kernel that
branches internally on cur_rank. This produces a single kernel dispatch
per rank, which is required for downstream capture tools (e.g. GUMMi).

Producer rank writes source data to consumer's buffer and signals via flag.
Consumer rank spin-waits on flag then reads and doubles the data.
Requires exactly 2 ranks.

Run with:
    torchrun --nproc_per_node=2 --standalone example.py [--validate]
"""

import argparse
import os
import random

import torch
import torch.distributed as dist
import triton
import triton.language as tl

import iris
from iris.host.platform.utils import is_simulation_env


@triton.jit
def message_passing_kernel(
    source_buffer,
    target_buffer,
    flag,
    buffer_size,
    cur_rank,
    producer_rank: tl.constexpr,
    consumer_rank: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
    heap_bases_ptr: tl.tensor,
):
    pid = tl.program_id(0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < buffer_size

    is_producer = cur_rank == producer_rank

    if is_producer:
        values = iris.load(source_buffer + offsets, producer_rank, producer_rank, heap_bases_ptr, mask=mask)
        iris.store(target_buffer + offsets, values, producer_rank, consumer_rank, heap_bases_ptr, mask=mask)
        iris.atomic_cas(flag + pid, 0, 1, producer_rank, consumer_rank, heap_bases_ptr, sem="release", scope="sys")
    else:
        done = 0
        while done == 0:
            done = iris.atomic_cas(
                flag + pid, 1, 0, consumer_rank, consumer_rank, heap_bases_ptr, sem="acquire", scope="sys"
            )
        values = iris.load(target_buffer + offsets, consumer_rank, consumer_rank, heap_bases_ptr, mask=mask)
        values = values * 2
        iris.store(target_buffer + offsets, values, consumer_rank, consumer_rank, heap_bases_ptr, mask=mask)


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
        raise ValueError(f"Unknown datatype: {datatype}")


def parse_args():
    parser = argparse.ArgumentParser(
        description="Message passing producer-consumer example with single kernel (2 ranks).",
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
    parser.add_argument("-s", "--buffer_size", type=int, default=4096, help="Buffer size")
    parser.add_argument("-b", "--block_size", type=int, default=512, help="Block size")
    parser.add_argument("--heap_size", type=int, default=1 << 16, help="Iris heap size")
    parser.add_argument("-v", "--validate", action="store_true", help="Validate output against reference")
    return vars(parser.parse_args())


def main():
    args = parse_args()

    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    torch.cuda.set_device(local_rank)
    backend = "gloo" if is_simulation_env() else "nccl"
    dist.init_process_group(backend=backend)

    ctx = iris.iris(heap_size=args["heap_size"])
    cur_rank = ctx.get_rank()
    world_size = ctx.get_num_ranks()

    if world_size != 2:
        raise ValueError("This example requires exactly two processes. Use: torchrun --nproc_per_node=2 ...")

    dtype = torch_dtype_from_str(args["datatype"])
    producer_rank = 0
    consumer_rank = 1

    source_buffer = ctx.zeros(args["buffer_size"], device="cuda", dtype=dtype)
    if dtype.is_floating_point:
        destination_buffer = ctx.randn(args["buffer_size"], device="cuda", dtype=dtype)
    else:
        ii = torch.iinfo(dtype)
        destination_buffer = ctx.randint(ii.min, ii.max, (args["buffer_size"],), device="cuda", dtype=dtype)

    n_elements = source_buffer.numel()
    grid = lambda meta: (triton.cdiv(n_elements, meta["BLOCK_SIZE"]),)
    num_blocks = triton.cdiv(n_elements, args["block_size"])
    flags = ctx.zeros((num_blocks,), device="cuda", dtype=torch.int32)

    heap_bases = ctx.get_heap_bases()

    ctx.info(f"Rank {cur_rank} launching message_passing_kernel.")
    message_passing_kernel[grid](
        source_buffer,
        destination_buffer,
        flags,
        n_elements,
        cur_rank,
        producer_rank,
        consumer_rank,
        args["block_size"],
        heap_bases,
    )

    ctx.barrier()
    ctx.info(f"Rank {cur_rank} has finished.")

    if args["validate"]:
        ctx.info("Validating output...")
        if cur_rank == consumer_rank:
            expected = source_buffer * 2
            if not torch.allclose(destination_buffer, expected, atol=1):
                max_diff = (destination_buffer - expected).abs().max().item()
                ctx.error(f"Validation failed. Max absolute difference: {max_diff}")
            else:
                ctx.info("Validation successful.")

    ctx.barrier()
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
