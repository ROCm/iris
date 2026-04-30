#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
# Copyright (c) 2025 Advanced Micro Devices, Inc. All rights reserved.
"""
Example: Ring-Based All-Reduce (standalone, no GEMM)

Demonstrates: Each rank has a partial result matrix. The ring-based all-reduce
kernel sums all partial results across ranks using a ring topology with
iris.store for data transfer and iris.atomic_cas/atomic_xchg for signaling.

Run with:
    torchrun --nproc_per_node=2 --standalone example.py --validate
"""

import argparse
import os

import torch
import torch.distributed as dist
import triton

from all_reduce_ring_based import persistent_all_reduce

import iris


def parse_args():
    parser = argparse.ArgumentParser(
        description="Ring-Based All-Reduce example.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("-m", type=int, default=8192, help="Number of rows in input/output matrix")
    parser.add_argument("-n", type=int, default=4608, help="Number of columns in input/output matrix")
    parser.add_argument("-v", "--validate", action="store_true", help="Enable validation mode")
    parser.add_argument(
        "--datatype",
        type=str,
        default="fp16",
        choices=["fp16", "fp32", "bf16"],
        help="Datatype of computation",
    )
    parser.add_argument("--BLK_M", type=int, default=128, help="Block size M")
    parser.add_argument("--BLK_N", type=int, default=128, help="Block size N")
    parser.add_argument("--gsize_m", type=int, default=6, help="L2-cache locality swizzle parameter")
    parser.add_argument("--num_sms", type=int, default=48, help="Number of SMs for All-Reduce kernel")
    parser.add_argument("--heap_size", type=int, default=1 << 33, help="Iris heap size")
    return vars(parser.parse_args())


def main():
    args = parse_args()
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    torch.cuda.set_device(local_rank)
    dist.init_process_group(backend="gloo")

    ctx = iris.iris(heap_size=args["heap_size"])
    rank = ctx.get_rank()
    world_size = ctx.get_num_ranks()
    num_xcds = iris.hip.get_num_xcc()

    # dtype
    dtype_map = {"fp16": torch.float16, "fp32": torch.float32, "bf16": torch.bfloat16}
    datatype = dtype_map[args["datatype"]]

    M = args["m"]
    N = args["n"]

    # Initialize partial with random data for each rank
    torch.manual_seed(123 + rank)
    partial = ctx.zeros((M, N), device="cuda", dtype=datatype)
    partial.copy_(torch.randn((M, N), device="cuda", dtype=datatype))

    output = ctx.zeros((M, N), device="cuda", dtype=datatype)

    total_blocks_M = triton.cdiv(M, args["BLK_M"])
    total_blocks_N = triton.cdiv(N, args["BLK_N"])
    total_tiles = total_blocks_M * total_blocks_N

    flags = ctx.zeros((total_tiles,), device="cuda", dtype=torch.int32)
    ring_buffer = ctx.zeros_like(partial, dtype=torch.float32)

    num_sms = args["num_sms"]

    def reset_buffers():
        ctx.barrier()
        flags.zero_()
        ring_buffer.zero_()
        ctx.barrier()

    def run():
        ctx.barrier()
        persistent_all_reduce[(num_sms,)](
            partial,
            ring_buffer,
            output,
            flags,
            M,
            N,
            output.stride(0),
            output.stride(1),
            args["BLK_M"],
            args["BLK_N"],
            args["gsize_m"],
            num_sms,
            num_xcds,
            ctx.get_heap_bases(),
            rank,
            world_size,
        )
        torch.cuda.synchronize()
        ctx.barrier()

    # Warmup
    run()
    reset_buffers()

    # Run for validation
    run()

    # Validate
    if args["validate"]:
        # Compute expected result using torch.distributed.all_reduce
        expected_output = partial.clone()
        dist.all_reduce(expected_output, op=dist.ReduceOp.SUM)

        if torch.allclose(output, expected_output, atol=2.0):
            if rank == 0:
                print("Validation PASSED")
        else:
            max_diff = (output - expected_output).abs().max().item()
            print(f"Rank {rank}: Validation FAILED (max diff: {max_diff})")

    ctx.barrier()
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
