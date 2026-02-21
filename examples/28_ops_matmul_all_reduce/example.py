#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
# Copyright (c) 2025 Advanced Micro Devices, Inc. All rights reserved.

"""
Example: iris.ops.matmul_all_reduce

Fused GEMM + all-reduce: output = all_reduce(A @ B).

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
        description="Fused matmul + all-reduce example",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("-m", type=int, default=512, help="Rows of A")
    parser.add_argument("-n", type=int, default=128, help="Columns of B")
    parser.add_argument("-k", type=int, default=256, help="Inner dimension")
    parser.add_argument("--heap_size", type=int, default=1 << 31, help="Iris heap size")
    parser.add_argument("--datatype", type=str, default="fp16", choices=["fp16", "fp32", "bf16"], help="Data type")
    parser.add_argument("-v", "--validate", action="store_true", help="Validate output against reference")
    return vars(parser.parse_args())


def main():
    args = parse_args()

    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    torch.cuda.set_device(local_rank)
    dist.init_process_group(backend="nccl")

    ctx = iris.iris(heap_size=args["heap_size"])
    rank = ctx.get_rank()
    world_size = ctx.get_num_ranks()

    dtype_map = {"fp16": torch.float16, "fp32": torch.float32, "bf16": torch.bfloat16}
    dtype = dtype_map[args["datatype"]]
    M, K, N = args["m"], args["k"], args["n"]

    torch.manual_seed(42)
    A = ctx.randn((M, K), dtype=dtype)
    B = ctx.randn((K, N), dtype=dtype)
    output = ctx.zeros((M, N), dtype=dtype)

    ctx.barrier()
    ctx.ops.matmul_all_reduce(output, A, B)
    torch.cuda.synchronize()

    if rank == 0:
        ctx.info(f"matmul_all_reduce: world_size={world_size}, A=({M},{K}), B=({K},{N}), dtype={dtype}")

    if args["validate"]:
        # Each rank computes the same GEMM; all-reduce sums world_size copies
        ref = torch.matmul(A.clone().float(), B.clone().float()).to(dtype) * world_size
        assert torch.allclose(output.float(), ref.float(), atol=1.0, rtol=0.05), (
            f"Rank {rank}: mismatch. Max diff: {(output.float() - ref.float()).abs().max().item():.4f}"
        )
        if rank == 0:
            ctx.info(f"Validation passed: output[0,0] = {output[0, 0].item():.4f}")

    ctx.barrier()
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
