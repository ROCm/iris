#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
# Copyright (c) 2025 Advanced Micro Devices, Inc. All rights reserved.

"""
Example: iris.ops.matmul_all_gather

Fused GEMM + all-gather along M: output = all_gather(A_local @ B).
A is row-sharded across ranks; every rank gets the full (M, N) output.

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
        description="Fused matmul + all-gather example",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("-m", type=int, default=4096, help="Total rows (must be divisible by world_size)")
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

    if M % world_size != 0:
        raise ValueError(
            f"M ({M}) must be divisible by world_size ({world_size}). Please adjust -m to be a multiple of {world_size}."
        )
    M_local = M // world_size

    torch.manual_seed(42 + rank)
    A_local = ctx.randn((M_local, K), dtype=dtype)
    torch.manual_seed(0)
    B = ctx.randn((K, N), dtype=dtype)
    output = ctx.zeros((M, N), dtype=dtype)

    ctx.barrier()
    ctx.ops.matmul_all_gather(output, A_local, B)
    torch.cuda.synchronize()

    if rank == 0:
        ctx.info(f"matmul_all_gather: world_size={world_size}, A_local=({M_local},{K}), B=({K},{N}), dtype={dtype}")

    if args["validate"]:
        C_local = torch.matmul(A_local.float(), B.clone().float()).to(dtype)
        C_shards = [torch.zeros(M_local, N, dtype=dtype, device=C_local.device) for _ in range(world_size)]
        dist.all_gather(C_shards, C_local)
        ref = torch.cat(C_shards, dim=0)
        assert torch.allclose(output.float(), ref.float(), atol=1.0, rtol=0.05), (
            f"Rank {rank}: mismatch. Max diff: {(output.float() - ref.float()).abs().max().item():.4f}"
        )
        if rank == 0:
            ctx.info(f"Validation passed: output[0,0] = {output[0, 0].item():.4f}")

    ctx.barrier()
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
