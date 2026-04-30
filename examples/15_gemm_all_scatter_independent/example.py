#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
# Copyright (c) 2025 Advanced Micro Devices, Inc. All rights reserved.
"""
Example: Independent GEMM + All-Scatter (N-split)

Demonstrates: Each rank computes full A x partial B (N-split) using a GEMM
kernel, then a separate all-scatter kernel sends completed tiles to all remote
ranks via iris.put. GEMM and scatter run as independent kernels on the same
stream (bulk synchronous). Optionally reads configs from a CSV file.

Run with:
    torchrun --nproc_per_node=2 --standalone example.py --validate
    torchrun --nproc_per_node=8 --standalone example.py --csv example_config.csv
"""
import argparse
import csv
import math
import os

import torch
import torch.distributed as dist
import triton

from matmul_wrapper import matmul
from gemm_all_scatter_bulk_synchronous import persistent_all_scatter

import iris


def parse_args():
    parser = argparse.ArgumentParser(
        description="Independent GEMM + All-Scatter example.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("-m", type=int, default=8192, help="Rows of A")
    parser.add_argument("-n", type=int, default=4608, help="Columns of B")
    parser.add_argument("-k", type=int, default=36864, help="Common dimension")
    parser.add_argument("-v", "--validate", action="store_true", help="Validate output")
    parser.add_argument(
        "--datatype",
        type=str,
        default="fp16",
        choices=["fp16", "fp32", "bf16"],
        help="Datatype of computation",
    )
    parser.add_argument("--BLK_M", type=int, default=256, help="Block size M")
    parser.add_argument("--BLK_N", type=int, default=64, help="Block size N")
    parser.add_argument("--BLK_K", type=int, default=64, help="Block size K")
    parser.add_argument("--gsize_m", type=int, default=6, help="L2-cache locality swizzle parameter")
    parser.add_argument("--num_stages", type=int, default=2, help="Number of stages")
    parser.add_argument("--comm_sms", type=int, default=48, help="SMs for communication kernel")
    parser.add_argument("--heap_size", type=int, default=1 << 33, help="Iris heap size")
    parser.add_argument("--csv", type=str, default=None, help="CSV file with per-run configs")
    parser.add_argument("--only_gemm", action="store_true", help="Run only GEMM (skip scatter)")
    parser.add_argument("--only_comm", action="store_true", help="Run only scatter (skip GEMM)")
    return vars(parser.parse_args())


def run_config(ctx, args, rank, world_size, cu_count):
    """Run one GEMM + all-scatter iteration with the given config."""
    dtype_map = {"fp16": torch.float16, "fp32": torch.float32, "bf16": torch.bfloat16}
    datatype = dtype_map[args["datatype"]]
    num_xcds = iris.hip.get_num_xcc()

    M, N, K = args["m"], args["n"], args["k"]
    gemm_sms = args.get("gemm_sms", 2 ** int(math.log2(cu_count)) if cu_count > 0 else 1)
    comm_sms = args["comm_sms"]
    only_gemm = args.get("only_gemm", False)
    only_comm = args.get("only_comm", False)

    assert N % world_size == 0, f"N ({N}) must be divisible by world size ({world_size})."

    # Allocate matrices
    A = ctx.randn(M, K, device="cuda", dtype=datatype)
    B = ctx.randn(N, K, device="cuda", dtype=datatype).T

    local_n = N // world_size
    local_B = B[:, rank * local_n : (rank + 1) * local_n].clone()

    # C is (M, N) -- GEMM writes local columns, scatter distributes to all ranks
    C = ctx.zeros((M, N), device="cuda", dtype=datatype)

    main_stream = torch.cuda.Stream()

    def run():
        ctx.barrier()
        with torch.cuda.stream(main_stream):
            if not only_comm:
                # GEMM: full A x local B -> columns [0, local_n) of C
                matmul._call(
                    A, local_B, C, None,
                    rank, world_size, gemm_sms,
                    args["BLK_M"], args["BLK_N"], args["BLK_K"],
                    args["gsize_m"], args["num_stages"],
                    ctx.get_heap_bases(), "gfx942",
                )
            if not only_gemm:
                # Scatter: put local columns to all remote ranks
                persistent_all_scatter[(comm_sms,)](
                    C,
                    M,
                    local_n,
                    C.stride(0),
                    C.stride(1),
                    args["BLK_M"],
                    args["BLK_N"],
                    args["gsize_m"],
                    comm_sms,
                    num_xcds,
                    ctx.get_heap_bases(),
                    rank,
                    world_size,
                )
        torch.cuda.synchronize()
        ctx.barrier()

    # Warmup
    run()

    # Run for validation
    run()

    if args.get("validate") and not only_gemm and not only_comm:
        expected = A @ B
        if torch.allclose(C, expected, atol=2.0):
            if rank == 0:
                print(f"Validation PASSED (M={M}, N={N}, K={K})")
        else:
            max_diff = (C - expected).abs().max().item()
            print(f"Rank {rank}: Validation FAILED (M={M}, N={N}, K={K}, max diff: {max_diff})")


def main():
    args = parse_args()
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    torch.cuda.set_device(local_rank)
    dist.init_process_group(backend="gloo")

    ctx = iris.iris(heap_size=args["heap_size"])
    rank = ctx.get_rank()
    world_size = ctx.get_num_ranks()

    cu_count = torch.cuda.get_device_properties(local_rank).multi_processor_count

    if args["csv"]:
        with open(args["csv"]) as f:
            reader = csv.DictReader(f)
            for row in reader:
                config = dict(args)
                config["m"] = int(row["m"])
                config["n"] = int(row["n"])
                config["k"] = int(row["k"])
                config["datatype"] = row.get("datatype", args["datatype"])
                config["BLK_M"] = int(row.get("blk_m", args["BLK_M"]))
                config["BLK_N"] = int(row.get("blk_n", args["BLK_N"]))
                config["BLK_K"] = int(row.get("blk_k", args["BLK_K"]))
                config["gemm_sms"] = int(row.get("gemm_sms", 2 ** int(math.log2(cu_count)) if cu_count > 0 else 1))
                config["comm_sms"] = int(row.get("comm_sms", args["comm_sms"]))
                run_config(ctx, config, rank, world_size, cu_count)
    else:
        run_config(ctx, args, rank, world_size, cu_count)

    ctx.barrier()
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
