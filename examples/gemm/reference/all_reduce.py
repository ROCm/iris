#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
# Copyright (c) 2025 Advanced Micro Devices, Inc. All rights reserved.


import torch
import torch.distributed as dist
import random
import iris
import argparse

from examples.gemm.utils import JSONWriter

torch.manual_seed(123)
random.seed(123)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Parse matrix dimensions and configuration.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("-m", type=int, default=8192, help="Number of rows in matrix A")
    parser.add_argument("-n", type=int, default=8192, help="Number of columns in matrix B")
    parser.add_argument("-k", type=int, default=30720, help="Common dimension between matrices A and B")
    parser.add_argument("-v", "--validate", action="store_true", help="Enable validation mode")
    parser.add_argument("-b", "--benchmark", action="store_true", help="Enable benchmarking mode")
    parser.add_argument(
        "-d",
        "--datatype",
        type=str,
        default="fp16",
        choices=["fp16", "fp32", "bf16"],
        help="Datatype of computation",
    )
    parser.add_argument(
        "-o",
        "--output_file",
        type=str,
        default="log.json",
        help="Output file",
    )
    return vars(parser.parse_args())


def main():
    args = parse_args()

    m = args["m"]
    n = args["n"]
    k = args["k"]
    validate = args["validate"]
    benchmark = args["benchmark"]

    dist.init_process_group("nccl")

    rank = dist.get_rank()
    world_size = dist.get_world_size()
    torch.cuda.set_device(rank)

    dtype_map = {
        "fp16": torch.float16,
        "fp32": torch.float32,
        "bf16": torch.bfloat16,
    }
    dtype = dtype_map[args["datatype"]]

    args["M"] = args["m"]
    args["N"] = args["n"]
    args["K"] = args["k"]

    json_writer = JSONWriter(args["output_file"])
    json_writer.add_field("world_size", world_size)

    print(f"Starting distributed GEMM on Rank {rank} of {world_size} on device cuda:{rank}")

    A_full = torch.randn(m, k, device=f"cuda:{rank}", dtype=dtype)
    B_full = torch.randn(k, n, device=f"cuda:{rank}", dtype=dtype)
    rows_per_gpu = k // world_size
    start_row = rank * rows_per_gpu
    end_row = start_row + rows_per_gpu
    B_local = B_full[start_row:end_row, :]
    A_local = A_full[:, start_row:end_row]
    C_global = torch.zeros((m, n), device=f"cuda:{rank}", dtype=dtype)

    args["k"] = args["k"] // world_size

    for key, value in args.items():
        json_writer.add_field(key, value)

    kernel_timing = {
        "gemm": {
            "start_event": torch.cuda.Event(enable_timing=True),
            "end_event": torch.cuda.Event(enable_timing=True),
            "ms": 0,
            "experiments": 0,
        },
        "communication": {
            "start_event": torch.cuda.Event(enable_timing=True),
            "end_event": torch.cuda.Event(enable_timing=True),
            "ms": 0,
            "experiments": 0,
        },
    }

    gemm_stream = torch.cuda.Stream()
    comm_stream = torch.cuda.Stream()

    def run_experiment():
        global C_partial
        nonlocal kernel_timing

        torch.cuda.nvtx.range_push("GEMM + Communication")
        torch.cuda.nvtx.range_push("GEMM")

        with torch.cuda.stream(gemm_stream):
            kernel_timing["gemm"]["start_event"].record()
            C_partial = A_local @ B_local
            kernel_timing["gemm"]["end_event"].record()
            kernel_timing["gemm"]["experiments"] += 1

        gemm_stream.synchronize()
        torch.cuda.nvtx.range_pop()
        torch.cuda.nvtx.range_push("Communication")

        with torch.cuda.stream(comm_stream):
            kernel_timing["communication"]["start_event"].record()
            dist.all_reduce(C_partial, op=dist.ReduceOp.SUM)
            kernel_timing["communication"]["end_event"].record()
            kernel_timing["communication"]["experiments"] += 1
        comm_stream.synchronize()
        torch.cuda.nvtx.range_pop()
        torch.cuda.nvtx.range_pop()

        for k in ["gemm", "communication"]:
            ms = kernel_timing[k]["start_event"].elapsed_time(kernel_timing[k]["end_event"])
            kernel_timing[k]["ms"] += ms

    run_experiment()

    for k in ["gemm", "communication"]:
        kernel_timing[k]["ms"] = 0
        kernel_timing[k]["experiments"] = 0

    C_global.copy_(C_partial)

    # Validation step
    if validate:
        C_full = A_full @ B_full
        valid = torch.allclose(C_global, C_full, atol=2)
        max_diff = torch.max(torch.abs(C_global - C_full))
        if valid:
            print(f"Rank {rank}: Validation passed! Distributed GEMM matches full GEMM.")
        else:
            print(f"Rank {rank}: Validation failed! Results do not match.")
            print(f"Max difference: {max_diff}")
    dist.barrier()

    if benchmark:
        perf = lambda ms: 2 * args["M"] * args["N"] * args["K"] * 1e-12 / (ms * 1e-3)
        ms = iris.do_bench(run_experiment, dist.barrier)
        flops = perf(ms)
        print(f"Rank {rank}: {ms:.3f} ms  {perf(ms):.3f} tflops")
        json_writer.add_field("ms", ms)
        json_writer.add_field("flops", flops)

        for k in ["gemm", "communication"]:
            json_writer.add_field(k + "_ms", kernel_timing[k]["ms"] / kernel_timing[k]["experiments"])
            json_writer.add_field(k + "_experiments", kernel_timing[k]["experiments"])

    dist.barrier()

    if rank == 0:
        json_writer.add_field("algorithm", "torch_dist_all_reduce")
        json_writer.flush()
        json_writer.display()

    dist.destroy_process_group()


if __name__ == "__main__":
    main()
