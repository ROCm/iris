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
        description="Distributed GEMM with reduce_scatter_tensor",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("-m", type=int, default=8192, help="Number of rows in matrix A")
    parser.add_argument("-n", type=int, default=8192, help="Number of columns in matrix B")
    parser.add_argument("-k", type=int, default=30720, help="Shared dimension")
    parser.add_argument("-v", "--validate", action="store_true", help="Enable validation")
    parser.add_argument("-b", "--benchmark", action="store_true", help="Enable benchmarking")
    parser.add_argument(
        "-d",
        "--datatype",
        type=str,
        default="fp16",
        choices=["fp16", "fp32", "bf16"],
        help="Datatype",
    )
    parser.add_argument("-o", "--output_file", type=str, default="log.json", help="Output file")
    return vars(parser.parse_args())


def main():
    args = parse_args()

    m, n, k = args["m"], args["n"], args["k"]
    validate, benchmark = args["validate"], args["benchmark"]

    dist.init_process_group("nccl")
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    torch.cuda.set_device(rank)

    args["M"], args["N"], args["K"] = m, n, k

    dtype_map = {
        "fp16": torch.float16,
        "fp32": torch.float32,
        "bf16": torch.bfloat16,
    }
    dtype = dtype_map[args["datatype"]]

    json_writer = JSONWriter(args["output_file"])
    json_writer.add_field("world_size", world_size)

    print(f"Starting distributed GEMM on Rank {rank} of {world_size} on device cuda:{rank}")

    # Shared input
    A = torch.randn(m, k, device=f"cuda:{rank}", dtype=dtype)
    B = torch.randn(k, n, device=f"cuda:{rank}", dtype=dtype)

    cols_per_rank = n // world_size
    start = rank * cols_per_rank
    end = start + cols_per_rank
    B_local = B[:, start:end]  # [k, n/world]

    # Partial GEMM
    C_partial = A @ B_local  # [m, n/world]
    output = torch.empty_like(C_partial)  # Output buffer

    # Stack all ranks contribute partial to same slot
    stacked = torch.zeros(world_size, *C_partial.shape, device=f"cuda:{rank}", dtype=dtype)
    stacked[rank].copy_(C_partial)

    args["n"] = cols_per_rank

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
        torch.cuda.nvtx.range_push("GEMM + Communication")

        torch.cuda.nvtx.range_push("GEMM")
        with torch.cuda.stream(gemm_stream):
            kernel_timing["gemm"]["start_event"].record()
            C_local = A @ B_local
            kernel_timing["gemm"]["end_event"].record()
            kernel_timing["gemm"]["experiments"] += 1
        gemm_stream.synchronize()
        torch.cuda.nvtx.range_pop()

        torch.cuda.nvtx.range_push("Communication")
        with torch.cuda.stream(comm_stream):
            kernel_timing["communication"]["start_event"].record()
            stacked.zero_()
            stacked[rank].copy_(C_local)
            dist.reduce_scatter_tensor(output, stacked, op=dist.ReduceOp.SUM)
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

    if validate:
        expected = A @ B
        expected_slice = expected[:, start:end]
        atol = 1e-1 if dtype in (torch.float16, torch.bfloat16) else 1e-5
        if torch.allclose(output, expected_slice, atol=atol):
            print(f"Rank {rank}: Validation passed.")
        else:
            max_err = (output - expected_slice).abs().max().item()
            print(f"Rank {rank}: Validation failed! Max error = {max_err:.1f}")

    dist.barrier()

    if benchmark:
        perf = lambda ms: 2 * args["M"] * args["N"] * args["K"] * 1e-12 / (ms * 1e-3)
        ms = iris.do_bench(run_experiment, dist.barrier)
        tflops = perf(ms)
        print(f"Rank {rank}: {ms:.3f} ms  {tflops:.3f} tflops")
        json_writer.add_field("ms", ms)
        json_writer.add_field("flops", tflops)
        for k in ["gemm", "communication"]:
            avg = kernel_timing[k]["ms"] / kernel_timing[k]["experiments"]
            json_writer.add_field(k + "_ms", avg)
            json_writer.add_field(k + "_experiments", kernel_timing[k]["experiments"])

    dist.barrier()
    if rank == 0:
        json_writer.add_field("algorithm", "reduce_scatter_tensor")
        json_writer.flush()
        json_writer.display()

    dist.destroy_process_group()


if __name__ == "__main__":
    main()
