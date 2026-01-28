#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
# Copyright (c) 2025 Advanced Micro Devices, Inc. All rights reserved.

import torch
import torch.distributed as dist
import torch.multiprocessing as mp
import triton
import random
import argparse

from examples.common.utils import (
    JSONWriter,
    Timestamps,
)

import iris

from all_reduce_ring_based import persistent_all_reduce

torch.manual_seed(123)
random.seed(123)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Parse matrix dimensions and configuration.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("-m", type=int, default=8192, help="Number of rows in input/output matrix")
    parser.add_argument("-n", type=int, default=4608, help="Number of columns in input/output matrix")
    parser.add_argument("-d", "--debug", action="store_true", help="Enable debug mode")
    parser.add_argument("-v", "--validate", action="store_true", help="Enable validation mode")
    parser.add_argument("-t", "--trace_tiles", action="store_true", help="Enable tile-tracing mode")
    parser.add_argument("-b", "--benchmark", action="store_true", help="Enable benchmarking mode")
    parser.add_argument(
        "--datatype",
        type=str,
        default="fp16",
        choices=["fp16", "fp32", "int8", "bf16"],
        help="Datatype of computation",
    )
    parser.add_argument(
        "--output_file",
        type=str,
        default="log.json",
        help="Output file",
    )
    # For All Scatter, use: 256x64x64
    # For One Shot, use: 256x256x64
    parser.add_argument("--BLK_M", type=int, default=128, help="Block size M")
    parser.add_argument("--BLK_N", type=int, default=128, help="Block size N")

    # Best to try 1, 6 or 8
    parser.add_argument("--gsize_m", type=int, default=6, help="Grid size M")
    parser.add_argument("--heap_size", type=int, default=1 << 33, help="Iris heap size")

    # For All Scatter, use: 288
    # For One Shot, use: 256
    parser.add_argument("--num_sms", type=int, default=48, help="Number of SMs for All-Reduce kernel")
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
    rank = shmem.get_rank()
    world_size = shmem.get_num_ranks()
    cu_count = shmem.get_cu_count()
    num_xcds = iris.hip.get_num_xcc()

    # datatypes
    datatype = torch.float32
    if args["datatype"] == "fp16":
        datatype = torch.float16
    elif args["datatype"] == "fp32":
        datatype = torch.float32
    elif args["datatype"] == "int8":
        datatype = torch.int8
    elif args["datatype"] == "bf16":
        datatype = torch.bfloat16
    else:
        print("Unknown datatype.")
        exit(1)

    args["M"] = args["m"]
    args["N"] = args["n"]

    json_writer = JSONWriter(args["output_file"])
    json_writer.add_field("world_size", world_size)

    for key, value in args.items():
        json_writer.add_field(key, value)

    # Initialize partial with random data for each rank
    # In all_reduce, each rank has a partial result that needs to be summed across all ranks
    torch.manual_seed(123 + rank)  # Different seed per rank for different data
    partial = shmem.zeros((args["M"], args["N"]), device="cuda", dtype=datatype)
    partial.copy_(torch.randn((args["M"], args["N"]), device="cuda", dtype=datatype))

    output = shmem.zeros((args["M"], args["N"]), device="cuda", dtype=datatype)

    total_blocks_M = triton.cdiv(args["m"], args["BLK_M"])
    total_blocks_N = triton.cdiv(args["n"], args["BLK_N"])
    total_tiles = total_blocks_M * total_blocks_N

    flags = shmem.zeros((total_tiles,), device="cuda", dtype=torch.int32)
    ring_buffer = shmem.zeros_like(partial, dtype=torch.float32)
    comm_stream = torch.cuda.Stream()

    json_writer.add_field("num_sms", args["num_sms"])

    kernel_timing = {
        "communication": {
            "start_event": torch.cuda.Event(enable_timing=True),
            "end_event": torch.cuda.Event(enable_timing=True),
            "ms": 0,
            "experiments": 0,
        },
    }

    # Timestamps
    timestamps = Timestamps(num_tiles=total_tiles)

    def preamble():
        shmem.barrier()
        flags.zero_()
        ring_buffer.zero_()
        shmem.barrier()

    def run_experiment():
        nonlocal output
        nonlocal partial
        nonlocal kernel_timing
        nonlocal ring_buffer

        shmem.barrier()

        if args["trace_tiles"]:
            timestamps.reset()
            shmem.barrier()

        torch.cuda.nvtx.range_push("Communication")
        with torch.cuda.stream(comm_stream):
            kernel_timing["communication"]["start_event"].record()
            ar = persistent_all_reduce[(args["num_sms"],)](
                partial,
                ring_buffer,
                output,
                flags,
                args["M"],
                args["N"],
                output.stride(0),
                output.stride(1),
                args["BLK_M"],
                args["BLK_N"],
                args["gsize_m"],
                args["num_sms"],
                num_xcds,
                shmem.get_heap_bases(),
                rank,
                world_size,
            )
            kernel_timing["communication"]["end_event"].record()
            kernel_timing["communication"]["experiments"] += 1
        torch.cuda.nvtx.range_pop()
        shmem.barrier()

        for k in ["communication"]:
            ms = kernel_timing[k]["start_event"].elapsed_time(kernel_timing[k]["end_event"])
            kernel_timing[k]["ms"] += ms

    # Synchronize across all GPUs
    shmem.barrier()

    # Warmup
    run_experiment()

    shmem.barrier()
    preamble()
    shmem.barrier()

    for k in ["communication"]:
        kernel_timing[k]["ms"] = 0
        kernel_timing[k]["experiments"] = 0

    if args["validate"]:
        shmem.info("Validating...")

        # Run the experiment once to populate output
        run_experiment()
        shmem.barrier()

        # Create a reference result using torch.distributed.all_reduce
        # Save original partial values for reference computation
        partial_copy = partial.clone()
        expected_output = partial_copy.clone()

        # Use NCCL all_reduce to compute the expected result
        dist.all_reduce(expected_output, op=dist.ReduceOp.SUM)

        # Compare the output from our kernel with the expected result
        success = torch.allclose(output, expected_output, atol=2)
        max_diff = torch.max(torch.abs(output - expected_output)).item()

        if success:
            shmem.info(f"Final validation passed. Max difference: {max_diff}")
        else:
            shmem.info(f"Final validation failed. Max difference: {max_diff}")

        # Wait for all to finish validation
        shmem.barrier()
        shmem.info("Validation completed")

        json_writer.add_field("success", success)

    if args["benchmark"]:
        shmem.info("Benchmarking...")
        # Calculate bandwidth instead of FLOPS since there's no GEMM
        # All-reduce moves 2 * (world_size - 1) / world_size * data_size bytes
        data_size_bytes = (
            args["M"] * args["N"] * 2
            if datatype == torch.float16 or datatype == torch.bfloat16
            else args["M"] * args["N"] * 4
        )
        perf = lambda ms: (2 * (world_size - 1) / world_size * data_size_bytes * 1e-9) / (ms * 1e-3)  # GB/s
        triton_ms = iris.do_bench(run_experiment, shmem.barrier, preamble)
        bandwidth_gbps = perf(triton_ms)
        algo_string = "all_reduce"
        shmem.info(f"{algo_string} (grid={total_tiles}): {triton_ms:.3f} ms  {bandwidth_gbps:.3f} GB/s")

        json_writer.add_field("bandwidth_gbps", bandwidth_gbps)
        json_writer.add_field("total_ms", triton_ms)

        for k in ["communication"]:
            json_writer.add_field(k + "_ms", kernel_timing[k]["ms"] / kernel_timing[k]["experiments"])
            json_writer.add_field(k + "_experiments", kernel_timing[k]["experiments"])

        # Wait for all to finish benchmarking
        shmem.barrier()

    if rank == 0:
        json_writer.flush()
        json_writer.display()

    if args["trace_tiles"] and rank == 0:
        gpu_freq = iris.hip.get_wall_clock_rate(rank) * 1e-3
        algo_string = "all_reduce"
        filename = f"comm_tiles_{algo_string}_trace_rank{rank}.json"
        timestamps.to_json(filename, gpu_freq)

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
