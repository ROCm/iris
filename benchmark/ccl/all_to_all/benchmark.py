#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
# Copyright (c) 2025 Advanced Micro Devices, Inc. All rights reserved.

"""
Benchmark for iris-ccl all-to-all collective operation.

This benchmark showcases the all-to-all collective and reports achieved bandwidth.
"""

import torch
import torch.distributed as dist
import torch.multiprocessing as mp
import random
import argparse

from examples.common.utils import JSONWriter

import iris
from iris.ccl import Config

# Conditional import for Gluon
try:
    import iris.experimental.iris_gluon as iris_gluon

    GLUON_AVAILABLE = True
except ImportError:
    GLUON_AVAILABLE = False

torch.manual_seed(123)
random.seed(123)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Benchmark all-to-all collective operation.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("-m", type=int, default=16384, help="Number of rows in tensors")
    parser.add_argument("-n", type=int, default=16384, help="Number of columns in tensors")
    parser.add_argument("-d", "--debug", action="store_true", help="Enable debug mode")
    parser.add_argument("-v", "--validate", action="store_true", help="Enable validation mode")
    parser.add_argument("-b", "--benchmark", action="store_true", help="Enable benchmarking mode")
    parser.add_argument(
        "--datatype",
        type=str,
        default="fp16",
        choices=["fp16", "fp32", "bf16"],
        help="Datatype of tensors",
    )
    parser.add_argument(
        "--output_file",
        type=str,
        default="log.json",
        help="Output file",
    )
    parser.add_argument("--heap_size", type=int, default=1 << 34, help="Iris heap size")
    parser.add_argument("--comm_sms", type=int, default=32, help="Number of SMs for all-to-all kernel")
    parser.add_argument("--block_size_m", type=int, default=None, help="Block size for M dimension tiling")
    parser.add_argument("--block_size_n", type=int, default=None, help="Block size for N dimension tiling")
    parser.add_argument("--swizzle_size", type=int, default=None, help="Number of tiles to swizzle together")
    parser.add_argument("--num_xcds", type=int, default=None, help="Number of XCDs (auto-detected if not set)")
    parser.add_argument("-r", "--num_ranks", type=int, default=8, help="Number of ranks/processes")
    parser.add_argument("--use_gluon", action="store_true", help="Use Gluon implementation with traffic shaping")

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

    # Use Gluon if requested and available
    if args.get("use_gluon", False):
        if not GLUON_AVAILABLE:
            raise RuntimeError("Gluon is not available. Install Triton with Gluon support or remove --use_gluon flag")
        shmem = iris_gluon.iris(args["heap_size"])
    else:
        shmem = iris.iris(args["heap_size"])

    rank = shmem.get_rank()
    world_size = shmem.get_num_ranks()

    # Datatype mapping
    datatype = torch.float32
    if args["datatype"] == "fp16":
        datatype = torch.float16
    elif args["datatype"] == "fp32":
        datatype = torch.float32
    elif args["datatype"] == "bf16":
        datatype = torch.bfloat16
    else:
        print("Unknown datatype.")
        exit(1)

    M = args["m"]
    N = args["n"]

    # Create config with optional block size parameters
    config_kwargs = {"comm_sms": args["comm_sms"]}
    if args["block_size_m"] is not None:
        config_kwargs["block_size_m"] = args["block_size_m"]
    if args["block_size_n"] is not None:
        config_kwargs["block_size_n"] = args["block_size_n"]
    if args["swizzle_size"] is not None:
        config_kwargs["swizzle_size"] = args["swizzle_size"]
    if args["num_xcds"] is not None:
        config_kwargs["num_xcds"] = args["num_xcds"]
    if args.get("use_gluon", False):
        config_kwargs["use_gluon"] = True

    config = Config(**config_kwargs)

    json_writer = JSONWriter(args["output_file"])
    json_writer.add_field("world_size", world_size)

    for key, value in args.items():
        json_writer.add_field(key, value)

    # Export config values to JSON (use actual values from config, including defaults)
    json_writer.add_field("block_size_m", config.block_size_m)
    json_writer.add_field("block_size_n", config.block_size_n)
    json_writer.add_field("swizzle_size", config.swizzle_size)
    json_writer.add_field("num_xcds", config.num_xcds)
    json_writer.add_field("use_gluon", config.use_gluon)

    # Create input and output tensor lists for all-to-all
    # Each rank sends a different tensor to each rank
    # Create concatenated input tensor: shape (M, N * world_size)
    # Each chunk of N columns corresponds to data sent to that rank
    # Note: Must use shmem.zeros() to allocate on Iris symmetric heap for iris.put() compatibility
    input_concat = shmem.zeros((M, N * world_size), dtype=datatype)
    output_concat = shmem.zeros((M, N * world_size), dtype=datatype)
    expected_concat = shmem.zeros((M, N * world_size), dtype=datatype)

    for target_rank in range(world_size):
        # Input: rank sends data at position (target_rank * N)
        val = float(rank * 1000 + target_rank)
        input_concat[:, target_rank * N : (target_rank + 1) * N] = val

        # Expected: receive from target_rank at position (target_rank * N)
        expected_val = float(target_rank * 1000 + rank)
        expected_concat[:, target_rank * N : (target_rank + 1) * N] = expected_val

    comm_stream = torch.cuda.Stream()

    kernel_timing = {
        "all_to_all": {
            "start_event": torch.cuda.Event(enable_timing=True),
            "end_event": torch.cuda.Event(enable_timing=True),
            "ms": 0,
            "experiments": 0,
        },
    }

    def run_experiment():
        nonlocal kernel_timing
        shmem.barrier()

        torch.cuda.nvtx.range_push("All-to-All")
        with torch.cuda.stream(comm_stream):
            kernel_timing["all_to_all"]["start_event"].record()
            shmem.ccl.all_to_all(output_concat, input_concat, config=config, async_op=False)
            kernel_timing["all_to_all"]["end_event"].record()
            kernel_timing["all_to_all"]["experiments"] += 1
        torch.cuda.nvtx.range_pop()

        # Synchronize before querying event timing
        shmem.barrier()

        # Update timing
        ms = kernel_timing["all_to_all"]["start_event"].elapsed_time(kernel_timing["all_to_all"]["end_event"])
        kernel_timing["all_to_all"]["ms"] += ms

    # Synchronize across all GPUs
    shmem.barrier()

    if args["validate"]:
        shmem.info("Validating...")

        # Reset output before validation
        output_concat.zero_()
        shmem.barrier()

        # Reinitialize input data
        for target_rank in range(world_size):
            val = float(rank * 1000 + target_rank)
            input_concat[:, target_rank * N : (target_rank + 1) * N] = val
        shmem.barrier()

        run_experiment()
        torch.cuda.synchronize()
        shmem.barrier()

        atol = 1e-3 if datatype == torch.float16 else 1e-5
        success = torch.allclose(output_concat, expected_concat, atol=atol)
        if not success:
            max_diff = torch.abs(output_concat - expected_concat).max().item()
            shmem.error(f"Rank {rank}: Validation failed, max diff: {max_diff}")

        if success:
            shmem.info("All-to-all validation passed!")
        else:
            shmem.error("All-to-all validation failed!")

        json_writer.add_field("success", success)

        # Wait for all to finish validation
        shmem.barrier()

    if args["benchmark"]:
        # Warmup for benchmarking
        run_experiment()
        shmem.barrier()

        for k in ["all_to_all"]:
            kernel_timing[k]["ms"] = 0
            kernel_timing[k]["experiments"] = 0

        # Reset output before benchmarking
        output_concat.zero_()
        shmem.barrier()

        # Reinitialize input data
        for target_rank in range(world_size):
            val = float(rank * 1000 + target_rank)
            input_concat[:, target_rank * N : (target_rank + 1) * N] = val
        shmem.barrier()

        shmem.info("Benchmarking...")

        # Calculate bandwidth
        # In all-to-all, each rank sends and receives world_size tensors
        # Total bytes = (world_size - 1) * M * N * element_size
        element_size = torch.tensor([], dtype=datatype).element_size()
        total_bytes = (world_size - 1) * M * N * element_size
        total_bytes_gb = total_bytes / (1024**3)

        triton_ms = iris.do_bench(run_experiment, shmem.barrier)
        bandwidth_gbps = total_bytes_gb / (
            (kernel_timing["all_to_all"]["ms"] / kernel_timing["all_to_all"]["experiments"]) * 1e-3
        )

        shmem.info(
            f"All-to-all (M={M}, N={N}, world_size={world_size}, dtype={args['datatype']}): "
            f"{triton_ms:.3f} ms, {bandwidth_gbps:.3f} GB/s"
        )

        json_writer.add_field("bandwidth_gbps", bandwidth_gbps)
        json_writer.add_field("total_ms", triton_ms)
        json_writer.add_field("total_bytes", total_bytes)
        json_writer.add_field("total_bytes_gb", total_bytes_gb)
        json_writer.add_field(
            "all_to_all_ms", kernel_timing["all_to_all"]["ms"] / kernel_timing["all_to_all"]["experiments"]
        )
        json_writer.add_field("all_to_all_experiments", kernel_timing["all_to_all"]["experiments"])

        # Wait for all to finish benchmarking
        shmem.barrier()

    if rank == 0:
        json_writer.flush()
        json_writer.display()

    shmem.barrier()
    dist.destroy_process_group()


def main():
    args = parse_args()
    num_ranks = args["num_ranks"]
    init_url = "tcp://127.0.0.1:29503"

    mp.spawn(
        fn=_worker,
        args=(num_ranks, init_url, args),
        nprocs=num_ranks,
        join=True,
    )


if __name__ == "__main__":
    main()
