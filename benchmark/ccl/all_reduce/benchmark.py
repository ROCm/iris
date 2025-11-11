#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
# Copyright (c) 2025 Advanced Micro Devices, Inc. All rights reserved.

"""
Benchmark for iris-ccl all-reduce collective operation.

This benchmark showcases the all-reduce collective and reports achieved bandwidth.
"""

import torch
import torch.distributed as dist
import torch.multiprocessing as mp
import random
import argparse

from examples.common.utils import JSONWriter

import iris
from iris.ccl import Config

torch.manual_seed(123)
random.seed(123)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Benchmark all-reduce collective operation.",
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
    parser.add_argument("--comm_sms", type=int, default=32, help="Number of SMs for all-reduce kernel")
    parser.add_argument("--block_size_m", type=int, default=None, help="Block size for M dimension tiling")
    parser.add_argument("--block_size_n", type=int, default=None, help="Block size for N dimension tiling")
    parser.add_argument("--swizzle_size", type=int, default=None, help="Number of tiles to swizzle together")
    parser.add_argument("--num_xcds", type=int, default=None, help="Number of XCDs (auto-detected if not set)")
    parser.add_argument("-r", "--num_ranks", type=int, default=8, help="Number of ranks/processes")
    parser.add_argument(
        "--variant",
        type=str,
        default="atomic",
        choices=["atomic", "ring", "two_shot", "one_shot", "spinlock"],
        help="All-reduce variant to use",
    )
    parser.add_argument(
        "--distribution",
        type=int,
        default=0,
        choices=[0, 1],
        help="Distribution for two-shot variant (0=striding, 1=block)",
    )
    parser.add_argument(
        "--num_rings",
        type=int,
        default=1,
        help="Number of concurrent rings for ring variant",
    )
    parser.add_argument(
        "--ring_slice_n",
        type=int,
        default=None,
        help="Column slice size for ring variant (power of two, must divide block_size_n)",
    )

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
    config_kwargs = {
        "comm_sms": args["comm_sms"],
        "all_reduce_variant": args["variant"],
    }
    if args["variant"] == "ring":
        config_kwargs["all_reduce_num_rings"] = args["num_rings"]
        if args["ring_slice_n"] is not None:
            config_kwargs["all_reduce_ring_slice_n"] = args["ring_slice_n"]
    if args["block_size_m"] is not None:
        config_kwargs["block_size_m"] = args["block_size_m"]
    if args["block_size_n"] is not None:
        config_kwargs["block_size_n"] = args["block_size_n"]
    if args["swizzle_size"] is not None:
        config_kwargs["swizzle_size"] = args["swizzle_size"]
    if args["num_xcds"] is not None:
        config_kwargs["num_xcds"] = args["num_xcds"]
    if args["variant"] == "two_shot":
        config_kwargs["all_reduce_distribution"] = args["distribution"]

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
    json_writer.add_field("all_reduce_variant", config.all_reduce_variant)
    if args["variant"] == "ring":
        json_writer.add_field("all_reduce_num_rings", config.all_reduce_num_rings)
        json_writer.add_field("all_reduce_ring_slice_n", config.all_reduce_ring_slice_n)
    if args["variant"] == "two_shot":
        json_writer.add_field("all_reduce_distribution", config.all_reduce_distribution)

    # Create input and output tensors for all-reduce
    # Each rank has its own M x N tensor
    # Note: Must use shmem.zeros() to allocate on Iris symmetric heap
    input_tensor = shmem.zeros((M, N), dtype=datatype)
    output_tensor = shmem.zeros((M, N), dtype=datatype)
    expected_tensor = None

    # Fill input with deterministic values
    val = float(rank + 1)
    input_tensor.fill_(val)

    # Expected result: sum of all ranks (1 + 2 + ... + world_size)
    expected_sum = float(world_size * (world_size + 1) / 2)
    if args["validate"]:
        expected_tensor = shmem.zeros((M, N), dtype=datatype)
        expected_tensor.fill_(expected_sum)

    comm_stream = torch.cuda.Stream()

    kernel_timing = {
        "all_reduce": {
            "start_event": torch.cuda.Event(enable_timing=True),
            "end_event": torch.cuda.Event(enable_timing=True),
            "ms": 0,
            "experiments": 0,
        },
    }

    workspace = None

    def run_experiment():
        nonlocal kernel_timing, workspace

        workspace = shmem.ccl.all_reduce_preamble(
            output_tensor,
            input_tensor,
            config=config,
            workspace=workspace,
        )

        shmem.barrier()

        torch.cuda.nvtx.range_push("All-Reduce")
        with torch.cuda.stream(comm_stream):
            kernel_timing["all_reduce"]["start_event"].record()
            shmem.ccl.all_reduce(
                output_tensor,
                input_tensor,
                config=config,
                async_op=False,
                workspace=workspace,
            )
            kernel_timing["all_reduce"]["end_event"].record()
            kernel_timing["all_reduce"]["experiments"] += 1
        torch.cuda.nvtx.range_pop()

        # Synchronize before querying event timing
        shmem.barrier()

        # Update timing
        ms = kernel_timing["all_reduce"]["start_event"].elapsed_time(kernel_timing["all_reduce"]["end_event"])
        kernel_timing["all_reduce"]["ms"] += ms

    # Synchronize across all GPUs
    shmem.barrier()

    if args["validate"]:
        shmem.info("Validating...")

        # Reset output before validation
        output_tensor.zero_()
        shmem.barrier()

        # Reinitialize input data
        input_tensor.fill_(float(rank + 1))
        shmem.barrier()

        run_experiment()
        torch.cuda.synchronize()
        shmem.barrier()

        atol = 1e-3 if datatype == torch.float16 else 1e-5
        success = torch.allclose(output_tensor, expected_tensor, atol=atol)
        if not success:
            max_diff = torch.abs(output_tensor - expected_tensor).max().item()
            shmem.error(f"Rank {rank}: Validation failed, max diff: {max_diff}")

        if success:
            shmem.info("All-reduce validation passed!")
        else:
            shmem.error("All-reduce validation failed!")

        json_writer.add_field("success", success)

        # Wait for all to finish validation
        shmem.barrier()

    if args["benchmark"]:
        # Warmup for benchmarking
        run_experiment()
        shmem.barrier()

        for k in ["all_reduce"]:
            kernel_timing[k]["ms"] = 0
            kernel_timing[k]["experiments"] = 0

        # Reset output before benchmarking
        output_tensor.zero_()
        shmem.barrier()

        # Reinitialize input data
        input_tensor.fill_(float(rank + 1))
        shmem.barrier()

        shmem.info("Benchmarking...")

        # Calculate bandwidth
        # All-reduce moves 2 * (world_size - 1) / world_size * data_size bytes
        # This accounts for the ring-based algorithm where data is transferred in (world_size - 1) steps
        # Each rank transfers 2 * (world_size - 1) / world_size * M * N * element_size bytes
        element_size = torch.tensor([], dtype=datatype).element_size()
        total_bytes = M * N * element_size * (2 * (world_size - 1)) / world_size
        total_bytes_gb = total_bytes / (1024**3)

        triton_ms = iris.do_bench(run_experiment, shmem.barrier)
        bandwidth_gbps = total_bytes_gb / (
            (kernel_timing["all_reduce"]["ms"] / kernel_timing["all_reduce"]["experiments"]) * 1e-3
        )

        shmem.info(
            f"All-reduce (M={M}, N={N}, world_size={world_size}, dtype={args['datatype']}, variant={args['variant']}): "
            f"{triton_ms:.3f} ms, {bandwidth_gbps:.3f} GB/s"
        )

        json_writer.add_field("bandwidth_gbps", bandwidth_gbps)
        json_writer.add_field("total_ms", triton_ms)
        json_writer.add_field("total_bytes", total_bytes)
        json_writer.add_field("total_bytes_gb", total_bytes_gb)
        json_writer.add_field(
            "all_reduce_ms", kernel_timing["all_reduce"]["ms"] / kernel_timing["all_reduce"]["experiments"]
        )
        json_writer.add_field("all_reduce_experiments", kernel_timing["all_reduce"]["experiments"])

        # Wait for all to finish benchmarking
        shmem.barrier()

    if rank == 0:
        if args["variant"] == "ring":
            json_writer.add_field("all_reduce_ring_slice_n", config.all_reduce_ring_slice_n)
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
