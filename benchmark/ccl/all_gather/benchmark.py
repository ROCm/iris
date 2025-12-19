#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
# Copyright (c) 2025 Advanced Micro Devices, Inc. All rights reserved.

"""
Benchmark for iris-ccl all-gather collective operation.

This benchmark showcases the all-gather collective and reports achieved bandwidth.
"""

import torch
import torch.distributed as dist
import torch.multiprocessing as mp
import random
import argparse

from examples.common.utils import JSONWriter

import iris
from iris.ccl import Config
import iris.experimental.iris_gluon as iris_gluon

torch.manual_seed(123)
random.seed(123)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Benchmark all-gather collective operation.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("-m", type=int, default=16384, help="Number of rows in input tensors")
    parser.add_argument("-n", type=int, default=16384, help="Number of columns in input tensors")
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
    parser.add_argument("--comm_sms", type=int, default=64, help="Number of SMs for all-gather kernel")
    parser.add_argument(
        "--benchmark_rccl",
        action="store_true",
        help="Also benchmark PyTorch RCCL (all_gather_into_tensor) for comparison",
    )
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

    # Use Gluon if requested
    if args["use_gluon"]:
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
    if args["use_gluon"]:
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

    # Create input and output tensors for all-gather
    # Input: each rank has (M, N) tensor
    # Output: (world_size * M, N) - concatenated along dimension 0
    # Note: Must use shmem.zeros() to allocate on Iris symmetric heap for iris.put() compatibility
    input_tensor = shmem.zeros((M, N), dtype=datatype)
    output_tensor = shmem.zeros((world_size * M, N), dtype=datatype)
    expected_tensor = shmem.zeros((world_size * M, N), dtype=datatype)

    # Fill input with deterministic values
    val = float(rank + 1)
    input_tensor.fill_(val)

    # Expected output: each rank's input appears at output[rank * M : (rank + 1) * M, :]
    for r in range(world_size):
        expected_val = float(r + 1)
        expected_tensor[r * M : (r + 1) * M, :] = expected_val

    comm_stream = torch.cuda.Stream()

    kernel_timing = {
        "all_gather": {
            "start_event": torch.cuda.Event(enable_timing=True),
            "end_event": torch.cuda.Event(enable_timing=True),
            "ms": 0,
            "experiments": 0,
        },
    }

    def run_experiment():
        nonlocal kernel_timing
        shmem.barrier()

        torch.cuda.nvtx.range_push("All-Gather")
        with torch.cuda.stream(comm_stream):
            kernel_timing["all_gather"]["start_event"].record()
            shmem.ccl.all_gather(output_tensor, input_tensor, config=config, async_op=False)
            kernel_timing["all_gather"]["end_event"].record()
            kernel_timing["all_gather"]["experiments"] += 1
        torch.cuda.nvtx.range_pop()

        # Synchronize before querying event timing
        shmem.barrier()

        # Update timing
        ms = kernel_timing["all_gather"]["start_event"].elapsed_time(kernel_timing["all_gather"]["end_event"])
        kernel_timing["all_gather"]["ms"] += ms

    # Synchronize across all GPUs
    shmem.barrier()

    if args["validate"]:
        shmem.info("Validating...")

        # Reset output before validation
        output_tensor.zero_()
        shmem.barrier()

        # Reinitialize input data
        val = float(rank + 1)
        input_tensor.fill_(val)
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
            shmem.info("All-gather validation passed!")
        else:
            shmem.error("All-gather validation failed!")

        json_writer.add_field("success", success)

        # Wait for all to finish validation
        shmem.barrier()

    if args["benchmark"]:
        # Warmup for benchmarking
        for k in ["all_gather"]:
            kernel_timing[k]["ms"] = 0
            kernel_timing[k]["experiments"] = 0

        iris.do_bench(run_experiment, shmem.barrier, n_warmup=25, n_repeat=1)

        for k in ["all_gather"]:
            kernel_timing[k]["ms"] = 0
            kernel_timing[k]["experiments"] = 0

        # Reset output before benchmarking
        output_tensor.zero_()
        shmem.barrier()

        # Reinitialize input data
        val = float(rank + 1)
        input_tensor.fill_(val)
        shmem.barrier()

        shmem.info("Benchmarking...")

        # Calculate bandwidth
        # In all-gather, each rank sends its (M, N) tensor to all ranks
        # Total bytes sent = (world_size - 1) * M * N * element_size (excluding local copy)
        # Total bytes received = (world_size - 1) * M * N * element_size
        # Total bytes = (world_size - 1) * M * N * element_size
        element_size = torch.tensor([], dtype=datatype).element_size()
        total_bytes = (world_size - 1) * M * N * element_size
        total_bytes_gb = total_bytes / (1024**3)

        triton_ms = iris.do_bench(run_experiment, shmem.barrier)
        bandwidth_gbps = total_bytes_gb / (
            (kernel_timing["all_gather"]["ms"] / kernel_timing["all_gather"]["experiments"]) * 1e-3
        )

        shmem.info(
            f"All-gather (M={M}, N={N}, world_size={world_size}, dtype={args['datatype']}): "
            f"{triton_ms:.3f} ms, {bandwidth_gbps:.3f} GB/s"
        )

        json_writer.add_field("bandwidth_gbps", bandwidth_gbps)
        json_writer.add_field("total_ms", triton_ms)
        json_writer.add_field("total_bytes", total_bytes)
        json_writer.add_field("total_bytes_gb", total_bytes_gb)
        json_writer.add_field(
            "all_gather_ms", kernel_timing["all_gather"]["ms"] / kernel_timing["all_gather"]["experiments"]
        )
        json_writer.add_field("all_gather_experiments", kernel_timing["all_gather"]["experiments"])

        # Wait for all to finish benchmarking
        shmem.barrier()

    # Benchmark RCCL (PyTorch all_gather_into_tensor) for comparison
    if args["benchmark_rccl"]:
        shmem.info("Benchmarking PyTorch RCCL (all_gather_into_tensor)...")

        # Create PyTorch tensors (not on Iris heap)
        pytorch_input = torch.zeros(M, N, dtype=datatype, device=f"cuda:{rank}")
        pytorch_input.fill_(float(rank + 1))
        pytorch_output = torch.zeros(world_size * M, N, dtype=datatype, device=f"cuda:{rank}")

        # Warmup
        for _ in range(10):
            dist.all_gather_into_tensor(pytorch_output, pytorch_input)
        torch.cuda.synchronize()
        dist.barrier()

        # Benchmark
        pytorch_output.zero_()
        pytorch_input.fill_(float(rank + 1))
        dist.barrier()

        def run_rccl_experiment():
            dist.all_gather_into_tensor(pytorch_output, pytorch_input)

        rccl_ms = iris.do_bench(run_rccl_experiment, dist.barrier)
        element_size = torch.tensor([], dtype=datatype).element_size()
        total_bytes = (world_size - 1) * M * N * element_size
        total_bytes_gb = total_bytes / (1024**3)
        rccl_bandwidth_gbps = total_bytes_gb / (rccl_ms * 1e-3)

        shmem.info(
            f"RCCL all_gather_into_tensor (M={M}, N={N}, world_size={world_size}, dtype={args['datatype']}): "
            f"{rccl_ms:.3f} ms, {rccl_bandwidth_gbps:.3f} GB/s"
        )

        if args["benchmark"]:
            # Calculate performance ratio
            iris_bandwidth = bandwidth_gbps
            rccl_ratio = (iris_bandwidth / rccl_bandwidth_gbps) * 100 if rccl_bandwidth_gbps > 0 else 0
            shmem.info(f"Performance ratio (Iris/RCCL): {rccl_ratio:.1f}%")

            json_writer.add_field("rccl_bandwidth_gbps", rccl_bandwidth_gbps)
            json_writer.add_field("rccl_ms", rccl_ms)
            json_writer.add_field("rccl_ratio_percent", rccl_ratio)

        # Wait for all to finish RCCL benchmarking
        shmem.barrier()

    if rank == 0:
        json_writer.flush()
        json_writer.display()

    shmem.barrier()
    dist.destroy_process_group()


def main():
    args = parse_args()
    num_ranks = args["num_ranks"]
    init_url = "tcp://127.0.0.1:29234"

    mp.spawn(
        fn=_worker,
        args=(num_ranks, init_url, args),
        nprocs=num_ranks,
        join=True,
    )


if __name__ == "__main__":
    main()
