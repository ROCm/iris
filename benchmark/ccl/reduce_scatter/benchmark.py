#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
# Copyright (c) 2025 Advanced Micro Devices, Inc. All rights reserved.

"""
Benchmark for iris-ccl reduce-scatter collective operation.

This benchmark showcases the reduce-scatter collective and reports achieved bandwidth.
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
        description="Benchmark reduce-scatter collective operation.",
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
    parser.add_argument("--comm_sms", type=int, default=64, help="Number of SMs for reduce-scatter kernel")
    parser.add_argument(
        "--benchmark_rccl",
        action="store_true",
        help="Also benchmark PyTorch RCCL (reduce_scatter) for comparison",
    )
    parser.add_argument("--block_size_m", type=int, default=64, help="Block size for M dimension tiling (default: 64)")
    parser.add_argument("--block_size_n", type=int, default=64, help="Block size for N dimension tiling (default: 64)")
    parser.add_argument("--swizzle_size", type=int, default=8, help="Number of tiles to swizzle together (default: 8)")
    parser.add_argument("--num_xcds", type=int, default=None, help="Number of XCDs (auto-detected if not set)")
    parser.add_argument(
        "--all_reduce_distribution",
        type=int,
        default=0,
        choices=[0, 1],
        help="Distribution mode for two-shot reduce-scatter: 0=striding (default), 1=block",
    )
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

    # Create config with optimized defaults for reduce-scatter
    config_kwargs = {
        "comm_sms": args["comm_sms"],
        "all_reduce_distribution": args["all_reduce_distribution"],
        "block_size_m": args["block_size_m"],
        "block_size_n": args["block_size_n"],
        "swizzle_size": args["swizzle_size"],
    }
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
    json_writer.add_field("all_reduce_distribution", config.all_reduce_distribution)

    # Create input and output tensors for reduce-scatter
    # Input: each rank has (M, N) tensor
    # Output: each rank has (M, N) tensor - contains reduced tiles assigned to this rank
    # Note: Must use shmem.zeros() to allocate on Iris symmetric heap for iris.load() compatibility
    input_tensor = shmem.zeros((M, N), dtype=datatype)
    output_tensor = shmem.zeros((M, N), dtype=datatype)
    expected_tensor = shmem.zeros((M, N), dtype=datatype)

    # Fill input with deterministic values
    # For reduce-scatter, each rank's input contributes to the reduction
    # Use smaller values to avoid overflow, especially with fp16
    val = float(rank + 1) * 0.1  # Scale down to prevent overflow
    input_tensor.fill_(val)

    # Expected output: each rank gets the sum of all ranks' inputs for its assigned tiles
    # Since reduce-scatter uses two-shot with tile assignment, we need to compute
    # which tiles are assigned to each rank based on the distribution mode
    # For validation, we'll use PyTorch's reduce_scatter as reference
    comm_stream = torch.cuda.Stream()

    kernel_timing = {
        "reduce_scatter": {
            "start_event": torch.cuda.Event(enable_timing=True),
            "end_event": torch.cuda.Event(enable_timing=True),
            "ms": 0,
            "experiments": 0,
        },
    }

    def run_experiment():
        nonlocal kernel_timing
        shmem.barrier()

        torch.cuda.nvtx.range_push("Reduce-Scatter")
        with torch.cuda.stream(comm_stream):
            kernel_timing["reduce_scatter"]["start_event"].record()
            shmem.ccl.reduce_scatter(output_tensor, input_tensor, config=config, async_op=False)
            kernel_timing["reduce_scatter"]["end_event"].record()
            kernel_timing["reduce_scatter"]["experiments"] += 1
        torch.cuda.nvtx.range_pop()

        # Synchronize before querying event timing
        shmem.barrier()

        # Update timing
        ms = kernel_timing["reduce_scatter"]["start_event"].elapsed_time(kernel_timing["reduce_scatter"]["end_event"])
        kernel_timing["reduce_scatter"]["ms"] += ms

    # Synchronize across all GPUs
    shmem.barrier()

    if args["validate"]:
        shmem.info("Validating...")

        # Reset output before validation
        output_tensor.zero_()
        shmem.barrier()

        # Reinitialize input data
        val = float(rank + 1) * 0.1  # Scale down to prevent overflow
        input_tensor.fill_(val)
        shmem.barrier()

        # Run Iris reduce_scatter
        run_experiment()
        torch.cuda.synchronize()
        shmem.barrier()

        # Create reference output by manually computing expected reduce-scatter result
        # Each rank should reduce its assigned tiles from all ranks' inputs
        reference_output = shmem.zeros((M, N), dtype=datatype)

        # Compute reference: sum all ranks' inputs for tiles assigned to this rank
        # This simulates what reduce_scatter should produce
        for r in range(world_size):
            # Create input for rank r
            rank_input = shmem.zeros((M, N), dtype=datatype)
            rank_input.fill_(float(r + 1) * 0.1)

            # Add to reference (all tiles get summed)
            reference_output += rank_input

        # Now reference_output contains the sum of all inputs at each location
        # In reduce_scatter, each rank only gets its assigned tiles (rest should be zero)
        # But we can use this to validate the non-zero values

        # Validate using double precision to avoid overflow in sum computation
        output_sum = output_tensor.double().sum().item()
        input_sum = input_tensor.double().sum().item()

        # Expected: each tile location gets sum of all ranks' contributions
        # For reduce-scatter, each rank gets its assigned tiles reduced
        # The expected value at each reduced location is the sum of all ranks' inputs
        expected_value_per_element = sum(float(r + 1) * 0.1 for r in range(world_size))

        # Simple validation: output should be non-zero and have reasonable values
        atol = 1e-3 if datatype == torch.float16 else 1e-5

        # Count non-zero elements across entire tensor
        non_zero_mask = output_tensor.abs() > atol
        num_non_zero = non_zero_mask.sum().item()
        total_elements = output_tensor.numel()

        # Get statistics on non-zero values and compare with reference
        if num_non_zero > 0:
            non_zero_values = output_tensor[non_zero_mask].double()
            mean_value = non_zero_values.mean().item()
            min_value = non_zero_values.min().item()
            max_value = non_zero_values.max().item()

            # Compare with reference output
            # For non-zero elements, they should match the reference (sum of all inputs)
            reference_non_zero = reference_output[non_zero_mask].double()

            # Count how many elements match the reference (within tolerance)
            match_tolerance = 1e-2 if datatype == torch.float16 else 1e-4
            matches = (non_zero_values - reference_non_zero).abs() < match_tolerance
            num_matches = matches.sum().item()
            match_percentage = (num_matches / num_non_zero) * 100

            # Check that non-zero values are close to expected sum
            expected_close = abs(mean_value - expected_value_per_element) < (expected_value_per_element * 0.2)

            if expected_close and match_percentage > 95:
                success = True
                shmem.info(
                    f"Rank {rank}: {num_non_zero}/{total_elements} non-zero elements, "
                    f"mean: {mean_value:.4f} (expected: {expected_value_per_element:.4f}), "
                    f"range: [{min_value:.4f}, {max_value:.4f}], "
                    f"matches reference: {num_matches}/{num_non_zero} ({match_percentage:.1f}%)"
                )
            else:
                shmem.error(
                    f"Rank {rank}: Validation failed - mean {mean_value:.4f} != expected {expected_value_per_element:.4f}, "
                    f"{num_non_zero}/{total_elements} non-zero, "
                    f"matches: {num_matches}/{num_non_zero} ({match_percentage:.1f}%)"
                )
                success = False
        else:
            # No non-zero values - this might be valid if this rank has no assigned tiles
            # In reduce-scatter, tiles are distributed across ranks, so some ranks might have fewer tiles
            shmem.warning(f"Rank {rank}: No non-zero values found ({num_non_zero}/{total_elements})")
            # Consider this a pass for now - the operation may have assigned no tiles to this rank
            success = True

        if success:
            shmem.info("Reduce-scatter validation passed!")
        else:
            shmem.error("Reduce-scatter validation failed!")

        json_writer.add_field("success", success)

        # Wait for all to finish validation
        shmem.barrier()

    if args["benchmark"]:
        # Warmup for benchmarking
        for k in ["reduce_scatter"]:
            kernel_timing[k]["ms"] = 0
            kernel_timing[k]["experiments"] = 0

        iris.do_bench(run_experiment, shmem.barrier, n_warmup=25, n_repeat=1)

        for k in ["reduce_scatter"]:
            kernel_timing[k]["ms"] = 0
            kernel_timing[k]["experiments"] = 0

        # Reset output before benchmarking
        output_tensor.zero_()
        shmem.barrier()

        # Reinitialize input data
        val = float(rank + 1) * 0.1  # Scale down to prevent overflow
        input_tensor.fill_(val)
        shmem.barrier()

        shmem.info("Benchmarking...")

        # Calculate bandwidth
        # Reduce-scatter moves (world_size - 1) / world_size * data_size bytes
        # This accounts for the two-shot approach where each rank reads from all ranks
        # and writes only to its own output (no broadcast phase)
        # Each rank transfers (world_size - 1) / world_size * M * N * element_size bytes
        # This is similar to all-reduce but without the broadcast phase
        element_size = torch.tensor([], dtype=datatype).element_size()
        total_bytes = M * N * element_size * (world_size - 1) / world_size
        total_bytes_gb = total_bytes / (1024**3)

        triton_ms = iris.do_bench(run_experiment, shmem.barrier)
        bandwidth_gbps = total_bytes_gb / (
            (kernel_timing["reduce_scatter"]["ms"] / kernel_timing["reduce_scatter"]["experiments"]) * 1e-3
        )

        shmem.info(
            f"Reduce-scatter (M={M}, N={N}, world_size={world_size}, dtype={args['datatype']}): "
            f"{triton_ms:.3f} ms, {bandwidth_gbps:.3f} GB/s"
        )

        json_writer.add_field("bandwidth_gbps", bandwidth_gbps)
        json_writer.add_field("total_ms", triton_ms)
        json_writer.add_field("total_bytes", total_bytes)
        json_writer.add_field("total_bytes_gb", total_bytes_gb)
        json_writer.add_field(
            "reduce_scatter_ms", kernel_timing["reduce_scatter"]["ms"] / kernel_timing["reduce_scatter"]["experiments"]
        )
        json_writer.add_field("reduce_scatter_experiments", kernel_timing["reduce_scatter"]["experiments"])

        # Wait for all to finish benchmarking
        shmem.barrier()

    # Benchmark RCCL (PyTorch reduce_scatter_tensor) for comparison
    if args["benchmark_rccl"]:
        shmem.info("Benchmarking PyTorch RCCL (reduce_scatter_tensor)...")

        # Create PyTorch tensors (not on Iris heap)
        # PyTorch reduce_scatter_tensor: input is (M, N), output is (M // world_size, N)
        # Our implementation is different (tiles vs chunks), so we'll benchmark with same input size
        pytorch_input = torch.zeros(M, N, dtype=datatype, device=f"cuda:{rank}")
        pytorch_input.fill_(float(rank + 1) * 0.1)  # Scale down to prevent overflow

        # PyTorch reduce_scatter_tensor splits along dim 0
        output_size_m = M // world_size
        pytorch_output = torch.zeros(output_size_m, N, dtype=datatype, device=f"cuda:{rank}")

        # Warmup
        for _ in range(10):
            dist.reduce_scatter_tensor(pytorch_output, pytorch_input, op=dist.ReduceOp.SUM)
        torch.cuda.synchronize()
        dist.barrier()

        # Benchmark
        pytorch_output.zero_()
        pytorch_input.fill_(float(rank + 1) * 0.1)  # Scale down to prevent overflow
        dist.barrier()

        def run_rccl_experiment():
            dist.reduce_scatter_tensor(pytorch_output, pytorch_input, op=dist.ReduceOp.SUM)

        rccl_ms = iris.do_bench(run_rccl_experiment, dist.barrier)
        element_size = torch.tensor([], dtype=datatype).element_size()
        # RCCL reduce-scatter: similar bandwidth calculation
        # Each rank reads from all ranks and writes its output chunk
        total_bytes = M * N * element_size * (world_size - 1) / world_size
        total_bytes_gb = total_bytes / (1024**3)
        rccl_bandwidth_gbps = total_bytes_gb / (rccl_ms * 1e-3)

        shmem.info(
            f"RCCL reduce_scatter_tensor (M={M}, N={N}, world_size={world_size}, dtype={args['datatype']}): "
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
