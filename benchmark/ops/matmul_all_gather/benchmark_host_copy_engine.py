#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
# Copyright (c) 2025 Advanced Micro Devices, Inc. All rights reserved.

"""
Benchmark for iris.ops matmul_all_gather_host_copy_engine fused operation.

This benchmark showcases the host-initiated SDMA variant where the host pre-queues
POLL+COPY packets and the device kernel just stores tiles and sets flags to trigger
the pre-queued SDMA transfers.
"""

import os
import torch
import torch.distributed as dist
import random
import argparse

from examples.common.utils import JSONWriter

import iris
from iris.ops import FusedConfig
from iris.ops.matmul_all_gather_host_copy_engine import (
    matmul_all_gather_host_copy_engine,
)

torch.manual_seed(123)
random.seed(123)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Benchmark matmul_all_gather_host_copy_engine fused operation.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("-m", type=int, default=16384, help="Number of rows per rank in matrix A (M_local)")
    parser.add_argument("-n", type=int, default=2048, help="Number of columns in matrix B (N)")
    parser.add_argument("-k", type=int, default=131072, help="Common dimension (K)")
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
        default="matmul_all_gather_host_copy_engine.json",
        help="Output file",
    )
    parser.add_argument("--heap_size", type=int, default=1 << 34, help="Iris heap size")
    parser.add_argument("--comm_sms", type=int, default=None, help="Number of SMs for operation (auto-detect if None)")
    parser.add_argument(
        "--benchmark_baseline",
        action="store_true",
        help="Also benchmark baseline (non-copy-engine) variant for comparison",
    )
    parser.add_argument("--block_size_m", type=int, default=256, help="Block size for M dimension")
    parser.add_argument("--block_size_n", type=int, default=64, help="Block size for N dimension")
    parser.add_argument("--block_size_k", type=int, default=64, help="Block size for K dimension")
    parser.add_argument("--group_size_m", type=int, default=1, help="Group size for M dimension tiling")
    parser.add_argument("--num_xcds", type=int, default=None, help="Number of XCDs (auto-detected if not set)")

    return vars(parser.parse_args())


def _worker(args: dict):
    """Worker function for PyTorch distributed execution."""
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    backend = "nccl" if torch.cuda.is_available() else "gloo"
    dist.init_process_group(backend=backend)

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

    M_local = args["m"]  # Local M dimension
    M = M_local * world_size  # Total M after gather
    N = args["n"]
    K = args["k"]

    # Create config with parameters
    config_kwargs = {
        "block_size_m": args["block_size_m"],
        "block_size_n": args["block_size_n"],
        "block_size_k": args["block_size_k"],
        "group_size_m": args["group_size_m"],
    }
    if args["comm_sms"] is not None:
        config_kwargs["num_sms"] = args["comm_sms"]
    if args["num_xcds"] is not None:
        config_kwargs["num_xcds"] = args["num_xcds"]

    config = FusedConfig(**config_kwargs)

    json_writer = JSONWriter(args["output_file"])
    json_writer.add_field("world_size", world_size)
    json_writer.add_field("operation", "matmul_all_gather_host_copy_engine")
    json_writer.add_field("m_local", M_local)
    json_writer.add_field("m_total", M)

    for key, value in args.items():
        json_writer.add_field(key, value)

    # Export actual config values to JSON (including defaults)
    json_writer.add_field("block_size_m", config.block_size_m)
    json_writer.add_field("block_size_n", config.block_size_n)
    json_writer.add_field("block_size_k", config.block_size_k)
    json_writer.add_field("group_size_m", config.group_size_m)
    json_writer.add_field("num_sms", config.num_sms)
    json_writer.add_field("num_xcds", config.num_xcds)

    # Create input and output tensors
    # A_local is M_local x K, output is M x N (gathered)
    A_local = shmem.zeros((M_local, K), dtype=datatype)
    B = shmem.zeros((K, N), dtype=datatype)
    C = shmem.zeros((M, N), dtype=datatype)
    expected_tensor = None

    # Fill inputs with deterministic values
    # Each rank has different A_local, same B
    torch.manual_seed(123 + rank)
    A_local_data = torch.randn((M_local, K), dtype=datatype, device=f"cuda:{rank}")
    A_local.copy_(A_local_data)

    torch.manual_seed(456)  # Same B for all ranks
    B_data = torch.randn((K, N), dtype=datatype, device=f"cuda:{rank}")
    B.copy_(B_data)

    # For validation: compute expected result
    if args["validate"]:
        # Gather all A_local matrices and compute expected result
        A_local_list = [torch.zeros((M_local, K), dtype=datatype, device=f"cuda:{rank}") for _ in range(world_size)]
        dist.all_gather(A_local_list, A_local_data)

        # Expected: [A_0 @ B; A_1 @ B; ...; A_n @ B] stacked along M
        expected_tensor = shmem.zeros((M, N), dtype=datatype)
        expected_parts = []
        for i, A_rank_local in enumerate(A_local_list):
            C_rank_local = torch.matmul(A_rank_local, B_data)
            expected_parts.append(C_rank_local)
        expected_result = torch.cat(expected_parts, dim=0)
        expected_tensor.copy_(expected_result)

    comm_stream = torch.cuda.Stream()

    kernel_timing = {
        "host_copy_engine": {
            "start_event": torch.cuda.Event(enable_timing=True),
            "end_event": torch.cuda.Event(enable_timing=True),
            "ms": 0,
            "experiments": 0,
        },
        "baseline": {
            "start_event": torch.cuda.Event(enable_timing=True),
            "end_event": torch.cuda.Event(enable_timing=True),
            "ms": 0,
            "experiments": 0,
        },
    }

    workspace = None

    def run_host_copy_engine_experiment():
        nonlocal kernel_timing, workspace

        shmem.barrier()

        torch.cuda.nvtx.range_push("Matmul-All-Gather-HostCopyEngine")
        with torch.cuda.stream(comm_stream):
            kernel_timing["host_copy_engine"]["start_event"].record()
            workspace = matmul_all_gather_host_copy_engine(
                shmem,
                C,
                A_local,
                B,
                config=config,
                async_op=False,
                workspace=workspace,
            )
            kernel_timing["host_copy_engine"]["end_event"].record()
            kernel_timing["host_copy_engine"]["experiments"] += 1
        torch.cuda.nvtx.range_pop()

        # Synchronize before querying event timing
        shmem.barrier()

        # Update timing
        ms = kernel_timing["host_copy_engine"]["start_event"].elapsed_time(
            kernel_timing["host_copy_engine"]["end_event"]
        )
        kernel_timing["host_copy_engine"]["ms"] += ms

    def run_baseline_experiment():
        nonlocal kernel_timing, workspace

        shmem.barrier()

        torch.cuda.nvtx.range_push("Matmul-All-Gather-Baseline")
        with torch.cuda.stream(comm_stream):
            kernel_timing["baseline"]["start_event"].record()
            shmem.ops.matmul_all_gather(
                C,
                A_local,
                B,
                config=config,
                async_op=False,
                workspace=workspace,
            )
            kernel_timing["baseline"]["end_event"].record()
            kernel_timing["baseline"]["experiments"] += 1
        torch.cuda.nvtx.range_pop()

        # Synchronize before querying event timing
        shmem.barrier()

        # Update timing
        ms = kernel_timing["baseline"]["start_event"].elapsed_time(kernel_timing["baseline"]["end_event"])
        kernel_timing["baseline"]["ms"] += ms

    # Synchronize across all GPUs
    shmem.barrier()

    if args["validate"]:
        shmem.info("Validating host copy engine variant...")

        # Reset output before validation
        C.zero_()
        shmem.barrier()

        run_host_copy_engine_experiment()
        torch.cuda.synchronize()
        shmem.barrier()

        atol = 1e-1 if datatype == torch.float16 else 1e-3
        success = torch.allclose(C, expected_tensor, atol=atol)
        if not success:
            max_diff = torch.abs(C - expected_tensor).max().item()
            shmem.error(f"Rank {rank}: Validation failed, max diff: {max_diff}")

        if success:
            shmem.info("Matmul-all-gather host copy engine validation passed!")
        else:
            shmem.error("Matmul-all-gather host copy engine validation failed!")

        json_writer.add_field("success", success)

        # Wait for all to finish validation
        shmem.barrier()

    if args["benchmark"]:
        # Warmup for benchmarking
        for k in ["host_copy_engine", "baseline"]:
            kernel_timing[k]["ms"] = 0
            kernel_timing[k]["experiments"] = 0

        iris.do_bench(run_host_copy_engine_experiment, shmem.barrier, n_warmup=25, n_repeat=1)

        for k in ["host_copy_engine", "baseline"]:
            kernel_timing[k]["ms"] = 0
            kernel_timing[k]["experiments"] = 0

        # Reset output before benchmarking
        C.zero_()
        shmem.barrier()

        shmem.info("Benchmarking host copy engine variant...")

        # Calculate TFLOPS: 2*M_local*N*K flops per rank (but total is same across all ranks)
        total_flops = 2 * M_local * N * K
        total_tflops_unit = total_flops * 1e-12

        triton_ms = iris.do_bench(run_host_copy_engine_experiment, shmem.barrier)
        tflops = total_tflops_unit / (
            (kernel_timing["host_copy_engine"]["ms"] / kernel_timing["host_copy_engine"]["experiments"]) * 1e-3
        )

        # Calculate bandwidth for all-gather part
        # All-gather moves (world_size - 1) * M_local * N * element_size bytes
        element_size = torch.tensor([], dtype=datatype).element_size()
        output_bytes = M_local * N * element_size
        total_bytes = output_bytes * (world_size - 1)
        total_bytes_gb = total_bytes / (1024**3)

        bandwidth_gbps = total_bytes_gb / (
            (kernel_timing["host_copy_engine"]["ms"] / kernel_timing["host_copy_engine"]["experiments"]) * 1e-3
        )

        shmem.info(
            f"Matmul-all-gather host copy engine (M_local={M_local}, M_total={M}, N={N}, K={K}, world_size={world_size}, dtype={args['datatype']}): "
            f"{triton_ms:.3f} ms, {tflops:.3f} TFLOPS, {bandwidth_gbps:.3f} GB/s"
        )

        json_writer.add_field("host_copy_engine_tflops", tflops)
        json_writer.add_field("host_copy_engine_bandwidth_gbps", bandwidth_gbps)
        json_writer.add_field("host_copy_engine_total_ms", triton_ms)
        json_writer.add_field("total_flops", total_flops)
        json_writer.add_field("total_bytes", total_bytes)
        json_writer.add_field("total_bytes_gb", total_bytes_gb)
        json_writer.add_field(
            "host_copy_engine_ms",
            kernel_timing["host_copy_engine"]["ms"] / kernel_timing["host_copy_engine"]["experiments"],
        )
        json_writer.add_field("host_copy_engine_experiments", kernel_timing["host_copy_engine"]["experiments"])

        # Wait for all to finish benchmarking
        shmem.barrier()

    # Benchmark baseline (compute scatter) for comparison
    if args["benchmark_baseline"] and args["benchmark"]:
        shmem.info("Benchmarking baseline (compute scatter) variant...")

        # Warmup
        iris.do_bench(run_baseline_experiment, shmem.barrier, n_warmup=25, n_repeat=1)

        kernel_timing["baseline"]["ms"] = 0
        kernel_timing["baseline"]["experiments"] = 0

        # Reset output before benchmarking
        C.zero_()
        shmem.barrier()

        # Calculate TFLOPS: 2*M_local*N*K flops per rank
        total_flops = 2 * M_local * N * K
        total_tflops_unit = total_flops * 1e-12

        baseline_ms = iris.do_bench(run_baseline_experiment, shmem.barrier)
        baseline_tflops = total_tflops_unit / (
            (kernel_timing["baseline"]["ms"] / kernel_timing["baseline"]["experiments"]) * 1e-3
        )

        # Calculate bandwidth for all-gather part
        element_size = torch.tensor([], dtype=datatype).element_size()
        output_bytes = M_local * N * element_size
        total_bytes = output_bytes * (world_size - 1)
        total_bytes_gb = total_bytes / (1024**3)

        baseline_bandwidth_gbps = total_bytes_gb / (
            (kernel_timing["baseline"]["ms"] / kernel_timing["baseline"]["experiments"]) * 1e-3
        )

        shmem.info(
            f"Matmul-all-gather baseline (M_local={M_local}, M_total={M}, N={N}, K={K}, world_size={world_size}, dtype={args['datatype']}): "
            f"{baseline_ms:.3f} ms, {baseline_tflops:.3f} TFLOPS, {baseline_bandwidth_gbps:.3f} GB/s"
        )

        # Calculate speedup
        host_copy_engine_tflops = tflops
        speedup = (host_copy_engine_tflops / baseline_tflops) if baseline_tflops > 0 else 0
        shmem.info(f"Speedup (HostCopyEngine/Baseline): {speedup:.2f}x")

        json_writer.add_field("baseline_tflops", baseline_tflops)
        json_writer.add_field("baseline_bandwidth_gbps", baseline_bandwidth_gbps)
        json_writer.add_field("baseline_ms", baseline_ms)
        json_writer.add_field("speedup_vs_baseline", speedup)

        # Wait for all to finish baseline benchmarking
        shmem.barrier()

    if rank == 0:
        json_writer.flush()
        json_writer.display()

    shmem.barrier()
    dist.destroy_process_group()


def main():
    print("Starting matmul_all_gather_host_copy_engine benchmark...")
    args = parse_args()
    if "RANK" in os.environ or "LOCAL_RANK" in os.environ:
        _worker(args)
    else:
        print(
            "Please run with torchrun:\n"
            "  torchrun --nproc_per_node=N "
            "benchmark/ops/matmul_all_gather/benchmark_host_copy_engine.py [OPTIONS]"
        )


if __name__ == "__main__":
    main()
