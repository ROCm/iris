#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
# Copyright (c) 2025 Advanced Micro Devices, Inc. All rights reserved.

import torch
import triton
import random
import sys
import os
import argparse
import json

from examples.common.utils import JSONWriter
from examples.common.validation import validate_gemm
import importlib
import iris

# Import the new pipelined push-based kernels
module_path = "examples.14_all_gather_gemm.all_gather_gemm_push_tile"  # Assuming new module name
ag_gemm_kernels_module = importlib.import_module(module_path)
push_shards_kernel = ag_gemm_kernels_module.push_shards_kernel
wait_and_compute_gemm_kernel = ag_gemm_kernels_module.wait_and_compute_gemm_kernel


torch.manual_seed(123)
random.seed(123)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Run a sweep of Iris Pipelined Push All-Gather GEMM benchmarks from a config file.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("-v", "--validate", action="store_true", help="Enable validation mode.")
    parser.add_argument("-b", "--benchmark", action="store_true", help="Enable benchmarking mode.")
    parser.add_argument(
        "--config_file",
        type=str,
        default="dataset/ag_gemm.json",
        help="Path to the JSON file with benchmark configurations.",
    )
    parser.add_argument(
        "--output_file", type=str, default="ag_gemm_pipelined_push_log.json", help="Base name for output files"
    )

    parser.add_argument("-m", type=int, default=1024, help="Number of rows in matrix A (M)")
    parser.add_argument("-n", type=int, default=3584, help="Total number of columns in matrix B (N)")
    parser.add_argument("-k", type=int, default=8192, help="Common dimension between matrices A and B (K)")

    parser.add_argument(
        "--datatype", type=str, default="fp16", choices=["fp16", "bf16", "fp32"], help="Datatype of computation"
    )
    parser.add_argument("--heap_size", type=int, default=1 << 33, help="Iris heap size in bytes")
    parser.add_argument("--BLK_M", type=int, default=256, help="Block size M for tiling")
    parser.add_argument("--BLK_N", type=int, default=64, help="Block size N for GEMM computation")
    parser.add_argument("--BLK_K", type=int, default=64, help="Block size K for tiling")
    parser.add_argument("--gsize_m", type=int, default=6, help="Group size in M dimension")
    parser.add_argument("--num_sms", type=int, default=304, help="Number of SMs for the kernel")

    return parser.parse_args()


def main():
    default_args = parse_args()

    shmem = iris.iris(default_args.heap_size)
    rank = shmem.get_rank()
    world_size = shmem.get_num_ranks()
    torch.cuda.set_device(rank)

    output_dir = "results/ag_gemm_push_tile"
    if rank == 0:
        os.makedirs(output_dir, exist_ok=True)
    shmem.barrier()

    with open(default_args.config_file, "r") as f:
        configs_to_run = json.load(f)

    shmem.log(f"Loaded {len(configs_to_run)} configurations from {default_args.config_file}")

    for config in configs_to_run:
        run_args = vars(default_args).copy()
        run_args.update(config)

        dtype_map = {"fp16": torch.float16, "bf16": torch.bfloat16, "fp32": torch.float32}
        datatype = dtype_map.get(run_args["datatype"])

        M, N, K = run_args["m"], run_args["n"], run_args["k"]
        shmem.log(f"\n--- Running Benchmark for M={M}, N={N}, K={K} ---")

        base_name, extension = os.path.splitext(default_args.output_file)
        unique_filename = f"{base_name}_m_{M}{extension}"
        full_output_path = os.path.join(output_dir, unique_filename)

        json_writer = JSONWriter(full_output_path)
        json_writer.add_field("world_size", world_size)
        for key, value in run_args.items():
            json_writer.add_field(key, value)

        K_local = K // world_size

        if rank == 0:
            A_global = torch.randn((M, K), dtype=datatype, device="cuda")
        else:
            A_global = torch.empty((M, K), dtype=datatype, device="cuda")

        A_global_broadcasted = (
            torch.from_numpy(shmem.broadcast_tensor(A_global.cpu().numpy(), source_rank=0)).to(datatype).to("cuda")
        )
        shmem.barrier()

        A_local = A_global_broadcasted[:, rank * K_local : (rank + 1) * K_local].contiguous()

        if rank == 0:
            B = torch.randn((K, N), device="cuda", dtype=datatype)
        else:
            B = torch.empty((K, N), device="cuda", dtype=datatype)

        B = torch.from_numpy(shmem.broadcast_tensor(B.cpu().numpy(), source_rank=0)).to(datatype).to("cuda")
        shmem.barrier()

        C = torch.empty((M, N), device="cuda", dtype=datatype)

        A_local_iris = shmem.empty((M, K_local), dtype=datatype)
        A_local_iris.copy_(A_local)
        A_inbox_iris = shmem.empty((world_size, M, K_local), dtype=datatype)

        num_m_tiles = (M + run_args["BLK_M"] - 1) // run_args["BLK_M"]
        num_k_tiles = (K_local + run_args["BLK_K"] - 1) // run_args["BLK_K"]
        signal_flags_iris = shmem.zeros((world_size, world_size, num_m_tiles, num_k_tiles), dtype=torch.int32)

        num_sms = torch.cuda.get_device_properties(rank).multi_processor_count

        main_stream = torch.cuda.Stream()
        kernel_timing = {
            "push_kernel": {
                "start_event": torch.cuda.Event(enable_timing=True),
                "end_event": torch.cuda.Event(enable_timing=True),
                "ms": 0,
                "experiments": 0,
            },
            "wait_and_compute_kernel": {
                "start_event": torch.cuda.Event(enable_timing=True),
                "end_event": torch.cuda.Event(enable_timing=True),
                "ms": 0,
                "experiments": 0,
            },
        }

        def run_experiment():
            nonlocal kernel_timing
            signal_flags_iris.zero_()
            shmem.barrier()

            with torch.cuda.stream(main_stream):
                push_grid = (num_m_tiles, num_k_tiles)

                kernel_timing["push_kernel"]["start_event"].record()
                push_shards_kernel[push_grid](
                    A_local_iris,
                    A_inbox_iris,
                    signal_flags_iris,
                    M,
                    K_local,
                    A_local_iris.stride(0),
                    A_local_iris.stride(1),
                    A_inbox_iris.stride(0),
                    A_inbox_iris.stride(1),
                    A_inbox_iris.stride(2),
                    signal_flags_iris.stride(0),
                    signal_flags_iris.stride(1),
                    signal_flags_iris.stride(2),
                    signal_flags_iris.stride(3),
                    run_args["BLK_M"],
                    run_args["BLK_K"],
                    rank,
                    world_size,
                    shmem.get_heap_bases(),
                )
                kernel_timing["push_kernel"]["end_event"].record()

                kernel_timing["wait_and_compute_kernel"]["start_event"].record()
                wait_and_compute_gemm_kernel[(num_sms,)](
                    A_inbox_iris,
                    B,
                    C,
                    M,
                    N,
                    K,
                    signal_flags_iris,
                    A_inbox_iris.stride(0),
                    A_inbox_iris.stride(1),
                    A_inbox_iris.stride(2),
                    B.stride(0),
                    B.stride(1),
                    C.stride(0),
                    C.stride(1),
                    signal_flags_iris.stride(0),
                    signal_flags_iris.stride(1),
                    signal_flags_iris.stride(2),
                    signal_flags_iris.stride(3),
                    run_args["BLK_M"],
                    run_args["BLK_N"],
                    run_args["BLK_K"],
                    run_args["gsize_m"],
                    num_sms,
                    1,
                    (K_local % run_args["BLK_K"] == 0),
                    rank,
                    world_size,
                )
                kernel_timing["wait_and_compute_kernel"]["end_event"].record()

            torch.cuda.synchronize()
            kernel_timing["push_kernel"]["ms"] += kernel_timing["push_kernel"]["start_event"].elapsed_time(
                kernel_timing["push_kernel"]["end_event"]
            )
            kernel_timing["push_kernel"]["experiments"] += 1
            kernel_timing["wait_and_compute_kernel"]["ms"] += kernel_timing["wait_and_compute_kernel"][
                "start_event"
            ].elapsed_time(kernel_timing["wait_and_compute_kernel"]["end_event"])
            kernel_timing["wait_and_compute_kernel"]["experiments"] += 1

        run_experiment()
        shmem.barrier()

        for key in kernel_timing:
            kernel_timing[key]["ms"] = 0
            kernel_timing[key]["experiments"] = 0

        if default_args.benchmark:
            triton_ms = iris.do_bench(run_experiment, barrier_fn=shmem.barrier)
            tflops = 2 * M * N * K * 1e-12 / (triton_ms * 1e-3)

            shmem.log_stats(f"Result (iris.do_bench): {triton_ms:.3f} ms, {tflops:.3f} TFLOPS")
            json_writer.add_field("total_ms", triton_ms)
            json_writer.add_field("tflops", tflops)

            for key in kernel_timing:
                if kernel_timing[key]["experiments"] > 0:
                    avg_kernel_ms = kernel_timing[key]["ms"] / kernel_timing[key]["experiments"]
                    json_writer.add_field(key + "_ms", avg_kernel_ms)
                    shmem.log_stats(f"Result (CUDA Events) - {key}: {avg_kernel_ms:.3f} ms")

        if default_args.validate:
            if not default_args.benchmark:
                run_experiment()
                shmem.barrier()

            zero_count = torch.sum(A_global_broadcasted == 0).item()
            shmem.log(f"Number of zeros {rank} in A_global: {zero_count}")
            shmem.log("Validating...")

            success = validate_gemm(A_global_broadcasted, B, C, shmem, atol=1.0)

            passed_str = "passed" if success else "failed"
            shmem.log(f"Final C validation {passed_str}.")
            json_writer.add_field("validation_passed", success)

        if rank == 0:
            json_writer.flush()
            shmem.log(f"Saved results to {full_output_path}")

    shmem.log("\nBenchmark sweep complete.")
    shmem.barrier()


if __name__ == "__main__":
    main()
