#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
# Copyright (c) 2025 Advanced Micro Devices, Inc. All rights reserved.

import torch
import torch.distributed as dist
import torch.multiprocessing as mp
import random
import sys
import os
import argparse
import json

from examples.common.utils import JSONWriter
import iris

torch.manual_seed(123)
random.seed(123)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Run a sweep of RCCL All-Gather + torch.matmul benchmarks from a config file.",
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
        "--output_file", type=str, default="rccl_torch_matmul_log.json", help="Base name for output files"
    )
    parser.add_argument(
        "--output_dir", type=str, default="results/all_gather_gemm_rccl", help="Name of the output directory"
    )
    parser.add_argument("--num_ranks", type=int, default=8, help="Number of GPUs to run on.")
    parser.add_argument("-m", type=int, default=1024)
    parser.add_argument("-n", type=int, default=3584)
    parser.add_argument("-k", type=int, default=8192)
    parser.add_argument("--datatype", type=str, default="fp16", choices=["fp16", "bf16", "fp32"])
    return parser.parse_args()


def worker(rank: int, world_size: int, init_url: str, args: argparse.Namespace):
    dist.init_process_group(backend="nccl", init_method=init_url, world_size=world_size, rank=rank)
    torch.cuda.set_device(rank)

    output_dir = args.output_dir
    if rank == 0:
        os.makedirs(output_dir, exist_ok=True)
    dist.barrier()

    with open(args.config_file, "r") as f:
        configs_to_run = json.load(f)

    if rank == 0:
        print(f"Loaded {len(configs_to_run)} configurations from {args.config_file}")

    for config in configs_to_run:
        run_args = vars(args).copy()
        run_args.update(config)

        dtype_map = {"fp16": torch.float16, "bf16": torch.bfloat16, "fp32": torch.float32}
        datatype = dtype_map.get(run_args["datatype"])

        M, N, K = run_args["m"], run_args["n"], run_args["k"]
        if rank == 0:
            print(f"\n--- Running Benchmark for M={M}, N={N}, K={K} ---")
            sys.stdout.flush()

        base_name, extension = os.path.splitext(args.output_file)
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
        dist.broadcast(A_global, src=0)

        A_local = A_global[:, rank * K_local : (rank + 1) * K_local].contiguous()

        if rank == 0:
            B = torch.randn((K, N), device="cuda", dtype=datatype)
        else:
            B = torch.empty((K, N), device="cuda", dtype=datatype)
        dist.broadcast(B, src=0)

        C = torch.empty((M, N), device="cuda", dtype=datatype)
        all_a_shards = [torch.empty_like(A_local) for _ in range(world_size)]

        main_stream = torch.cuda.Stream()
        kernel_timing = {
            "rccl_all_gather": {
                "start_event": torch.cuda.Event(enable_timing=True),
                "end_event": torch.cuda.Event(enable_timing=True),
                "ms": 0,
                "experiments": 0,
            },
            "torch_matmul": {
                "start_event": torch.cuda.Event(enable_timing=True),
                "end_event": torch.cuda.Event(enable_timing=True),
                "ms": 0,
                "experiments": 0,
            },
        }

        def run_experiment():
            nonlocal kernel_timing
            with torch.cuda.stream(main_stream):
                kernel_timing["rccl_all_gather"]["start_event"].record()
                dist.all_gather(all_a_shards, A_local)
                A_gathered = torch.cat(all_a_shards, dim=1)
                kernel_timing["rccl_all_gather"]["end_event"].record()

                kernel_timing["torch_matmul"]["start_event"].record()
                torch.matmul(A_gathered, B, out=C)
                kernel_timing["torch_matmul"]["end_event"].record()

            torch.cuda.synchronize()
            kernel_timing["rccl_all_gather"]["ms"] += kernel_timing["rccl_all_gather"]["start_event"].elapsed_time(
                kernel_timing["rccl_all_gather"]["end_event"]
            )
            kernel_timing["rccl_all_gather"]["experiments"] += 1
            kernel_timing["torch_matmul"]["ms"] += kernel_timing["torch_matmul"]["start_event"].elapsed_time(
                kernel_timing["torch_matmul"]["end_event"]
            )
            kernel_timing["torch_matmul"]["experiments"] += 1

        run_experiment()
        dist.barrier()

        for key in kernel_timing:
            kernel_timing[key]["ms"] = 0
            kernel_timing[key]["experiments"] = 0

        if args.benchmark:
            total_ms = iris.do_bench(run_experiment, barrier_fn=dist.barrier)
            tflops = 2 * M * N * K * 1e-12 / (total_ms * 1e-3)
            if rank == 0:
                print(f"Result (iris.do_bench): {total_ms:.3f} ms, {tflops:.3f} TFLOPS")
                json_writer.add_field("total_ms", total_ms)
                json_writer.add_field("tflops", tflops)

            for key in kernel_timing:
                if kernel_timing[key]["experiments"] > 0:
                    avg_kernel_ms = kernel_timing[key]["ms"] / kernel_timing[key]["experiments"]
                    json_writer.add_field(key + "_ms", avg_kernel_ms)
                    if rank == 0:
                        print(f"Result (CUDA Events) - {key}: {avg_kernel_ms:.3f} ms")

        if args.validate:
            if not args.benchmark:
                run_experiment()
                dist.barrier()

            if rank == 0:
                print("Validating...")

            C_ref = torch.matmul(A_global, B)
            success = torch.allclose(C, C_ref, atol=1.0, rtol=0.05)
            passed_str = "passed" if success else "failed"
            print(f"Final C validation for rank {rank} is {passed_str}.")
            json_writer.add_field("validation_passed", success)

        if rank == 0:
            json_writer.flush()
            print(f"Saved results to {full_output_path}")
            sys.stdout.flush()

    if rank == 0:
        print("\nBenchmark sweep complete.")

    dist.barrier()
    dist.destroy_process_group()


def main():
    args = parse_args()
    if not args.validate and not args.benchmark:
        print("Error: You must specify a mode to run. Use -v or -b.", file=sys.stderr)
        sys.exit(1)

    num_ranks = args.num_ranks
    init_url = "tcp://127.0.0.1:29505"
    mp.spawn(fn=worker, args=(num_ranks, init_url, args), nprocs=num_ranks, join=True)


if __name__ == "__main__":
    main()
