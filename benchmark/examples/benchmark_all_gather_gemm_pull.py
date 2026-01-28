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
from examples.common.validation import validate_gemm
import importlib.util
from pathlib import Path
import iris

current_dir = Path(__file__).parent
file_path = (current_dir / "../../examples/14_all_gather_gemm/all_gather_gemm_pull.py").resolve()
module_name = "all_gather_gemm_pull"

spec = importlib.util.spec_from_file_location(module_name, file_path)
ag_gemm_kernels_module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(ag_gemm_kernels_module)
persistent_ag_gemm = ag_gemm_kernels_module.persistent_ag_gemm


torch.manual_seed(123)
random.seed(123)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Run a sweep of All-Gather GEMM benchmarks from a config file.",
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
    parser.add_argument("--output_file", type=str, default="ag_gemm_pull.json", help="Base name for output files")
    parser.add_argument(
        "--output_dir", type=str, default="results/all_gather_gemm_pull", help="Name of the output directory"
    )

    parser.add_argument("-m", type=int, default=1024, help="Number of rows in matrix A (M)")
    parser.add_argument("-n", type=int, default=3584, help="Total number of columns in matrix B (N)")
    parser.add_argument("-k", type=int, default=8192, help="Common dimension between matrices A and B (K)")

    parser.add_argument(
        "--datatype", type=str, default="fp16", choices=["fp16", "bf16", "fp32"], help="Datatype of computation"
    )
    parser.add_argument("--heap_size", type=int, default=1 << 33, help="Iris heap size in bytes")

    parser.add_argument("--BLK_M", type=int, default=256, help="Block size M for the kernel")
    parser.add_argument("--BLK_N", type=int, default=64, help="Block size N for the kernel")
    parser.add_argument("--BLK_K", type=int, default=64, help="Block size K for the kernel")
    parser.add_argument("--gsize_m", type=int, default=6, help="Group size in M dimension")
    parser.add_argument(
        "--num_sms", type=int, default=None, help="Number of SMs for the kernel (default: auto-detected)"
    )

    parser.add_argument("--num_ranks", type=int, default=8, help="Number of GPUs to run the example on.")

    return parser.parse_args()


def worker(rank: int, world_size: int, init_url: str, args: argparse.Namespace):
    """
    This function will be executed by each spawned process.
    """
    backend = "nccl" if torch.cuda.is_available() else "gloo"
    dist.init_process_group(
        backend=backend, init_method=init_url, world_size=world_size, rank=rank, device_id=torch.device(f"cuda:{rank}")
    )

    shmem = iris.iris(args.heap_size)
    torch.cuda.set_device(rank)
    world_size = shmem.get_num_ranks()
    torch.cuda.set_device(rank)

    output_dir = args.output_dir

    if rank == 0:
        os.makedirs(output_dir, exist_ok=True)
    shmem.barrier()

    with open(args.config_file, "r") as f:
        configs_to_run = json.load(f)

    shmem.info(f"Loaded {len(configs_to_run)} configurations from {args.config_file}")

    for config in configs_to_run:
        run_args = vars(args).copy()
        run_args.update(config)

        dtype_map = {"fp16": torch.float16, "bf16": torch.bfloat16, "fp32": torch.float32}
        datatype = dtype_map.get(run_args["datatype"])

        M, N, K = run_args["m"], run_args["n"], run_args["k"]
        shmem.info(f"\n--- Running Benchmark for M={M}, N={N}, K={K} ---")

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

        A_global_broadcasted = (
            torch.from_numpy(shmem.broadcast(A_global.cpu().numpy(), source_rank=0)).to(datatype).to("cuda")
        )
        shmem.barrier()

        A_local = A_global_broadcasted[:, rank * K_local : (rank + 1) * K_local].contiguous()

        if rank == 0:
            B = torch.randn((K, N), device="cuda", dtype=datatype)
        else:
            B = torch.empty((K, N), device="cuda", dtype=datatype)

        B = torch.from_numpy(shmem.broadcast(B.cpu().numpy(), source_rank=0)).to(datatype).to("cuda")
        shmem.barrier()

        C = torch.empty((M, N), device="cuda", dtype=datatype)
        A_local_iris = shmem.empty((M, K_local), dtype=datatype)
        A_local_iris.copy_(A_local)

        # Use provided num_sms or auto-detect
        if run_args["num_sms"] is None:
            num_sms = torch.cuda.get_device_properties(rank).multi_processor_count
            run_args["num_sms"] = num_sms
        else:
            num_sms = run_args["num_sms"]

        main_stream = torch.cuda.Stream()
        kernel_timing = {
            "fused_ag_gemm": {
                "start_event": torch.cuda.Event(enable_timing=True),
                "end_event": torch.cuda.Event(enable_timing=True),
                "ms": 0,
                "experiments": 0,
            }
        }

        def run_experiment():
            nonlocal kernel_timing
            with torch.cuda.stream(main_stream):
                kernel_timing["fused_ag_gemm"]["start_event"].record()
                persistent_ag_gemm[(num_sms,)](
                    A_local_iris,
                    B,
                    C,
                    M,
                    N,
                    K,
                    A_local_iris.stride(0),
                    A_local_iris.stride(1),
                    B.stride(0),
                    B.stride(1),
                    C.stride(0),
                    C.stride(1),
                    run_args["BLK_M"],
                    run_args["BLK_N"],
                    run_args["BLK_K"],
                    run_args["gsize_m"],
                    run_args["num_sms"],
                    1,  # NUM_XCDs
                    (K % run_args["BLK_K"] == 0),
                    shmem.get_heap_bases(),
                    rank,
                    world_size,
                )
                kernel_timing["fused_ag_gemm"]["end_event"].record()
                kernel_timing["fused_ag_gemm"]["experiments"] += 1

            shmem.barrier()

            ms = kernel_timing["fused_ag_gemm"]["start_event"].elapsed_time(kernel_timing["fused_ag_gemm"]["end_event"])
            kernel_timing["fused_ag_gemm"]["ms"] += ms

        run_experiment()
        shmem.barrier()
        kernel_timing["fused_ag_gemm"]["ms"] = 0
        kernel_timing["fused_ag_gemm"]["experiments"] = 0

        if args.benchmark:
            triton_ms = iris.do_bench(run_experiment, barrier_fn=shmem.barrier)
            tflops = 2 * M * N * K * 1e-12 / (triton_ms * 1e-3)

            shmem.info(f"Result (iris.do_bench): {triton_ms:.3f} ms, {tflops:.3f} TFLOPS")
            json_writer.add_field("total_ms", triton_ms)
            json_writer.add_field("tflops", tflops)

            key = "fused_ag_gemm"
            avg_kernel_ms = kernel_timing[key]["ms"] / kernel_timing[key]["experiments"]
            json_writer.add_field(key + "_ms", avg_kernel_ms)
            shmem.info(f"Result (CUDA Events): {avg_kernel_ms:.3f} ms for the kernel")

        if args.validate:
            if not args.benchmark:
                run_experiment()
                shmem.barrier()

            success = validate_gemm(A_global_broadcasted, B, C, shmem, atol=1.0)

            passed_str = "passed" if success else "failed"
            shmem.info(f"Final C validation {passed_str}.")
            json_writer.add_field("validation_passed", success)

        if rank == 0:
            json_writer.flush()
            shmem.info(f"Saved results to {full_output_path}")

    shmem.info("\nBenchmark sweep complete.")

    shmem.barrier()
    dist.destroy_process_group()


def main():
    args = parse_args()
    if not args.validate and not args.benchmark:
        print("Error: You must specify a mode to run.")
        print("Please use -v for validation or -b for benchmarking.")
        sys.exit(1)
    num_ranks = args.num_ranks
    init_url = "tcp://127.0.0.1:29501"
    mp.spawn(
        fn=worker,
        args=(num_ranks, init_url, args),
        nprocs=num_ranks,
        join=True,
    )


if __name__ == "__main__":
    main()
