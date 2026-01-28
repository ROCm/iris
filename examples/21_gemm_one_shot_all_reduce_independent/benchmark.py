#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
# Copyright (c) 2025 Advanced Micro Devices, Inc. All rights reserved.

import torch
import torch.distributed as dist
import torch.multiprocessing as mp
import triton
import random
import os
import argparse
import csv

from examples.common.utils import JSONWriter, Timestamps, is_triton_interpret_set
from examples.common.validation import validate_gemm, validate_all_reduce

import iris

from matmul_wrapper import matmul
from all_reduce_wrapper import all_reduce_kernel

torch.manual_seed(123)
random.seed(123)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Parse matrix dimensions and configuration.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("-m", type=int, default=8192, help="Number of rows in matrix A (GEMM)")
    parser.add_argument("-n", type=int, default=4608, help="Number of columns in matrix B (GEMM)")
    parser.add_argument("-k", type=int, default=36864, help="Common dimension between matrices A and B (GEMM)")
    parser.add_argument("--m_comm", type=int, default=None, help="Number of rows for all-reduce tensor (defaults to m)")
    parser.add_argument(
        "--n_comm", type=int, default=None, help="Number of columns for all-reduce tensor (defaults to n)"
    )
    parser.add_argument("-d", "--debug", action="store_true", help="Enable debug mode")
    parser.add_argument("-v", "--validate", action="store_true", help="Enable validation mode")
    parser.add_argument("-t", "--trace_tiles", action="store_true", help="Enable tile-tracing mode")
    parser.add_argument("-b", "--benchmark", action="store_true", help="Enable benchmarking mode")
    parser.add_argument(
        "--datatype",
        type=str,
        default="fp16",
        choices=["fp16", "fp32", "bf16"],
        help="Datatype of computation",
    )
    parser.add_argument(
        "--output_file",
        type=str,
        default="log.json",
        help="Output file",
    )
    parser.add_argument("--BLK_M", type=int, default=256, help="Block size M")
    parser.add_argument("--BLK_N", type=int, default=64, help="Block size N")
    parser.add_argument("--BLK_K", type=int, default=64, help="Block size K")
    parser.add_argument("--gsize_m", type=int, default=6, help="L2-cache locality swizzle parameter")
    parser.add_argument("--num_stages", type=int, default=2, help="Number of stages")
    parser.add_argument("--heap_size", type=int, default=1 << 33, help="Iris heap size")
    parser.add_argument("--gemm_sms", type=int, default=256, help="Number of SMs for GEMM kernel")
    parser.add_argument("--comm_sms", type=int, default=48, help="Number of SMs for All-Reduce kernel")
    parser.add_argument("-r", "--num_ranks", type=int, default=8, help="Number of ranks/processes")
    parser.add_argument(
        "--csv",
        type=str,
        default=None,
        help="Path to CSV file with configurations (columns: m, n, k, datatype, blk_m, blk_n, blk_k, gemm_sms, comm_sms)",
    )
    parser.add_argument(
        "--only_gemm",
        action="store_true",
        help="Run only GEMM operation (cannot be used with --only_comm)",
    )
    parser.add_argument(
        "--only_comm",
        action="store_true",
        help="Run only communication (all-reduce) operation (cannot be used with --only_gemm)",
    )
    parser.add_argument(
        "--distribution",
        type=int,
        default=0,
        choices=[0, 1],
        help="Distribution mode for all-reduce: 0=striding, 1=block",
    )

    args = vars(parser.parse_args())

    # Validate mutually exclusive flags
    if args["only_gemm"] and args["only_comm"]:
        parser.error("--only_gemm and --only_comm cannot be used together")

    return args


def load_configs_from_csv(csv_path):
    """Load configurations from a CSV file.

    Expected CSV format:
    m,n,k,datatype,blk_m,blk_n,blk_k,gemm_sms,comm_sms
    8192,4608,36864,fp16,128,128,64,256,48
    8192,4096,12288,fp32,256,128,64,256,48
    ...

    Args:
        csv_path: Path to the CSV file

    Returns:
        List of configuration dictionaries
    """
    configs = []
    with open(csv_path, "r") as f:
        reader = csv.DictReader(f)
        for row in reader:
            config = {
                "m": int(row["m"]),
                "n": int(row["n"]),
                "k": int(row["k"]),
                "datatype": row["datatype"],
                "BLK_M": int(row["blk_m"]),
                "BLK_N": int(row["blk_n"]),
                "BLK_K": int(row["blk_k"]),
                "gemm_sms": int(row["gemm_sms"]),
                "comm_sms": int(row["comm_sms"]),
            }
            configs.append(config)
    return configs


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
    cu_count = shmem.get_cu_count()

    # GEMM
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

    # Set default values for all-reduce dimensions if not provided
    if args["m_comm"] is None:
        args["m_comm"] = args["m"]
    if args["n_comm"] is None:
        args["n_comm"] = args["n"]

    A = shmem.randn(args["m"], args["k"], device="cuda", dtype=datatype)
    B = shmem.randn(args["n"], args["k"], device="cuda", dtype=datatype).T

    json_writer = JSONWriter(args["output_file"])
    json_writer.add_field("world_size", world_size)

    local_A = A
    local_B = B

    for key, value in args.items():
        json_writer.add_field(key, value)

    C = shmem.zeros((args["m"], args["n"]), device="cuda", dtype=A.dtype)

    # Create all-reduce tensors (independent from GEMM)
    # Each rank has a value of rank+1
    all_reduce_local = shmem.full((args["m_comm"], args["n_comm"]), rank + 1.0, device="cuda", dtype=datatype)
    all_reduce_result = shmem.zeros((args["m_comm"], args["n_comm"]), device="cuda", dtype=datatype)

    total_blocks_M = triton.cdiv(args["m"], args["BLK_M"])
    total_blocks_N = triton.cdiv(args["n"], args["BLK_N"])
    total_tiles = total_blocks_M * total_blocks_N

    bias = None

    num_xcds = iris.hip.get_num_xcc()

    # Independent streams for GEMM and all-reduce
    gemm_stream = torch.cuda.Stream()
    comm_stream = torch.cuda.Stream()

    json_writer.add_field("gemm_sms", args["gemm_sms"])
    json_writer.add_field("comm_sms", args["comm_sms"])

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

    # Allocate Timestamps
    timestamps = Timestamps(num_tiles=total_tiles)

    def run_experiment():
        nonlocal C
        nonlocal all_reduce_result
        nonlocal kernel_timing

        shmem.barrier()

        if args["trace_tiles"]:
            timestamps.reset()
            shmem.barrier()

        # Determine what to run based on flags
        run_gemm = not args["only_comm"]
        run_comm = not args["only_gemm"]

        # Set NVTX range name based on what we're running
        if run_gemm and run_comm:
            nvtx_name = "GEMM + All-Reduce (Independent)"
        elif run_gemm:
            nvtx_name = "GEMM"
        else:
            nvtx_name = "All-Reduce"

        torch.cuda.nvtx.range_push(nvtx_name)

        if run_gemm:
            torch.cuda.nvtx.range_push("GEMM")
            with torch.cuda.stream(gemm_stream):
                kernel_timing["gemm"]["start_event"].record()
                C = matmul.apply(
                    local_A,
                    local_B,
                    C,
                    bias,
                    rank,
                    world_size,
                    args["gemm_sms"],
                    args["BLK_M"],
                    args["BLK_N"],
                    args["BLK_K"],
                    args["gsize_m"],
                    args["num_stages"],
                    shmem.get_heap_bases(),
                    "gfx942",
                    args["trace_tiles"],
                    timestamps.mm_begin_timestamp,
                    timestamps.mm_end_timestamp,
                )
                kernel_timing["gemm"]["end_event"].record()
                kernel_timing["gemm"]["experiments"] += 1
            torch.cuda.nvtx.range_pop()

        if run_comm:
            torch.cuda.nvtx.range_push("All-Reduce")
            with torch.cuda.stream(comm_stream):
                kernel_timing["communication"]["start_event"].record()
                all_reduce_kernel.run(
                    all_reduce_local,
                    all_reduce_result,
                    args["m_comm"],
                    args["n_comm"],
                    all_reduce_local.stride(0),
                    all_reduce_local.stride(1),
                    all_reduce_result.stride(0),
                    all_reduce_result.stride(1),
                    args["BLK_M"],
                    args["BLK_N"],
                    args["gsize_m"],
                    args["comm_sms"],
                    num_xcds,
                    shmem.get_heap_bases(),
                    rank,
                    world_size,
                    args["distribution"],
                    args["trace_tiles"],
                    timestamps.mm_begin_timestamp,
                    timestamps.mm_end_timestamp,
                )
                kernel_timing["communication"]["end_event"].record()
                kernel_timing["communication"]["experiments"] += 1
            torch.cuda.nvtx.range_pop()

        shmem.barrier()

        # Update timing for operations that were run
        if run_gemm:
            ms = kernel_timing["gemm"]["start_event"].elapsed_time(kernel_timing["gemm"]["end_event"])
            kernel_timing["gemm"]["ms"] += ms
        if run_comm:
            ms = kernel_timing["communication"]["start_event"].elapsed_time(kernel_timing["communication"]["end_event"])
            kernel_timing["communication"]["ms"] += ms

        torch.cuda.nvtx.range_pop()

    # Synchronize across all GPUs
    shmem.barrier()

    # Warmup
    run_experiment()

    shmem.barrier()

    for k in ["gemm", "communication"]:
        kernel_timing[k]["ms"] = 0
        kernel_timing[k]["experiments"] = 0

    if args["validate"]:
        # Ensure all GPU kernels have completed before validation
        torch.cuda.synchronize()
        shmem.barrier()

        shmem.info("Validating...")
        matmul.set_debug(True)
        all_reduce_kernel.set_debug(True)

        # Determine what to validate based on flags
        validate_gemm_op = not args["only_comm"]
        validate_comm_op = not args["only_gemm"]

        success_gemm = True
        success_comm = True

        # Validate GEMM result if it was run
        if validate_gemm_op:
            shmem.info("Validating GEMM operation...")
            success_gemm = validate_gemm(A, B, C, shmem)
            passed_str = "passed" if success_gemm else "failed"
            shmem.info(f"GEMM validation {passed_str}.")
            # Wait for all to finish GEMM validation
            shmem.barrier()

        # Validate all-reduce result if it was run
        if validate_comm_op:
            shmem.info("Validating all-reduce operation...")
            success_comm = validate_all_reduce(all_reduce_local, all_reduce_result, shmem)
            passed_str = "passed" if success_comm else "failed"
            shmem.info(f"All-reduce validation {passed_str}.")
            # Wait for all to finish communication validation
            shmem.barrier()

        # Overall success
        success = success_gemm and success_comm
        overall_str = "passed" if success else "failed"
        shmem.info(f"Overall validation {overall_str}.")

        # Wait for all to finish validation
        shmem.barrier()

        json_writer.add_field("success", success)
        if validate_gemm_op:
            json_writer.add_field("success_gemm", success_gemm)
        if validate_comm_op:
            json_writer.add_field("success_comm", success_comm)

        if not is_triton_interpret_set():
            if validate_gemm_op:
                gemm_registers = matmul.get_matmul_registers()
                gemm_spills = matmul.get_matmul_spills()

                json_writer.add_field("gemm_registers", gemm_registers)
                json_writer.add_field("gemm_spills", gemm_spills)

            if validate_comm_op:
                comm_registers = all_reduce_kernel.get_registers()
                comm_spills = all_reduce_kernel.get_spills()

                json_writer.add_field("comm_registers", comm_registers)
                json_writer.add_field("comm_spills", comm_spills)

        shmem.info("Validation completed")

    if args["benchmark"]:
        matmul.set_debug(False)
        all_reduce_kernel.set_debug(False)
        shmem.info("Benchmarking...")
        perf = lambda ms: 2 * args["m"] * args["n"] * args["k"] * 1e-12 / (ms * 1e-3)
        triton_ms = iris.do_bench(run_experiment, shmem.barrier)
        triton_tflops = perf(triton_ms)

        # Determine what was run based on flags
        run_gemm = not args["only_comm"]
        run_comm = not args["only_gemm"]

        if run_gemm and run_comm:
            op_string = "tile matmul + one_shot_all_reduce (independent)"
        elif run_gemm:
            op_string = "tile matmul"
        else:
            op_string = "one_shot_all_reduce"

        shmem.info(f"{op_string} (total_tiles={total_tiles}): {triton_ms:.3f} ms  {triton_tflops:.3f} tflops")

        json_writer.add_field("tflops", triton_tflops)
        json_writer.add_field("total_ms", triton_ms)

        # Only add timing for operations that were run
        if run_gemm:
            json_writer.add_field("gemm_ms", kernel_timing["gemm"]["ms"] / kernel_timing["gemm"]["experiments"])
            json_writer.add_field("gemm_experiments", kernel_timing["gemm"]["experiments"])
        if run_comm:
            json_writer.add_field(
                "communication_ms", kernel_timing["communication"]["ms"] / kernel_timing["communication"]["experiments"]
            )
            json_writer.add_field("communication_experiments", kernel_timing["communication"]["experiments"])

        # Wait for all to finish benchmarking
        shmem.barrier()

    if rank == 0:
        json_writer.flush()
        json_writer.display()

    if args["trace_tiles"] and rank == 0:
        gpu_freq = iris.hip.get_wall_clock_rate(rank) * 1e-3
        algo_string = "one_shot_all_reduce_independent"
        filename = f"gemm_tiles_{algo_string}_trace_rank{rank}.json"
        timestamps.to_json(filename, gpu_freq)

    shmem.barrier()
    dist.destroy_process_group()


def main():
    args = parse_args()
    num_ranks = args["num_ranks"]
    init_url = "tcp://127.0.0.1:29500"

    # If CSV is provided, run sweep with configurations from CSV
    if args["csv"] is not None:
        configs = load_configs_from_csv(args["csv"])
        print(f"Loaded {len(configs)} configurations from {args['csv']}")

        for i, config in enumerate(configs):
            # Create a copy of args and update with CSV config
            run_args = args.copy()
            run_args.update(config)

            print(
                f"\nRunning configuration {i + 1}/{len(configs)}:\n"
                + "\n".join(
                    f"\t{k}={config[k]}"
                    for k in ["m", "n", "k", "datatype", "BLK_M", "BLK_N", "BLK_K", "gemm_sms", "comm_sms"]
                )
            )
            # Generate unique output filename for this configuration
            base_name, ext = os.path.splitext(args["output_file"])
            run_args["output_file"] = (
                f"{base_name}_m{config['m']}_n{config['n']}_k{config['k']}_{config['datatype']}_{config['BLK_M']}_{config['BLK_N']}_{config['BLK_K']}_{config['gemm_sms']}_{config['comm_sms']}{ext}"
            )

            mp.spawn(
                fn=_worker,
                args=(num_ranks, init_url, run_args),
                nprocs=num_ranks,
                join=True,
            )
    else:
        # Single run with command line arguments
        mp.spawn(
            fn=_worker,
            args=(num_ranks, init_url, args),
            nprocs=num_ranks,
            join=True,
        )


if __name__ == "__main__":
    main()
