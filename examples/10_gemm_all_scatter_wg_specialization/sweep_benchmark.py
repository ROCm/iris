#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
# Copyright (c) 2025 Advanced Micro Devices, Inc. All rights reserved.
#
# Benchmark sweep for example 10: GEMM + All-Scatter with Workgroup Specialization
#
# Usage:
#   torchrun --rdzv-backend=c10d --rdzv-endpoint=localhost:0 --nnodes=1 \
#            --nproc_per_node=<num_ranks> \
#            sweep_benchmark.py [--output_file sweep_results.csv]
#
# This script sweeps over matrix sizes, block sizes, datatypes, and num_stages,
# collecting TFLOPS, latency, and arithmetic intensity for a roofline analysis.

import argparse
import csv
import json
import math
import os
import random
import sys

import torch
import torch.distributed as dist
import triton

import iris
from matmul_wrapper import matmul

torch.manual_seed(123)
random.seed(123)

# Peak TFLOPS for AMD Instinct MI325X (per GPU, matrix cores)
MI325X_PEAK_MATRIX_TFLOPS = 1307.4  # FP16/BF16 matrix cores (CDNA3)

DTYPE_BYTES = {"fp16": 2, "bf16": 2, "fp32": 4}

# Sweep parameters
MATRIX_SIZES = [
    (4096, 4608, 36864),
    (8192, 4608, 36864),
    (16384, 4608, 36864),
    (8192, 9216, 36864),
    (8192, 4608, 18432),
]
BLOCK_SIZES = [
    (256, 64, 64),
    (128, 128, 64),
    (256, 128, 64),
]
DATATYPES = ["fp16", "bf16"]
NUM_STAGES_LIST = [2, 3]

HEAP_SIZE = 1 << 33
GSIZE_M = 6


def _nearest_power_of_two_floor(n):
    """Return the largest power of two less than or equal to n."""
    return 2 ** int(math.log2(n)) if n > 0 else 1


def arithmetic_intensity(M, N, K, dtype_bytes):
    """Compute arithmetic intensity (flops/byte) for GEMM C = A * B."""
    flops = 2 * M * N * K
    bytes_io = (M * K + K * N + M * N) * dtype_bytes
    return flops / bytes_io


def run_single_benchmark(rank, world_size, M, N, K, BLK_M, BLK_N, BLK_K, gsize_m, num_stages, datatype_str):
    """Run a single benchmark configuration and return timing results."""
    datatype = {"fp16": torch.float16, "bf16": torch.bfloat16, "fp32": torch.float32}[datatype_str]

    if N % world_size != 0 or K % world_size != 0:
        return None  # Skip invalid config

    shmem = iris.iris(HEAP_SIZE)
    assert shmem.get_rank() == rank

    cu_count = torch.cuda.get_device_properties(rank).multi_processor_count
    gemm_sms = _nearest_power_of_two_floor(cu_count)
    num_sms = cu_count

    n_per_rank = N // world_size

    A = shmem.randn(M, K, device="cuda", dtype=datatype)
    B = shmem.randn(N, K, device="cuda", dtype=datatype).T
    local_B = B[:, rank * n_per_rank : (rank + 1) * n_per_rank].clone()
    local_A = A

    global_C = shmem.zeros((M, N), device="cuda", dtype=datatype)
    local_C = shmem.zeros((M, n_per_rank), device="cuda", dtype=datatype)

    total_blocks_M = triton.cdiv(M, BLK_M)
    total_blocks_N = triton.cdiv(n_per_rank, BLK_N)
    total_tiles = total_blocks_M * total_blocks_N

    locks = shmem.zeros((total_tiles,), device="cuda", dtype=torch.int8)
    bias = None
    gemm_stream = torch.cuda.Stream()

    def run_experiment():
        nonlocal local_C
        nonlocal shmem
        locks.zero_()
        shmem.barrier()
        with torch.cuda.stream(gemm_stream):
            local_C = matmul.apply(
                local_A,
                local_B,
                local_C,
                global_C,
                bias,
                locks,
                rank,
                world_size,
                gemm_sms,
                num_sms,
                BLK_M,
                BLK_N,
                BLK_K,
                gsize_m,
                num_stages,
                shmem.get_heap_bases(),
                "gfx942",
                False,
                None,
                None,
            )
        shmem.barrier()

    shmem.barrier()
    try:
        triton_ms = iris.do_bench(run_experiment, shmem.barrier)
    except Exception as e:
        shmem.barrier()
        return None

    shmem.barrier()
    del shmem
    return triton_ms


def _worker(rank, world_size, init_url, output_file):
    backend = "nccl" if torch.cuda.is_available() else "gloo"
    dist.init_process_group(
        backend=backend,
        init_method=init_url,
        world_size=world_size,
        rank=rank,
        device_id=torch.device(f"cuda:{rank}"),
    )
    torch.cuda.set_device(rank)

    results = []

    for M, N, K in MATRIX_SIZES:
        for BLK_M, BLK_N, BLK_K in BLOCK_SIZES:
            for dtype_str in DATATYPES:
                for num_stages in NUM_STAGES_LIST:
                    # Skip configs where N or K not divisible by world_size
                    if N % world_size != 0 or K % world_size != 0:
                        if rank == 0:
                            print(f"  Skipping ({M},{N},{K}) with {world_size} ranks: N/K not divisible")
                        continue

                    # Skip configs where block sizes don't fit
                    n_per_rank = N // world_size
                    if n_per_rank < BLK_N:
                        if rank == 0:
                            print(f"  Skipping ({M},{N},{K}) BLK_N={BLK_N}: n_per_rank={n_per_rank} < BLK_N")
                        continue

                    if rank == 0:
                        print(
                            f"Running: M={M} N={N} K={K} "
                            f"BLK=({BLK_M},{BLK_N},{BLK_K}) dtype={dtype_str} "
                            f"stages={num_stages} ranks={world_size} ...",
                            flush=True,
                        )

                    ms = run_single_benchmark(
                        rank, world_size, M, N, K, BLK_M, BLK_N, BLK_K, GSIZE_M, num_stages, dtype_str
                    )

                    if ms is not None and rank == 0:
                        tflops = 2 * M * N * K * 1e-12 / (ms * 1e-3)
                        dtype_bytes = DTYPE_BYTES[dtype_str]
                        ai = arithmetic_intensity(M, N, K, dtype_bytes)
                        peak = MI325X_PEAK_MATRIX_TFLOPS
                        efficiency = tflops / (peak * world_size) * 100

                        row = {
                            "M": M,
                            "N": N,
                            "K": K,
                            "BLK_M": BLK_M,
                            "BLK_N": BLK_N,
                            "BLK_K": BLK_K,
                            "dtype": dtype_str,
                            "num_stages": num_stages,
                            "num_ranks": world_size,
                            "total_ms": round(ms, 3),
                            "tflops": round(tflops, 2),
                            "arith_intensity": round(ai, 2),
                            "peak_tflops_per_gpu": peak,
                            "peak_tflops_total": peak * world_size,
                            "efficiency_pct": round(efficiency, 2),
                        }
                        results.append(row)
                        print(
                            f"  -> {tflops:.1f} TFLOPS  {ms:.2f} ms  AI={ai:.1f}  eff={efficiency:.1f}%",
                            flush=True,
                        )

    if rank == 0 and results:
        base, _ = os.path.splitext(output_file)

        # Write CSV
        csv_file = base + ".csv"
        with open(csv_file, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=results[0].keys())
            writer.writeheader()
            writer.writerows(results)
        print(f"\nResults written to {csv_file}")

        # Write JSON
        json_file = base + ".json"
        with open(json_file, "w") as f:
            json.dump(results, f, indent=2)
        print(f"Results written to {json_file}")

        # Print roofline summary table
        print("\n" + "=" * 110)
        print(
            f"{'M':>6} {'N':>6} {'K':>6} {'BLK':>12} {'dtype':>5} "
            f"{'stg':>3} {'rnks':>4} {'ms':>7} {'TFLOPS':>8} {'AI':>7} {'peak':>8} {'eff%':>6}"
        )
        print("-" * 110)
        for r in sorted(results, key=lambda x: -x["tflops"]):
            blk_str = f"{r['BLK_M']}x{r['BLK_N']}x{r['BLK_K']}"
            print(
                f"{r['M']:>6} {r['N']:>6} {r['K']:>6} {blk_str:>12} {r['dtype']:>5} "
                f"{r['num_stages']:>3} {r['num_ranks']:>4} "
                f"{r['total_ms']:>7.2f} {r['tflops']:>8.1f} "
                f"{r['arith_intensity']:>7.1f} {r['peak_tflops_total']:>8.0f} "
                f"{r['efficiency_pct']:>6.1f}"
            )
        print("=" * 110)
        print(f"\nPeak per-GPU: {MI325X_PEAK_MATRIX_TFLOPS} TFLOPS (fp16/bf16 matrix cores, MI325X)")

    dist.destroy_process_group()


def main():
    parser = argparse.ArgumentParser(description="Benchmark sweep for example 10")
    parser.add_argument("--output_file", type=str, default="sweep_results.json", help="Output file (JSON)")
    args = parser.parse_args()

    if "RANK" in os.environ and "LOCAL_RANK" in os.environ:
        rank = int(os.environ["RANK"])
        world_size = int(os.environ["WORLD_SIZE"])
        master_addr = os.environ.get("MASTER_ADDR", "127.0.0.1")
        master_port = os.environ.get("MASTER_PORT", "29500")
        init_url = f"tcp://{master_addr}:{master_port}"
        _worker(rank, world_size, init_url, args.output_file)
    else:
        print("Error: This script must be launched with torchrun.")
        print(
            "Example: torchrun --rdzv-backend=c10d --rdzv-endpoint=localhost:0 "
            "--nnodes=1 --nproc_per_node=2 sweep_benchmark.py"
        )
        sys.exit(1)


if __name__ == "__main__":
    main()
