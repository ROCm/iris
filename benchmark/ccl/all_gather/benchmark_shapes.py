#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
# Copyright (c) 2025 Advanced Micro Devices, Inc. All rights reserved.

"""
Comprehensive all-gather benchmark: shape sweep x CU sweep.

Compares RCCL (default channels), Iris Triton persistent, and Iris Gluon flat-2D
across multiple tensor shapes and CU counts.

Usage:
    torchrun --nproc_per_node=8 benchmark/ccl/all_gather/benchmark_shapes.py [--csv results.csv]
"""

import argparse
import csv
import io
import os

import torch
import torch.distributed as dist

import iris
from iris.ccl import Config
import iris.experimental.iris_gluon as iris_gluon


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

SHAPES = [
    (1024, 1024),    # 2 MB   - small activations
    (2048, 4096),    # 16 MB  - medium MLP
    (4096, 4096),    # 32 MB  - GPT-scale hidden
    (8192, 8192),    # 128 MB - large MLP / standard bench
    (16384, 8192),   # 256 MB - long sequences
    (16384, 16384),  # 512 MB - large model partitions
]

CU_COUNTS = [8, 16, 32, 64, 96]

DTYPE = torch.float16
DTYPE_STR = "fp16"
ELEMENT_SIZE = 2  # bytes per fp16 element

# Default benchmark parameters
DEFAULT_N_WARMUP = 25
DEFAULT_N_REPEAT = 100


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def calc_bandwidth_gbps(M, N, world_size, ms):
    """Calculate all-gather bandwidth in GB/s."""
    total_bytes = (world_size - 1) * M * N * ELEMENT_SIZE
    total_gb = total_bytes / (1024**3)
    return total_gb / (ms * 1e-3) if ms > 0 else 0.0


def bench_rccl(M, N, rank, world_size, n_warmup, n_repeat):
    """Benchmark RCCL all_gather_into_tensor at default channel config."""
    inp = torch.zeros(M, N, dtype=DTYPE, device=f"cuda:{rank}")
    inp.fill_(float(rank + 1))
    out = torch.zeros(world_size * M, N, dtype=DTYPE, device=f"cuda:{rank}")

    # Warmup
    for _ in range(10):
        dist.all_gather_into_tensor(out, inp)
    torch.cuda.synchronize()
    dist.barrier()

    out.zero_()
    inp.fill_(float(rank + 1))
    dist.barrier()

    def fn():
        dist.all_gather_into_tensor(out, inp)

    ms = iris.do_bench(fn, dist.barrier, n_warmup=n_warmup, n_repeat=n_repeat)
    return ms


def bench_iris(M, N, shmem, config, n_warmup, n_repeat):
    """Benchmark Iris all-gather (Triton or Gluon depending on config)."""
    world_size = shmem.get_num_ranks()
    rank = shmem.get_rank()

    inp = shmem.zeros((M, N), dtype=DTYPE)
    out = shmem.zeros((world_size * M, N), dtype=DTYPE)

    inp.fill_(float(rank + 1))
    shmem.barrier()

    def fn():
        shmem.ccl.all_gather(out, inp, config=config, async_op=False)

    ms = iris.do_bench(fn, shmem.barrier, n_warmup=n_warmup, n_repeat=n_repeat)

    # Free symmetric heap memory for next iteration
    del inp, out
    torch.cuda.empty_cache()
    shmem.barrier()

    return ms


def validate_iris(M, N, shmem, config):
    """Quick correctness check for an Iris config. Returns True if correct."""
    world_size = shmem.get_num_ranks()
    rank = shmem.get_rank()

    inp = shmem.zeros((M, N), dtype=DTYPE)
    out = shmem.zeros((world_size * M, N), dtype=DTYPE)

    inp.fill_(float(rank + 1))
    out.zero_()
    shmem.barrier()

    shmem.ccl.all_gather(out, inp, config=config, async_op=False)
    shmem.barrier()
    torch.cuda.synchronize()

    ok = True
    for r in range(world_size):
        expected = float(r + 1)
        chunk = out[r * M : (r + 1) * M, :]
        if not torch.allclose(chunk, torch.full_like(chunk, expected), atol=1e-3):
            ok = False
            break

    del inp, out
    torch.cuda.empty_cache()
    shmem.barrier()
    return ok


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="All-gather shape + CU sweep benchmark")
    parser.add_argument("--csv", type=str, default=None, help="Output CSV file path")
    parser.add_argument("--heap_size", type=int, default=1 << 34, help="Iris heap size (default 16 GB)")
    parser.add_argument("--validate", action="store_true", help="Validate correctness before benchmarking")
    parser.add_argument("--n_warmup", type=int, default=DEFAULT_N_WARMUP, help="Warmup iterations")
    parser.add_argument("--n_repeat", type=int, default=DEFAULT_N_REPEAT, help="Benchmark iterations")
    args = parser.parse_args()

    n_warmup = args.n_warmup
    n_repeat = args.n_repeat

    # torchrun sets these env vars
    local_rank = int(os.environ["LOCAL_RANK"])
    world_size = int(os.environ["WORLD_SIZE"])

    torch.cuda.set_device(local_rank)
    dist.init_process_group(
        backend="nccl",
        device_id=torch.device(f"cuda:{local_rank}"),
    )

    rank = dist.get_rank()
    is_root = rank == 0

    # Initialize both Triton and Gluon shmem contexts.
    # They share the same underlying symmetric heap / distributed state,
    # but dispatch to different kernel backends.
    shmem_triton = iris.iris(args.heap_size)
    shmem_gluon = iris_gluon.iris(args.heap_size)

    # Collect results: list of dicts
    results = []

    # Table header
    if is_root:
        hdr = (
            f"{'Shape':>14s}  {'Size':>7s}  {'Backend':>10s}  {'CUs':>4s}  "
            f"{'Time(ms)':>9s}  {'BW(GB/s)':>9s}  {'vs RCCL':>7s}"
        )
        print("=" * len(hdr))
        print(hdr)
        print("=" * len(hdr))

    for M, N in SHAPES:
        data_mb = M * N * ELEMENT_SIZE / (1024**2)
        shape_str = f"{M}x{N}"

        # --- Optional validation ---
        if args.validate:
            for cu in [CU_COUNTS[-1]]:  # validate at highest CU count only
                triton_cfg = Config(comm_sms=cu)
                gluon_cfg = Config(comm_sms=cu, use_gluon=True)

                ok_t = validate_iris(M, N, shmem_triton, triton_cfg)
                ok_g = validate_iris(M, N, shmem_gluon, gluon_cfg)

                if is_root:
                    if not ok_t:
                        print(f"WARNING: Triton validation FAILED for {shape_str} cu={cu}")
                    if not ok_g:
                        print(f"WARNING: Gluon  validation FAILED for {shape_str} cu={cu}")

        # --- RCCL baseline ---
        rccl_ms = bench_rccl(M, N, rank, world_size, n_warmup, n_repeat)
        rccl_bw = calc_bandwidth_gbps(M, N, world_size, rccl_ms)

        row = {
            "shape": shape_str, "M": M, "N": N, "data_mb": data_mb,
            "backend": "RCCL", "cus": "-", "time_ms": rccl_ms,
            "bw_gbps": rccl_bw, "vs_rccl_pct": 100.0,
        }
        results.append(row)

        if is_root:
            print(
                f"{shape_str:>14s}  {data_mb:6.0f}M  {'RCCL':>10s}  {'-':>4s}  "
                f"{rccl_ms:9.3f}  {rccl_bw:9.1f}  {100.0:6.1f}%"
            )

        # --- Iris Triton + Gluon at each CU count ---
        for cu in CU_COUNTS:
            for backend_name, shmem, use_gluon in [
                ("Triton", shmem_triton, False),
                ("Gluon", shmem_gluon, True),
            ]:
                cfg = Config(comm_sms=cu, use_gluon=use_gluon)
                ms = bench_iris(M, N, shmem, cfg, n_warmup, n_repeat)
                bw = calc_bandwidth_gbps(M, N, world_size, ms)
                vs_rccl = (bw / rccl_bw * 100) if rccl_bw > 0 else 0.0

                row = {
                    "shape": shape_str, "M": M, "N": N, "data_mb": data_mb,
                    "backend": backend_name, "cus": cu, "time_ms": ms,
                    "bw_gbps": bw, "vs_rccl_pct": vs_rccl,
                }
                results.append(row)

                if is_root:
                    print(
                        f"{shape_str:>14s}  {data_mb:6.0f}M  {backend_name:>10s}  {cu:4d}  "
                        f"{ms:9.3f}  {bw:9.1f}  {vs_rccl:6.1f}%"
                    )

        # Separator between shapes
        if is_root:
            print("-" * 72)

    # --- Summary CSV ---
    if is_root:
        buf = io.StringIO()
        writer = csv.DictWriter(
            buf,
            fieldnames=["shape", "M", "N", "data_mb", "backend", "cus", "time_ms", "bw_gbps", "vs_rccl_pct"],
        )
        writer.writeheader()
        writer.writerows(results)

        csv_text = buf.getvalue()

        if args.csv:
            with open(args.csv, "w") as f:
                f.write(csv_text)
            print(f"\nResults written to {args.csv}")
        else:
            print("\n--- CSV ---")
            print(csv_text)

    dist.barrier()
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
