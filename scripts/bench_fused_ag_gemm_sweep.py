#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
# Copyright (c) 2025 Advanced Micro Devices, Inc. All rights reserved.

"""
Parameter sweep benchmark for fused all-gather + GEMM.

Compares the iris fused kernel against sequential RCCL all-gather + rocBLAS GEMM
across multiple tiling configurations and problem sizes.

Usage:
    torchrun --nproc_per_node=4 scripts/bench_fused_ag_gemm_sweep.py
"""

import gc
import os
import sys
import time

import torch
import torch.distributed as dist

# Ensure iris is importable
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import iris
from iris.ccl import Config


def do_bench(fn, warmup=50, rep=200):
    """Benchmark a function using CUDA events. Returns median latency in us."""
    # Warmup
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()

    # Measure
    timings = []
    for _ in range(rep):
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        fn()
        end.record()
        torch.cuda.synchronize()
        timings.append(start.elapsed_time(end) * 1000)  # ms -> us

    timings.sort()
    return timings[len(timings) // 2]  # median


def benchmark_sequential(A_shard, weight, rank, world_size, warmup=50, rep=200):
    """Benchmark sequential: RCCL all_gather + torch.matmul."""
    gathered = [torch.zeros_like(A_shard) for _ in range(world_size)]

    def fn():
        dist.all_gather(gathered, A_shard)
        A_full = torch.cat(gathered, dim=1)
        torch.matmul(A_full, weight)

    return do_bench(fn, warmup=warmup, rep=rep)


def benchmark_fused(shmem, A_shard_sym, weight_sym, output, config, block_size_k, warmup=50, rep=200):
    """Benchmark fused AG+GEMM."""
    def fn():
        shmem.ccl.all_gather_gemm(output, A_shard_sym, weight_sym, config=config, block_size_k=block_size_k)

    return do_bench(fn, warmup=warmup, rep=rep)


def main():
    if not dist.is_initialized():
        print("Run with: torchrun --nproc_per_node=<N> scripts/bench_fused_ag_gemm_sweep.py")
        sys.exit(1)

    rank = dist.get_rank()
    world_size = dist.get_world_size()
    torch.cuda.set_device(rank)
    device = torch.device(f"cuda:{rank}")

    # Get hardware info
    props = torch.cuda.get_device_properties(device)
    total_sms = props.multi_processor_count

    if rank == 0:
        print(f"GPU: {props.name}")
        print(f"SMs: {total_sms}")
        print(f"World size: {world_size}")
        print(f"ROCm/CUDA: {torch.version.hip if hasattr(torch.version, 'hip') else torch.version.cuda}")
        print()

    # Problem sizes: (M, K_local, N)
    problem_sizes = [
        (128, 64, 64),       # Small — batch-1 inference
        (256, 128, 256),     # Small-medium
        (1024, 1024, 1024),  # Medium
        (2048, 2048, 4096),  # Large TP (H=8192 with world_size=4)
        (4096, 512, 512),    # Tall-skinny
        (4096, 2048, 4096),  # Large TP
        # LLM-relevant sizes (Llama-70B style with TP=4)
        (1, 2048, 8192),     # Batch-1 inference, TP linear
        (32, 2048, 8192),    # Small batch inference
        (1024, 2048, 8192),  # Medium batch
        (4096, 3584, 3584),  # Near-square
    ]

    # Tiling configs to sweep: (BLK_M, BLK_N, BLK_K, GROUP_SIZE_M, num_warps, num_stages, num_sms)
    tiling_configs = [
        # Original defaults
        (32, 64, 64, 4, 4, 1, 64, "original"),
        # Use all SMs with original tiles
        (32, 64, 64, 4, 4, 1, total_sms, "all_sms_32x64"),
        # Larger tiles matching the example benchmark
        (256, 64, 64, 6, 4, 1, total_sms, "256x64_gm6"),
        # Wider tiles
        (128, 128, 64, 4, 4, 1, total_sms, "128x128"),
        # Software pipelining
        (256, 64, 64, 6, 4, 2, total_sms, "256x64_s2"),
        (256, 64, 64, 6, 4, 4, total_sms, "256x64_s4"),
        # More warps
        (256, 64, 64, 6, 8, 1, total_sms, "256x64_w8"),
        (128, 128, 64, 4, 8, 1, total_sms, "128x128_w8"),
        # Different K block sizes
        (256, 64, 128, 6, 4, 1, total_sms, "256x64_k128"),
        (128, 64, 64, 4, 4, 1, total_sms, "128x64"),
        # Small tiles for small problems
        (64, 64, 32, 4, 4, 1, total_sms, "64x64_k32"),
    ]

    dtype = torch.float16
    heap_size = 2**33  # 8 GB
    shmem = iris.iris(heap_size)

    if rank == 0:
        print("=" * 140)
        print(f"{'M':>6} {'K_local':>7} {'N':>6} {'K':>6} | {'Config':>16} | {'BLK_M':>5} {'BLK_N':>5} {'BLK_K':>5} {'GSM':>3} {'W':>2} {'S':>2} {'SMS':>4} | {'Fused(us)':>10} {'Seq(us)':>10} {'Speedup':>8} {'TFLOPS':>7}")
        print("=" * 140)

    results = []

    for M, K_local, N in problem_sizes:
        K = K_local * world_size

        # Allocate tensors
        torch.manual_seed(42 + rank)
        A_shard = torch.randn(M, K_local, dtype=dtype, device=device)
        torch.manual_seed(123)
        weight = torch.randn(K, N, dtype=dtype, device=device)

        # Sequential baseline (only measure once per problem size)
        seq_us = benchmark_sequential(A_shard, weight, rank, world_size, warmup=30, rep=100)

        # Allocate symmetric tensors
        A_shard_sym = shmem.zeros((M, K_local), dtype=dtype)
        A_shard_sym.copy_(A_shard)
        weight_sym = shmem.zeros((K, N), dtype=dtype)
        weight_sym.copy_(weight)
        output = shmem.zeros((M, N), dtype=dtype)
        shmem.barrier()

        best_fused_us = float('inf')
        best_config_name = ""

        for blk_m, blk_n, blk_k, gsm, nw, ns, nsms, name in tiling_configs:
            # Skip configs where block sizes are larger than problem dimensions
            if blk_m > M or blk_n > N or blk_k > K_local:
                continue

            try:
                config = Config(
                    block_size_m=blk_m,
                    block_size_n=blk_n,
                    swizzle_size=gsm,
                    comm_sms=nsms,
                    num_warps=nw,
                    num_stages=ns,
                )

                fused_us = benchmark_fused(shmem, A_shard_sym, weight_sym, output, config, blk_k, warmup=30, rep=100)
                speedup = seq_us / fused_us
                tflops = 2 * M * N * K / (fused_us * 1e-6) / 1e12

                if fused_us < best_fused_us:
                    best_fused_us = fused_us
                    best_config_name = name

                if rank == 0:
                    marker = " <-- best" if fused_us == best_fused_us else ""
                    print(f"{M:>6} {K_local:>7} {N:>6} {K:>6} | {name:>16} | {blk_m:>5} {blk_n:>5} {blk_k:>5} {gsm:>3} {nw:>2} {ns:>2} {nsms:>4} | {fused_us:>10.1f} {seq_us:>10.1f} {speedup:>7.2f}x {tflops:>7.2f}{marker}")

                results.append({
                    'M': M, 'K_local': K_local, 'N': N, 'K': K,
                    'config': name, 'blk_m': blk_m, 'blk_n': blk_n, 'blk_k': blk_k,
                    'gsm': gsm, 'nw': nw, 'ns': ns, 'nsms': nsms,
                    'fused_us': fused_us, 'seq_us': seq_us, 'speedup': speedup, 'tflops': tflops,
                })

            except Exception as e:
                if rank == 0:
                    print(f"{M:>6} {K_local:>7} {N:>6} {K:>6} | {name:>16} | {blk_m:>5} {blk_n:>5} {blk_k:>5} {gsm:>3} {nw:>2} {ns:>2} {nsms:>4} | FAILED: {e}")

        if rank == 0:
            print(f"{'':>6} {'':>7} {'':>6} {'':>6} | {'** BEST **':>16} | {best_config_name:>31} | {best_fused_us:>10.1f} {seq_us:>10.1f} {seq_us/best_fused_us:>7.2f}x")
            print("-" * 140)

        # Cleanup symmetric tensors for this problem size
        shmem.barrier()
        del A_shard_sym, weight_sym, output
        gc.collect()

    # Print summary: best config per problem size
    if rank == 0:
        print()
        print("=" * 100)
        print("SUMMARY: Best config per problem size")
        print("=" * 100)
        print(f"{'M':>6} {'K_local':>7} {'N':>6} | {'Best Config':>16} | {'Fused(us)':>10} {'Seq(us)':>10} {'Speedup':>8} {'TFLOPS':>7}")
        print("-" * 100)

        seen = set()
        for r in results:
            key = (r['M'], r['K_local'], r['N'])
            if key in seen:
                continue
            # Find best for this problem size
            best = min([x for x in results if (x['M'], x['K_local'], x['N']) == key], key=lambda x: x['fused_us'])
            seen.add(key)
            print(f"{best['M']:>6} {best['K_local']:>7} {best['N']:>6} | {best['config']:>16} | {best['fused_us']:>10.1f} {best['seq_us']:>10.1f} {best['speedup']:>7.2f}x {best['tflops']:>7.2f}")

        # Compute MI325X roofline
        print()
        print("MI325X Theoretical Peaks:")
        print("  FP16 Matrix: 1307.4 TFLOPS")
        print("  HBM bandwidth: 6.0 TB/s")
        print("  XGMI bandwidth: 896 GB/s (bidirectional, 4 links)")

    shmem.barrier()
    del shmem
    gc.collect()


if __name__ == "__main__":
    main()
