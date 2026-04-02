#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
# Copyright (c) 2025-2026 Advanced Micro Devices, Inc. All rights reserved.

"""
Parameter derivation for matmul_all_gather_copy_engine.

Given a problem size (M, N, K), world size, derives kernel parameters
for the fused GEMM → scatter pattern where each rank:
1. Computes local GEMM: (M_local, K) @ (K, N) → (M_local, N)
2. Scatters result tiles to all other ranks via SDMA

Key differences from all_gather_matmul:
- SMALLER local GEMM (M_local instead of M_total)
- SCATTER communication (send results vs gather inputs)
- PERSISTENT kernel with per-tile GEMM+scatter fusion
- SDMA overhead dominates (not GEMM)

Usage:
    python derive_params.py -m 131072 -n 2048 -k 16384
    python derive_params.py -m 16384 -n 2048 -k 16384 --link_bw 50
"""

import argparse
import math
import time

# ── MI300X hardware defaults (COPIED from all_gather_matmul/derive_params.py) ──
DEFAULT_NUM_CUS = 304
DEFAULT_PEAK_TFLOPS_FP16 = 1300.0
DEFAULT_HBM_BW_GBPS = 5300.0
DEFAULT_L2_SIZE_BYTES = 256 * 1024 * 1024
DEFAULT_NUM_XCDS = 8
DEFAULT_WORLD_SIZE = 8

# Calibrated from MI300X trace data
DEFAULT_SCHEDULING_FACTOR = 4.5

# SDMA latency parameters (specific to copy engine)
SDMA_LATENCY_US = 2.0
DEVICE_POST_OVERHEAD_US = 1.0
HOST_POST_OVERHEAD_US = 0.5
FLAG_POLL_LATENCY_US = 0.1
REMOTE_WRITE_SLOWDOWN_PER_RANK = 0.05
BIDIRECTIONAL_TRAFFIC_FACTOR = 1.5


def profile_link_bandwidth(world_size=DEFAULT_WORLD_SIZE):
    """Measure per-link unidirectional XGMI bandwidth.

    COPIED from all_gather_matmul/derive_params.py
    """
    import torch

    n_gpus = torch.cuda.device_count()
    if n_gpus < 2:
        raise RuntimeError(
            f"Need >= 2 visible GPUs for bandwidth profiling, found {n_gpus}. Pass --link_bw explicitly instead."
        )

    n_peers = min(world_size, n_gpus) - 1
    size_bytes = 256 * 1024 * 1024
    numel = size_bytes // 2
    warmup_iters = 10
    timed_iters = 40

    print(f"\n── Link Bandwidth Profiling {'─' * 43}")
    print(f"  GPUs visible:   {n_gpus}")
    print(f"  Testing:        GPU 0 → GPUs 1..{n_peers}")
    print(f"  Transfer size:  {size_bytes // (1024**2)} MB × {timed_iters} iterations\n")

    src = torch.empty(numel, dtype=torch.float16, device="cuda:0").normal_()
    bandwidths = []

    for peer in range(1, n_peers + 1):
        dst = torch.empty(numel, dtype=torch.float16, device=f"cuda:{peer}")

        for _ in range(warmup_iters):
            dst.copy_(src)
        torch.cuda.synchronize(0)
        torch.cuda.synchronize(peer)

        t_start = time.perf_counter()
        for _ in range(timed_iters):
            dst.copy_(src)
        torch.cuda.synchronize(peer)
        elapsed_s = time.perf_counter() - t_start

        bw = size_bytes * timed_iters / elapsed_s / 1e9
        bandwidths.append(bw)
        print(f"    GPU 0 → GPU {peer}:  {bw:6.1f} GB/s")

        del dst

    del src
    torch.cuda.empty_cache()

    bw_min = min(bandwidths)
    bw_max = max(bandwidths)
    bw_avg = sum(bandwidths) / len(bandwidths)
    print(f"\n  min = {bw_min:.1f}   avg = {bw_avg:.1f}   max = {bw_max:.1f} GB/s")
    print(f"  Using conservative (min): {bw_min:.1f} GB/s per link")

    return bw_min


# ── Tile / block size heuristics (COPIED from all_gather_matmul/derive_params.py) ──

def _choose_block_sizes(M_local, N, K):
    """Heuristic tile-size selection for MI300X MFMA."""
    bk = 64

    bm = 256 if M_local >= 8192 else 128
    while M_local % bm != 0 and bm > 64:
        bm //= 2

    if N >= 512:
        bn = 256
    elif N >= 256:
        bn = 256 if N % 256 == 0 else 128
    else:
        bn = 128
    while N % bn != 0 and bn > 32:
        bn //= 2

    while K % bk != 0 and bk > 16:
        bk //= 2

    nw = 8 if bm * bn >= 256 * 256 else 4
    return bm, bn, bk, nw


# ── Per-tile roofline model (COPIED from all_gather_matmul/derive_params.py) ──

def _tile_roofline(bm, bn, bk, M_local, K, N, dtype_bytes, peak_tflops, hbm_bw_gbps, l2_size):
    """Compute achievable per-CU TFLOPS from tile arithmetic intensity.

    For matmul_all_gather: A and B are both local (no remote reads),
    so we only check if they fit in L2.
    """
    tile_flops = 2 * bm * bn * bk
    a_bytes = bm * bk * dtype_bytes
    b_bytes = bk * bn * dtype_bytes

    a_total = M_local * K * dtype_bytes
    b_total = K * N * dtype_bytes

    # Both A and B are local - check if they fit in L2
    b_in_l2 = (a_total + b_total) <= l2_size

    hbm_bytes = a_bytes + (0 if b_in_l2 else b_bytes)
    intensity = tile_flops / max(hbm_bytes, 1)

    ridge = peak_tflops * 1e3 / hbm_bw_gbps
    if intensity >= ridge:
        roofline = peak_tflops
    else:
        roofline = hbm_bw_gbps * intensity / 1e3

    return roofline, intensity, ridge, b_in_l2


# ── matmul_all_gather specific models ──

def _gemm_wg_time_us(bm, bn, bk, K, roofline_tflops, num_cus):
    """Estimate per-WG local GEMM execution time.

    Each rank does (M_local, K) @ (K, N) where M_local is local to that rank.
    """
    num_k_blocks = K // bk
    total_flops = 2 * bm * bn * K
    per_cu_tflops = roofline_tflops / num_cus

    # Roofline-ideal per-WG time
    ideal_us = total_flops / (per_cu_tflops * 1e6)

    # Single-occupancy overhead
    occupancy_factor = 1.25 if bm * bn >= 256 * 256 else 1.10

    # Signaling overhead per output tile
    signal_us = 2.5 # TODO use parameters
    return ideal_us * occupancy_factor + signal_us


def _scatter_sdma_time_us(bm, bn, world_size, link_bw, dtype_bytes):
    """Estimate per-WG scatter time for one output tile.

    Scatters (bm × bn) tile to (world_size - 1) remote ranks via SDMA.
    """
    tile_bytes = bm * bn * dtype_bytes
    scatters_per_tile = world_size - 1

    # Effective XGMI bandwidth with bidirectional traffic
    # All ranks scatter simultaneously
    effective_bw = link_bw / (math.sqrt(world_size) * BIDIRECTIONAL_TRAFFIC_FACTOR)

    # XGMI transfer time
    xgmi_us = (tile_bytes * scatters_per_tile) / (effective_bw * 1e3)

    # SDMA posting overhead (iris.put per remote rank)
    sdma_overhead_us = scatters_per_tile * (DEVICE_POST_OVERHEAD_US + SDMA_LATENCY_US)

    # Remote write contention
    remote_write_slowdown = 1 + (world_size - 1) * REMOTE_WRITE_SLOWDOWN_PER_RANK
    remote_write_us = (tile_bytes / (DEFAULT_HBM_BW_GBPS * 1e3)) * remote_write_slowdown

    # Total scatter cost (serialized)
    total_us = xgmi_us + sdma_overhead_us + remote_write_us

    return total_us


def _estimate_kernel_time(num_tiles, gemm_wg_us, scatter_wg_us, num_cus, scheduling_factor):
    """Estimate kernel time for persistent fused GEMM+scatter.

    Persistent kernel: NUM_CUS tiles in flight, each doing GEMM then scatter.
    """
    # Per-tile time (serialized GEMM + scatter)
    tile_time_us = gemm_wg_us + scatter_wg_us

    # Persistent kernel work queue model
    total_work_us = num_tiles * tile_time_us
    ideal_time_us = total_work_us / num_cus

    # Apply scheduling overhead
    kernel_time_us = ideal_time_us * scheduling_factor

    return kernel_time_us


# ── Main derivation function ──

def derive(
    M,
    N,
    K,
    world_size=DEFAULT_WORLD_SIZE,
    link_bw=50.0,
    num_cus=DEFAULT_NUM_CUS,
    peak_tflops=DEFAULT_PEAK_TFLOPS_FP16,
    hbm_bw_gbps=DEFAULT_HBM_BW_GBPS,
    l2_size=DEFAULT_L2_SIZE_BYTES,
    scheduling_factor=DEFAULT_SCHEDULING_FACTOR,
    dtype_bytes=2,
):
    """Derive optimal parameters for matmul_all_gather_copy_engine.

    Args:
        M, N, K: Problem dimensions for a SINGLE rank
                 M is M_local (sharded), total M across all ranks = M * world_size
                 K is full K dimension (NOT sharded)
        world_size: Number of ranks
        link_bw: XGMI link bandwidth (GB/s per link)
        ...hardware params...

    Returns:
        dict with kernel parameters and performance estimates
    """
    M_local = M  # Input M is already the local dimension
    M_total = M_local * world_size

    # 1. Tile sizes
    bm, bn, bk, nw = _choose_block_sizes(M_local, N, K)
    gm = 4  # M-dimension grouping for L2 cache reuse (matches all_gather_matmul)

    # 2. Per-tile roofline
    roofline_tflops, intensity, ridge, b_in_l2 = _tile_roofline(
        bm, bn, bk, M_local, K, N, dtype_bytes, peak_tflops, hbm_bw_gbps, l2_size
    )

    # Number of output tiles (per rank)
    num_m_tiles = M_local // bm
    num_n_tiles = (N + bn - 1) // bn
    total_tiles = num_m_tiles * num_n_tiles

    # Per-WG times
    gemm_wg_us_val = _gemm_wg_time_us(bm, bn, bk, K, roofline_tflops, num_cus)
    scatter_sdma_us_val = _scatter_sdma_time_us(bm, bn, world_size, link_bw, dtype_bytes)

    # Kernel time estimate
    kernel_time_us = _estimate_kernel_time(total_tiles, gemm_wg_us_val, scatter_sdma_us_val, num_cus, scheduling_factor)

    # Sequential baseline (GEMM then separate scatter)
    total_flops = 2 * M_local * N * K
    gemm_only_us = (total_flops / 1e12) / (peak_tflops * 0.85) * 1e6

    total_scatter_bytes = M_local * N * dtype_bytes * (world_size - 1)
    effective_scatter_bw = link_bw / (math.sqrt(world_size) * BIDIRECTIONAL_TRAFFIC_FACTOR)
    scatter_only_us = (total_scatter_bytes / 1e9) / effective_scatter_bw * 1e6
    sequential_us = gemm_only_us + scatter_only_us

    # Speedup
    speedup = sequential_us / kernel_time_us

    return dict(
        block_size_m=bm,
        block_size_n=bn,
        block_size_k=bk,
        group_size_m=gm,
        num_warps=nw,
        device_initiated=True,  # Always use device-initiated for persistent kernel
        # derived
        M_local=M_local,
        M_total=M_total,
        num_m_tiles=num_m_tiles,
        num_tiles_n=num_n_tiles,
        total_tiles=total_tiles,
        # roofline
        roofline_tflops=roofline_tflops,
        tile_intensity=intensity,
        ridge_point=ridge,
        b_in_l2=b_in_l2,
        # per-WG timing
        gemm_wg_us=gemm_wg_us_val,
        scatter_wg_us=scatter_sdma_us_val,
        tile_time_us=gemm_wg_us_val + scatter_sdma_us_val,
        # estimates
        kernel_time_us=kernel_time_us,
        kernel_time_ms=kernel_time_us / 1000,
        sequential_us=sequential_us,
        sequential_ms=sequential_us / 1000,
        speedup=speedup,
    )


# ── CLI ──

def main():
    parser = argparse.ArgumentParser(
        description="Derive parameters for matmul_all_gather_copy_engine",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("-m", type=int, required=True, help="M dimension (M_local, rows per rank)")
    parser.add_argument("-n", type=int, required=True, help="N dimension (columns)")
    parser.add_argument("-k", type=int, required=True, help="K dimension (full K, NOT sharded)")
    parser.add_argument("-w", "--world_size", type=int, default=DEFAULT_WORLD_SIZE, help="Number of ranks")
    parser.add_argument("--link_bw", type=float, default=None, help="XGMI link BW (GB/s). Auto-profile if None.")
    parser.add_argument("--dtype", choices=["fp16", "fp32", "bf16"], default="fp16", help="Data type")
    parser.add_argument("-v", "--verbose", action="store_true", help="Verbose output")

    args = parser.parse_args()

    dtype_bytes = 2 if args.dtype in ["fp16", "bf16"] else 4

    # Profile link bandwidth if not provided
    if args.link_bw is None:
        try:
            args.link_bw = profile_link_bandwidth(args.world_size)
        except Exception as e:
            print(f"Auto-profiling failed ({e}), using default 50 GB/s")
            args.link_bw = 50.0

    # Derive parameters
    params = derive(
        args.m, args.n, args.k, args.world_size,
        link_bw=args.link_bw,
        dtype_bytes=dtype_bytes,
    )

    # Print results
    print("\n" + "="*80)
    print("MATMUL_ALL_GATHER_COPY_ENGINE: DERIVED PARAMETERS")
    print("="*80)

    print(f"\nProblem:")
    print(f"  M_local = {params['M_local']}, M_total = {params['M_total']}, N = {args.n}, K = {args.k}")
    print(f"  world_size = {args.world_size}")
    print(f"  Local GEMM: ({params['M_local']}, {args.k}) @ ({args.k}, {args.n})")

    print(f"\nDerived Kernel Parameters:")
    print(f"  block_size_m:      {params['block_size_m']}")
    print(f"  block_size_n:      {params['block_size_n']}")
    print(f"  block_size_k:      {params['block_size_k']}")
    print(f"  num_warps:         {params['num_warps']}")
    print(f"  device_initiated:  {params['device_initiated']}")

    print(f"\nPerformance Model:")
    print(f"  Roofline TFLOPS:   {params['roofline_tflops']:.1f}")
    print(f"  Arith. Intensity:  {params['arithmetic_intensity']:.1f} FLOPs/byte")
    print(f"  B in L2:           {params['b_in_l2']}")
    print(f"  GEMM per tile:     {params['gemm_wg_us']:.2f} μs")
    print(f"  Scatter per tile:  {params['scatter_wg_us']:.2f} μs")
    print(f"  Total per tile:    {params['tile_time_us']:.2f} μs")
    print(f"  Total tiles:       {params['total_tiles']}")

    print(f"\nEstimated Times:")
    print(f"  Kernel (fused):    {params['kernel_time_ms']:.2f} ms")
    print(f"  Sequential:        {params['sequential_ms']:.2f} ms (GEMM then scatter)")
    print(f"  Speedup:           {params['speedup']:.2f}x")

    if params['scatter_wg_us'] > params['gemm_wg_us']:
        ratio = params['scatter_wg_us'] / params['gemm_wg_us']
        print(f"\n  ⚠ Scatter dominates ({ratio:.1f}x slower than GEMM per tile)")
        print(f"    → Communication-bound workload")
    else:
        print(f"\n  ✓ GEMM dominates")
        print(f"    → Compute-bound workload")

    print("="*80)

    print("\nBenchmark command:")
    print(f"  torchrun --nproc_per_node={args.world_size} \\")
    print(f"    benchmark/ops/matmul_all_gather/benchmark_copy_engine.py \\")
    print(f"    -m {params['M_local']} -n {args.n} -k {args.k} \\")
    print(f"    --block_size_m {params['block_size_m']} \\")
    print(f"    --block_size_n {params['block_size_n']} \\")
    print(f"    --block_size_k {params['block_size_k']} \\")
    print(f"    --device_initiated \\")
    print(f"    --benchmark --validate")


if __name__ == "__main__":
    main()
