#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
# Copyright (c) 2025-2026 Advanced Micro Devices, Inc. All rights reserved.

"""
Parameter derivation for all_gather_matmul_copy_engine.

This variant uses SDMA (copy engine) for data movement instead of fetch workgroups:
  - Host (or device SDMA WGs) orchestrate remote tile transfers
  - GEMM workgroups poll flags and compute, no fetch WGs
  - Batch-based orchestration with coarse per-(batch, K-flag-group) synchronization

Key differences from HBM buffer variant (derive_params.py):
  - No fetcher workgroups (GEMM only)
  - Two modes: host-initiated (default) vs device-initiated SDMA
  - Batch-based transfer scheduling instead of wave-based pipeline
  - Coarser synchronization granularity

Usage:
    python derive_params_copy_engine.py -m 16384 -n 2048 -k 16384 --world_size 8
    python derive_params_copy_engine.py -m 16384 -n 2048 -k 16384 -v --mode device
"""

import argparse
import math
import time

# ── MI300X hardware defaults ──────────────────────────────────────────────
DEFAULT_NUM_CUS = 304
DEFAULT_PEAK_TFLOPS_FP16 = 1300.0
DEFAULT_HBM_BW_GBPS = 5300.0
DEFAULT_L2_SIZE_BYTES = 256 * 1024 * 1024
DEFAULT_NUM_XCDS = 8
DEFAULT_WORLD_SIZE = 8

# Calibrated from MI300X trace data: the ratio of measured wall time to
# the CU-work-queue lower bound.  Captures WG dispatch overhead,
# cross-XCD coherence latency, and pipeline bubble effects.
DEFAULT_SCHEDULING_FACTOR = 4.5


# SDMA/copy engine specific latencies (calibrated from MI300X traces)
DEFAULT_SDMA_LATENCY_US = 2.0  # SDMA packet submission latency
DEFAULT_HOST_POST_OVERHEAD_US = 0.5  # Host API overhead per transfer
DEFAULT_DEVICE_POST_OVERHEAD_US = 1.0  # Device SDMA WG posting overhead
DEFAULT_FLAG_POLL_LATENCY_US = 0.1  # Flag detection latency

# Performance parameters (to be calibrated)
DEFAULT_TFLOPS_ACHIEVED_RATIO = 0.85  # Achieved vs peak TFLOPS
DEFAULT_XGMI_BW_GBPS = 896.0  # Total XGMI bandwidth


# ── Block size heuristics (matches HBM buffer logic) ─────────────────────────


def _choose_block_sizes(M, N, K, K_local):
    """Heuristic tile-size selection for MI300X MFMA.

    Matches the logic from derive_params.py to ensure shared memory limits
    are respected and block sizes are consistent across variants.
    """
    bk = 64

    bm = 256 if M >= 8192 else 128
    while M % bm != 0 and bm > 64:
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
    while K_local % bk != 0 and bk > 16:
        bk //= 2

    nw = 8 if bm * bn >= 256 * 256 else 4
    return bm, bn, bk, nw


def _choose_k_per_flag(num_k_blocks, num_k_blocks_local, target_groups=8):
    """Pick k_per_flag so that flag groups align to rank boundaries when possible.

    Identical logic to HBM buffer variant - rank alignment is important for
    efficient all-gather patterns regardless of the transfer mechanism.
    """
    if num_k_blocks % num_k_blocks_local == 0:
        candidate = num_k_blocks_local
        groups = num_k_blocks // candidate
        if groups >= 4:
            return candidate

    kpf = max(1, num_k_blocks // target_groups)
    while num_k_blocks % kpf != 0 and kpf > 1:
        kpf -= 1
    return kpf


def _choose_m_tiles_per_batch(
    num_m_tiles, num_n_tiles, tile_gemm_us, tile_transfer_us
):
    """Choose m_tiles_per_batch to minimize exposed communication time.

    Wave-based model:
    - Wave 0: Transfer for first batch is fully exposed
    - Wave 1+: If gemm_time >= transfer_time per batch, transfer is hidden

    Args:
        num_m_tiles: Total M tiles
        num_n_tiles: Total N tiles
        tile_gemm_us: Per-tile GEMM time (microseconds)
        tile_transfer_us: Per-M-tile transfer time (microseconds)

    Returns:
        Optimal m_tiles_per_batch
    """
    best_batch_size = num_m_tiles  # Default: all tiles in one batch
    best_time_us = float("inf")

    # Try divisors of num_m_tiles
    candidates = []
    for d in range(1, num_m_tiles + 1):
        if num_m_tiles % d == 0:
            candidates.append(num_m_tiles // d)

    for m_batch in candidates:
        num_batches = num_m_tiles // m_batch

        # Per-batch transfer time (for m_batch tiles across all K)
        transfer_per_batch_us = m_batch * tile_transfer_us

        # Per-batch GEMM time (m_batch × num_n_tiles tiles)
        gemm_per_batch_us = m_batch * num_n_tiles * tile_gemm_us

        # Wave 0: First batch - transfer exposed
        wave0_us = transfer_per_batch_us + gemm_per_batch_us

        # Wave 1+: Remaining batches
        if num_batches > 1:
            if gemm_per_batch_us >= transfer_per_batch_us:
                # Transfer hidden by GEMM
                wave_rest_us = (num_batches - 1) * gemm_per_batch_us
            else:
                # Transfer exposed
                wave_rest_us = (num_batches - 1) * (
                    transfer_per_batch_us + gemm_per_batch_us
                )
        else:
            wave_rest_us = 0

        total_time_us = wave0_us + wave_rest_us

        if total_time_us < best_time_us:
            best_time_us = total_time_us
            best_batch_size = m_batch

    return best_batch_size


# ── Roofline model (reused from HBM buffer) ──────────────────────────────────


def _tile_roofline(bm, bn, bk, M, K, N, dtype_bytes, peak_tflops, hbm_bw_gbps, l2_size):
    """Compute achievable per-CU TFLOPS from tile arithmetic intensity.

    Identical to HBM buffer variant - roofline analysis is architecture-independent.
    """
    tile_flops = 2 * bm * bn * bk
    a_bytes = bm * bk * dtype_bytes
    b_bytes = bk * bn * dtype_bytes

    b_total = K * N * dtype_bytes
    staged_a_total = M * K * dtype_bytes
    b_in_l2 = (staged_a_total <= l2_size) and (b_total <= l2_size)

    hbm_bytes = a_bytes + (0 if b_in_l2 else b_bytes)
    intensity = tile_flops / max(hbm_bytes, 1)

    ridge = peak_tflops * 1e3 / hbm_bw_gbps
    if intensity >= ridge:
        roofline = peak_tflops
    else:
        roofline = hbm_bw_gbps * intensity / 1e3

    return roofline, intensity, ridge, b_in_l2


# ── Per-WG execution time models ────────────────────────────────────────


def _gemm_wg_time_us(bm, bn, bk, K, num_flag_groups, roofline_tflops, num_cus):
    """Estimate per-WG GEMM execution time in microseconds.

    Identical to HBM buffer variant (lines 192-213 of derive_params.py).
    """
    total_flops = 2 * bm * bn * K
    per_cu_tflops = roofline_tflops / num_cus

    # Roofline-ideal per-WG time
    ideal_us = total_flops / (per_cu_tflops * 1e6)

    # Single-occupancy overhead: imperfect latency hiding, instruction
    # scheduling gaps, cross-XCD coherence on staged_a reads.
    # Calibrated from MI300X traces: actual/ideal ≈ 1.2-1.3.
    occupancy_factor = 1.25 if bm * bn >= 256 * 256 else 1.10

    # Flag polling: acquire-semantics atomic per flag group
    flag_us = num_flag_groups * 2.5

    return ideal_us * occupancy_factor + flag_us

# ── Per-SDMA-WG timing (for device-initiated mode) ───────────────────────────

def _sdma_wg_time_us(num_transfers_per_wg, bytes_per_transfer, sdma_latency_us,
                     device_overhead_us, xgmi_bw_gbps):
    """Estimate per-SDMA-WG execution time in microseconds.

    Analogous to _fetch_wg_time_us in HBM buffer variant. Each SDMA WG posts
    transfers for one remote rank.

    Args:
        num_transfers_per_wg: Number of transfers this WG must post
        bytes_per_transfer: Size of each transfer in bytes
        sdma_latency_us: SDMA packet submission latency
        device_overhead_us: Device-side posting overhead per transfer
        xgmi_bw_gbps: Available XGMI bandwidth (shared across all SDMA WGs)

    Returns:
        Estimated execution time in microseconds
    """
    # Posting overhead: WG serially posts all its transfers
    post_time_us = num_transfers_per_wg * (device_overhead_us + sdma_latency_us)

    # Bandwidth time: SDMA engine executes transfers
    # (Note: bandwidth is shared across all SDMA WGs; this is per-WG share)
    total_bytes = num_transfers_per_wg * bytes_per_transfer
    bandwidth_time_us = total_bytes / (xgmi_bw_gbps * 1e3)

    # Conservative: assume posting and transfer happen sequentially
    # (In reality, some overlap possible depending on SDMA engine scheduling)
    return post_time_us + bandwidth_time_us



# ── Kernel time estimation (reused from HBM buffer) ──────────────────────────


def _estimate_kernel_time(total_gemm_wgs, gemm_wg_us, total_sdma_wgs, sdma_wg_us, num_cus, scheduling_factor):
    """Estimate kernel wall-clock time from the CU work queue model.

    Identical to HBM buffer variant. total_CU_work / num_CUs gives the ideal
    (work-conserving) lower bound. The scheduling_factor captures GPU dispatch
    overhead, cross-XCD coherence, and pipeline bubble effects.
    """
    total_cu_work_us = total_gemm_wgs * gemm_wg_us + total_sdma_wgs * sdma_wg_us

    ideal_ms = total_cu_work_us / num_cus / 1e3
    estimated_ms = ideal_ms * scheduling_factor
    return estimated_ms, ideal_ms






def derive(
    M,
    N,
    K,
    world_size,
    link_bw,
    num_cus,
    peak_tflops,
    hbm_bw_gbps,
    l2_size,
    scheduling_factor,
    dtype_bytes,
    device_initiated=None,
):
    """Derive optimal parameters for all_gather_matmul_copy_engine.

    Matches the interface of derive_params.py (HBM buffer variant) but returns
    copy-engine-specific parameters.

    Args:
        M, N, K: GEMM dimensions (K is total across all ranks)
        world_size: Number of GPUs
        link_bw: XGMI bandwidth (not used - using hardcoded XGMI_BW_GBPS)
        num_cus: Number of compute units
        peak_tflops: Peak TFLOPS (not used - using hardcoded value)
        hbm_bw_gbps: HBM bandwidth (not used - using hardcoded value)
        l2_size: L2 cache size
        scheduling_factor: Not used in copy engine (no work queue model)
        dtype_bytes: 2 for fp16/bf16, 4 for fp32
        device_initiated: None (auto-select), True (force device mode), False (force host mode)

    Returns:
        dict with kernel parameters and performance estimates
    """
    K_local = K // world_size

    # 1. Tile sizes (matches HBM buffer logic)
    bm, bn, bk, nw = _choose_block_sizes(M, N, K, K_local)
    gm = 4
    num_m_tiles = M // bm
    num_tiles_n = math.ceil(N / bn)
    num_k_blocks = K // bk
    num_k_blocks_local = K_local // bk

    # 2. Per-tile roofline (reused from HBM buffer)
    roofline_tflops, intensity, ridge, b_in_l2 = _tile_roofline(
        bm, bn, bk, M, K, N, dtype_bytes, peak_tflops, hbm_bw_gbps, l2_size
    )

    # 3. Communication model (link-limited)
    total_remote_bytes = M * K_local * (world_size - 1) * dtype_bytes
    total_link_bw = link_bw * (world_size - 1)
    comm_time_ms = total_remote_bytes / (total_link_bw * 1e9) * 1e3

    # 4. Compute model (roofline-limited)
    total_flops = 2 * M * N * K
    compute_time_ms = total_flops / (roofline_tflops * 1e12) * 1e3

    ratio = comm_time_ms / compute_time_ms if compute_time_ms > 0 else 999

    # 5. k_per_flag selection (matches HBM buffer)
    kpf = _choose_k_per_flag(num_k_blocks, num_k_blocks_local)
    num_flag_groups_k = num_k_blocks // kpf

    # 6. Per-tile GEMM time
    gemm_wg_us_val = _gemm_wg_time_us(bm, bn, bk, K, num_flag_groups_k, roofline_tflops, num_cus)

    # 7. Per-M-tile transfer time
    # Each M-tile needs K_local data from (world_size-1) ranks
    bytes_per_m_tile = bm * K_local * dtype_bytes * (world_size - 1)
    transfer_bw_gbps = DEFAULT_XGMI_BW_GBPS / math.sqrt(world_size)  # Congestion model
    tile_transfer_us = bytes_per_m_tile / (transfer_bw_gbps * 1e3)

    # 8. m_tiles_per_batch selection (minimize exposed communication)
    m_tiles_per_batch = _choose_m_tiles_per_batch(
        num_m_tiles, num_tiles_n, gemm_wg_us_val, tile_transfer_us
    )

    # Sanity check
    assert num_m_tiles % m_tiles_per_batch == 0, f"m_tiles_per_batch={m_tiles_per_batch} must divide num_m_tiles={num_m_tiles}"
    num_batches = num_m_tiles // m_tiles_per_batch

    # 9. device_initiated default
    if device_initiated is None:
        device_initiated = True

    # 10. Grid geometry
    total_gemm_wgs = num_m_tiles * num_tiles_n
    num_sdma_wgs = world_size - 1 if device_initiated else 0

    # 11. Per-SDMA-WG time (for device-initiated mode)
    if device_initiated:
        num_transfers = num_batches * num_flag_groups_k * (world_size - 1)
        transfers_per_sdma_wg = num_transfers // num_sdma_wgs
        batch_m = m_tiles_per_batch * bm
        flag_group_k = kpf * bk
        bytes_per_transfer = batch_m * flag_group_k * dtype_bytes
        sdma_wg_us = _sdma_wg_time_us(
            transfers_per_sdma_wg,
            bytes_per_transfer,
            DEFAULT_SDMA_LATENCY_US,
            DEFAULT_DEVICE_POST_OVERHEAD_US,
            DEFAULT_XGMI_BW_GBPS,
        )
    else:
        sdma_wg_us = 0

    # 12. Kernel time estimate (CU-work model)
    if device_initiated:
        est_kernel_ms, est_ideal_ms = _estimate_kernel_time(
            total_gemm_wgs, gemm_wg_us_val, num_sdma_wgs, sdma_wg_us, num_cus, scheduling_factor
        )
    else:
        # Host-initiated: no device WG orchestration
        est_kernel_ms = compute_time_ms
        est_ideal_ms = compute_time_ms

    # 13. Pipeline time (from wave model)
    transfer_per_batch_us = m_tiles_per_batch * tile_transfer_us
    gemm_per_batch_us = m_tiles_per_batch * num_tiles_n * gemm_wg_us_val
    wave0_us = transfer_per_batch_us + gemm_per_batch_us
    if num_batches > 1:
        if gemm_per_batch_us >= transfer_per_batch_us:
            wave_rest_us = (num_batches - 1) * gemm_per_batch_us
        else:
            wave_rest_us = (num_batches - 1) * (transfer_per_batch_us + gemm_per_batch_us)
    else:
        wave_rest_us = 0
    pipeline_time_ms = (wave0_us + wave_rest_us) / 1000
    overlap_efficiency = compute_time_ms / pipeline_time_ms if pipeline_time_ms > 0 else 0

    # 13. Staged A size


    # 14. Standalone GEMM estimate (rocBLAS-class efficiency for comparison)
    standalone_gemm_eff = 0.30
    standalone_tflops = roofline_tflops * standalone_gemm_eff
    standalone_gemm_ms = total_flops / (standalone_tflops * 1e12) * 1e3
    pytorch_est_ms = comm_time_ms + standalone_gemm_ms

    staged_a_gb = M * K * dtype_bytes / (1024**3)

    return dict(
        block_size_m=bm,
        block_size_n=bn,
        block_size_k=bk,
        group_size_m=gm,
        num_warps=nw,
        k_per_flag=kpf,
        m_tiles_per_batch=m_tiles_per_batch,
        device_initiated=device_initiated,
        # derived
        K_local=K_local,
        num_m_tiles=num_m_tiles,
        num_tiles_n=num_tiles_n,
        num_k_blocks=num_k_blocks,
        num_flag_groups_k=num_flag_groups_k,
        num_batches=num_batches,
        num_k_flag_groups=num_flag_groups_k,
        # roofline
        roofline_tflops=roofline_tflops,
        tile_intensity=intensity,
        ridge_point=ridge,
        b_in_l2=b_in_l2,
        # per-WG timing
        gemm_wg_us=gemm_wg_us_val,
        sdma_wg_us=sdma_wg_us,
        # grid
        total_gemm_wgs=total_gemm_wgs,
        num_sdma_wgs=num_sdma_wgs,
        # estimates
        total_remote_bytes=total_remote_bytes,
        total_link_bw=total_link_bw,
        comm_time_ms=comm_time_ms,
        total_flops=total_flops,
        compute_time_ms=compute_time_ms,
        ratio=ratio,

        est_kernel_ms=est_kernel_ms,
        est_ideal_ms=est_ideal_ms,
        standalone_gemm_ms=standalone_gemm_ms,
        pytorch_est_ms=pytorch_est_ms,
        staged_a_gb=staged_a_gb,
        scheduling_factor=scheduling_factor,
        # copy engine performance estimates
        pipeline_time_ms=pipeline_time_ms,
        overlap_efficiency=overlap_efficiency,
        transfer_time_per_batch_us=transfer_per_batch_us,
        gemm_time_per_batch_us=gemm_per_batch_us,
    )


# ── Formatting helpers ───────────────────────────────────────────────────


def _fmt_bytes(n):
    if n >= 1024**3:
        return f"{n / 1024**3:.2f} GB"
    if n >= 1024**2:
        return f"{n / 1024**2:.1f} MB"
    return f"{n / 1024:.1f} KB"


def _fmt_flops(n):
    if n >= 1e15:
        return f"{n / 1e15:.2f} PFLOPs"
    return f"{n / 1e12:.2f} TFLOPs"


def _fmt_tflops(t):
    return f"{t:.0f} TFLOPS"


# ── Analysis output ──────────────────────────────────────────────────────


def print_analysis(p, M, N, K_local, world_size):
    """Print detailed performance analysis (matches HBM buffer format)."""
    K = K_local * world_size

    print("\n" + "=" * 80)
    print("ALL_GATHER_MATMUL_COPY_ENGINE: PERFORMANCE MODEL & DERIVED PARAMETERS")
    print("=" * 80)

    print(f"\nProblem Size:")
    print(f"  M = {M}, N = {N}, K_local = {K_local}, K_total = {K}, world_size = {world_size}")
    print(f"  Total GEMM: ({M}, {K}) @ ({K}, {N}) = ({M}, {N})")
    print(f"  Staged A buffer: {p['staged_a_gb']:.2f} GB")

    print(f"\nBlock Sizes:")
    print(f"  BLOCK_M = {p['block_size_m']}, BLOCK_N = {p['block_size_n']}, BLOCK_K = {p['block_size_k']}")
    print(f"  num_warps = {p['num_warps']}")

    print(f"\nGEMM Analysis (Roofline):")
    print(f"  Arithmetic Intensity: {p['tile_intensity']:.2f} FLOPs/byte")
    print(f"  Ridge Point: {p['ridge_point']:.2f} FLOPs/byte")
    print(f"  B in L2: {p['b_in_l2']}")
    print(f"  Achieved TFLOPS: {p['roofline_tflops']:.0f}")
    print(f"  Compute Time: {p['compute_time_ms']:.2f} ms")

    print(f"\nCommunication Analysis:")
    print(f"  Total Recv: {p['total_remote_bytes'] / 1e9:.2f} GB per rank")
    print(f"  Comm Time: {p['comm_time_ms']:.2f} ms")

    print(f"\nCopy Engine Parameters:")
    print(f"  k_per_flag:        {p['k_per_flag']}")
    print(f"  m_tiles_per_batch: {p['m_tiles_per_batch']}")
    print(f"  num_batches:       {p['num_batches']}")
    print(f"  device_initiated:  {p['device_initiated']}")

    print(f"\nPipeline Performance:")
    print(f"  Pipeline Time:     {p['pipeline_time_ms']:.2f} ms")
    print(f"  Overlap Efficiency: {p['overlap_efficiency'] * 100:.1f}%")

    # Comparison
    baseline = p["comm_time_ms"] + p["compute_time_ms"]
    speedup = baseline / p["pipeline_time_ms"] if p["pipeline_time_ms"] > 0 else 0
    print(f"\nSpeedup vs Sequential (AllGather→GEMM): {speedup:.2f}x")
    print(f"  Sequential: {baseline:.2f} ms")
    print(f"  Pipelined:  {p['pipeline_time_ms']:.2f} ms")
    print("=" * 80 + "\n")


def main():
    parser = argparse.ArgumentParser(
        description="Derive optimal parameters for all_gather_matmul_copy_engine",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("-m", type=int, required=True, help="M dimension (rows of output)")
    parser.add_argument("-n", type=int, required=True, help="N dimension (cols of output)")
    parser.add_argument("-k", type=int, required=True, help="K dimension (total reduction dim)")
    parser.add_argument("-w", "--world_size", type=int, default=DEFAULT_WORLD_SIZE, help="Number of GPUs")
    parser.add_argument("--mode", type=str, default="auto", choices=["auto", "host", "device"])
    parser.add_argument(
        "--link_bw",
        type=float,
        default=None,
        help="Per-link XGMI bandwidth in GB/s (one direction). Omit to auto-profile via GPU-to-GPU copies.",
    )
    parser.add_argument("--num_cus", type=int, default=DEFAULT_NUM_CUS, help="Number of compute units")
    parser.add_argument("--peak_tflops", type=float, default=DEFAULT_PEAK_TFLOPS_FP16, help="Peak fp16 TFLOPS")
    parser.add_argument("--hbm_bw", type=float, default=DEFAULT_HBM_BW_GBPS, help="HBM bandwidth in GB/s")
    parser.add_argument(
        "--scheduling_factor",
        type=float,
        default=DEFAULT_SCHEDULING_FACTOR,
        help="CU scheduling overhead factor (calibrated from traces)",
    )

    args, passthrough = parser.parse_known_args()

    if args.k % args.world_size != 0:
        parser.error(f"K ({args.k}) must be divisible by world_size ({args.world_size})")

    # Convert mode string to device_initiated parameter
    if args.mode == "auto":
        device_initiated = None
    elif args.mode == "device":
        device_initiated = True
    else:  # host
        device_initiated = False

    p = derive(
        args.m,
        args.n,
        args.k,
        args.world_size,
        args.link_bw,
        args.num_cus,
        args.peak_tflops,
        args.hbm_bw,
        DEFAULT_L2_SIZE_BYTES,
        args.scheduling_factor,
        dtype_bytes=2,
        device_initiated=device_initiated,
    )

    # Print analysis
    print_analysis(p, args.m, args.n, args.k_local, args.world_size)

    # Print benchmark command
    print("Benchmark command:")
    print(f"  torchrun --nproc_per_node={args.world_size} \\")
    print("    benchmark/ops/all_gather_matmul/benchmark_copy_engine.py \\")
    print(f"    -m {args.m} -n {args.n} -k {args.k_local} \\")
    print(
        f"    --block_size_m {p['block_size_m']} --block_size_n {p['block_size_n']} --block_size_k {p['block_size_k']} \\"
    )
    print(f"    --k_per_flag {p['k_per_flag']} \\")
    print(f"    --m_tiles_per_batch {p['m_tiles_per_batch']} \\")
    if p["device_initiated"]:
        print("    --device_initiated \\")
    print("    --benchmark --validate")


if __name__ == "__main__":
    main()
