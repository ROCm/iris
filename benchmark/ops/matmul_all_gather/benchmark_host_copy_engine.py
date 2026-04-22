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
import numpy as np

from examples.common.utils import JSONWriter

import iris
from iris.ops.matmul_all_gather_host_copy_engine import (
    matmul_all_gather_host_copy_engine,
    matmul_all_gather_host_copy_engine_preamble,
)
from iris.ops import FusedConfig

_DERIVE_AVAILABLE = False
try:
    import sys as _sys

    _script_dir = os.path.dirname(os.path.abspath(__file__))
    if _script_dir not in _sys.path:
        _sys.path.insert(0, _script_dir)
    from derive_params import (
        _tile_roofline,
        _gemm_wg_time_us,
        _scatter_sdma_time_us,
        DEFAULT_NUM_CUS,
        DEFAULT_PEAK_TFLOPS_FP16,
        DEFAULT_HBM_BW_GBPS,
        DEFAULT_L2_SIZE_BYTES,
    )

    _DERIVE_AVAILABLE = True
except Exception:
    pass

torch.manual_seed(123)
random.seed(123)

TICKS_PER_US = 100  # s_memrealtime runs at 100 MHz: 1 tick = 10 ns = 0.01 us


def _plot_trace(trace_data, output_path, rank, M, N, K):
    """Generate a Gantt chart showing GEMM and SDMA workgroup activity.

    Y-axis: workgroup (sorted by start time)
    X-axis: time in microseconds
    Colors:
      - GEMM: wait (red), compute (green)
      - SDMA: SDMA posting (blue)
    """
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D

    starts = trace_data["start"].numpy().astype(np.int64)
    ends = trace_data["end"].numpy().astype(np.int64)
    waits = trace_data["wait"].numpy().astype(np.int64)
    xcds = trace_data["xcd"].numpy().astype(np.int32)
    grid_size = trace_data["grid_size"]

    # Check for SDMA timestamp traces (host-initiated SDMA)
    has_sdma_timestamps = "sdma_timestamps" in trace_data
    if has_sdma_timestamps:
        # sdma_timestamps: (world_size, 2) where [:,0] = start, [:,1] = end
        sdma_ts = trace_data["sdma_timestamps"].numpy().astype(np.int64)
        num_sdma = sdma_ts.shape[0]  # world_size (includes current rank with zeros)
        # Filter out invalid timestamps (zeros indicate no transfer to self)
        valid_mask = (sdma_ts[:, 0] > 0) & (sdma_ts[:, 1] > 0)
        sdma_starts = sdma_ts[valid_mask, 0]
        sdma_ends = sdma_ts[valid_mask, 1]
        sdma_rank_ids = np.arange(num_sdma)[valid_mask]
        num_sdma = len(sdma_starts)
    else:
        # Check for old-style SDMA WG traces (device-initiated)
        has_sdma = "sdma_start" in trace_data and trace_data.get("num_sdma", 0) > 0
        if has_sdma:
            sdma_starts = trace_data["sdma_start"].numpy().astype(np.int64)
            sdma_ends = trace_data["sdma_end"].numpy().astype(np.int64)
            sdma_xcds = trace_data["sdma_xcd"].numpy().astype(np.int32)
            num_sdma = trace_data["num_sdma"]
            sdma_rank_ids = None
        else:
            num_sdma = 0
            sdma_rank_ids = None

    # Convert to microseconds relative to earliest start
    t_min = starts.min()
    if num_sdma > 0 and len(sdma_starts) > 0:
        t_min = min(t_min, sdma_starts.min())

    starts_us = (starts - t_min) / TICKS_PER_US
    ends_us = (ends - t_min) / TICKS_PER_US
    waits_us = waits / TICKS_PER_US

    if num_sdma > 0:
        sdma_starts_us = (sdma_starts - t_min) / TICKS_PER_US
        sdma_ends_us = (sdma_ends - t_min) / TICKS_PER_US

    # Sort GEMM by start time
    order = np.argsort(starts_us)

    # Figure sizing (SDMA bars disabled due to clock domain mismatch)
    total_rows = grid_size  # + num_sdma (disabled)
    row_h = 0.012
    fig_h = max(12, total_rows * row_h + 2)
    fig, ax = plt.subplots(figsize=(18, fig_h))

    wait_color = "#F44336"  # red
    compute_color = "#4CAF50"  # green
    sdma_color = "#2196F3"  # blue (for SDMA posting)

    y_offset = 0

    # Skip plotting SDMA bars (clock domain mismatch with GPU timestamps)
    # But we'll still show SDMA stats in the summary
    # if num_sdma > 0:
    #     for sdma_idx in range(num_sdma):
    #         s = sdma_starts_us[sdma_idx]
    #         e = sdma_ends_us[sdma_idx]
    #         dur = e - s
    #         ax.barh(y_offset + sdma_idx, dur, left=s, height=0.8, color=sdma_color, edgecolor="none", linewidth=0)
    #         # Add rank label if available (use start position for label)
    #         if sdma_rank_ids is not None:
    #             label_text = f"R{sdma_rank_ids[sdma_idx]}"
    #             ax.text(s - 5, y_offset + sdma_idx, label_text, ha='right', va='center', fontsize=7)
    #     y_offset = num_sdma

    # Plot GEMM WGs
    for y_idx, wg_idx in enumerate(order):
        s = starts_us[wg_idx]
        e = ends_us[wg_idx]
        dur = e - s

        # Split into wait (red) and compute (green)
        w = waits_us[wg_idx]
        comp = max(0, dur - w)
        ax.barh(y_offset + y_idx, w, left=s, height=0.8, color=wait_color, edgecolor="none", linewidth=0)
        ax.barh(y_offset + y_idx, comp, left=s + w, height=0.8, color=compute_color, edgecolor="none", linewidth=0)

    # XCD annotations
    xcd_set = sorted(set(xcds.tolist()))
    xcd_cmap = {}
    if len(xcd_set) > 1:
        cmap = matplotlib.colormaps.get_cmap("tab10").resampled(len(xcd_set))
        for i, x in enumerate(xcd_set):
            xcd_cmap[x] = cmap(i)

    x_max = ends_us.max() * 1.02
    # Don't include SDMA in x_max (clock domain mismatch)
    # if num_sdma > 0:
    #     x_max = max(x_max, sdma_ends_us.max() * 1.02)

    for y_idx, wg_idx in enumerate(order):
        xcd_id = xcds[wg_idx]
        if xcd_id in xcd_cmap:
            ax.plot(x_max, y_offset + y_idx, marker="s", markersize=1.5, color=xcd_cmap[xcd_id], clip_on=False)

    # Only plot XCD markers for old-style SDMA WG traces (not host timestamps)
    if num_sdma > 0 and sdma_rank_ids is None and "sdma_xcds" in locals():
        for sdma_idx in range(num_sdma):
            xcd_id = sdma_xcds[sdma_idx]
            if xcd_id in xcd_cmap:
                ax.plot(x_max, sdma_idx, marker="s", markersize=1.5, color=xcd_cmap[xcd_id], clip_on=False)

    ax.set_xlabel("Time (us)", fontsize=12)
    ylabel = "Tile ID (sorted by start)" if has_sdma_timestamps else "Workgroup (sorted by start)"
    ax.set_ylabel(ylabel, fontsize=12)

    title = f"Rank {rank}  |  Host Copy Engine Matmul+AG Trace  |  M={M} N={N} K={K}  |  {grid_size} GEMM tiles"
    # Note: SDMA bars not shown due to clock domain mismatch, but stats included below
    ax.set_title(title, fontsize=13)
    ax.set_ylim(-1, total_rows + 1)
    ax.set_xlim(0, x_max)
    ax.invert_yaxis()

    # Legend
    legend_elements = [
        Line2D([0], [0], color=wait_color, lw=6, label="GEMM: waiting on remote data"),
        Line2D([0], [0], color=compute_color, lw=6, label="GEMM: compute"),
    ]
    # Add SDMA to legend (stats shown below) even though bars not plotted
    if has_sdma_timestamps or num_sdma > 0:
        legend_elements.append(
            Line2D([0], [0], color=sdma_color, lw=6, label="SDMA: see timing stats below", linestyle="--")
        )
    ax.legend(handles=legend_elements, loc="upper right", fontsize=10)

    # Summary stats
    gemm_dur = ends_us - starts_us
    gemm_wait = waits_us
    gemm_compute = gemm_dur - gemm_wait

    # Wall time (GEMM only, SDMA not included due to clock domain mismatch)
    wall_time_us = ends_us.max()
    # Don't include SDMA in wall time calculation due to clock mismatch
    # if num_sdma > 0 and len(sdma_ends_us) > 0:
    #     wall_time_us = max(wall_time_us, sdma_ends_us.max())

    stats_lines = [
        f"GEMM total: {gemm_dur.mean():.1f} us avg  ({gemm_dur.min():.1f}-{gemm_dur.max():.1f})",
        f"  wait: {gemm_wait.mean():.1f} us avg  ({gemm_wait.min():.1f}-{gemm_wait.max():.1f})",
        f"  compute: {gemm_compute.mean():.1f} us avg  ({gemm_compute.min():.1f}-{gemm_compute.max():.1f})",
        f"  wait%: {100 * gemm_wait.sum() / gemm_dur.sum():.1f}%",
    ]

    if num_sdma > 0 and len(sdma_starts_us) > 0:
        sdma_dur = sdma_ends_us - sdma_starts_us
        label = "SDMA (per-rank)" if has_sdma_timestamps else "SDMA"
        stats_lines.append(f"{label}: {sdma_dur.mean():.1f} us avg  ({sdma_dur.min():.1f}-{sdma_dur.max():.1f})")

    stats_lines.append(f"Wall time: {wall_time_us:.1f} us")
    stats_text = "\n".join(stats_lines)
    ax.text(
        0.01,
        0.99,
        stats_text,
        transform=ax.transAxes,
        fontsize=9,
        verticalalignment="top",
        fontfamily="monospace",
        bbox=dict(boxstyle="round,pad=0.4", facecolor="white", alpha=0.85),
    )

    plt.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  [Rank {rank}] Trace plot saved to: {output_path}")
    print(f"  {stats_text}")


def parse_args():
    parser = argparse.ArgumentParser(
        description="Benchmark matmul_all_gather_host_copy_engine fused operation.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("-m", type=int, default=16384, help="Number of rows per rank in matrix A (M_local)")
    parser.add_argument("-n", type=int, default=2048, help="Number of columns in matrix B (N)")
    parser.add_argument("-k", type=int, default=131072, help="Common dimension (K)")
    parser.add_argument("-v", "--validate", action="store_true", help="Enable validation mode")
    parser.add_argument("-b", "--benchmark", action="store_true", help="Enable benchmarking mode")
    parser.add_argument(
        "--trace",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Collect per-workgroup trace and save Gantt chart PNG",
    )
    parser.add_argument(
        "--datatype",
        type=str,
        default="fp16",
        choices=["fp16", "fp32", "bf16"],
        help="Tensor datatype",
    )
    parser.add_argument("--heap_size", type=int, default=1 << 34, help="Iris heap size")
    parser.add_argument("--num_sms", type=int, default=None, help="Number of SMs for operation (auto-detect if None)")
    parser.add_argument("--block_size_m", type=int, default=None, help="Block size M (model-derived if omitted)")
    parser.add_argument("--block_size_n", type=int, default=None, help="Block size N (model-derived if omitted)")
    parser.add_argument("--block_size_k", type=int, default=None, help="Block size K (model-derived if omitted)")
    parser.add_argument("--group_size_m", type=int, default=None, help="Group size M (model-derived if omitted)")
    parser.add_argument("--num_xcds", type=int, default=None, help="Number of XCDs (auto if None)")
    parser.add_argument(
        "--m_tiles_per_batch", type=int, default=None, help="Number of M-tiles to batch together for SDMA"
    )
    parser.add_argument(
        "--trace_output", type=str, default="trace_host_copy_engine.png", help="Output file for trace plot"
    )
    parser.add_argument(
        "--output_file",
        type=str,
        default="matmul_all_gather_host_copy_engine.json",
        help="Output file",
    )
    parser.add_argument(
        "--link_bw",
        type=float,
        default=50.0,
        help="XGMI link bandwidth in GB/s (for performance model)",
    )

    return vars(parser.parse_args())


def _apply_model_defaults(args, world_size, dtype, device, dtype_bytes=2):
    """Fill None-valued kernel parameters using tritonBLAS selector.

    Priority:
    1. User-specified values (keep as-is)
    2. tritonBLAS selector (always)

    Returns a list of parameter names that were set by the selector.
    """
    applied = []

    # Import tritonBLAS selector
    from tritonblas.matmul import _make_matmul_selector

    # Always use tritonBLAS selector for block sizes
    selector = _make_matmul_selector(
        args["m"],  # M_local
        args["n"],
        args["k"],
        dtype,
        dtype,
        dtype,
        device,
        streamk=False,
    )

    # Only apply if not user-specified
    if args.get("block_size_m") is None:
        args["block_size_m"] = selector.block_m
        applied.append("block_size_m (tritonBLAS)")
    if args.get("block_size_n") is None:
        args["block_size_n"] = selector.block_n
        applied.append("block_size_n (tritonBLAS)")
    if args.get("block_size_k") is None:
        args["block_size_k"] = selector.block_k
        applied.append("block_size_k (tritonBLAS)")
    if args.get("group_size_m") is None:
        args["group_size_m"] = selector.group_m
        applied.append("group_size_m (tritonBLAS)")

    return selector, applied


def _analyze_performance(bm, bn, bk, M_local, N, K, world_size, link_bw, dtype_bytes):
    """Analyze GEMM vs scatter bottleneck using performance model from derive_params.

    Returns dict with performance metrics (to be written to JSON):
        - gemm_wg_us: Per-workgroup GEMM time
        - scatter_wg_us: Per-workgroup scatter time
        - bottleneck: "compute" or "communication"
        - ratio: scatter_wg_us / gemm_wg_us
        - roofline_tflops: Achievable TFLOPS from roofline model
        - intensity: Arithmetic intensity (FLOPs/byte)
    """
    if not _DERIVE_AVAILABLE:
        return None

    try:
        # Roofline model for GEMM
        roofline_tflops, intensity, ridge, b_in_l2 = _tile_roofline(
            bm,
            bn,
            bk,
            M_local,
            K,
            N,
            dtype_bytes,
            DEFAULT_PEAK_TFLOPS_FP16,
            DEFAULT_HBM_BW_GBPS,
            DEFAULT_L2_SIZE_BYTES,
        )

        # Per-WG GEMM time
        gemm_wg_us = _gemm_wg_time_us(
            bm,
            bn,
            bk,
            K,
            roofline_tflops,
            DEFAULT_NUM_CUS,
        )

        # Per-WG scatter time
        scatter_wg_us = _scatter_sdma_time_us(bm, bn, world_size, link_bw, dtype_bytes)

        # Determine bottleneck
        ratio = scatter_wg_us / gemm_wg_us
        bottleneck = "communication" if ratio > 1.0 else "compute"

        return {
            "gemm_wg_us": gemm_wg_us,
            "scatter_wg_us": scatter_wg_us,
            "bottleneck": bottleneck,
            "ratio": ratio,
            "roofline_tflops": roofline_tflops,
            "intensity": intensity,
        }
    except Exception as e:
        print(f"Warning: Performance analysis failed ({e})")
        return None


def _auto_tune_batching(perf_analysis, num_m_tiles, num_n_tiles, group_size_m, bm, bn, dtype_bytes):
    """Auto-tune m_tiles_per_batch based on bottleneck analysis.

    Baseline heuristic: choose one wave-aligned batch worth of M-tiles.

    Returns: (m_tiles_per_batch, adjusted_block_m, adjusted_block_n)
    """
    if perf_analysis is None:
        return 1, bm, bn  # Conservative default

    num_cus = DEFAULT_NUM_CUS

    # Batch one wave's worth of M-tiles when possible.
    tiles_per_group = group_size_m * num_n_tiles
    groups_per_wave = max(1, num_cus // tiles_per_group)
    m_tiles_per_batch = max(1, groups_per_wave * group_size_m)

    if perf_analysis["bottleneck"] == "compute":
        # Compute-bound: keep original block sizes
        # Larger tiles = more compute per tile = better amortization
        adjusted_bm, adjusted_bn = bm, bn
    else:
        # Communication-bound: reduce tile sizes to increase overlap
        ratio = perf_analysis["ratio"]

        # TODO need a better way to determine if we can halve
        # TODO this breaks the selector
        # if ratio > 1.0:  # Severely communication-bound
        #     print(f"Communication-bound (ratio={ratio:.2f}x), halving tile sizes for better overlap")
        #     # Halve block sizes to create MORE tiles (but enforce minimum)
        #     MIN_BLOCK_SIZE = 64
        #     adjusted_bm = MIN_BLOCK_SIZE #max(MIN_BLOCK_SIZE, bm // 4)
        #     adjusted_bn = MIN_BLOCK_SIZE #max(MIN_BLOCK_SIZE, bn // 4)
        # else:
        adjusted_bm, adjusted_bn = bm, bn

    # Clamp to valid range
    m_tiles_per_batch = max(1, min(m_tiles_per_batch, num_m_tiles))

    return m_tiles_per_batch, adjusted_bm, adjusted_bn


def _derive_batch_metadata(selector, m_local, n, m_tiles_per_batch):
    """Return selector- and batching-derived metadata for JSON/reporting."""
    block_size_m = selector.block_m
    block_size_n = selector.block_n
    block_size_k = selector.block_k
    group_size_m = selector.group_m
    num_stages = getattr(selector, "num_stages", 2)
    waves_per_eu = getattr(selector, "waves_per_eu", 0)
    active_cus = getattr(selector, "_ACTIVE_CU", None)
    if active_cus is None:
        active_cus = getattr(selector._hardware, "N_CU", getattr(selector._hardware, "NUM_XCD", 1))

    num_tiles_m = (m_local + block_size_m - 1) // block_size_m
    num_tiles_n = (n + block_size_n - 1) // block_size_n
    tiles_per_group = max(1, group_size_m * num_tiles_n)
    groups_per_wave = max(1, active_cus // tiles_per_group)
    m_tiles_per_wave = min(num_tiles_m, groups_per_wave * group_size_m)
    num_batches = (num_tiles_m + m_tiles_per_batch - 1) // m_tiles_per_batch
    last_batch_m_tiles = num_tiles_m - max(0, num_batches - 1) * m_tiles_per_batch

    return {
        "output_tile_size_m": block_size_m,
        "output_tile_size_n": block_size_n,
        "output_tile_size_k": block_size_k,
        "group_size_m": group_size_m,
        "num_stages": num_stages,
        "waves_per_eu": waves_per_eu,
        "active_cus": active_cus,
        "num_tiles_m": num_tiles_m,
        "num_tiles_n": num_tiles_n,
        "tiles_per_group": tiles_per_group,
        "groups_per_wave": groups_per_wave,
        "m_tiles_per_wave": m_tiles_per_wave,
        "num_batches": num_batches,
        "last_batch_m_tiles": last_batch_m_tiles,
        "m_tiles_per_batch_over_wave": m_tiles_per_batch / max(1, m_tiles_per_wave),
    }


def _worker(args: dict):
    """Worker function for PyTorch distributed execution."""
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    backend = "nccl" if torch.cuda.is_available() else "gloo"
    dist.init_process_group(backend=backend)

    shmem = iris.iris(args["heap_size"])
    rank = shmem.get_rank()
    world_size = shmem.get_num_ranks()

    datatype_map = {"fp16": torch.float16, "fp32": torch.float32, "bf16": torch.bfloat16}
    datatype = datatype_map.get(args["datatype"], torch.float16)
    dtype_bytes = torch.tensor([], dtype=datatype).element_size()

    # Get device for tritonBLAS selector
    device = torch.device(f"cuda:{local_rank}")

    # Apply tritonBLAS selector to get block sizes
    selector, model_applied = _apply_model_defaults(args, world_size, datatype, device, dtype_bytes)
    if rank == 0 and model_applied:
        shmem.info(f"tritonBLAS selector applied: {', '.join(model_applied)}")

    M_local = args["m"]  # Local M dimension
    M = M_local * world_size  # Total M after gather
    N = args["n"]
    K = args["k"]

    # if args.get("m_tiles_per_batch") is None:
    #     args["m_tiles_per_batch"] = 1

    # Performance analysis
    perf_analysis = _analyze_performance(
        args["block_size_m"],
        args["block_size_n"],
        args["block_size_k"],
        M_local,
        N,
        K,
        world_size,
        args.get("link_bw", 50.0),
        dtype_bytes,
    )

    # Auto-tune m_tiles_per_batch if not user-specified
    if args.get("m_tiles_per_batch") is None:
        num_m_tiles = (M_local + args["block_size_m"] - 1) // args["block_size_m"]
        num_n_tiles = (N + args["block_size_n"] - 1) // args["block_size_n"]

        m_tiles_per_batch, adj_bm, adj_bn = _auto_tune_batching(
            perf_analysis,
            num_m_tiles,
            num_n_tiles,
            args["group_size_m"],
            args["block_size_m"],
            args["block_size_n"],
            dtype_bytes,
        )

        # Apply tuned values
        args["m_tiles_per_batch"] = m_tiles_per_batch

        # If block sizes were adjusted, update config
        if adj_bm != args["block_size_m"] or adj_bn != args["block_size_n"]:
            if rank == 0:
                shmem.info(f"Adjusted block sizes for communication-bound: {adj_bm}×{adj_bn}")
            args["block_size_m"] = adj_bm
            args["block_size_n"] = adj_bn

        if rank == 0:
            shmem.info(f"Auto-tuned m_tiles_per_batch: {m_tiles_per_batch}")

    # Print performance analysis
    if rank == 0 and perf_analysis is not None:
        shmem.info("\n" + "=" * 80)
        shmem.info("PERFORMANCE ANALYSIS")
        shmem.info("=" * 80)
        shmem.info(f"Block sizes: {args['block_size_m']}×{args['block_size_n']}×{args['block_size_k']}")
        num_m_tiles = (M_local + args["block_size_m"] - 1) // args["block_size_m"]
        num_n_tiles = (N + args["block_size_n"] - 1) // args["block_size_n"]
        shmem.info(f"Tiles: {num_m_tiles} M-tiles × {num_n_tiles} N-tiles = {num_m_tiles * num_n_tiles} total")
        shmem.info("\nPer-tile timing:")
        shmem.info(f"  GEMM:    {perf_analysis['gemm_wg_us']:.2f} μs")
        shmem.info(f"  Scatter: {perf_analysis['scatter_wg_us']:.2f} μs")
        shmem.info(f"  Ratio:   {perf_analysis['ratio']:.2f}x")
        shmem.info(f"\nBottleneck: {perf_analysis['bottleneck'].upper()}")
        shmem.info(f"m_tiles_per_batch: {args['m_tiles_per_batch']}")
        shmem.info("=" * 80 + "\n")

    batch_metadata = _derive_batch_metadata(selector, M_local, N, args["m_tiles_per_batch"])

    # Create config with parameters
    config_kwargs = {
        "block_size_m": args["block_size_m"],
        "block_size_n": args["block_size_n"],
        "block_size_k": args["block_size_k"],
        "group_size_m": args["group_size_m"],
    }
    if args["num_sms"] is not None:
        config_kwargs["num_sms"] = args["num_sms"]
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

    # Write performance analysis to JSON
    if perf_analysis is not None:
        for key, value in perf_analysis.items():
            json_writer.add_field(f"perf_analysis_{key}", value)

    # Export actual config values to JSON (including defaults)
    json_writer.add_field("block_size_m", config.block_size_m)
    json_writer.add_field("block_size_n", config.block_size_n)
    json_writer.add_field("block_size_k", config.block_size_k)
    json_writer.add_field("group_size_m", config.group_size_m)
    json_writer.add_field("num_sms", config.num_sms)
    json_writer.add_field("num_xcds", config.num_xcds)
    json_writer.add_field("m_tiles_per_batch", args["m_tiles_per_batch"])
    for key, value in batch_metadata.items():
        json_writer.add_field(key, value)

    # Create input and output tensors
    # A_local is M_local x K, output is M x N (gathered)
    A_local = shmem.zeros((M_local, K), dtype=datatype)
    B = shmem.zeros((K, N), dtype=datatype)
    C = shmem.zeros((M, N), dtype=datatype)

    # Fill inputs with deterministic values
    # Each rank has different A_local, same B
    torch.manual_seed(123 + rank)
    A_local_data = torch.randn((M_local, K), dtype=datatype, device=f"cuda:{rank}")
    A_local.copy_(A_local_data)

    torch.manual_seed(456)  # Same B for all ranks
    B_data = torch.randn((K, N), dtype=datatype, device=f"cuda:{rank}")
    B.copy_(B_data)

    # Expected
    expected_tensor = None
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

    # Pre-allocate workspace
    workspace = matmul_all_gather_host_copy_engine_preamble(
        shmem,
        A_local,
        B,
        config,
        m_tiles_per_batch=args["m_tiles_per_batch"],
    )
    workspace.selector = selector

    # ── Timing ───────────────────────────────────────────────────────────
    comm_stream = torch.cuda.Stream()
    anvil_lib = shmem.copy_engines
    start_ev = torch.cuda.Event(enable_timing=True)
    end_ev = torch.cuda.Event(enable_timing=True)
    total_ms = 0.0
    num_experiments = 0
    flag_iteration = 0

    def run_experiment():
        nonlocal total_ms, num_experiments, flag_iteration
        shmem.barrier()
        with torch.cuda.stream(comm_stream):
            start_ev.record()
            matmul_all_gather_host_copy_engine(
                shmem,
                C,
                A_local,
                B,
                config=config,
                m_tiles_per_batch=args["m_tiles_per_batch"],
                async_op=False,
                workspace=workspace,
                flag_iteration=flag_iteration,
            )
            end_ev.record()
            num_experiments += 1
            flag_iteration += 1
        shmem.barrier(comm_stream)
        total_ms += start_ev.elapsed_time(end_ev)

    shmem.barrier()

    # ── Validate ─────────────────────────────────────────────────────────
    if args["validate"]:
        shmem.info("Validating...")
        C.zero_()
        workspace.locks.zero_()
        workspace.completion_signals.zero_()
        shmem.barrier()

        # Run validation with verbose output to show SDMA timing
        matmul_all_gather_host_copy_engine(
            shmem,
            C,
            A_local,
            B,
            config=config,
            m_tiles_per_batch=args["m_tiles_per_batch"],
            async_op=False,
            workspace=workspace,
            verbose=True,
        )
        torch.cuda.synchronize()
        # for remote_rank in range(world_size):
        #     if remote_rank != rank:
        #         anvil_lib.host_quiet(rank, remote_rank, 0)
        shmem.barrier()

        atol = 1e-1 if datatype == torch.float16 else 1e-3
        rtol = 1e-2 if datatype == torch.float16 else 1e-5
        success = torch.allclose(C, expected_tensor, atol=atol, rtol=rtol)
        if not success:
            max_diff = torch.abs(C - expected_tensor).max().item()
            shmem.error(f"Rank {rank}: Validation FAILED, max diff: {max_diff}")
        else:
            shmem.info("Validation PASSED!")
        shmem.barrier()
        json_writer.add_field("success", success)

    # ── Benchmark ────────────────────────────────────────────────────────
    if args["benchmark"]:
        if args.get("single_run"):
            n_warmup, n_repeat = 0, 1
        else:
            n_warmup, n_repeat = 25, 100

        # Warmup
        total_ms = 0.0
        num_experiments = 0
        flag_iteration = 0
        workspace.locks.zero_()
        workspace.completion_signals.zero_()
        if n_warmup > 0:
            iris.do_bench(run_experiment, shmem.barrier, n_warmup=n_warmup, n_repeat=1)

        total_ms = 0.0
        num_experiments = 0
        flag_iteration = 0
        C.zero_()
        workspace.locks.zero_()
        workspace.completion_signals.zero_()
        shmem.barrier()

        iris.do_bench(run_experiment, shmem.barrier, n_warmup=0, n_repeat=n_repeat)
        torch.cuda.synchronize()
        # for remote_rank in range(world_size):
        #     if remote_rank != rank:
        #         anvil_lib.host_quiet(rank, remote_rank, 0)
        shmem.barrier()
        avg_ms = total_ms / num_experiments if num_experiments > 0 else 0

        total_flops = 2 * M_local * N * K
        tflops = (total_flops * 1e-12) / (avg_ms * 1e-3) if avg_ms > 0 else 0
        element_size = torch.tensor([], dtype=datatype).element_size()
        output_bytes = M_local * N * element_size
        total_bytes = output_bytes * (world_size - 1)
        total_bytes_gb = total_bytes / (1024**3)
        bw_gbps = (total_bytes / (1024**3)) / (avg_ms * 1e-3) if avg_ms > 0 else 0

        shmem.info(
            f"Matmul-all-gather host copy engine (M_local={M_local}, M_total={M}, N={N}, K={K}, world_size={world_size}, dtype={args['datatype']}): "
            f"{avg_ms:.3f} ms, {tflops:.3f} TFLOPS, {bw_gbps:.3f} GB/s"
        )

        json_writer.add_field("tflops", tflops)
        json_writer.add_field("bandwidth_gbps", bw_gbps)
        json_writer.add_field("avg_ms", avg_ms)
        json_writer.add_field("total_flops", total_flops)
        json_writer.add_field("total_bytes", total_bytes)
        json_writer.add_field("total_bytes_gb", total_bytes_gb)

        shmem.barrier()

    # ── Trace ────────────────────────────────────────────────────────────
    if args["trace"]:
        # Warmup: compile the TRACE=True kernel variant before the real run
        shmem.info("Trace warmup (compiling traced kernel variant)...")
        C.zero_()
        workspace.locks.zero_()
        workspace.completion_signals.zero_()
        shmem.barrier()
        matmul_all_gather_host_copy_engine(
            shmem,
            C,
            A_local,
            B,
            config=config,
            m_tiles_per_batch=args["m_tiles_per_batch"],
            async_op=False,
            workspace=workspace,
            trace=True,
        )
        torch.cuda.synchronize()
        # for remote_rank in range(world_size):
        #     if remote_rank != rank:
        #         anvil_lib.host_quiet(rank, remote_rank, 0)
        shmem.barrier()

        # Actual traced run (post-compilation, clean state)
        shmem.info("Running single traced iteration...")
        C.zero_()
        workspace.locks.zero_()
        workspace.completion_signals.zero_()
        shmem.barrier()
        matmul_all_gather_host_copy_engine(
            shmem,
            C,
            A_local,
            B,
            config=config,
            m_tiles_per_batch=args["m_tiles_per_batch"],
            async_op=False,
            workspace=workspace,
            trace=True,
        )
        torch.cuda.synchronize()
        # for remote_rank in range(world_size):
        #     if remote_rank != rank:
        #         anvil_lib.host_quiet(rank, remote_rank, 0)
        shmem.barrier()

        if rank == 0 and hasattr(workspace, "trace_data"):
            trace_out = args.get("trace_output", "trace_gantt.png")
            try:
                _plot_trace(workspace.trace_data, trace_out, rank, M, N, K)
            except ImportError:
                print("  (matplotlib not available -- skipping trace plot)")
            except Exception as e:
                print(f"  (Trace plot failed: {e})")
        shmem.barrier()

    if rank == 0:
        json_writer.flush()
        json_writer.display()

    # Synchronize device before exiting
    shmem.barrier(sync_copy_engine=True)
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
