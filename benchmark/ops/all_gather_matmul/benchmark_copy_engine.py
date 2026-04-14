#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
# Copyright (c) 2025-2026 Advanced Micro Devices, Inc. All rights reserved.

"""
Benchmark for the copy engine all_gather_matmul variant.

This variant uses the copy engine (SDMA) for data movement. SMs only perform GEMM:
- Host orchestrates SDMA transfers of remote tiles to HBM buffer
- GEMM workgroups process local K-blocks first (from A_sharded), then remote K-blocks
- No fetcher workgroups - SMs focus purely on computation

Usage with torchrun:
    torchrun --nproc_per_node=8 benchmark/ops/all_gather_matmul/benchmark_copy_engine.py \\
        -m 2048 -n 16384 -k 131072 --benchmark

    torchrun --nproc_per_node=8 benchmark/ops/all_gather_matmul/benchmark_copy_engine.py \\
        -m 2048 -n 16384 -k 131072 --benchmark --benchmark_pytorch --validate
"""

import os
import time
import torch
import torch.distributed as dist
import random
import argparse
import numpy as np

from examples.common.utils import JSONWriter

import iris
from iris.ops.all_gather_matmul_copy_engine import (
    all_gather_matmul_copy_engine,
    all_gather_matmul_copy_engine_preamble,
)
from iris.ops import FusedConfig
from tritonblas.matmul import _make_matmul_selector

_DERIVE_AVAILABLE = False
try:
    import sys as _sys

    _script_dir = os.path.dirname(os.path.abspath(__file__))
    if _script_dir not in _sys.path:
        _sys.path.insert(0, _script_dir)
    from derive_params_copy_engine import (
        derive as _derive_params,
        DEFAULT_NUM_CUS,
        DEFAULT_PEAK_TFLOPS_FP16,
        DEFAULT_HBM_BW_GBPS,
        DEFAULT_L2_SIZE_BYTES,
        DEFAULT_SCHEDULING_FACTOR,
    )

    _DERIVE_AVAILABLE = True
except Exception:
    pass

_MODEL_PARAMS = (
    "block_size_m",
    "block_size_n",
    "block_size_k",
    "group_size_m",
    "k_per_flag",
    "m_tiles_per_batch",
    "device_initiated",
    "num_warps",
)

_FALLBACK_DEFAULTS = {
    "block_size_m": 256,
    "block_size_n": 64,
    "block_size_k": 64,
    "group_size_m": 4,  # M-grouping for L2 cache reuse
    "k_per_flag": 4,  # Copy engine default (larger batches)
    "m_tiles_per_batch": 8,  # Default batch size
    "device_initiated": True,  # Fallback: device mode (derive() should set this)
    "num_warps": 4,
}

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

    # Check for SDMA WG traces
    has_sdma = "sdma_start" in trace_data and trace_data.get("num_sdma", 0) > 0
    if has_sdma:
        sdma_starts = trace_data["sdma_start"].numpy().astype(np.int64)
        sdma_ends = trace_data["sdma_end"].numpy().astype(np.int64)
        sdma_xcds = trace_data["sdma_xcd"].numpy().astype(np.int32)
        num_sdma = trace_data["num_sdma"]
    else:
        num_sdma = 0

    # Convert to microseconds relative to earliest start
    t_min = starts.min()
    if has_sdma and len(sdma_starts) > 0:
        t_min = min(t_min, sdma_starts.min())

    starts_us = (starts - t_min) / TICKS_PER_US
    ends_us = (ends - t_min) / TICKS_PER_US
    waits_us = waits / TICKS_PER_US

    if has_sdma:
        sdma_starts_us = (sdma_starts - t_min) / TICKS_PER_US
        sdma_ends_us = (sdma_ends - t_min) / TICKS_PER_US

    # Sort GEMM by start time
    order = np.argsort(starts_us)

    # Figure sizing - add SDMA rows at top
    total_rows = grid_size + num_sdma
    row_h = 0.012
    fig_h = max(12, total_rows * row_h + 2)
    fig, ax = plt.subplots(figsize=(18, fig_h))

    wait_color = "#F44336"  # red
    compute_color = "#4CAF50"  # green
    sdma_color = "#2196F3"  # blue (for SDMA posting)

    y_offset = 0

    # Plot SDMA WGs at top (if any)
    if has_sdma:
        for sdma_idx in range(num_sdma):
            s = sdma_starts_us[sdma_idx]
            e = sdma_ends_us[sdma_idx]
            dur = e - s
            ax.barh(y_offset + sdma_idx, dur, left=s, height=0.8, color=sdma_color, edgecolor="none", linewidth=0)
        y_offset = num_sdma

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
    if has_sdma:
        x_max = max(x_max, sdma_ends_us.max() * 1.02)

    for y_idx, wg_idx in enumerate(order):
        xcd_id = xcds[wg_idx]
        if xcd_id in xcd_cmap:
            ax.plot(x_max, y_offset + y_idx, marker="s", markersize=1.5, color=xcd_cmap[xcd_id], clip_on=False)

    if has_sdma:
        for sdma_idx in range(num_sdma):
            xcd_id = sdma_xcds[sdma_idx]
            if xcd_id in xcd_cmap:
                ax.plot(x_max, sdma_idx, marker="s", markersize=1.5, color=xcd_cmap[xcd_id], clip_on=False)

    ax.set_xlabel("Time (us)", fontsize=12)
    ax.set_ylabel("Workgroup (SDMA WGs at top, then GEMM sorted by start)", fontsize=12)
    title = f"Rank {rank}  |  Copy Engine All-Gather GEMM Trace  |  M={M} N={N} K={K}  |  {grid_size} GEMM"
    if has_sdma:
        title += f" + {num_sdma} SDMA WGs"
    else:
        title += " workgroups"
    ax.set_title(title, fontsize=13)
    ax.set_ylim(-1, total_rows + 1)
    ax.set_xlim(0, x_max)
    ax.invert_yaxis()

    # Legend
    legend_elements = [
        Line2D([0], [0], color=wait_color, lw=6, label="GEMM: waiting on remote data"),
        Line2D([0], [0], color=compute_color, lw=6, label="GEMM: compute"),
    ]
    if has_sdma:
        legend_elements.append(Line2D([0], [0], color=sdma_color, lw=6, label="SDMA: posting transfers"))
    ax.legend(handles=legend_elements, loc="upper right", fontsize=10)

    # Summary stats
    gemm_dur = ends_us - starts_us
    gemm_wait = waits_us
    gemm_compute = gemm_dur - gemm_wait

    # Wall time is max(GEMM end, SDMA end) - min(GEMM start, SDMA start)
    wall_time_us = ends_us.max()
    if has_sdma and len(sdma_ends_us) > 0:
        wall_time_us = max(wall_time_us, sdma_ends_us.max())

    stats_lines = [
        f"GEMM total: {gemm_dur.mean():.1f} us avg  ({gemm_dur.min():.1f}-{gemm_dur.max():.1f})",
        f"  wait: {gemm_wait.mean():.1f} us avg  ({gemm_wait.min():.1f}-{gemm_wait.max():.1f})",
        f"  compute: {gemm_compute.mean():.1f} us avg  ({gemm_compute.min():.1f}-{gemm_compute.max():.1f})",
        f"  wait%: {100 * gemm_wait.sum() / gemm_dur.sum():.1f}%",
    ]

    if has_sdma and len(sdma_starts_us) > 0:
        sdma_dur = sdma_ends_us - sdma_starts_us
        stats_lines.append(f"SDMA: {sdma_dur.mean():.1f} us avg  ({sdma_dur.min():.1f}-{sdma_dur.max():.1f})")

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
        description="Benchmark copy engine all_gather_matmul.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("-m", type=int, default=2048, help="M dimension")
    parser.add_argument("-n", type=int, default=16384, help="N dimension")
    parser.add_argument("-k", type=int, default=131072, help="K dimension (total)")
    parser.add_argument("-v", "--validate", action="store_true", help="Validate correctness")
    parser.add_argument("-b", "--benchmark", action="store_true", help="Run benchmark")
    parser.add_argument(
        "--datatype",
        type=str,
        default="fp16",
        choices=["fp16", "fp32", "bf16"],
        help="Tensor datatype",
    )
    parser.add_argument("--heap_size", type=int, default=1 << 34, help="Iris heap size")
    parser.add_argument("--num_sms", type=int, default=None, help="Number of SMs (auto if None)")
    parser.add_argument(
        "--benchmark_pytorch",
        action="store_true",
        help="Also benchmark PyTorch (all_gather_into_tensor + matmul)",
    )
    parser.add_argument("--block_size_m", type=int, default=None, help="Block size M (model-derived if omitted)")
    parser.add_argument("--block_size_n", type=int, default=None, help="Block size N (model-derived if omitted)")
    parser.add_argument("--block_size_k", type=int, default=None, help="Block size K (model-derived if omitted)")
    parser.add_argument("--group_size_m", type=int, default=None, help="Group size M (model-derived if omitted)")
    parser.add_argument("--num_xcds", type=int, default=None, help="Number of XCDs (auto if None)")
    parser.add_argument("--b_col_major", action="store_true", help="B col-major (K-contiguous)")
    parser.add_argument("--a_col_major", action="store_true", help="A col-major (M-contiguous)")
    parser.add_argument("--single-run", action="store_true", help="1 iteration (for profiling)")
    parser.add_argument("--k_per_flag", type=int, default=4, help="K-blocks per ready flag")
    parser.add_argument(
        "--m_tiles_per_batch",
        type=int,
        default=None,
        help="M-tiles per batch for K-block batching (None = all M-tiles)",
    )
    parser.add_argument("--num_warps", type=int, default=None, help="Triton num_warps (auto if None)")
    parser.add_argument("--num_stages", type=int, default=None, help="Triton num_stages (auto if None)")
    parser.add_argument(
        "--trace",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Collect per-workgroup trace and save Gantt chart PNG",
    )
    parser.add_argument("--trace_output", type=str, default="trace_copy_engine.png", help="Output path for trace plot")
    parser.add_argument(
        "--force-device-initiated",
        action="store_true",
        dest="force_device_mode",
        help="Force device-initiated SDMA mode (overrides model)",
    )
    parser.add_argument(
        "--force-host-initiated",
        action="store_true",
        dest="force_host_mode",
        help="Force host-initiated SDMA mode (overrides model)",
    )
    parser.add_argument(
        "--output_file", type=str, default="all_gather_matmul_copy_engine.json", help="Output JSON file"
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


def _worker(args):
    """Worker function for torchrun."""
    local_rank = int(os.environ.get("RANK", os.environ.get("LOCAL_RANK", 0)))
    world_size_env = int(os.environ.get("WORLD_SIZE", 1))

    t0 = time.perf_counter()

    backend = "nccl" if torch.cuda.is_available() else "gloo"

    if "RANK" in os.environ or "LOCAL_RANK" in os.environ:
        dist.init_process_group(
            backend=backend,
            init_method="env://",
            device_id=torch.device(f"cuda:{local_rank}") if torch.cuda.is_available() else None,
        )
    else:
        dist.init_process_group(
            backend=backend,
            init_method="tcp://127.0.0.1:29530",
            world_size=world_size_env,
            rank=local_rank,
            device_id=torch.device(f"cuda:{local_rank}") if torch.cuda.is_available() else None,
        )

    t1 = time.perf_counter()

    shmem = iris.iris(args["heap_size"])
    rank = shmem.get_rank()
    world_size = shmem.get_num_ranks()

    t2 = time.perf_counter()
    shmem.info(f"Startup: dist.init={t1 - t0:.1f}s, iris.init={t2 - t1:.1f}s, total={t2 - t0:.1f}s")

    datatype_map = {"fp16": torch.float16, "fp32": torch.float32, "bf16": torch.bfloat16}
    datatype = datatype_map.get(args["datatype"], torch.float16)
    dtype_bytes = torch.tensor([], dtype=datatype).element_size()

    # Get device for tritonBLAS selector
    device = torch.device(f"cuda:{local_rank}")

    # TODO why are we passing datatype and dtype?
    selector, model_applied = _apply_model_defaults(args, world_size, datatype, device, dtype_bytes)
    if rank == 0 and model_applied:
        shmem.info(f"tritonBLAS selector applied: {', '.join(model_applied)}")

    if args.get("force_device_mode") and args.get("force_host_mode"):
        raise ValueError("Cannot set both --force-device-initiated and --force-host-initiated")
    if args.get("force_device_mode"):
        args["device_initiated"] = True
    elif args.get("force_host_mode"):
        args["device_initiated"] = False
    elif "device_initiated" not in args:
        args["device_initiated"] = _FALLBACK_DEFAULTS["device_initiated"]

    if rank == 0:
        param_summary = " ".join(f"{k}={args.get(k)}" for k in _MODEL_PARAMS)
        shmem.info(f"Kernel params: {param_summary}")

    M = args["m"]
    N = args["n"]
    K = args["k"]
    K_local = K // world_size

    perf_analysis = None
    if args.get("m_tiles_per_batch") is None:
        args["m_tiles_per_batch"] = 1


    # Print performance analysis
    if rank == 0 and perf_analysis is not None:
        shmem.info("\n" + "=" * 80)
        shmem.info("PERFORMANCE ANALYSIS")
        shmem.info("=" * 80)
        shmem.info(f"Block sizes: {args['block_size_m']}×{args['block_size_n']}×{args['block_size_k']}")
        num_m_tiles = (M + args["block_size_m"] - 1) // args["block_size_m"]
        num_n_tiles = (N + args["block_size_n"] - 1) // args["block_size_n"]
        shmem.info(f"Tiles: {num_m_tiles} M-tiles × {num_n_tiles} N-tiles = {num_m_tiles * num_n_tiles} total")
        shmem.info(f"\nPer-tile timing:")
        shmem.info(f"  GEMM:    {perf_analysis['gemm_wg_us']:.2f} μs")
        shmem.info(f"  Scatter: {perf_analysis['scatter_wg_us']:.2f} μs")
        shmem.info(f"  Ratio:   {perf_analysis['ratio']:.2f}x")
        shmem.info(f"\nBottleneck: {perf_analysis['bottleneck'].upper()}")
        shmem.info(f"m_tiles_per_batch: {args['m_tiles_per_batch']}")
        shmem.info("=" * 80 + "\n")

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
    json_writer.add_field("operation", "all_gather_matmul_copy_engine")
    json_writer.add_field("variant", "copy_engine")
    json_writer.add_field("k_local", K_local)

    for key, value in args.items():
        json_writer.add_field(key, value)

    # Write performance analysis to JSON
    if perf_analysis is not None:
        for key, value in perf_analysis.items():
            json_writer.add_field(f"perf_analysis_{key}", value)

    # Export actual config values
    json_writer.add_field("block_size_m", config.block_size_m)
    json_writer.add_field("block_size_n", config.block_size_n)
    json_writer.add_field("block_size_k", config.block_size_k)
    json_writer.add_field("num_sms", config.num_sms)
    json_writer.add_field("num_xcds", config.num_xcds)

    buffer_mb = M * K * torch.tensor([], dtype=datatype).element_size() / (1024**2)
    num_m_tiles = M // config.block_size_m
    num_tiles_n = (N + config.block_size_n - 1) // config.block_size_n
    num_k_blocks = K // config.block_size_k
    num_k_blocks_local = K_local // config.block_size_k
    k_per_flag = args["k_per_flag"]
    num_k_block_groups = (num_k_blocks_local + k_per_flag - 1) // k_per_flag

    # Calculate flag count for info message
    # Note: num_sms is auto-detected by the kernel if not specified
    total_gemm_tiles = num_m_tiles * num_tiles_n
    num_flags = num_m_tiles * num_k_block_groups * (world_size - 1)
    m_tiles_per_batch = args["m_tiles_per_batch"]

    shmem.info(
        f"Copy Engine variant: M={M} N={N} K={K} K_local={K_local} "
        f"block=({config.block_size_m},{config.block_size_n},{config.block_size_k}) "
        f"buffer={buffer_mb:.0f}MB tiles={total_gemm_tiles} flags={num_flags} "
        f"(k_per_flag={k_per_flag}, m_tiles_per_batch={m_tiles_per_batch})"
    )

    # ── Allocate tensors ─────────────────────────────────────────────────
    C = shmem.zeros((M, N), dtype=datatype)

    if args["a_col_major"]:
        A_storage = shmem.zeros((K_local, M), dtype=datatype)
        A_sharded = A_storage.T
    else:
        A_sharded = shmem.zeros((M, K_local), dtype=datatype)

    if args["b_col_major"]:
        B_storage = shmem.zeros((N, K), dtype=datatype)
        B = B_storage.T
    else:
        B = shmem.zeros((K, N), dtype=datatype)

    shmem.info(f"A strides={A_sharded.stride()}, B strides={B.stride()}")

    # Fill
    torch.manual_seed(123 + rank)
    A_data = torch.randn((M, K_local), dtype=datatype, device=f"cuda:{rank}")
    A_sharded.copy_(A_data)

    torch.manual_seed(456)
    B_data = torch.randn((K, N), dtype=datatype, device=f"cuda:{rank}")
    B.copy_(B_data)

    # Expected
    expected_tensor = None
    if args["validate"]:
        A_list = [torch.zeros((M, K_local), dtype=datatype, device=f"cuda:{rank}") for _ in range(world_size)]
        dist.all_gather(A_list, A_data)
        A_gathered = torch.cat(A_list, dim=1)
        expected_tensor = shmem.zeros((M, N), dtype=datatype)
        expected_tensor.copy_(torch.matmul(A_gathered, B_data))

    # Pre-allocate workspace
    workspace = all_gather_matmul_copy_engine_preamble(
        shmem, A_sharded, B, config, k_per_flag=k_per_flag, m_tiles_per_batch=args["m_tiles_per_batch"]
    )
    # selector = _make_matmul_selector(
    #     M, N, K, A_sharded.dtype, B.dtype, A_sharded.dtype, A_sharded.device, streamk=False
    # )
    workspace.selector = selector
    # workspace.tb_num_tiles_m = (M + selector.block_m - 1) // selector.block_m
    # workspace.tb_num_tiles_n = (N + selector.block_n - 1) // selector.block_n
    # workspace.tb_num_batches = (workspace.tb_num_tiles_m + args["m_tiles_per_batch"] - 1) // args["m_tiles_per_batch"]
    # if workspace.locks.numel() != workspace.tb_num_batches:
    #     workspace.locks = shmem.zeros((workspace.tb_num_batches,), dtype=torch.int32)

    # ── Timing ───────────────────────────────────────────────────────────
    comm_stream = torch.cuda.Stream()
    start_ev = torch.cuda.Event(enable_timing=True)
    end_ev = torch.cuda.Event(enable_timing=True)
    total_ms = 0.0
    num_experiments = 0
    flag_iteration = 0

    num_warps = args["num_warps"]
    num_stages = args["num_stages"]

    def run_experiment():
        nonlocal total_ms, num_experiments, flag_iteration
        shmem.barrier()
        with torch.cuda.stream(comm_stream):
            start_ev.record()
            all_gather_matmul_copy_engine(
                shmem,
                C,
                A_sharded,
                B,
                config=config,
                async_op=False,
                workspace=workspace,
                flag_iteration=flag_iteration,
                k_per_flag=k_per_flag,
                m_tiles_per_batch=args["m_tiles_per_batch"],
                num_warps=num_warps,
                num_stages=num_stages,
                device_initiated=args.get("device_initiated", False),
            )
            end_ev.record()
            num_experiments += 1
            flag_iteration += 1
        shmem.barrier()
        total_ms += start_ev.elapsed_time(end_ev)

    shmem.barrier()

    # ── Validate ─────────────────────────────────────────────────────────
    if args["validate"]:
        shmem.info("Validating...")
        C.zero_()
        shmem.barrier()
        run_experiment()
        torch.cuda.synchronize()
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
        if n_warmup > 0:
            iris.do_bench(run_experiment, shmem.barrier, n_warmup=n_warmup, n_repeat=1)

        total_ms = 0.0
        num_experiments = 0
        C.zero_()
        shmem.barrier()

        iris.do_bench(run_experiment, shmem.barrier, n_warmup=0, n_repeat=n_repeat)
        avg_ms = total_ms / num_experiments if num_experiments > 0 else 0

        total_flops = 2 * M * N * K
        tflops = (total_flops * 1e-12) / (avg_ms * 1e-3) if avg_ms > 0 else 0
        element_size = torch.tensor([], dtype=datatype).element_size()
        total_bytes = M * K_local * element_size * (world_size - 1)
        bw_gbps = (total_bytes / (1024**3)) / (avg_ms * 1e-3) if avg_ms > 0 else 0

        shmem.info(
            f"Copy Engine (M={M}, K_local={K_local}, K={K}, N={N}, "
            f"ws={world_size}, dtype={args['datatype']}): "
            f"{avg_ms:.3f} ms, {tflops:.3f} TFLOPS, {bw_gbps:.3f} GB/s"
        )

        json_writer.add_field("tflops", tflops)
        json_writer.add_field("bandwidth_gbps", bw_gbps)
        json_writer.add_field("avg_ms", avg_ms)
        json_writer.add_field("total_flops", total_flops)
        json_writer.add_field("total_bytes", total_bytes)
        json_writer.add_field("total_bytes_gb", total_bytes / (1024**3))

        shmem.barrier()

        # ── Per-rank finish time measurement ─────────────────────────────
        shmem.barrier()
        torch.cuda.synchronize()
        dist.barrier()

        # Synchronized start
        dist.barrier()
        t_start = time.perf_counter()

        all_gather_matmul_copy_engine(
            shmem,
            C,
            A_sharded,
            B,
            config=config,
            async_op=False,
            workspace=workspace,
            k_per_flag=k_per_flag,
            m_tiles_per_batch=args["m_tiles_per_batch"],
            num_warps=num_warps,
            num_stages=num_stages,
            device_initiated=args.get("device_initiated", False),
        )
        torch.cuda.synchronize()
        t_end = time.perf_counter()

        finish_ms = (t_end - t_start) * 1000.0

        # Gather all finish times to rank 0
        finish_tensor = torch.tensor([finish_ms], dtype=torch.float64, device=f"cuda:{rank}")
        all_finish = [torch.zeros(1, dtype=torch.float64, device=f"cuda:{rank}") for _ in range(world_size)]
        dist.all_gather(all_finish, finish_tensor)

        if rank == 0:
            times = [t.item() for t in all_finish]
            min_t = min(times)
            max_t = max(times)
            print("\n  Per-rank finish times (single run):")
            print(f"  {'Rank':>6}  {'Finish ms':>10}  {'Delta ms':>10}")
            print(f"  {'-' * 30}")
            for r, t in enumerate(times):
                delta = t - min_t
                print(f"  {r:>6}  {t:>10.3f}  {delta:>+10.3f}")
            print(f"  {'-' * 30}")
            print(f"  Spread (max - min): {max_t - min_t:.3f} ms")
            print()

        shmem.barrier()

    # ── Trace ────────────────────────────────────────────────────────────
    if args["trace"]:
        shmem.info("Trace warmup (compiling traced kernel variant)...")
        C.zero_()
        workspace.locks.zero_()
        shmem.barrier()
        all_gather_matmul_copy_engine(
            shmem,
            C,
            A_sharded,
            B,
            config=config,
            async_op=False,
            workspace=workspace,
            k_per_flag=k_per_flag,
            m_tiles_per_batch=args["m_tiles_per_batch"],
            num_warps=num_warps,
            num_stages=num_stages,
            device_initiated=args.get("device_initiated", False),
            trace=True,
        )
        torch.cuda.synchronize()
        shmem.barrier()

        # Actual traced run
        shmem.info("Running single traced iteration...")
        C.zero_()
        workspace.locks.zero_()
        shmem.barrier()

        all_gather_matmul_copy_engine(
            shmem,
            C,
            A_sharded,
            B,
            config=config,
            async_op=False,
            workspace=workspace,
            k_per_flag=k_per_flag,
            m_tiles_per_batch=args["m_tiles_per_batch"],
            num_warps=num_warps,
            num_stages=num_stages,
            device_initiated=args.get("device_initiated", False),
            trace=True,
        )
        torch.cuda.synchronize()
        shmem.barrier()

        if rank == 0 and hasattr(workspace, "trace_data"):
            trace_out = args.get("trace_output", "trace_copy_engine.png")
            try:
                _plot_trace(workspace.trace_data, trace_out, rank, M, N, K)
            except ImportError:
                print("  (matplotlib not available -- skipping trace plot)")
            except Exception as e:
                print(f"  (Trace plot failed: {e})")
        shmem.barrier()

    # ── PyTorch baseline ─────────────────────────────────────────────────
    if args["benchmark_pytorch"]:
        shmem.info("Benchmarking PyTorch (all_gather_into_tensor + matmul)...")

        pt_A = torch.randn(M, K_local, dtype=datatype, device=f"cuda:{rank}")
        pt_B = torch.randn(K, N, dtype=datatype, device=f"cuda:{rank}")
        pt_Ag = torch.zeros(M, K, dtype=datatype, device=f"cuda:{rank}")

        for _ in range(10):
            dist.all_gather_into_tensor(pt_Ag, pt_A)
            _ = torch.matmul(pt_Ag, pt_B)
        torch.cuda.synchronize()
        dist.barrier()

        def run_pt():
            dist.all_gather_into_tensor(pt_Ag, pt_A)
            _ = torch.matmul(pt_Ag, pt_B)

        total_flops = 2 * M * N * K
        element_size = torch.tensor([], dtype=datatype).element_size()
        total_bytes = M * K_local * element_size * (world_size - 1)

        pt_ms = iris.do_bench(run_pt, dist.barrier)
        pt_tflops = (total_flops * 1e-12) / (pt_ms * 1e-3) if pt_ms > 0 else 0
        pt_bw = (total_bytes / (1024**3)) / (pt_ms * 1e-3) if pt_ms > 0 else 0

        shmem.info(
            f"PyTorch (M={M}, K_local={K_local}, K={K}, N={N}, ws={world_size}, "
            f"dtype={args['datatype']}): "
            f"{pt_ms:.3f} ms, {pt_tflops:.3f} TFLOPS, {pt_bw:.3f} GB/s"
        )

        if args["benchmark"]:
            avg_ms = total_ms / num_experiments if num_experiments > 0 else 0
            iris_tflops = (total_flops * 1e-12) / (avg_ms * 1e-3) if avg_ms > 0 else 0
            speedup = iris_tflops / pt_tflops if pt_tflops > 0 else 0
            shmem.info(f"Speedup (Copy Engine / PyTorch): {speedup:.2f}x")

        shmem.barrier()

    if rank == 0:
        json_writer.flush()
        json_writer.display()

    shmem.barrier()
    dist.destroy_process_group()


def main():
    print("Starting copy engine all_gather_matmul benchmark...")
    args = parse_args()
    if "RANK" in os.environ or "LOCAL_RANK" in os.environ:
        _worker(args)
    else:
        print(
            "Please run with torchrun:\n"
            "  torchrun --nproc_per_node=N "
            "benchmark/ops/all_gather_matmul/benchmark_copy_engine.py [OPTIONS]"
        )


if __name__ == "__main__":
    main()
