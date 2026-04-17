#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
# Copyright (c) 2025-2026 Advanced Micro Devices, Inc. All rights reserved.

"""
Benchmark for the HBM-buffered all_gather_matmul variant.

This variant cooperatively gathers A into a local HBM buffer with per-tile
ready flags, then runs GEMM from local memory. No global barriers -- CUs
that finish gathering early start GEMM immediately, spinning on flags for
any tile not yet available.

Usage with torchrun:
    torchrun --nproc_per_node=8 benchmark/ops/all_gather_matmul/benchmark_hbm_buffer.py \\
        -m 2048 -n 16384 -k 131072 --benchmark

    torchrun --nproc_per_node=8 benchmark/ops/all_gather_matmul/benchmark_hbm_buffer.py \\
        -m 2048 -n 16384 -k 131072 --benchmark --benchmark_pytorch --b_col_major
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
from iris.ops.all_gather_matmul_hbm_buffer import (
    all_gather_matmul_hbm_buffer,
    all_gather_matmul_hbm_buffer_preamble,
)
from iris.ops import FusedConfig

_DERIVE_AVAILABLE = False
try:
    import sys as _sys

    _script_dir = os.path.dirname(os.path.abspath(__file__))
    if _script_dir not in _sys.path:
        _sys.path.insert(0, _script_dir)
    from derive_params import (
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
    "num_fetch_sms",
    "k_per_flag",
    "num_warps",
    "num_fetch_stages",
    "first_stage_fetch_sms",
)

_FALLBACK_DEFAULTS = {
    "block_size_m": 256,
    "block_size_n": 64,
    "block_size_k": 64,
    "group_size_m": 1,
    "k_per_flag": 1,
    "num_fetch_stages": 1,
}

torch.manual_seed(123)
random.seed(123)

TICKS_PER_US = 100  # s_memrealtime runs at 100 MHz: 1 tick = 10 ns = 0.01 us


def _plot_trace(trace_data, output_path, rank, M, N, K, num_fetch_sms_cfg):
    """Generate a tall Gantt chart showing per-workgroup activity over time.

    Y-axis: workgroup (sorted by start time)
    X-axis: time in microseconds
    Colors: fetcher stages (blue shades), GEMM wait (red), GEMM compute (green)
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
    n_fetch_per_stage = trace_data["num_fetch_sms"]
    n_stages = trace_data.get("num_fetch_stages", 1)
    total_fetch = trace_data.get("total_fetch_wgs", n_fetch_per_stage)
    first_stage_fetch = trace_data.get("first_stage_fetch_sms", n_fetch_per_stage)
    first_stage_size = trace_data.get("first_stage_size", grid_size)
    rest_stage_size = trace_data.get("rest_stage_size", grid_size)

    # Convert to microseconds relative to earliest start
    t_min = starts.min()
    starts_us = (starts - t_min) / TICKS_PER_US
    ends_us = (ends - t_min) / TICKS_PER_US
    waits_us = waits / TICKS_PER_US

    # Build role array: stage index for fetchers (0..S-1), S for GEMM
    # Asymmetric layout: [fetch0 (P)] [gemm0] [fetch1 (F)] [gemm1] ...
    roles = np.empty(grid_size, dtype=np.int32)
    for i in range(grid_size):
        if i < first_stage_size:
            stage = 0
            local = i
            fetch_thresh = first_stage_fetch
        else:
            adjusted = i - first_stage_size
            stage = 1 + adjusted // rest_stage_size
            local = adjusted % rest_stage_size
            fetch_thresh = n_fetch_per_stage
        if local < fetch_thresh:
            roles[i] = stage  # fetcher for this stage
        else:
            roles[i] = n_stages  # GEMM

    # Sort by start time
    order = np.argsort(starts_us)

    # Compute figure height: ~0.012 inches per row, min 12 inches
    row_h = 0.012
    fig_h = max(12, grid_size * row_h + 2)
    fig, ax = plt.subplots(figsize=(18, fig_h))

    # One color per fetch stage (blue palette), plus GEMM colors
    fetch_blues = ["#1565C0", "#42A5F5", "#90CAF9", "#BBDEFB"]
    wait_color = "#F44336"  # red
    compute_color = "#4CAF50"  # green

    for y_idx, wg_idx in enumerate(order):
        s = starts_us[wg_idx]
        e = ends_us[wg_idx]
        dur = e - s
        role = roles[wg_idx]

        if role < n_stages:
            # Fetcher: color by stage
            c = fetch_blues[role % len(fetch_blues)]
            ax.barh(y_idx, dur, left=s, height=0.8, color=c, edgecolor="none", linewidth=0)
        else:
            # GEMM: split into wait (red) and compute (green)
            w = waits_us[wg_idx]
            comp = max(0, dur - w)
            ax.barh(y_idx, w, left=s, height=0.8, color=wait_color, edgecolor="none", linewidth=0)
            ax.barh(y_idx, comp, left=s + w, height=0.8, color=compute_color, edgecolor="none", linewidth=0)

    # XCD annotations on the right margin
    xcd_set = sorted(set(xcds.tolist()))
    xcd_cmap = {}
    if len(xcd_set) > 1:
        cmap = matplotlib.colormaps.get_cmap("tab10").resampled(len(xcd_set))
        for i, x in enumerate(xcd_set):
            xcd_cmap[x] = cmap(i)

    x_max = ends_us.max() * 1.02
    for y_idx, wg_idx in enumerate(order):
        xcd_id = xcds[wg_idx]
        if xcd_id in xcd_cmap:
            ax.plot(x_max, y_idx, marker="s", markersize=1.5, color=xcd_cmap[xcd_id], clip_on=False)

    n_gemm = grid_size - total_fetch
    if n_stages > 1 and first_stage_fetch != n_fetch_per_stage:
        stage_info = f"{first_stage_fetch}+{n_stages - 1}x{n_fetch_per_stage}"
    elif n_stages > 1:
        stage_info = f"{n_stages}x{n_fetch_per_stage}"
    else:
        stage_info = str(first_stage_fetch)
    ax.set_xlabel("Time (us)", fontsize=12)
    ax.set_ylabel("Workgroup (sorted by start time)", fontsize=12)
    ax.set_title(
        f"Rank {rank}  |  All-Gather GEMM Trace  |  "
        f"M={M} N={N} K={K}  |  "
        f"{stage_info} fetchers + {n_gemm} GEMM workgroups",
        fontsize=13,
    )
    ax.set_ylim(-1, grid_size + 1)
    ax.set_xlim(0, x_max)

    # Invert y so earliest-starting workgroups are at top
    ax.invert_yaxis()

    # Legend
    legend_elements = []
    for s_idx in range(min(n_stages, len(fetch_blues))):
        legend_elements.append(Line2D([0], [0], color=fetch_blues[s_idx], lw=6, label=f"Fetch stage {s_idx}"))
    legend_elements.append(Line2D([0], [0], color=wait_color, lw=6, label="GEMM: waiting on data"))
    legend_elements.append(Line2D([0], [0], color=compute_color, lw=6, label="GEMM: compute"))
    ax.legend(handles=legend_elements, loc="upper right", fontsize=10)

    # Summary stats
    fetch_mask = roles < n_stages
    gemm_mask = roles == n_stages
    fetch_dur = (ends_us - starts_us)[fetch_mask]
    gemm_dur = (ends_us - starts_us)[gemm_mask]
    gemm_wait = waits_us[gemm_mask]
    gemm_compute = gemm_dur - gemm_wait

    stats_lines = []
    for s_idx in range(n_stages):
        s_mask = roles == s_idx
        s_dur = (ends_us - starts_us)[s_mask]
        s_start = starts_us[s_mask]
        if len(s_dur) > 0:
            stats_lines.append(
                f"Fetch stg{s_idx}: {s_dur.mean():.1f} us avg  "
                f"({s_dur.min():.1f}-{s_dur.max():.1f})  "
                f"first@{s_start.min():.0f}us"
            )
    stats_lines += [
        f"GEMM total: {gemm_dur.mean():.1f} us avg  ({gemm_dur.min():.1f}-{gemm_dur.max():.1f})",
        f"  wait: {gemm_wait.mean():.1f} us avg  ({gemm_wait.min():.1f}-{gemm_wait.max():.1f})",
        f"  compute: {gemm_compute.mean():.1f} us avg  ({gemm_compute.min():.1f}-{gemm_compute.max():.1f})",
        f"  wait%: {100 * gemm_wait.sum() / gemm_dur.sum():.1f}%",
        f"Wall time: {ends_us.max():.1f} us",
    ]
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
        description="Benchmark HBM-buffered all_gather_matmul (per-tile flags).",
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
    parser.add_argument("--comm_sms", type=int, default=None, help="Number of SMs (auto if None)")
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
    parser.add_argument("--num_fetch_sms", type=int, default=None, help="Fetcher SMs (auto if None)")
    parser.add_argument(
        "--k_per_flag", type=int, default=None, help="K-blocks per ready flag (model-derived if omitted)"
    )
    parser.add_argument("--num_warps", type=int, default=None, help="Triton num_warps (auto if None)")
    parser.add_argument("--num_stages", type=int, default=None, help="Triton num_stages (auto if None)")
    parser.add_argument(
        "--num_fetch_stages",
        type=int,
        default=None,
        help="Number of fetch stages (model-derived if omitted)",
    )
    parser.add_argument(
        "--first_stage_fetch_sms",
        type=int,
        default=None,
        help="Fetcher WGs for stage 0 (fills first GPU wave; defaults to num_fetch_sms)",
    )
    parser.add_argument(
        "--trace",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Collect per-workgroup trace and save Gantt chart PNG",
    )
    parser.add_argument("--trace_output", type=str, default="trace.png", help="Output path for trace plot")
    parser.add_argument("--output_file", type=str, default="all_gather_matmul_hbm_buffer.json", help="Output JSON file")
    return vars(parser.parse_args())


def _apply_model_defaults(args, world_size, dtype_bytes=2):
    """Fill None-valued kernel parameters with model-derived predictions.

    Returns a list of parameter names that were set by the model.
    """
    applied = []
    if _DERIVE_AVAILABLE:
        try:
            p = _derive_params(
                args["m"],
                args["n"],
                args["k"],
                world_size,
                link_bw=50.0,
                num_cus=DEFAULT_NUM_CUS,
                peak_tflops=DEFAULT_PEAK_TFLOPS_FP16,
                hbm_bw_gbps=DEFAULT_HBM_BW_GBPS,
                l2_size=DEFAULT_L2_SIZE_BYTES,
                scheduling_factor=DEFAULT_SCHEDULING_FACTOR,
                dtype_bytes=dtype_bytes,
            )
            for name in _MODEL_PARAMS:
                if args.get(name) is None and name in p:
                    args[name] = p[name]
                    applied.append(name)
        except Exception:
            pass

    for name, fallback in _FALLBACK_DEFAULTS.items():
        if args.get(name) is None:
            args[name] = fallback

    return applied


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

    model_applied = _apply_model_defaults(args, world_size, dtype_bytes)
    if rank == 0 and model_applied:
        shmem.info(f"Model-derived defaults: {', '.join(model_applied)}")
    if rank == 0:
        param_summary = " ".join(f"{k}={args[k]}" for k in _MODEL_PARAMS)
        shmem.info(f"Kernel params: {param_summary}")

    M = args["m"]
    N = args["n"]
    K = args["k"]
    K_local = K // world_size

    config_kwargs = {
        "block_size_m": args["block_size_m"],
        "block_size_n": args["block_size_n"],
        "block_size_k": args["block_size_k"],
        "group_size_m": args["group_size_m"],
    }
    if args["comm_sms"] is not None:
        config_kwargs["num_sms"] = args["comm_sms"]
    if args["num_xcds"] is not None:
        config_kwargs["num_xcds"] = args["num_xcds"]
    config = FusedConfig(**config_kwargs)

    json_writer = JSONWriter(args["output_file"])
    json_writer.add_field("world_size", world_size)
    json_writer.add_field("operation", "all_gather_matmul_hbm_buffer")
    json_writer.add_field("variant", "hbm_buffer")
    json_writer.add_field("k_local", K_local)

    for key, value in args.items():
        json_writer.add_field(key, value)

    # Export actual config values
    json_writer.add_field("block_size_m", config.block_size_m)
    json_writer.add_field("block_size_n", config.block_size_n)
    json_writer.add_field("block_size_k", config.block_size_k)
    json_writer.add_field("num_sms", config.num_sms)
    json_writer.add_field("num_xcds", config.num_xcds)

    buffer_mb = M * K * torch.tensor([], dtype=datatype).element_size() / (1024**2)
    num_m_tiles = M // config.block_size_m
    num_k_blocks = K // config.block_size_k
    shmem.info(
        f"HBM-Buffer variant: M={M} N={N} K={K} K_local={K_local} "
        f"block=({config.block_size_m},{config.block_size_n},{config.block_size_k}) "
        f"buffer={buffer_mb:.0f}MB flags={num_m_tiles}x{num_k_blocks}"
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
        expected_tensor = torch.matmul(A_gathered, B_data)

    # Pre-allocate workspace
    k_per_flag = args["k_per_flag"]
    workspace = all_gather_matmul_hbm_buffer_preamble(shmem, A_sharded, B, config, k_per_flag=k_per_flag)

    # ── Timing ───────────────────────────────────────────────────────────
    comm_stream = torch.cuda.Stream()
    start_ev = torch.cuda.Event(enable_timing=True)
    end_ev = torch.cuda.Event(enable_timing=True)
    total_ms = 0.0
    num_experiments = 0
    flag_iteration = 0

    num_fetch_sms = args["num_fetch_sms"]
    num_warps = args["num_warps"]
    num_stages = args["num_stages"]
    num_fetch_stages = args["num_fetch_stages"]
    first_stage_fetch_sms = args["first_stage_fetch_sms"]

    def run_experiment():
        nonlocal total_ms, num_experiments, flag_iteration
        shmem.barrier()
        with torch.cuda.stream(comm_stream):
            start_ev.record()
            all_gather_matmul_hbm_buffer(
                shmem,
                C,
                A_sharded,
                B,
                config=config,
                async_op=False,
                workspace=workspace,
                flag_iteration=flag_iteration,
                num_fetch_sms=num_fetch_sms,
                k_per_flag=k_per_flag,
                num_warps=num_warps,
                num_stages=num_stages,
                num_fetch_stages=num_fetch_stages,
                first_stage_fetch_sms=first_stage_fetch_sms,
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
            f"HBM-Buffer (M={M}, K_local={K_local}, K={K}, N={N}, "
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
        # Run a single iteration and record wall-clock finish time per rank
        # to see if ranks complete at different times (load imbalance).
        shmem.barrier()
        torch.cuda.synchronize()
        dist.barrier()

        # Synchronized start
        dist.barrier()
        t_start = time.perf_counter()

        all_gather_matmul_hbm_buffer(
            shmem,
            C,
            A_sharded,
            B,
            config=config,
            async_op=False,
            workspace=workspace,
            num_fetch_sms=num_fetch_sms,
            k_per_flag=k_per_flag,
            num_warps=num_warps,
            num_stages=num_stages,
            num_fetch_stages=num_fetch_stages,
            first_stage_fetch_sms=first_stage_fetch_sms,
        )
        torch.cuda.synchronize()
        t_end = time.perf_counter()

        finish_ms = (t_end - t_start) * 1000.0

        # Gather all finish times to rank 0 for display
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
        # Warmup: compile the TRACE=True kernel variant before the real run
        shmem.info("Trace warmup (compiling traced kernel variant)...")
        C.zero_()
        workspace.locks.zero_()
        shmem.barrier()
        all_gather_matmul_hbm_buffer(
            shmem,
            C,
            A_sharded,
            B,
            config=config,
            async_op=False,
            workspace=workspace,
            num_fetch_sms=num_fetch_sms,
            k_per_flag=k_per_flag,
            num_warps=num_warps,
            num_stages=num_stages,
            num_fetch_stages=num_fetch_stages,
            first_stage_fetch_sms=first_stage_fetch_sms,
            trace=True,
        )
        torch.cuda.synchronize()
        shmem.barrier()

        # Actual traced run (post-compilation, clean state)
        shmem.info("Running single traced iteration...")
        C.zero_()
        workspace.locks.zero_()
        shmem.barrier()

        all_gather_matmul_hbm_buffer(
            shmem,
            C,
            A_sharded,
            B,
            config=config,
            async_op=False,
            workspace=workspace,
            num_fetch_sms=num_fetch_sms,
            k_per_flag=k_per_flag,
            num_warps=num_warps,
            num_stages=num_stages,
            num_fetch_stages=num_fetch_stages,
            first_stage_fetch_sms=first_stage_fetch_sms,
            trace=True,
        )
        torch.cuda.synchronize()
        shmem.barrier()

        if rank == 0 and hasattr(workspace, "trace_data"):
            trace_out = args.get("trace_output", "trace_gantt.png")
            try:
                _plot_trace(workspace.trace_data, trace_out, rank, M, N, K, num_fetch_sms)
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
            shmem.info(f"Speedup (HBM-Buffer / PyTorch): {speedup:.2f}x")

        shmem.barrier()

    if rank == 0:
        json_writer.flush()
        json_writer.display()

    shmem.barrier()
    dist.destroy_process_group()


def main():
    print("Starting HBM-buffer all_gather_matmul benchmark...")
    args = parse_args()
    if "RANK" in os.environ or "LOCAL_RANK" in os.environ:
        _worker(args)
    else:
        print(
            "Please run with torchrun:\n"
            "  torchrun --nproc_per_node=N "
            "benchmark/ops/all_gather_matmul/benchmark_hbm_buffer.py [OPTIONS]"
        )


if __name__ == "__main__":
    main()
