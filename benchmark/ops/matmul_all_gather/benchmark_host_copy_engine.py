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

from examples.common.utils import JSONWriter

import iris
from iris.ops.matmul_all_gather_host_copy_engine import (
    matmul_all_gather_host_copy_engine,
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
    "num_warps",
)

_FALLBACK_DEFAULTS = {
    "block_size_m": 256,
    "block_size_n": 128,
    "block_size_k": 64,
    "group_size_m": 4,
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
        description="Benchmark matmul_all_gather_host_copy_engine fused operation.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("-m", type=int, default=16384, help="Number of rows per rank in matrix A (M_local)")
    parser.add_argument("-n", type=int, default=2048, help="Number of columns in matrix B (N)")
    parser.add_argument("-k", type=int, default=131072, help="Common dimension (K)")
    parser.add_argument("-v", "--validate", action="store_true", help="Enable validation mode")
    parser.add_argument("-b", "--benchmark", action="store_true", help="Enable benchmarking mode")
    parser.add_argument(
        "--datatype",
        type=str,
        default="fp16",
        choices=["fp16", "fp32", "bf16"],
        help="Datatype of tensors",
    )
    parser.add_argument(
        "--output_file",
        type=str,
        default="matmul_all_gather_host_copy_engine.json",
        help="Output file",
    )
    parser.add_argument("--heap_size", type=int, default=1 << 34, help="Iris heap size")
    parser.add_argument("--num_sms", type=int, default=None, help="Number of SMs for operation (auto-detect if None)")
    parser.add_argument(
        "--benchmark_baseline",
        action="store_true",
        help="Also benchmark baseline (non-copy-engine) variant for comparison",
    )
    parser.add_argument("--block_size_m", type=int, default=256, help="Block size for M dimension")
    parser.add_argument("--block_size_n", type=int, default=64, help="Block size for N dimension")
    parser.add_argument("--block_size_k", type=int, default=64, help="Block size for K dimension")
    parser.add_argument("--group_size_m", type=int, default=1, help="Group size for M dimension tiling")
    parser.add_argument("--num_xcds", type=int, default=None, help="Number of XCDs (auto-detected if not set)")

    return vars(parser.parse_args())


def _apply_model_defaults(args, world_size, dtype_bytes=2):
    """Fill None-valued kernel parameters with model-derived predictions.

    Returns a list of parameter names that were set by the model.
    """
    applied = []
    if _DERIVE_AVAILABLE:
        try:
            # For matmul_all_gather, M is local dimension (sharded)
            # Total M = M_local * world_size
            M_local = args["m"]
            M_total = M_local * world_size

            p = _derive_params(
                M_total,
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


def _worker(args: dict):
    """Worker function for PyTorch distributed execution."""
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    backend = "nccl" if torch.cuda.is_available() else "gloo"
    dist.init_process_group(backend=backend)

    shmem = iris.iris(args["heap_size"])
    rank = shmem.get_rank()
    world_size = shmem.get_num_ranks()

    # Datatype mapping
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

    M_local = args["m"]  # Local M dimension
    M = M_local * world_size  # Total M after gather
    N = args["n"]
    K = args["k"]

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

    # Export actual config values to JSON (including defaults)
    json_writer.add_field("block_size_m", config.block_size_m)
    json_writer.add_field("block_size_n", config.block_size_n)
    json_writer.add_field("block_size_k", config.block_size_k)
    json_writer.add_field("group_size_m", config.group_size_m)
    json_writer.add_field("num_sms", config.num_sms)
    json_writer.add_field("num_xcds", config.num_xcds)

    # Create input and output tensors
    # A_local is M_local x K, output is M x N (gathered)
    A_local = shmem.zeros((M_local, K), dtype=datatype)
    B = shmem.zeros((K, N), dtype=datatype)
    C = shmem.zeros((M, N), dtype=datatype)
    expected_tensor = None

    # Fill inputs with deterministic values
    # Each rank has different A_local, same B
    torch.manual_seed(123 + rank)
    A_local_data = torch.randn((M_local, K), dtype=datatype, device=f"cuda:{rank}")
    A_local.copy_(A_local_data)

    torch.manual_seed(456)  # Same B for all ranks
    B_data = torch.randn((K, N), dtype=datatype, device=f"cuda:{rank}")
    B.copy_(B_data)

    # For validation: compute expected result
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
    workspace = matmul_all_gather_host_copy_engine_preamble(shmem, A_local, B, config)

    # ── Timing ───────────────────────────────────────────────────────────
    comm_stream = torch.cuda.Stream()
    start_ev = torch.cuda.Event(enable_timing=True)
    end_ev = torch.cuda.Event(enable_timing=True)
    total_ms = 0.0
    num_experiments = 0

    def run_experiment():
        nonlocal total_ms, num_experiments
        shmem.barrier()

        with torch.cuda.stream(comm_stream):
            start_ev.record()
            matmul_all_gather_host_copy_engine(
                shmem,
                C,
                A_local,
                B,
                config=config,
                async_op=False,
                workspace=workspace,
            )
            end_ev.record()
            num_experiments += 1
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

    if rank == 0:
        json_writer.flush()
        json_writer.display()

    shmem.barrier()
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
