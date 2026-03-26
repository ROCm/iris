#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
# Copyright (c) 2025 Advanced Micro Devices, Inc. All rights reserved.

"""
Benchmark for iris-ccl all-reduce collective operation.

Supports two modes:

1. Single-point mode (original):
     python benchmark.py -m 16384 -n 16384 --variant two_shot -b

2. Sweep mode (from a YAML config or CLI):
     python benchmark.py --config configs/vllm_shapes.yaml
     python benchmark.py --sweep-ms 1,32,64,128 -n 2880 --variants rccl,two_shot

Sweep mode measures multiple (M, variant) combinations in one process and
produces a markdown summary table alongside the JSON output.
"""

from __future__ import annotations

import argparse
import itertools
import json
import random
import statistics
from pathlib import Path

import torch
import torch.distributed as dist
import torch.multiprocessing as mp

from examples.common.utils import JSONWriter

import iris
from iris.ccl import Config

torch.manual_seed(123)
random.seed(123)

# ── Tuning grids ─────────────────────────────────────────────────────────
#
# BLOCK_SIZE_M, BLOCK_SIZE_N, COMM_SMS are tl.constexpr in the Triton
# kernels — each unique combo triggers a JIT compilation.  Keep grids small.
#
# all_reduce_distribution is a runtime param (free to sweep).

TUNE_GRIDS: dict[str, dict] = {
    "two_shot": dict(
        comm_sms=[64, 128, 256],
        block_size_m=[4, 16, 64],
        block_size_n=[32, 64],
        all_reduce_distribution=[0, 1],
    ),
    "one_shot": dict(
        comm_sms=[64, 128, 256],
        block_size_m=[4, 16, 64],
        block_size_n=[32, 64],
    ),
    "ring": dict(
        comm_sms=[64, 128, 256],
        block_size_n=[32, 64],
        all_reduce_num_rings=[1, 4],
    ),
}


# ── Helpers ──────────────────────────────────────────────────────────────


def parse_int_list(value: str) -> list[int]:
    return [int(x.strip()) for x in value.split(",") if x.strip()]


def parse_str_list(value: str) -> list[str]:
    return [x.strip() for x in value.split(",") if x.strip()]


def format_bytes(num_bytes: int) -> str:
    if num_bytes >= 1024**2:
        return f"{num_bytes / (1024**2):.2f} MB"
    if num_bytes >= 1024:
        return f"{num_bytes / 1024:.2f} KB"
    return f"{num_bytes} B"


def dtype_from_name(name: str) -> torch.dtype:
    mapping = {"fp16": torch.float16, "fp32": torch.float32, "bf16": torch.bfloat16}
    if name not in mapping:
        raise ValueError(f"Unsupported dtype: {name}")
    return mapping[name]


def barrier_for_variant(variant: str, shmem) -> None:
    if variant == "rccl":
        dist.barrier()
    else:
        shmem.barrier()


def make_config(args: dict, variant: str) -> Config:
    config_kwargs = {
        "comm_sms": args["comm_sms"],
        "all_reduce_variant": variant,
        "block_size_m": args["block_size_m"],
        "block_size_n": args["block_size_n"],
        "swizzle_size": args["swizzle_size"],
    }
    if args["num_xcds"] is not None:
        config_kwargs["num_xcds"] = args["num_xcds"]
    if variant == "ring":
        config_kwargs["all_reduce_num_rings"] = args["num_rings"]
        if args["ring_slice_n"] is not None:
            config_kwargs["all_reduce_ring_slice_n"] = args["ring_slice_n"]
    if variant == "two_shot":
        config_kwargs["all_reduce_distribution"] = args["distribution"]
    return Config(**config_kwargs)


def _config_from_grid_params(variant: str, params: dict, args: dict) -> Config | None:
    config_kwargs: dict = {
        "all_reduce_variant": variant,
        "comm_sms": params.get("comm_sms", args["comm_sms"]),
        "block_size_m": params.get("block_size_m", args["block_size_m"]),
        "block_size_n": params.get("block_size_n", args["block_size_n"]),
        "swizzle_size": params.get("swizzle_size", args["swizzle_size"]),
    }
    if args["num_xcds"] is not None:
        config_kwargs["num_xcds"] = args["num_xcds"]
    if variant == "two_shot":
        config_kwargs["all_reduce_distribution"] = params.get("all_reduce_distribution", args["distribution"])
    if variant == "ring":
        config_kwargs["all_reduce_num_rings"] = params.get("all_reduce_num_rings", args["num_rings"])
    try:
        return Config(**config_kwargs)
    except (ValueError, Exception):
        return None


def _build_grid_configs(variant: str, args: dict) -> list[tuple[dict, Config]]:
    grid = TUNE_GRIDS.get(variant)
    if grid is None:
        return []
    keys = list(grid.keys())
    combos = list(itertools.product(*[grid[k] for k in keys]))
    result = []
    for combo in combos:
        params = dict(zip(keys, combo))
        cfg = _config_from_grid_params(variant, params, args)
        if cfg is not None:
            result.append((params, cfg))
    return result


# ── Measurement ──────────────────────────────────────────────────────────


def measure_case(
    *,
    phase: str,
    M: int,
    N: int,
    dtype: torch.dtype,
    variant: str,
    rank: int,
    world_size: int,
    shmem,
    args: dict,
    config: Config | None = None,
    warmup: int | None = None,
    iters: int | None = None,
) -> dict | None:
    warmup = warmup if warmup is not None else args["warmup"]
    iters = iters if iters is not None else args["iters"]

    device = torch.device(f"cuda:{rank}")
    element_size = torch.tensor([], dtype=dtype).element_size()
    payload_bytes = M * N * element_size
    effective_bytes = payload_bytes * (2 * (world_size - 1) / world_size)
    effective_gb = effective_bytes / (1024**3)

    stream = torch.cuda.Stream(device=device)
    latencies_ms: list[float] = []

    if variant == "rccl":
        input_tensor = torch.full((M, N), float(rank + 1), dtype=dtype, device=device)

        def prepare():
            return None

        def run():
            dist.all_reduce(input_tensor, op=dist.ReduceOp.SUM)
    else:
        if config is None:
            config = make_config(args, variant)
        input_tensor = shmem.zeros((M, N), dtype=dtype)
        output_tensor = shmem.zeros((M, N), dtype=dtype)
        input_tensor.fill_(float(rank + 1))
        workspace = None

        def prepare():
            nonlocal workspace
            workspace = shmem.ccl.all_reduce_preamble(
                output_tensor,
                input_tensor,
                config=config,
                workspace=workspace,
            )

        def run():
            shmem.ccl.all_reduce(
                output_tensor,
                input_tensor,
                config=config,
                async_op=False,
                workspace=workspace,
            )

    for _ in range(warmup):
        prepare()
        barrier_for_variant(variant, shmem)
        with torch.cuda.stream(stream):
            run()
        stream.synchronize()
        barrier_for_variant(variant, shmem)

    for _ in range(iters):
        prepare()
        barrier_for_variant(variant, shmem)
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        with torch.cuda.stream(stream):
            start.record(stream)
            run()
            end.record(stream)
        end.synchronize()
        barrier_for_variant(variant, shmem)
        latencies_ms.append(start.elapsed_time(end))

    rank_median_ms = statistics.median(latencies_ms)
    rank_tensor = torch.tensor([rank_median_ms], dtype=torch.float64, device=device)
    gather = [torch.zeros_like(rank_tensor) for _ in range(world_size)]
    dist.all_gather(gather, rank_tensor)
    rank_medians_ms = [float(t.item()) for t in gather]

    if rank != 0:
        return None

    min_ms = min(rank_medians_ms)
    max_ms = max(rank_medians_ms)
    mean_ms = statistics.mean(rank_medians_ms)
    median_of_ranks_ms = statistics.median(rank_medians_ms)
    imbalance_pct = 0.0 if mean_ms == 0 else ((max_ms - mean_ms) / mean_ms) * 100.0
    achieved_gbps = 0.0 if median_of_ranks_ms == 0 else effective_gb / (median_of_ranks_ms * 1e-3)

    return {
        "phase": phase,
        "M": M,
        "N": N,
        "payload_bytes": payload_bytes,
        "effective_bytes": effective_bytes,
        "variant": variant,
        "median_rank_ms": median_of_ranks_ms,
        "min_rank_ms": min_ms,
        "max_rank_ms": max_ms,
        "mean_rank_ms": mean_ms,
        "imbalance_pct": imbalance_pct,
        "achieved_gbps": achieved_gbps,
        "rank_medians_ms": rank_medians_ms,
    }


# ── Validation ───────────────────────────────────────────────────────────


def validate_all_reduce(
    M: int,
    N: int,
    dtype: torch.dtype,
    variant: str,
    rank: int,
    world_size: int,
    shmem,
    args: dict,
):
    config = make_config(args, variant)
    input_tensor = shmem.zeros((M, N), dtype=dtype)
    output_tensor = shmem.zeros((M, N), dtype=dtype)
    input_tensor.fill_(float(rank + 1))
    expected_sum = float(world_size * (world_size + 1) / 2)
    expected_tensor = shmem.zeros((M, N), dtype=dtype)
    expected_tensor.fill_(expected_sum)

    workspace = shmem.ccl.all_reduce_preamble(
        output_tensor,
        input_tensor,
        config=config,
    )
    shmem.barrier()

    torch.cuda.nvtx.range_push("All-Reduce-Validate")
    shmem.ccl.all_reduce(
        output_tensor,
        input_tensor,
        config=config,
        async_op=False,
        workspace=workspace,
    )
    torch.cuda.nvtx.range_pop()
    torch.cuda.synchronize()
    shmem.barrier()

    atol = 1e-3 if dtype == torch.float16 else 1e-5
    success = torch.allclose(output_tensor, expected_tensor, atol=atol)
    if not success:
        max_diff = torch.abs(output_tensor - expected_tensor).max().item()
        shmem.error(f"Rank {rank}: Validation failed, max diff: {max_diff}")
    elif rank == 0:
        shmem.info(f"Validation passed ({variant}, M={M}, N={N})")
    shmem.barrier()
    return success


# ── Tuning ───────────────────────────────────────────────────────────────


def tune_variant_for_m(
    *,
    variant: str,
    phase: str,
    M: int,
    N: int,
    dtype: torch.dtype,
    rank: int,
    world_size: int,
    shmem,
    args: dict,
) -> Config:
    fallback = make_config(args, variant)
    grid_configs = _build_grid_configs(variant, args)
    if not grid_configs:
        return fallback

    device = torch.device(f"cuda:{rank}")
    best_idx = 0
    best_latency_ms = float("inf")

    for idx, (params, cfg) in enumerate(grid_configs):
        try:
            result = measure_case(
                phase=phase,
                M=M,
                N=N,
                dtype=dtype,
                variant=variant,
                rank=rank,
                world_size=world_size,
                shmem=shmem,
                args=args,
                config=cfg,
                warmup=args["tune_warmup"],
                iters=args["tune_iters"],
            )
        except Exception:
            continue

        if rank == 0 and result is not None:
            lat = result["median_rank_ms"]
            if lat < best_latency_ms:
                best_latency_ms = lat
                best_idx = idx

    idx_tensor = torch.tensor([best_idx], dtype=torch.long, device=device)
    dist.broadcast(idx_tensor, src=0)
    best_idx = int(idx_tensor.item())

    best_params, best_cfg = grid_configs[best_idx]
    if rank == 0:
        lat_us = best_latency_ms * 1000
        print(
            f"  tune  M={M:5d}  {variant:10s}  best={lat_us:.1f} us  "
            + "  ".join(f"{k}={v}" for k, v in best_params.items())
        )
    return best_cfg


# ── Markdown rendering ───────────────────────────────────────────────────


def render_markdown(
    results: list[dict],
    world_size: int,
    dtype_name: str,
    best_configs: dict[tuple, Config] | None = None,
) -> str:
    lines: list[str] = []
    lines.append("# Iris All-Reduce Benchmark Results")
    lines.append("")
    lines.append(f"- World size: `{world_size}`")
    lines.append(f"- Dtype: `{dtype_name}`")
    N = results[0]["N"] if results else "?"
    lines.append(f"- Tensor shape convention: `[M, {N}]`")
    lines.append("- Effective bandwidth model: `2 * (p - 1) / p * payload_bytes / latency`")
    lines.append("")

    phases_seen = []
    for r in results:
        if r["phase"] not in phases_seen:
            phases_seen.append(r["phase"])

    for phase in phases_seen:
        phase_rows = [r for r in results if r["phase"] == phase]
        if not phase_rows:
            continue
        lines.append(f"## {phase.replace('_', ' ').title()}")
        lines.append("")
        lines.append(
            "| M | Payload | Variant | Median Rank Latency (us) | "
            "Min Rank (us) | Max Rank (us) | Imbalance % | Achieved GB/s |"
        )
        lines.append("|---:|---:|---|---:|---:|---:|---:|---:|")
        for row in phase_rows:
            lines.append(
                f"| {row['M']} | "
                f"{format_bytes(row['payload_bytes'])} | "
                f"`{row['variant']}` | "
                f"{row['median_rank_ms'] * 1000:.1f} | "
                f"{row['min_rank_ms'] * 1000:.1f} | "
                f"{row['max_rank_ms'] * 1000:.1f} | "
                f"{row['imbalance_pct']:.1f} | "
                f"{row['achieved_gbps']:.2f} |"
            )
        lines.append("")

    if best_configs:
        lines.append("## Tuned Configs")
        lines.append("")
        lines.append("| M | Variant | comm_sms | block_size_m | block_size_n | swizzle_size | extra |")
        lines.append("|---:|---|---:|---:|---:|---:|---|")
        for (M, variant), cfg in sorted(best_configs.items()):
            extra_parts = []
            if variant == "two_shot":
                extra_parts.append(f"distribution={cfg.all_reduce_distribution}")
            if variant == "ring":
                extra_parts.append(f"num_rings={cfg.all_reduce_num_rings}")
            extra = ", ".join(extra_parts) if extra_parts else "—"
            lines.append(
                f"| {M} | `{variant}` | {cfg.comm_sms} | {cfg.block_size_m} | "
                f"{cfg.block_size_n} | {cfg.swizzle_size} | {extra} |"
            )
        lines.append("")

    return "\n".join(lines) + "\n"


# ── Argument parsing ─────────────────────────────────────────────────────


def parse_args() -> dict:
    parser = argparse.ArgumentParser(
        description="Benchmark iris-ccl all-reduce (single-point or sweep mode).",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # YAML config — overrides defaults, CLI flags override YAML
    parser.add_argument(
        "--config",
        type=str,
        default=None,
        help="Path to YAML config file (see configs/ for examples)",
    )

    # Shape
    parser.add_argument("-m", type=int, default=16384, help="Number of rows (single-point mode)")
    parser.add_argument("-n", type=int, default=16384, help="Number of columns")
    parser.add_argument(
        "--sweep-ms",
        type=str,
        default=None,
        help="Comma-separated M values to sweep (enables sweep mode, ignores -m)",
    )

    # Variant
    parser.add_argument(
        "--variant",
        type=str,
        default="two_shot",
        choices=["atomic", "ring", "two_shot", "one_shot", "spinlock"],
        help="All-reduce variant (single-point mode)",
    )
    parser.add_argument(
        "--variants",
        type=str,
        default=None,
        help="Comma-separated variants for sweep mode (e.g. rccl,two_shot,ring,one_shot)",
    )

    # Dtype
    parser.add_argument(
        "--datatype",
        type=str,
        default="fp16",
        choices=["fp16", "fp32", "bf16"],
        help="Datatype of tensors",
    )

    # Kernel config
    parser.add_argument("--heap-size", type=int, default=1 << 34, help="Iris heap size")
    parser.add_argument("--comm-sms", type=int, default=64, help="SMs for comm kernel")
    parser.add_argument("--block-size-m", type=int, default=64, help="Block size M")
    parser.add_argument("--block-size-n", type=int, default=64, help="Block size N")
    parser.add_argument("--swizzle-size", type=int, default=4, help="Swizzle size")
    parser.add_argument("--num-xcds", type=int, default=None, help="Number of XCDs")
    parser.add_argument("--distribution", type=int, default=0, choices=[0, 1], help="Two-shot distribution")
    parser.add_argument("--num-rings", type=int, default=1, help="Ring variant: number of rings")
    parser.add_argument("--ring-slice-n", type=int, default=None, help="Ring slice width")

    # Measurement
    parser.add_argument("--warmup", type=int, default=50, help="Warmup iterations")
    parser.add_argument("--iters", type=int, default=200, help="Measured iterations")
    parser.add_argument("-r", "--num-ranks", type=int, default=8, help="Number of ranks")
    parser.add_argument(
        "--init-url",
        type=str,
        default="tcp://127.0.0.1:29527",
        help="Initialization URL for distributed setup",
    )

    # Modes (single-point)
    parser.add_argument("-d", "--debug", action="store_true", help="Enable debug mode")
    parser.add_argument("-v", "--validate", action="store_true", help="Enable validation")
    parser.add_argument("-b", "--benchmark", action="store_true", help="Enable benchmarking (single-point)")
    parser.add_argument("--benchmark-rccl", action="store_true", help="Also benchmark RCCL (single-point)")

    # Tuning (sweep mode)
    parser.add_argument("--tune", action="store_true", help="Auto-tune config per (M, variant)")
    parser.add_argument("--tune-warmup", type=int, default=3, help="Tune pass: warmup iters")
    parser.add_argument("--tune-iters", type=int, default=10, help="Tune pass: measured iters")

    # Output
    parser.add_argument("--output-file", type=str, default="log.json", help="JSON output path")
    parser.add_argument("--json-output", type=str, default=None, help="JSON output (sweep mode)")
    parser.add_argument("--markdown-output", type=str, default=None, help="Markdown output (sweep mode)")

    raw_args = parser.parse_args()
    args = vars(raw_args)

    # ── Apply YAML config as base, CLI overrides on top ──
    if args["config"] is not None:
        import yaml

        with open(args["config"]) as f:
            yaml_cfg = yaml.safe_load(f)
        if yaml_cfg is None:
            yaml_cfg = {}

        yaml_key_map = {
            "n": "n",
            "datatype": "datatype",
            "num_ranks": "num_ranks",
            "comm_sms": "comm_sms",
            "block_size_m": "block_size_m",
            "block_size_n": "block_size_n",
            "swizzle_size": "swizzle_size",
            "distribution": "distribution",
            "num_rings": "num_rings",
            "ring_slice_n": "ring_slice_n",
            "heap_size": "heap_size",
            "warmup": "warmup",
            "iters": "iters",
            "tune_warmup": "tune_warmup",
            "tune_iters": "tune_iters",
        }

        defaults = parser.parse_args([])
        for yaml_key, args_key in yaml_key_map.items():
            if yaml_key in yaml_cfg and getattr(raw_args, args_key) == getattr(defaults, args_key):
                args[args_key] = yaml_cfg[yaml_key]

        if "variants" in yaml_cfg and args["variants"] is None:
            args["variants"] = ",".join(yaml_cfg["variants"])

        if "sweep_ms" in yaml_cfg and args["sweep_ms"] is None:
            all_ms = []
            for phase_key in ["decode_like", "prefill_like"]:
                if phase_key in yaml_cfg["sweep_ms"]:
                    all_ms.extend(yaml_cfg["sweep_ms"][phase_key])
            if all_ms:
                args["sweep_ms"] = ",".join(str(m) for m in all_ms)
                args["_sweep_phases"] = yaml_cfg["sweep_ms"]

    # Normalize
    args.setdefault("_sweep_phases", None)

    return args


def _is_sweep_mode(args: dict) -> bool:
    return args.get("sweep_ms") is not None or args.get("variants") is not None


# ── Single-point worker (original benchmark.py logic) ────────────────────


def _single_point_worker(local_rank: int, world_size: int, init_url: str, args: dict):
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
    datatype = dtype_from_name(args["datatype"])
    M, N = args["m"], args["n"]
    variant = args["variant"]
    config = make_config(args, variant)

    json_writer = JSONWriter(args["output_file"])
    json_writer.add_field("world_size", world_size)
    for key, value in args.items():
        if not key.startswith("_"):
            json_writer.add_field(key, value)
    json_writer.add_field("block_size_m", config.block_size_m)
    json_writer.add_field("block_size_n", config.block_size_n)
    json_writer.add_field("swizzle_size", config.swizzle_size)
    json_writer.add_field("num_xcds", config.num_xcds)
    json_writer.add_field("all_reduce_variant", config.all_reduce_variant)
    if variant == "ring":
        json_writer.add_field("all_reduce_num_rings", config.all_reduce_num_rings)
        json_writer.add_field("all_reduce_ring_slice_n", config.all_reduce_ring_slice_n)
    if variant == "two_shot":
        json_writer.add_field("all_reduce_distribution", config.all_reduce_distribution)

    input_tensor = shmem.zeros((M, N), dtype=datatype)
    output_tensor = shmem.zeros((M, N), dtype=datatype)
    input_tensor.fill_(float(rank + 1))
    expected_sum = float(world_size * (world_size + 1) / 2)

    comm_stream = torch.cuda.Stream()
    kernel_timing = {
        "all_reduce": {
            "start_event": torch.cuda.Event(enable_timing=True),
            "end_event": torch.cuda.Event(enable_timing=True),
            "ms": 0,
            "experiments": 0,
        },
    }

    workspace = None

    def run_experiment():
        nonlocal kernel_timing, workspace
        workspace = shmem.ccl.all_reduce_preamble(
            output_tensor,
            input_tensor,
            config=config,
            workspace=workspace,
        )
        shmem.barrier()
        torch.cuda.nvtx.range_push("All-Reduce")
        with torch.cuda.stream(comm_stream):
            kernel_timing["all_reduce"]["start_event"].record()
            shmem.ccl.all_reduce(
                output_tensor,
                input_tensor,
                config=config,
                async_op=False,
                workspace=workspace,
            )
            kernel_timing["all_reduce"]["end_event"].record()
            kernel_timing["all_reduce"]["experiments"] += 1
        torch.cuda.nvtx.range_pop()
        shmem.barrier()
        ms = kernel_timing["all_reduce"]["start_event"].elapsed_time(kernel_timing["all_reduce"]["end_event"])
        kernel_timing["all_reduce"]["ms"] += ms

    shmem.barrier()

    if args["validate"]:
        shmem.info("Validating...")
        output_tensor.zero_()
        shmem.barrier()
        input_tensor.fill_(float(rank + 1))
        shmem.barrier()
        run_experiment()
        torch.cuda.synchronize()
        shmem.barrier()
        expected_tensor = shmem.zeros((M, N), dtype=datatype)
        expected_tensor.fill_(expected_sum)
        atol = 1e-3 if datatype == torch.float16 else 1e-5
        success = torch.allclose(output_tensor, expected_tensor, atol=atol)
        if not success:
            max_diff = torch.abs(output_tensor - expected_tensor).max().item()
            shmem.error(f"Rank {rank}: Validation failed, max diff: {max_diff}")
        if success:
            shmem.info("All-reduce validation passed!")
        else:
            shmem.error("All-reduce validation failed!")
        json_writer.add_field("success", success)
        shmem.barrier()

    if args["benchmark"]:
        for k in ["all_reduce"]:
            kernel_timing[k]["ms"] = 0
            kernel_timing[k]["experiments"] = 0

        iris.do_bench(run_experiment, shmem.barrier, n_warmup=25, n_repeat=1)

        for k in ["all_reduce"]:
            kernel_timing[k]["ms"] = 0
            kernel_timing[k]["experiments"] = 0

        output_tensor.zero_()
        shmem.barrier()
        input_tensor.fill_(float(rank + 1))
        shmem.barrier()

        shmem.info("Benchmarking...")

        element_size = torch.tensor([], dtype=datatype).element_size()
        total_bytes = M * N * element_size * (2 * (world_size - 1)) / world_size
        total_bytes_gb = total_bytes / (1024**3)

        triton_ms = iris.do_bench(run_experiment, shmem.barrier)
        bandwidth_gbps = total_bytes_gb / (
            (kernel_timing["all_reduce"]["ms"] / kernel_timing["all_reduce"]["experiments"]) * 1e-3
        )

        shmem.info(
            f"All-reduce (M={M}, N={N}, world_size={world_size}, "
            f"dtype={args['datatype']}, variant={variant}): "
            f"{triton_ms:.3f} ms, {bandwidth_gbps:.3f} GB/s"
        )

        json_writer.add_field("bandwidth_gbps", bandwidth_gbps)
        json_writer.add_field("total_ms", triton_ms)
        json_writer.add_field("total_bytes", total_bytes)
        json_writer.add_field("total_bytes_gb", total_bytes_gb)
        json_writer.add_field(
            "all_reduce_ms",
            kernel_timing["all_reduce"]["ms"] / kernel_timing["all_reduce"]["experiments"],
        )
        json_writer.add_field("all_reduce_experiments", kernel_timing["all_reduce"]["experiments"])
        shmem.barrier()

    if args["benchmark_rccl"]:
        shmem.info("Benchmarking PyTorch RCCL (all_reduce)...")
        pytorch_tensor = torch.zeros(M, N, dtype=datatype, device=f"cuda:{rank}")
        pytorch_tensor.fill_(float(rank + 1))
        for _ in range(10):
            dist.all_reduce(pytorch_tensor, op=dist.ReduceOp.SUM)
        torch.cuda.synchronize()
        dist.barrier()

        pytorch_tensor.fill_(float(rank + 1))
        dist.barrier()

        def run_rccl_experiment():
            dist.all_reduce(pytorch_tensor, op=dist.ReduceOp.SUM)

        rccl_ms = iris.do_bench(run_rccl_experiment, dist.barrier)
        element_size = torch.tensor([], dtype=datatype).element_size()
        total_bytes = M * N * element_size * (2 * (world_size - 1)) / world_size
        total_bytes_gb = total_bytes / (1024**3)
        rccl_bandwidth_gbps = total_bytes_gb / (rccl_ms * 1e-3)

        shmem.info(
            f"RCCL all_reduce (M={M}, N={N}, world_size={world_size}, "
            f"dtype={args['datatype']}): {rccl_ms:.3f} ms, {rccl_bandwidth_gbps:.3f} GB/s"
        )

        if args["benchmark"]:
            iris_bandwidth = bandwidth_gbps
            rccl_ratio = (iris_bandwidth / rccl_bandwidth_gbps) * 100 if rccl_bandwidth_gbps > 0 else 0
            shmem.info(f"Performance ratio (Iris/RCCL): {rccl_ratio:.1f}%")
            json_writer.add_field("rccl_bandwidth_gbps", rccl_bandwidth_gbps)
            json_writer.add_field("rccl_ms", rccl_ms)
            json_writer.add_field("rccl_ratio_percent", rccl_ratio)
        shmem.barrier()

    if rank == 0:
        if variant == "ring":
            json_writer.add_field("all_reduce_ring_slice_n", config.all_reduce_ring_slice_n)
        json_writer.flush()
        json_writer.display()

    shmem.barrier()
    dist.destroy_process_group()


# ── Sweep worker ─────────────────────────────────────────────────────────


def _sweep_worker(local_rank: int, world_size: int, init_url: str, args: dict):
    backend = "nccl" if torch.cuda.is_available() else "gloo"
    dist.init_process_group(
        backend=backend,
        init_method=init_url,
        world_size=world_size,
        rank=local_rank,
        device_id=torch.device(f"cuda:{local_rank}"),
    )
    torch.cuda.set_device(local_rank)

    shmem = iris.iris(args["heap_size"])
    dtype = dtype_from_name(args["datatype"])

    # Build (phase, M) case list
    variants = parse_str_list(args["variants"]) if isinstance(args["variants"], str) else args["variants"]

    sweep_phases = args.get("_sweep_phases")
    if sweep_phases is not None:
        cases: list[tuple[str, int]] = []
        for phase_key in ["decode_like", "prefill_like"]:
            if phase_key in sweep_phases:
                cases.extend([(phase_key, m) for m in sweep_phases[phase_key]])
    else:
        all_ms = parse_int_list(args["sweep_ms"]) if isinstance(args["sweep_ms"], str) else args["sweep_ms"]
        cases = [("sweep", m) for m in all_ms]

    # Tuning pass
    best_configs: dict[tuple, Config] = {}
    if args["tune"]:
        iris_variants = [v for v in variants if v != "rccl"]
        if local_rank == 0:
            n_grid = {v: len(_build_grid_configs(v, args)) for v in iris_variants}
            total = sum(n_grid[v] * len(cases) for v in iris_variants)
            print(
                f"\n=== Tuning pass: {total} trials "
                f"({args['tune_warmup']} warmup + {args['tune_iters']} iters each) ===\n"
                + "  "
                + "  ".join(f"{v}: {n_grid[v]} configs" for v in iris_variants)
            )
        for phase, m in cases:
            for variant in iris_variants:
                cfg = tune_variant_for_m(
                    variant=variant,
                    phase=phase,
                    M=m,
                    N=args["n"],
                    dtype=dtype,
                    rank=local_rank,
                    world_size=world_size,
                    shmem=shmem,
                    args=args,
                )
                best_configs[(m, variant)] = cfg
        if local_rank == 0:
            print("\n=== Tuning complete — running full benchmark ===\n")

    # Validation pass
    if args["validate"]:
        iris_variants = [v for v in variants if v != "rccl"]
        for variant in iris_variants:
            config = best_configs.get((cases[0][1], variant)) if best_configs else None
            M_val = cases[0][1]
            validate_all_reduce(
                M_val,
                args["n"],
                dtype,
                variant,
                local_rank,
                world_size,
                shmem,
                args,
            )

    # Full measurement
    results: list[dict] = []
    for phase, m in cases:
        for variant in variants:
            cfg = best_configs.get((m, variant))
            row = measure_case(
                phase=phase,
                M=m,
                N=args["n"],
                dtype=dtype,
                variant=variant,
                rank=local_rank,
                world_size=world_size,
                shmem=shmem,
                args=args,
                config=cfg,
            )
            if local_rank == 0 and row is not None:
                results.append(row)

    if local_rank == 0:
        for row in results:
            key = (row["M"], row["variant"])
            cfg = best_configs.get(key)
            if cfg is not None:
                row["tuned_config"] = {
                    "comm_sms": cfg.comm_sms,
                    "block_size_m": cfg.block_size_m,
                    "block_size_n": cfg.block_size_n,
                    "swizzle_size": cfg.swizzle_size,
                    "all_reduce_distribution": cfg.all_reduce_distribution,
                    "all_reduce_num_rings": cfg.all_reduce_num_rings,
                }

        json_path = Path(args.get("json_output") or args["output_file"])
        json_path.write_text(json.dumps(results, indent=2) + "\n")

        md_content = render_markdown(
            results,
            world_size,
            args["datatype"],
            best_configs=best_configs if args["tune"] else None,
        )
        md_path = args.get("markdown_output")
        if md_path:
            Path(md_path).write_text(md_content)

        print(md_content)

    dist.barrier()
    dist.destroy_process_group()


# ── Entry point ──────────────────────────────────────────────────────────


def main():
    args = parse_args()

    if _is_sweep_mode(args):
        worker_fn = _sweep_worker
    else:
        worker_fn = _single_point_worker

    mp.spawn(
        fn=worker_fn,
        args=(args["num_ranks"], args["init_url"], args),
        nprocs=args["num_ranks"],
        join=True,
    )


if __name__ == "__main__":
    main()
