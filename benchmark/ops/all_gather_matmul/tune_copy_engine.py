#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
# Copyright (c) 2025-2026 Advanced Micro Devices, Inc. All rights reserved.

"""
Tune ``m_tiles_per_batch`` for ``all_gather_matmul_copy_engine``.

Unlike the more general matmul tuners, this script keeps the GEMM tile geometry
under the tritonBLAS selector and only sweeps the batch size used by the
host/device copy-engine path.
"""

import argparse
import importlib.util
import json
import math
import os
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

import torch

from tritonblas.matmul import _make_matmul_selector


BENCHMARK_TARGETS = {
    "device": {
        "script": "benchmark/ops/all_gather_matmul/benchmark_copy_engine.py",
        "operation_name": "all_gather_matmul_copy_engine",
        "output_stem": "tune_copy_engine",
        "flags": ["--force-device-initiated"],
    },
    "host": {
        "script": "benchmark/ops/all_gather_matmul/benchmark_copy_engine.py",
        "operation_name": "all_gather_matmul_host_copy_engine",
        "output_stem": "tune_host_copy_engine",
        "flags": ["--force-host-initiated", "--no-trace"],
    },
}


def _load_sweep_dimension_configs():
    """Load the shared sweep dimension list from benchmark/ops/sweep_benchmarks.py."""
    sweep_path = Path(__file__).resolve().parents[1] / "sweep_benchmarks.py"
    module_name = "_shared_sweep_benchmarks"
    spec = importlib.util.spec_from_file_location(module_name, sweep_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Unable to load sweep benchmark config from {sweep_path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    dimension_configs = []
    for config in module.DIMENSION_CONFIGS:
        if isinstance(config, dict):
            dimension_configs.append((config["m_local"], config["n"], config["k"]))
        else:
            dimension_configs.append(tuple(config))
    return dimension_configs


def parse_args():
    parser = argparse.ArgumentParser(
        description="Tune m_tiles_per_batch for all_gather_matmul_copy_engine.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("-m", type=int, default=16384, help="M_local dimension")
    parser.add_argument("-n", type=int, default=2048, help="N dimension")
    parser.add_argument("-k", type=int, default=131072, help="K dimension")
    parser.add_argument(
        "--benchmark",
        type=str,
        default="device",
        choices=sorted(BENCHMARK_TARGETS.keys()),
        help="Which copy-engine benchmark to tune",
    )
    parser.add_argument(
        "--use_sweep_dimensions",
        action="store_true",
        default=True,
        help="Use the shared dimension list from benchmark/ops/sweep_benchmarks.py",
    )
    parser.add_argument(
        "--single_shape",
        dest="use_sweep_dimensions",
        action="store_false",
        help="Only tune the single shape given by -m/-n/-k",
    )
    parser.add_argument("--nproc", type=int, default=8, help="Number of ranks / GPUs")
    parser.add_argument(
        "--datatype",
        type=str,
        default="fp16",
        choices=["fp16", "fp32", "bf16"],
        help="Datatype passed through to the benchmark",
    )
    parser.add_argument("--heap_size", type=int, default=1 << 34, help="Iris heap size passed to the benchmark")
    parser.add_argument("--num_sms", type=int, default=None, help="Optional NUM_SMS override for the benchmark")
    parser.add_argument("--num_xcds", type=int, default=None, help="Optional NUM_XCDS override for the benchmark")
    parser.add_argument(
        "--m_tiles_per_batch",
        type=int,
        nargs="+",
        default=None,
        help="Explicit sweep values. If omitted, derive a candidate set from selector geometry.",
    )
    parser.add_argument(
        "--all_values",
        action="store_true",
        help="Sweep every value from 1..num_m_tiles instead of the heuristic candidate set",
    )
    parser.add_argument("--output_dir", type=str, default=None, help="Output directory (auto-generated if unset)")
    parser.add_argument("--dry_run", action="store_true", help="Print the candidate list and exit")
    parser.add_argument("--skip_validation", action="store_true", help="Skip validation for faster sweeps")
    parser.add_argument("--timeout", type=int, default=600, help="Per-run timeout in seconds")
    return parser.parse_args()


def _dtype_from_name(name: str) -> torch.dtype:
    return {
        "fp16": torch.float16,
        "fp32": torch.float32,
        "bf16": torch.bfloat16,
    }[name]


def _selector_metadata(m_local: int, n: int, k: int, dtype: torch.dtype):
    device = torch.device("cuda:0")
    selector = _make_matmul_selector(
        m_local,
        n,
        k,
        dtype,
        dtype,
        dtype,
        device,
        streamk=False,
    )

    block_m = selector.block_m
    block_n = selector.block_n
    block_k = selector.block_k
    group_size_m = selector.group_m
    num_stages = getattr(selector, "num_stages", 2)
    waves_per_eu = getattr(selector, "waves_per_eu", 0)
    active_cus = getattr(selector, "_ACTIVE_CU", None)
    if active_cus is None:
        active_cus = getattr(selector._hardware, "N_CU", getattr(selector._hardware, "NUM_XCD", 1))

    num_tiles_m = math.ceil(m_local / block_m)
    num_tiles_n = math.ceil(n / block_n)
    tiles_per_group = max(1, group_size_m * num_tiles_n)
    groups_per_wave = max(1, active_cus // tiles_per_group)
    m_tiles_per_wave = min(num_tiles_m, groups_per_wave * group_size_m)

    return {
        "selector": selector,
        "block_size_m": block_m,
        "block_size_n": block_n,
        "block_size_k": block_k,
        "group_size_m": group_size_m,
        "num_stages": num_stages,
        "waves_per_eu": waves_per_eu,
        "active_cus": active_cus,
        "num_tiles_m": num_tiles_m,
        "num_tiles_n": num_tiles_n,
        "tiles_per_group": tiles_per_group,
        "groups_per_wave": groups_per_wave,
        "m_tiles_per_wave": m_tiles_per_wave,
    }


def _candidate_values(
    num_tiles_m: int, group_size_m: int, groups_per_wave: int, m_tiles_per_wave: int, sweep_all: bool
):
    if sweep_all:
        return list(range(1, num_tiles_m + 1))

    values = {1, num_tiles_m}

    power = 1
    while power <= num_tiles_m:
        values.add(power)
        power *= 2

    # Keep a small number of shape-aware anchors even in sparse mode.
    for candidate in (group_size_m, groups_per_wave, m_tiles_per_wave):
        if 1 <= candidate <= num_tiles_m:
            values.add(candidate)

    return sorted(values)


def _build_command(args, output_path: str, m_tiles_per_batch: int):
    target = BENCHMARK_TARGETS[args.benchmark]
    cmd = [
        "torchrun",
        "--nproc_per_node",
        str(args.nproc),
        target["script"],
        "-m",
        str(args.m),
        "-n",
        str(args.n),
        "-k",
        str(args.k),
        "--datatype",
        args.datatype,
        "--heap_size",
        str(args.heap_size),
        "--m_tiles_per_batch",
        str(m_tiles_per_batch),
        "--output_file",
        output_path,
        "-b",
    ]
    cmd.extend(target["flags"])

    if not args.skip_validation:
        cmd.append("-v")
    if args.num_sms is not None:
        cmd.extend(["--num_sms", str(args.num_sms)])
    if args.num_xcds is not None:
        cmd.extend(["--num_xcds", str(args.num_xcds)])

    return cmd


def _parse_json_output(json_path: Path):
    result = {
        "iris_ms": None,
        "iris_tflops": None,
        "iris_bw_gbps": None,
        "validation": None,
        "group_size_m": None,
        "block_size_m": None,
        "block_size_n": None,
        "block_size_k": None,
        "m_tiles_per_batch": None,
        "output_tile_size_m": None,
        "output_tile_size_n": None,
        "output_tile_size_k": None,
        "num_stages": None,
        "waves_per_eu": None,
        "active_cus": None,
        "num_tiles_m": None,
        "num_tiles_n": None,
        "tiles_per_group": None,
        "groups_per_wave": None,
        "m_tiles_per_wave": None,
        "m_tiles_first_wave": None,
        "schedule_iterations": None,
        "num_batches": None,
        "last_batch_m_tiles": None,
        "m_tiles_per_batch_over_wave": None,
        "gemm_wg_us": None,
        "scatter_wg_us": None,
        "bottleneck": None,
        "ratio": None,
        "roofline_tflops": None,
        "intensity": None,
    }

    try:
        with open(json_path, "r") as f:
            data = json.load(f)

        result["iris_ms"] = data.get("avg_ms")
        result["iris_tflops"] = data.get("tflops")
        result["iris_bw_gbps"] = data.get("bandwidth_gbps")
        result["validation"] = "PASSED" if data.get("success") is True else ("FAILED" if "success" in data else None)
        result["group_size_m"] = data.get("group_size_m")
        result["block_size_m"] = data.get("block_size_m")
        result["block_size_n"] = data.get("block_size_n")
        result["block_size_k"] = data.get("block_size_k")
        result["m_tiles_per_batch"] = data.get("m_tiles_per_batch")
        result["output_tile_size_m"] = data.get("output_tile_size_m", data.get("block_size_m"))
        result["output_tile_size_n"] = data.get("output_tile_size_n", data.get("block_size_n"))
        result["output_tile_size_k"] = data.get("output_tile_size_k", data.get("block_size_k"))
        result["num_stages"] = data.get("num_stages")
        result["waves_per_eu"] = data.get("waves_per_eu")
        result["active_cus"] = data.get("active_cus")
        result["num_tiles_m"] = data.get("num_tiles_m")
        result["num_tiles_n"] = data.get("num_tiles_n")
        result["tiles_per_group"] = data.get("tiles_per_group")
        result["groups_per_wave"] = data.get("groups_per_wave")
        result["m_tiles_per_wave"] = data.get("m_tiles_per_wave")
        result["m_tiles_first_wave"] = data.get("m_tiles_first_wave")
        result["schedule_iterations"] = data.get("schedule_iterations")
        result["num_batches"] = data.get("num_batches")
        result["last_batch_m_tiles"] = data.get("last_batch_m_tiles")
        result["m_tiles_per_batch_over_wave"] = data.get("m_tiles_per_batch_over_wave")
        result["gemm_wg_us"] = data.get("gemm_wg_us")
        result["scatter_wg_us"] = data.get("scatter_wg_us")
        result["bottleneck"] = data.get("bottleneck")
        result["ratio"] = data.get("ratio")
        result["roofline_tflops"] = data.get("roofline_tflops")
        result["intensity"] = data.get("intensity")
    except Exception:
        pass

    return result


def _print_selector_summary(meta, candidates):
    tile_shape = f"{meta['block_size_m']}x{meta['block_size_n']}x{meta['block_size_k']}"
    print("\nSelector-derived geometry")
    print(f"  output tile size           : {tile_shape}")
    print(f"  group_size_m               : {meta['group_size_m']}")
    print(f"  num_stages                 : {meta['num_stages']}")
    print(f"  waves_per_eu               : {meta['waves_per_eu']}")
    print(f"  active CUs                 : {meta['active_cus']}")
    print(f"  tile grid                  : {meta['num_tiles_m']} M-tiles x {meta['num_tiles_n']} N-tiles")
    print(f"  tiles per group            : {meta['tiles_per_group']}")
    print(f"  groups per wave/stage      : {meta['groups_per_wave']}")
    print(f"  M-tiles per wave/stage     : {meta['m_tiles_per_wave']}")
    print(f"  sweep m_tiles_per_batch    : {candidates}")


def _shape_tag(m_local: int, n: int, k: int):
    return f"M{m_local}_N{n}_K{k}"


def main():
    args = parse_args()
    dtype = _dtype_from_name(args.datatype)
    if args.use_sweep_dimensions:
        dimension_configs = _load_sweep_dimension_configs()
    else:
        dimension_configs = [(args.m, args.n, args.k)]

    if args.output_dir:
        output_dir = Path(args.output_dir)
    else:
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_stem = BENCHMARK_TARGETS[args.benchmark]["output_stem"]
        output_dir = Path(f"benchmark/ops/all_gather_matmul/{output_stem}_{ts}")
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n{'=' * 100}")
    print(f"  {BENCHMARK_TARGETS[args.benchmark]['operation_name']}  —  m_tiles_per_batch tuning")
    if args.use_sweep_dimensions:
        print(f"  Shapes: sweep_benchmarks.py DIMENSION_CONFIGS ({len(dimension_configs)} shapes)")
    else:
        print(f"  M_local={args.m}  N={args.n}  K={args.k}  nproc={args.nproc}  dtype={args.datatype}")
    print(f"  benchmark={args.benchmark}  nproc={args.nproc}  dtype={args.datatype}")
    print(f"  Output dir: {output_dir}")
    print(f"  Validation: {'OFF' if args.skip_validation else 'ON'}")
    print(f"{'=' * 100}")

    if args.dry_run:
        print("")
        for m_local, n, k in dimension_configs:
            meta = _selector_metadata(m_local, n, k, dtype)
            if args.m_tiles_per_batch is not None:
                candidates = sorted({value for value in args.m_tiles_per_batch if 1 <= value <= meta["num_tiles_m"]})
            else:
                candidates = _candidate_values(
                    meta["num_tiles_m"],
                    meta["group_size_m"],
                    meta["groups_per_wave"],
                    meta["m_tiles_per_wave"],
                    args.all_values,
                )
            print(f"Shape {_shape_tag(m_local, n, k)}")
            _print_selector_summary(meta, candidates)
            print("")
        print("Dry run only; no benchmarks launched.")
        return

    env = os.environ.copy()
    env["HSA_NO_SCRATCH_RECLAIM"] = "1"

    results = []
    total_start = time.time()

    for shape_idx, (m_local, n, k) in enumerate(dimension_configs, start=1):
        meta = _selector_metadata(m_local, n, k, dtype)
        if args.m_tiles_per_batch is not None:
            candidates = sorted({value for value in args.m_tiles_per_batch if 1 <= value <= meta["num_tiles_m"]})
        else:
            candidates = _candidate_values(
                meta["num_tiles_m"],
                meta["group_size_m"],
                meta["groups_per_wave"],
                meta["m_tiles_per_wave"],
                args.all_values,
            )

        if not candidates:
            raise ValueError(f"No valid m_tiles_per_batch values to test for shape {_shape_tag(m_local, n, k)}")

        shape_tag = _shape_tag(m_local, n, k)
        shape_output_dir = output_dir / shape_tag
        shape_output_dir.mkdir(parents=True, exist_ok=True)

        print(f"\n{'=' * 100}")
        print(f"[{shape_idx}/{len(dimension_configs)}] Shape {shape_tag}")
        _print_selector_summary(meta, candidates)

        for idx, m_tiles_per_batch in enumerate(candidates, start=1):
            label = f"{shape_tag}_mtpb{m_tiles_per_batch}"
            json_path = shape_output_dir / f"results_mtpb{m_tiles_per_batch}.json"
            log_path = shape_output_dir / f"log_mtpb{m_tiles_per_batch}.txt"
            cmd_args = argparse.Namespace(**vars(args))
            cmd_args.m = m_local
            cmd_args.n = n
            cmd_args.k = k
            cmd = _build_command(cmd_args, str(json_path), m_tiles_per_batch)
            cmd_str = " ".join(cmd)

            print(f"\n{'-' * 80}")
            print(f"[{idx}/{len(candidates)}] m_tiles_per_batch={m_tiles_per_batch}")
            print(f"  $ HSA_NO_SCRATCH_RECLAIM=1 {cmd_str}")

            started = time.time()
            try:
                proc = subprocess.run(
                    cmd,
                    env=env,
                    capture_output=True,
                    text=True,
                    timeout=args.timeout,
                )
                elapsed = time.time() - started
                parsed = _parse_json_output(json_path)
                json_ok = json_path.exists()

                results.append(
                    {
                        "shape": {"m_local": m_local, "n": n, "k": k},
                        "shape_tag": shape_tag,
                        "label": label,
                        "m_tiles_per_batch": m_tiles_per_batch,
                        "iris_ms": parsed["iris_ms"],
                        "iris_tflops": parsed["iris_tflops"],
                        "iris_bw_gbps": parsed["iris_bw_gbps"],
                        "validation": parsed["validation"],
                        "benchmark_json": parsed,
                        "returncode": proc.returncode,
                        "elapsed_s": round(elapsed, 1),
                        "json_path": str(json_path) if json_ok else None,
                    }
                )

                summary = []
                if parsed["iris_tflops"] is not None:
                    summary.append(f"{parsed['iris_tflops']:.2f} TFLOPS")
                if parsed["iris_ms"] is not None:
                    summary.append(f"{parsed['iris_ms']:.3f} ms")
                if parsed["iris_bw_gbps"] is not None:
                    summary.append(f"{parsed['iris_bw_gbps']:.1f} GB/s")
                if parsed["validation"] is not None:
                    summary.append(f"valid={parsed['validation']}")
                summary.append("json=OK" if json_ok else "json=MISSING")
                if proc.returncode != 0:
                    summary.append(f"EXIT={proc.returncode}")
                print(f"  => {' | '.join(summary)}  ({elapsed:.0f}s)")

                with open(log_path, "w") as f:
                    f.write(f"COMMAND: HSA_NO_SCRATCH_RECLAIM=1 {cmd_str}\n")
                    f.write(f"EXIT CODE: {proc.returncode}\n")
                    f.write(f"ELAPSED: {elapsed:.1f}s\n\n")
                    f.write("=== STDOUT ===\n")
                    f.write(proc.stdout)
                    f.write("\n=== STDERR ===\n")
                    f.write(proc.stderr)

            except subprocess.TimeoutExpired as exc:
                elapsed = time.time() - started
                results.append(
                    {
                        "shape": {"m_local": m_local, "n": n, "k": k},
                        "shape_tag": shape_tag,
                        "label": label,
                        "m_tiles_per_batch": m_tiles_per_batch,
                        "iris_ms": None,
                        "iris_tflops": None,
                        "iris_bw_gbps": None,
                        "validation": "TIMEOUT",
                        "benchmark_json": {},
                        "returncode": -1,
                        "elapsed_s": round(elapsed, 1),
                        "json_path": None,
                    }
                )
                print(f"  => TIMEOUT after {args.timeout}s")
                with open(log_path, "w") as f:
                    f.write(f"COMMAND: HSA_NO_SCRATCH_RECLAIM=1 {cmd_str}\n")
                    f.write(f"TIMEOUT: {args.timeout}s\n\n")
                    f.write(getattr(exc, "stdout", "") or "")
                    f.write("\n")
                    f.write(getattr(exc, "stderr", "") or "")

    total_elapsed = time.time() - total_start

    print(f"\n{'=' * 112}")
    print(f"  TUNING RESULTS  |  {len(dimension_configs)} shapes  |  {len(results)} runs in {total_elapsed:.0f}s")
    print(f"{'=' * 112}")
    print(
        f"  {'#':>3}  {'Shape':<24}  {'m_tiles_per_batch':>17}  {'ms':>8}  {'TFLOPS':>8}  "
        f"{'GB/s':>8}  {'Valid':>8}  {'JSON':>4}"
    )
    print(f"  {'-' * 108}")

    for idx, result in enumerate(results, start=1):
        ms_s = f"{result['iris_ms']:.3f}" if result["iris_ms"] is not None else "--"
        tf_s = f"{result['iris_tflops']:.2f}" if result["iris_tflops"] is not None else "--"
        bw_s = f"{result['iris_bw_gbps']:.1f}" if result["iris_bw_gbps"] is not None else "--"
        valid_s = (result["validation"] or "--")[:8]
        json_s = "Y" if result["json_path"] else "N"
        best_tag = ""
        if result["iris_tflops"] is not None:
            best_value = max((x["iris_tflops"] for x in results if x["iris_tflops"] is not None), default=None)
            if best_value is not None and result["iris_tflops"] == best_value:
                best_tag = " *"

        print(
            f"  {idx:>3}  {result['shape_tag']:<24}  {result['m_tiles_per_batch']:>17}  {ms_s:>8}  {tf_s:>8}  "
            f"{bw_s:>8}  {valid_s:>8}  {json_s:>4}{best_tag}"
        )

    valid_results = [result for result in results if result["iris_tflops"] is not None]
    if valid_results:
        best = max(valid_results, key=lambda result: result["iris_tflops"])
        worst = min(valid_results, key=lambda result: result["iris_tflops"])
        best_json = best["benchmark_json"]
        tile_m = best_json.get("output_tile_size_m") or meta["block_size_m"]
        tile_n = best_json.get("output_tile_size_n") or meta["block_size_n"]
        tile_k = best_json.get("output_tile_size_k") or meta["block_size_k"]
        best_group_size_m = best_json.get("group_size_m") or meta["group_size_m"]
        best_m_tiles_per_wave = best_json.get("m_tiles_per_wave") or meta["m_tiles_per_wave"]
        tile_shape = f"{tile_m}x{tile_n}x{tile_k}"

        print("\nBest configuration")
        print(f"  m_tiles_per_batch          : {best['m_tiles_per_batch']}")
        print(f"  avg_ms                     : {best['iris_ms']:.3f}")
        print(f"  tflops                     : {best['iris_tflops']:.2f}")
        print(f"  bandwidth_gbps             : {best['iris_bw_gbps']:.1f}")
        print(f"  output tile size           : {tile_shape}")
        print(f"  group_size_m               : {best_group_size_m}")
        print(f"  M-tiles per wave/stage     : {best_m_tiles_per_wave}")
        if best_json.get("groups_per_wave") is not None:
            print(f"  groups per wave/stage      : {best_json['groups_per_wave']}")
        if best_json.get("num_batches") is not None:
            print(f"  num_batches                : {best_json['num_batches']}")
        if best_json.get("last_batch_m_tiles") is not None:
            print(f"  last_batch_m_tiles         : {best_json['last_batch_m_tiles']}")
        if best_json.get("ratio") is not None:
            print(f"  scatter/gemm ratio         : {best_json['ratio']:.2f}x")
        if best_json.get("bottleneck") is not None:
            print(f"  bottleneck                 : {best_json['bottleneck']}")

        print("\nSpread")
        print(
            f"  best                       : {best['iris_tflops']:.2f} TFLOPS @ m_tiles_per_batch={best['m_tiles_per_batch']}"
        )
        print(
            f"  worst                      : {worst['iris_tflops']:.2f} TFLOPS @ m_tiles_per_batch={worst['m_tiles_per_batch']}"
        )
        if worst["iris_tflops"] and best["iris_tflops"]:
            print(f"  best / worst               : {best['iris_tflops'] / worst['iris_tflops']:.2f}x")

    results_path = output_dir / "results.json"
    with open(results_path, "w") as f:
        json.dump(
            {
                "meta": {
                    "dimension_configs": [{"m_local": m_local, "n": n, "k": k} for m_local, n, k in dimension_configs],
                    "use_sweep_dimensions": args.use_sweep_dimensions,
                    "benchmark": args.benchmark,
                    "benchmark_script": BENCHMARK_TARGETS[args.benchmark]["script"],
                    "nproc": args.nproc,
                    "datatype": args.datatype,
                    "timestamp": datetime.now().isoformat(),
                    "total_elapsed_s": round(total_elapsed, 1),
                    "candidate_generation": {
                        "block_size_m": meta["block_size_m"],
                        "block_size_n": meta["block_size_n"],
                        "block_size_k": meta["block_size_k"],
                        "group_size_m": meta["group_size_m"],
                        "num_stages": meta["num_stages"],
                        "waves_per_eu": meta["waves_per_eu"],
                        "active_cus": meta["active_cus"],
                        "num_tiles_m": meta["num_tiles_m"],
                        "num_tiles_n": meta["num_tiles_n"],
                        "tiles_per_group": meta["tiles_per_group"],
                        "groups_per_wave": meta["groups_per_wave"],
                        "m_tiles_per_wave": meta["m_tiles_per_wave"],
                    },
                    "candidates": candidates,
                },
                "results": results,
            },
            f,
            indent=2,
        )

    print(f"\nSummary JSON : {results_path}")
    print(f"Per-run JSON : {output_dir}/results_*.json")
    print(f"Per-run logs : {output_dir}/log_*.txt")
    print()


if __name__ == "__main__":
    main()
