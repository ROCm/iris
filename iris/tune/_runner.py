# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.

"""Distributed runner, cache, output formatters, and CLI entry point for iris.tune."""

from __future__ import annotations

import argparse
import hashlib
import itertools
import json
import os
import re
import statistics
import sys
import time
from pathlib import Path
from typing import Any, Callable

import torch
import torch.distributed as dist
from torch.distributed.launcher.api import LaunchConfig, elastic_launch

from ._core import (
    SearchAxis,
    TuneDef,
    TuneResult,
    TuneState,
    _registry,
)

_NUM_RANKS_AXIS = "num_ranks"
_DEFAULT_NUM_RANKS = 8

_CACHE_DIR = Path.home() / ".iris" / "tune_cache"


# Cache key
def _build_cache_key(
    tune_name: str,
    fixed_params: dict[str, Any],
    search_axes: list[SearchAxis],
    world_size: int,
    gpu_arch: str,
) -> str:
    """Build a deterministic cache key from tuning parameters."""
    key_parts = [
        tune_name,
        str(sorted((k, _param_str(v)) for k, v in fixed_params.items())),
        str([(a.name, sorted(str(v) for v in a.values)) for a in search_axes]),
        f"world_size={world_size}",
        f"arch={gpu_arch}",
    ]
    return hashlib.sha256("|".join(key_parts).encode()).hexdigest()[:16]


def _param_str(v: Any) -> str:
    if isinstance(v, torch.dtype):
        return str(v)
    return str(v)


def _load_cache(cache_key: str, tune_name: str) -> list[dict] | None:
    """Load cached tuning results if they exist."""
    cache_file = _CACHE_DIR / f"{tune_name}_{cache_key}.json"
    if cache_file.exists():
        with open(cache_file) as f:
            return json.load(f)
    return None


def _save_cache(cache_key: str, tune_name: str, results: list[dict]) -> Path:
    """Save tuning results to cache."""
    _CACHE_DIR.mkdir(parents=True, exist_ok=True)
    cache_file = _CACHE_DIR / f"{tune_name}_{cache_key}.json"
    with open(cache_file, "w") as f:
        json.dump(results, f, indent=2)
    return cache_file


# Generate and prune config space
def _generate_configs(
    search_axes: list[SearchAxis],
    prune_fns: list[Callable[[dict[str, Any]], bool]],
) -> list[dict[str, Any]]:
    """Generate the Cartesian product of search axes, then prune."""
    if not search_axes:
        return [{}]

    axis_names = [a.name for a in search_axes]
    axis_values = [a.values for a in search_axes]
    all_configs = []

    for combo in itertools.product(*axis_values):
        config = dict(zip(axis_names, combo))
        # Apply all prune functions — keep if all return True
        if all(fn(config) for fn in prune_fns):
            all_configs.append(config)

    return all_configs


# Output formatters
def _dtype_str(v: Any) -> str:
    if isinstance(v, torch.dtype):
        return {
            torch.float16: "fp16",
            torch.float32: "fp32",
            torch.bfloat16: "bf16",
        }.get(v, str(v))
    return str(v)


def _format_console(results: list[TuneResult], top_k: int) -> str:
    """Render tuning results as a ranked console table."""
    if not results:
        return "No results.\n"

    # Group by tune name
    by_tune: dict[str, list[TuneResult]] = {}
    for r in results:
        by_tune.setdefault(r.tune_name, []).append(r)

    lines: list[str] = []
    for tune_name, tune_results in by_tune.items():
        # Sort by GPU time (ascending = fastest first), skip skipped
        valid = [r for r in tune_results if not r.skipped]
        valid.sort(key=lambda r: r.gpu_time_ms)
        skipped_count = sum(1 for r in tune_results if r.skipped)

        lines.append(f"\n{tune_name}")
        lines.append(
            f"  {len(valid)} configs benchmarked, {skipped_count} pruned/skipped"
        )

        if not valid:
            lines.append("  No valid configs found.")
            continue

        # Show fixed params
        param_strs = [
            f"{k}={_dtype_str(v)}" for k, v in valid[0].params.items()
        ]
        if param_strs:
            lines.append(f"  params: {', '.join(param_strs)}")
        lines.append(f"  world_size: {valid[0].world_size}")
        lines.append("")

        # Build column specs from config kwargs
        config_keys = list(valid[0].config_kwargs.keys())
        cols: list[tuple[str, Callable[[TuneResult], str]]] = [
            ("rank", lambda r, _i=[0]: str((_i.__setitem__(0, _i[0] + 1), _i[0])[1])),
        ]
        for ck in config_keys:
            cols.append(
                (ck, lambda r, _ck=ck: str(r.config_kwargs.get(_ck, "")))
            )
        cols.append(("GPU Time (ms)", lambda r: f"{r.gpu_time_ms:.3f}"))
        if any(r.bandwidth_gbps is not None for r in valid):
            cols.append(
                (
                    "BW (GB/s)",
                    lambda r: f"{r.bandwidth_gbps:.1f}"
                    if r.bandwidth_gbps is not None
                    else "",
                )
            )
        if any(r.tflops is not None for r in valid):
            cols.append(
                (
                    "TFLOPS",
                    lambda r: f"{r.tflops:.1f}" if r.tflops is not None else "",
                )
            )

        # Only show top_k
        shown = valid[:top_k]

        header_strs = [c[0] for c in cols]
        row_strs: list[list[str]] = []
        # Reset rank counter
        for i, r in enumerate(shown):
            row = []
            for j, (_, fmt_fn) in enumerate(cols):
                if j == 0:  # rank column
                    row.append(str(i + 1))
                else:
                    row.append(fmt_fn(r))
            row_strs.append(row)

        widths = [len(h) for h in header_strs]
        for row in row_strs:
            for i, cell in enumerate(row):
                if i < len(widths):
                    widths[i] = max(widths[i], len(cell))

        fmt = "  ".join(f"{{:>{w}}}" for w in widths)
        lines.append("  " + fmt.format(*header_strs))
        for row in row_strs:
            lines.append("  " + fmt.format(*row))

        # Best config summary
        best = valid[0]
        lines.append("")
        lines.append(f"  Best config: Config({', '.join(f'{k}={v}' for k, v in best.config_kwargs.items())})")
        bw_str = f", {best.bandwidth_gbps:.1f} GB/s" if best.bandwidth_gbps else ""
        lines.append(f"  Best time: {best.gpu_time_ms:.3f} ms{bw_str}")

        if len(valid) > top_k:
            lines.append(f"  ({len(valid) - top_k} more configs not shown)")

    return "\n".join(lines) + "\n"


def _format_json(results: list[TuneResult]) -> str:
    """Structured JSON output."""
    records = []
    for r in results:
        rec: dict[str, Any] = {
            "tune_name": r.tune_name,
            "config": {k: _dtype_str(v) for k, v in r.config_kwargs.items()},
            "params": {k: _dtype_str(v) for k, v in r.params.items()},
            "gpu_time_ms": r.gpu_time_ms,
            "world_size": r.world_size,
        }
        if r.bandwidth_gbps is not None:
            rec["bandwidth_gbps"] = r.bandwidth_gbps
        if r.tflops is not None:
            rec["tflops"] = r.tflops
        if r.counters:
            rec["counters"] = r.counters
        if r.skipped:
            rec["skipped"] = True
            rec["skip_reason"] = r.skip_reason
        records.append(rec)

    # Sort by gpu_time_ms for convenience
    records.sort(key=lambda r: r.get("gpu_time_ms", float("inf")))
    return json.dumps(records, indent=2) + "\n"


# Distributed worker
def _run_tune_worker(
    tunables: list[TuneDef],
    heap_size: int,
    use_gluon: bool,
    n_warmup: int,
    n_repeat: int,
    tune_filter: str | None,
) -> list[TuneResult]:
    """Worker that runs inside each rank via ``elastic_launch``."""
    import iris as _iris

    local_rank = int(os.environ["LOCAL_RANK"])
    world_size = int(os.environ["WORLD_SIZE"])

    backend = "nccl" if torch.cuda.is_available() else "gloo"
    if backend == "nccl":
        torch.cuda.set_device(local_rank)
    dist.init_process_group(backend=backend)

    ctx = _iris.iris(heap_size)
    rank = ctx.get_rank()

    all_results: list[TuneResult] = []

    for tdef in tunables:
        if tune_filter and not re.search(tune_filter, tdef.name):
            continue

        # Generate and prune config space
        configs = _generate_configs(tdef.search_axes, tdef.prune_fns)

        if rank == 0:
            total = 1
            for ax in tdef.search_axes:
                total *= len(ax.values)
            print(
                f"[tune] {tdef.name}: {len(configs)} configs "
                f"(pruned {total - len(configs)} of {total})",
                file=sys.stderr,
            )

        for i, config_kwargs in enumerate(configs):
            if rank == 0:
                print(
                    f"  [{i + 1}/{len(configs)}] {config_kwargs}",
                    end="",
                    file=sys.stderr,
                    flush=True,
                )

            state = TuneState(
                params=dict(tdef.fixed_params),
                config_trial=config_kwargs,
                n_warmup=n_warmup,
                n_repeat=n_repeat,
            )

            skipped = False
            skip_reason = ""
            try:
                tdef.fn(state, ctx)
            except Exception as exc:
                skipped = True
                skip_reason = str(exc)

            if skipped:
                if rank == 0:
                    print(f" -> SKIP ({skip_reason})", file=sys.stderr)
                all_results.append(
                    TuneResult(
                        tune_name=tdef.name,
                        config_kwargs=config_kwargs,
                        params=dict(tdef.fixed_params),
                        gpu_time_ms=float("inf"),
                        all_times_ms=[],
                        skipped=True,
                        skip_reason=skip_reason,
                        world_size=world_size,
                    )
                )
                continue

            if state._exec_fn is None:
                raise RuntimeError(
                    f"Tunable '{tdef.name}' with config {config_kwargs} "
                    f"did not call state.exec(fn)."
                )

            try:
                times = _iris.do_bench(
                    state._exec_fn,
                    barrier_fn=ctx.barrier,
                    preamble_fn=state._preamble_fn,
                    n_warmup=state._n_warmup,
                    n_repeat=state._n_repeat,
                    return_mode="all",
                )
                mean_ms = statistics.mean(times)
            except Exception as exc:
                # Config caused a runtime error (e.g., resource exhaustion)
                if rank == 0:
                    print(f" -> ERROR ({exc})", file=sys.stderr)
                all_results.append(
                    TuneResult(
                        tune_name=tdef.name,
                        config_kwargs=config_kwargs,
                        params=dict(tdef.fixed_params),
                        gpu_time_ms=float("inf"),
                        all_times_ms=[],
                        skipped=True,
                        skip_reason=str(exc),
                        world_size=world_size,
                    )
                )
                continue

            bw = None
            if state._bytes is not None and mean_ms > 0:
                bw = (state._bytes / 1e9) / (mean_ms * 1e-3)

            tflops = None
            if state._flops is not None and mean_ms > 0:
                tflops = (state._flops / 1e12) / (mean_ms * 1e-3)

            if rank == 0:
                bw_str = f", {bw:.1f} GB/s" if bw else ""
                print(f" -> {mean_ms:.3f} ms{bw_str}", file=sys.stderr)

            all_results.append(
                TuneResult(
                    tune_name=tdef.name,
                    config_kwargs=config_kwargs,
                    params=dict(tdef.fixed_params),
                    gpu_time_ms=mean_ms,
                    all_times_ms=times,
                    bandwidth_gbps=bw,
                    tflops=tflops,
                    counters=dict(state._counters),
                    world_size=world_size,
                )
            )

    ctx.barrier()
    dist.destroy_process_group()

    return all_results if rank == 0 else []


# CLI
def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="iris.tune — autotuning framework for collective operations",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--tune_filter",
        type=str,
        default=None,
        help="Regex filter for tunable names",
    )
    parser.add_argument(
        "--benchmark_format",
        type=str,
        default="console",
        choices=["console", "json"],
        help="Output format",
    )
    parser.add_argument(
        "--benchmark_out",
        type=str,
        default=None,
        help="Write results to this file",
    )
    parser.add_argument(
        "--top_k",
        type=int,
        default=10,
        help="Show top-K configs in console output",
    )
    parser.add_argument(
        "--heap_size",
        type=int,
        default=1 << 34,
        help="Iris symmetric heap size in bytes",
    )
    parser.add_argument(
        "--use_gluon",
        action="store_true",
        help="Use Gluon backend",
    )
    parser.add_argument(
        "--n_warmup",
        type=int,
        default=10,
        help="Number of warmup iterations per config",
    )
    parser.add_argument(
        "--n_repeat",
        type=int,
        default=50,
        help="Number of timed iterations per config",
    )
    parser.add_argument(
        "--num_ranks",
        type=int,
        default=None,
        help="Number of ranks (GPUs). Default: auto-detect",
    )
    parser.add_argument(
        "--no_cache",
        action="store_true",
        help="Skip cache lookup, always re-tune",
    )
    return parser


def main(argv: list[str] | None = None) -> None:
    """CLI entry point. Call from ``if __name__ == '__main__': tune.main()``."""
    parser = _build_parser()
    args = parser.parse_args(argv)

    tunables = list(_registry)
    if not tunables:
        print("No tunables registered.", file=sys.stderr)
        sys.exit(1)

    # Determine num_ranks
    if args.num_ranks is not None:
        num_ranks = args.num_ranks
    else:
        num_ranks = torch.cuda.device_count() if torch.cuda.is_available() else 1

    print(
        f"[tune] Starting with {num_ranks} ranks, "
        f"{len(tunables)} tunable(s) registered",
        file=sys.stderr,
    )

    config = LaunchConfig(
        min_nodes=1,
        max_nodes=1,
        nproc_per_node=num_ranks,
        rdzv_backend="c10d",
        rdzv_endpoint="localhost:0",
        max_restarts=0,
    )

    start_time = time.time()
    results_by_rank = elastic_launch(config, _run_tune_worker)(
        tunables,
        args.heap_size,
        args.use_gluon,
        args.n_warmup,
        args.n_repeat,
        args.tune_filter,
    )
    elapsed = time.time() - start_time

    all_results: list[TuneResult] = results_by_rank[0]

    print(f"\n[tune] Completed in {elapsed:.1f}s", file=sys.stderr)

    # Cache results
    if not args.no_cache and all_results:
        gpu_arch = "unknown"
        try:
            import iris.hip

            gpu_arch = iris.hip.get_arch_string(0)
        except Exception:
            pass

        for tdef in tunables:
            tune_results = [r for r in all_results if r.tune_name == tdef.name]
            if not tune_results:
                continue

            world_size = tune_results[0].world_size
            cache_key = _build_cache_key(
                tdef.name, tdef.fixed_params, tdef.search_axes, world_size, gpu_arch
            )
            cache_data = []
            for r in tune_results:
                cache_data.append(
                    {
                        "config": r.config_kwargs,
                        "gpu_time_ms": r.gpu_time_ms,
                        "bandwidth_gbps": r.bandwidth_gbps,
                        "tflops": r.tflops,
                        "skipped": r.skipped,
                        "skip_reason": r.skip_reason,
                    }
                )
            cache_file = _save_cache(cache_key, tdef.name, cache_data)
            print(f"[tune] Cached {tdef.name} -> {cache_file}", file=sys.stderr)

    # Format output
    if args.benchmark_format == "json":
        output = _format_json(all_results)
    else:
        output = _format_console(all_results, top_k=args.top_k)

    print(output, end="")

    if args.benchmark_out:
        with open(args.benchmark_out, "w") as f:
            f.write(output)
