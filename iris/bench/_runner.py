# SPDX-License-Identifier: MIT
# Copyright (c) 2025-2026 Advanced Micro Devices, Inc. All rights reserved.

"""Distributed runner, output formatters, and CLI entry point for iris.bench."""

from __future__ import annotations

import argparse
import csv
import io
import itertools
import json
import os
import re
import statistics
import sys
from typing import Any, Callable

import torch
import torch.distributed as dist
import torch.multiprocessing as mp

from ._core import (
    AxisDef,
    BenchmarkDef,
    Result,
    State,
    _SkipCombination,
    _registry,
    power_of_two,
    linear_range,
)


# ---------------------------------------------------------------------------
# Axis override parsing
# ---------------------------------------------------------------------------

_DTYPE_MAP = {
    "fp16": torch.float16,
    "float16": torch.float16,
    "fp32": torch.float32,
    "float32": torch.float32,
    "bf16": torch.bfloat16,
    "bfloat16": torch.bfloat16,
}


def _parse_axis_override(raw: str, axis_name: str) -> list[Any]:
    """Parse a CLI ``--axis_<name>=<value>`` string.

    Formats:
    - ``1024,2048`` — explicit list
    - ``pow2:8:13``  — ``power_of_two(8, 13)``
    - ``lin:64:256:64`` — ``linear_range(64, 256, 64)``
    - dtype names: ``fp16``, ``fp32``, ``bf16``
    """
    raw = raw.strip()

    if raw.startswith("pow2:"):
        parts = raw.split(":")
        return power_of_two(int(parts[1]), int(parts[2]))

    if raw.startswith("lin:"):
        parts = raw.split(":")
        return linear_range(int(parts[1]), int(parts[2]), int(parts[3]))

    tokens = [t.strip() for t in raw.split(",")]

    # Check if they look like dtype names
    if axis_name == "dtype" or all(t.lower() in _DTYPE_MAP for t in tokens):
        return [_DTYPE_MAP[t.lower()] for t in tokens]

    # Try integers
    try:
        return [int(t) for t in tokens]
    except ValueError:
        pass

    # Try floats
    try:
        return [float(t) for t in tokens]
    except ValueError:
        pass

    # Fall back to strings
    return tokens


# ---------------------------------------------------------------------------
# Output formatters
# ---------------------------------------------------------------------------


def _dtype_str(v: Any) -> str:
    """Short string for a torch dtype, passthrough for anything else."""
    if isinstance(v, torch.dtype):
        return {
            torch.float16: "float16",
            torch.float32: "float32",
            torch.bfloat16: "bfloat16",
            torch.float64: "float64",
            torch.int8: "int8",
            torch.int16: "int16",
            torch.int32: "int32",
            torch.int64: "int64",
        }.get(v, str(v))
    return str(v)


def _format_console(results: list[Result]) -> str:
    """Render results as an aligned console table."""
    if not results:
        return ""

    # Group by benchmark name
    by_bench: dict[str, list[Result]] = {}
    for r in results:
        by_bench.setdefault(r.benchmark_name, []).append(r)

    lines: list[str] = []
    for bench_name, bench_results in by_bench.items():
        ws = bench_results[0].world_size
        lines.append(f"\n{bench_name} ({ws} ranks)")

        # Build column specs
        param_names = list(bench_results[0].params.keys())
        cols: list[tuple[str, Callable[[Result], str]]] = []
        for pn in param_names:
            cols.append((pn, lambda r, _pn=pn: _dtype_str(r.params[_pn])))
        cols.append(("GPU Time (ms)", lambda r: f"{r.gpu_time_ms:.3f}"))
        if any(r.bandwidth_gbps is not None for r in bench_results):
            cols.append(("BW (GB/s)", lambda r: f"{r.bandwidth_gbps:.1f}" if r.bandwidth_gbps is not None else ""))
        if any(r.tflops is not None for r in bench_results):
            cols.append(("TFLOPS", lambda r: f"{r.tflops:.1f}" if r.tflops is not None else ""))

        # Gather counter names across all results
        counter_names: list[str] = []
        seen: set[str] = set()
        for r in bench_results:
            for cn in r.counters:
                if cn not in seen:
                    counter_names.append(cn)
                    seen.add(cn)
        for cn in counter_names:
            cols.append((cn, lambda r, _cn=cn: f"{r.counters[_cn]:.3f}" if _cn in r.counters else ""))

        # Compute column widths
        header_strs = [c[0] for c in cols]
        row_strs: list[list[str]] = []
        for r in bench_results:
            if r.skipped:
                row = [cols[0][1](r)] + ["(skipped)" + (f" {r.skip_reason}" if r.skip_reason else "")]
                row += [""] * (len(cols) - 2)
                row_strs.append(row)
            else:
                row_strs.append([c[1](r) for c in cols])

        widths = [len(h) for h in header_strs]
        for row in row_strs:
            for i, cell in enumerate(row):
                if i < len(widths):
                    widths[i] = max(widths[i], len(cell))

        fmt = "  ".join(f"{{:>{w}}}" for w in widths)
        lines.append(fmt.format(*header_strs))
        for row in row_strs:
            while len(row) < len(widths):
                row.append("")
            lines.append(fmt.format(*row))

    return "\n".join(lines) + "\n"


def _format_json(results: list[Result]) -> str:
    """Structured JSON output for CI."""
    records = []
    for r in results:
        rec: dict[str, Any] = {
            "benchmark": r.benchmark_name,
            "world_size": r.world_size,
            "params": {k: _dtype_str(v) for k, v in r.params.items()},
            "gpu_time_ms": r.gpu_time_ms,
            "all_times_ms": r.all_times_ms,
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
    return json.dumps(records, indent=2) + "\n"


def _format_csv(results: list[Result]) -> str:
    """Flat CSV output."""
    if not results:
        return ""
    buf = io.StringIO()

    # Collect all param names and counter names
    param_names: list[str] = []
    counter_names: list[str] = []
    seen_p: set[str] = set()
    seen_c: set[str] = set()
    for r in results:
        for k in r.params:
            if k not in seen_p:
                param_names.append(k)
                seen_p.add(k)
        for k in r.counters:
            if k not in seen_c:
                counter_names.append(k)
                seen_c.add(k)

    fieldnames = (
        ["benchmark", "world_size"]
        + param_names
        + ["gpu_time_ms", "bandwidth_gbps", "tflops"]
        + counter_names
        + ["skipped", "skip_reason"]
    )
    writer = csv.DictWriter(buf, fieldnames=fieldnames)
    writer.writeheader()
    for r in results:
        row: dict[str, Any] = {
            "benchmark": r.benchmark_name,
            "world_size": r.world_size,
            "gpu_time_ms": f"{r.gpu_time_ms:.4f}" if not r.skipped else "",
            "bandwidth_gbps": f"{r.bandwidth_gbps:.2f}" if r.bandwidth_gbps is not None else "",
            "tflops": f"{r.tflops:.2f}" if r.tflops is not None else "",
            "skipped": r.skipped,
            "skip_reason": r.skip_reason,
        }
        for pn in param_names:
            row[pn] = _dtype_str(r.params.get(pn, ""))
        for cn in counter_names:
            row[cn] = f"{r.counters[cn]:.4f}" if cn in r.counters else ""
        writer.writerow(row)
    return buf.getvalue()


# ---------------------------------------------------------------------------
# Distributed runner
# ---------------------------------------------------------------------------


def _run_benchmarks_worker(
    local_rank: int,
    world_size: int,
    init_url: str,
    benchmarks: list[BenchmarkDef],
    axis_overrides: dict[str, list[Any]],
    heap_size: int,
    use_gluon: bool,
    n_warmup: int,
    n_repeat: int,
    benchmark_filter: str | None,
    benchmark_format: str,
    benchmark_out: str | None,
    _already_initialized: bool = False,
):
    """Worker that runs inside each rank (via mp.spawn or torchrun)."""
    import iris as _iris

    if not _already_initialized:
        backend = "nccl" if torch.cuda.is_available() else "gloo"
        dist.init_process_group(
            backend=backend,
            init_method=init_url,
            world_size=world_size,
            rank=local_rank,
            device_id=torch.device(f"cuda:{local_rank}"),
        )

    # Create iris context
    if use_gluon:
        import iris.experimental.iris_gluon as iris_gluon

        shmem = iris_gluon.iris(heap_size)
    else:
        shmem = _iris.iris(heap_size)

    rank = shmem.get_rank()
    actual_world_size = shmem.get_num_ranks()

    all_results: list[Result] = []

    for bdef in benchmarks:
        # Filter
        if benchmark_filter and not re.search(benchmark_filter, bdef.name):
            continue

        # Apply axis overrides
        axes = []
        for ax in bdef.axes:
            if ax.name in axis_overrides:
                axes.append(AxisDef(ax.name, axis_overrides[ax.name]))
            else:
                axes.append(ax)

        # Generate Cartesian product
        if axes:
            axis_names = [a.name for a in axes]
            axis_values = [a.values for a in axes]
            combos = list(itertools.product(*axis_values))
        else:
            axis_names = []
            combos = [()]

        for combo in combos:
            params = dict(zip(axis_names, combo))
            state = State(params, n_warmup=n_warmup, n_repeat=n_repeat)

            skipped = False
            skip_reason = ""
            try:
                bdef.fn(state, shmem)
            except _SkipCombination as exc:
                skipped = True
                skip_reason = exc.reason

            if skipped:
                result = Result(
                    benchmark_name=bdef.name,
                    params=params,
                    gpu_time_ms=0.0,
                    all_times_ms=[],
                    skipped=True,
                    skip_reason=skip_reason,
                    world_size=actual_world_size,
                )
                all_results.append(result)
                continue

            if state._exec_fn is None:
                raise RuntimeError(
                    f"Benchmark '{bdef.name}' with params {params} "
                    f"did not call state.exec(fn). Every benchmark must "
                    f"register a callable to time."
                )

            # Time with do_bench
            times = _iris.do_bench(
                state._exec_fn,
                barrier_fn=shmem.barrier,
                preamble_fn=state._preamble_fn,
                n_warmup=state._n_warmup,
                n_repeat=state._n_repeat,
                return_mode="all",
            )

            mean_ms = statistics.mean(times)

            # Compute derived metrics
            bw = None
            if state._bytes is not None and mean_ms > 0:
                bw = (state._bytes / 1e9) / (mean_ms * 1e-3)

            tflops = None
            if state._flops is not None and mean_ms > 0:
                tflops = (state._flops / 1e12) / (mean_ms * 1e-3)

            result = Result(
                benchmark_name=bdef.name,
                params=params,
                gpu_time_ms=mean_ms,
                all_times_ms=times,
                bandwidth_gbps=bw,
                tflops=tflops,
                counters=dict(state._counters),
                world_size=actual_world_size,
            )
            all_results.append(result)

    # Only rank 0 prints / writes output
    if rank == 0:
        if benchmark_format == "json":
            output = _format_json(all_results)
        elif benchmark_format == "csv":
            output = _format_csv(all_results)
        else:
            output = _format_console(all_results)

        print(output, end="")

        if benchmark_out:
            with open(benchmark_out, "w") as f:
                f.write(output)

    shmem.barrier()

    if not _already_initialized:
        dist.destroy_process_group()

    return all_results


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="iris.bench — GPU benchmarking framework",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--benchmark_filter",
        type=str,
        default=None,
        help="Regex filter for benchmark names",
    )
    parser.add_argument(
        "--benchmark_format",
        type=str,
        default="console",
        choices=["console", "json", "csv"],
        help="Output format",
    )
    parser.add_argument(
        "--benchmark_out",
        type=str,
        default=None,
        help="Write results to this file",
    )
    parser.add_argument(
        "-r",
        "--num_ranks",
        type=int,
        default=8,
        help="Number of GPUs (ignored under torchrun)",
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
        default=25,
        help="Number of warmup iterations",
    )
    parser.add_argument(
        "--n_repeat",
        type=int,
        default=100,
        help="Number of timed iterations",
    )
    return parser


def main(argv: list[str] | None = None) -> None:
    """CLI entry point.  Call from ``if __name__ == '__main__': bench.main()``."""
    parser = _build_parser()
    args, remaining = parser.parse_known_args(argv)

    # Parse --axis_<name>=<value> from remaining args
    axis_overrides: dict[str, list[Any]] = {}
    for token in remaining:
        m = re.match(r"^--axis_(\w+)=(.+)$", token)
        if m:
            axis_overrides[m.group(1)] = _parse_axis_override(m.group(2), m.group(1))
        else:
            parser.error(f"Unrecognized argument: {token}")

    benchmarks = list(_registry)
    if not benchmarks:
        print("No benchmarks registered.", file=sys.stderr)
        sys.exit(1)

    # Detect torchrun: if dist is already initialized, we're inside torchrun
    if dist.is_initialized():
        _run_benchmarks_worker(
            local_rank=int(os.environ.get("LOCAL_RANK", 0)),
            world_size=dist.get_world_size(),
            init_url="",
            benchmarks=benchmarks,
            axis_overrides=axis_overrides,
            heap_size=args.heap_size,
            use_gluon=args.use_gluon,
            n_warmup=args.n_warmup,
            n_repeat=args.n_repeat,
            benchmark_filter=args.benchmark_filter,
            benchmark_format=args.benchmark_format,
            benchmark_out=args.benchmark_out,
            _already_initialized=True,
        )
    elif "RANK" in os.environ and "WORLD_SIZE" in os.environ:
        # torchrun sets these env vars but dist may not be initialized yet
        local_rank = int(os.environ["LOCAL_RANK"])
        world_size = int(os.environ["WORLD_SIZE"])
        torch.cuda.set_device(local_rank)
        _run_benchmarks_worker(
            local_rank=local_rank,
            world_size=world_size,
            init_url="env://",
            benchmarks=benchmarks,
            axis_overrides=axis_overrides,
            heap_size=args.heap_size,
            use_gluon=args.use_gluon,
            n_warmup=args.n_warmup,
            n_repeat=args.n_repeat,
            benchmark_filter=args.benchmark_filter,
            benchmark_format=args.benchmark_format,
            benchmark_out=args.benchmark_out,
        )
    else:
        # Standalone: use mp.spawn
        num_ranks = args.num_ranks
        init_url = "tcp://127.0.0.1:29500"
        mp.spawn(
            fn=_run_benchmarks_worker,
            args=(
                num_ranks,
                init_url,
                benchmarks,
                axis_overrides,
                args.heap_size,
                args.use_gluon,
                args.n_warmup,
                args.n_repeat,
                args.benchmark_filter,
                args.benchmark_format,
                args.benchmark_out,
            ),
            nprocs=num_ranks,
            join=True,
        )
