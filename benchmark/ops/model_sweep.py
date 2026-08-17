#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.

"""Model-centric benchmark sweep across operations and batch sizes.

This script generates dimension configurations dynamically from model definitions
and runs benchmarks for systematic evaluation across:
- Models: DeepSeek-V3, Llama 3/4, GPT-OSS 120B
- Operations: attention output, MLP down-projection, MLP up-projection
- Batch sizes: decode (1, 32, 128), prefill (2048, 4096), training (16384, 32768)
- TP degree: 8 GPUs (configurable)

Usage:
    # Run all shapes for all_gather_matmul
    python model_sweep.py --operation all_gather_matmul

    # Run specific models only
    python model_sweep.py --operation all_gather_matmul --models llama3_8b,deepseek_v3

    # Run specific batch sizes
    python model_sweep.py --operation matmul_all_reduce --batch-sizes 16384,32768

    # Custom TP degree (must evenly divide all model dimensions)
    python model_sweep.py --operation all_gather_matmul --tp-degree 4

    # Run GPT-OSS 120B only
    python model_sweep.py --operation all_gather_matmul --models gpt_oss_120b --batch-sizes 16384
"""

import argparse
import json
import os
import signal
import subprocess
import sys
import tempfile
from dataclasses import asdict
from pathlib import Path
from typing import Any, Dict, List, Optional

from model_configs import MODELS, compute_dimensions, OperationType, SweepOperation
from sweep_bench import (
    BENCHMARK_CONFIGS,
    NUM_GPUS,
    PROJECT_ROOT,
    TIMEOUT_SECONDS,
    _calculate_heap_size,
    _dimension_label,
    _dimension_values,
    log,
)

# Batch size categories (in tokens)
DECODE_BATCH_SIZES = [32, 128]  # Online inference decode
PREFILL_BATCH_SIZES = [2048, 4096]  # Prefill / context ingestion
TRAINING_BATCH_SIZES = [16384, 32768]  # Long-context training
ALL_BATCH_SIZES = [32, 128, 512, 2048, 8192] #DECODE_BATCH_SIZES + PREFILL_BATCH_SIZES + TRAINING_BATCH_SIZES

# Operation mapping: sweep operation type → which OperationTypes to benchmark
# Different sweep operations use different TP sharding patterns even for the same layer
OPERATION_TYPE_MAP = {
    SweepOperation.ALL_GATHER_MATMUL: [
        OperationType.ATTN_OUT,
        OperationType.MLP_DOWN,
        OperationType.EXPERT_MLP_DOWN,
        # OperationType.ACTIVE_MOE_MLP_DOWN,
    ],  # K-sharding
    SweepOperation.MATMUL_ALL_GATHER: [
        OperationType.MLP_DOWN,
        OperationType.EXPERT_MLP_DOWN,
        OperationType.ACTIVE_MOE_MLP_DOWN,
    ],  # N-sharding (different from all_gather_matmul)
    SweepOperation.MATMUL_ALL_REDUCE: [
        OperationType.ATTN_OUT,
        OperationType.MLP_DOWN,
        OperationType.EXPERT_MLP_DOWN,
        # OperationType.ACTIVE_MOE_MLP_DOWN,
    ],  # K-sharding
}


def _run_bench_benchmark(
    benchmark_name: str,
    script: str,
    m: int,
    n: int,
    k: int,
    benchmark_filter: str,
    axes: Dict[str, str],
    heap_size: int,
    operation: str,
    variant: Optional[str] = None,
    tp_degree: int = NUM_GPUS,
) -> Optional[Dict[str, Any]]:
    """Run a benchmark with dimensions already computed by compute_dimensions().

    Dimensions passed in are already correct (no mapping needed).
    compute_dimensions() handles all TP sharding logic based on operation type and sweep operation.
    """
    with tempfile.NamedTemporaryFile(
        mode="w",
        suffix=f"_{benchmark_name}_M{m}_N{n}_K{k}.json",
        dir=str(PROJECT_ROOT),
        delete=False,
    ) as tmp_file:
        benchmark_out = tmp_file.name

    cmd = [
        sys.executable,
        script,
        "--benchmark_format=json",
        f"--benchmark_out={benchmark_out}",
        f"--benchmark_filter={benchmark_filter}",
        f"--axis_num_ranks={tp_degree}",
        f"--axis_{axes['m']}={m}",
        f"--axis_{axes['n']}={n}",
        f"--axis_{axes['k']}={k}",
        "--axis_dtype=fp16",
        f"--heap_size={heap_size}",
    ]

    # Add variant parameter if present (for matmul_all_reduce)
    if variant is not None and "variant" in axes:
        cmd.append(f"--axis_{axes['variant']}={variant}")

    log(f"  Running {benchmark_name}: M={m}, N={n}, K={k}")
    log(f"    Heap size: {heap_size / (1 << 30):.2f} GB")
    log(f"    Command: {' '.join(cmd)}")

    process = None
    try:
        process = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            cwd=str(PROJECT_ROOT),
            preexec_fn=os.setsid,
        )
        stdout, stderr = process.communicate(timeout=TIMEOUT_SECONDS)
        result = subprocess.CompletedProcess(cmd, process.returncode, stdout, stderr)

        if result.returncode != 0:
            log("    ✗ Failed: Non-zero return code")
            log(f"    Return code: {result.returncode}")
            error_log_file = PROJECT_ROOT / f"benchmark_error_{operation}_{benchmark_name}_M{m}_N{n}_K{k}.log"
            with open(error_log_file, "w") as f:
                f.write(f"Operation: {operation}\n")
                f.write(f"Benchmark: {benchmark_name}\n")
                f.write(f"Dimensions: M={m}, N={n}, K={k}\n")
                f.write(f"Command: {' '.join(cmd)}\n")
                f.write(f"Return code: {result.returncode}\n\n")
                f.write("=" * 80 + "\n")
                f.write("STDOUT:\n")
                f.write("=" * 80 + "\n")
                f.write(result.stdout)
                f.write("\n" + "=" * 80 + "\n")
                f.write("STDERR:\n")
                f.write("=" * 80 + "\n")
                f.write(result.stderr)
            log(f"    Full output saved to: {error_log_file}")
            lines = (result.stdout + result.stderr).strip().split("\n")
            log("    Last output lines:")
            for line in lines[-5:]:
                log(f"      {line}")
            return None

        with open(benchmark_out, "r") as f:
            records = json.load(f)
        if not isinstance(records, list) or not records:
            log("    ✗ Failed: bench JSON output was empty")
            return None

        record = next((r for r in records if not r.get("skipped")), None)
        if record is None:
            skip_reason = records[0].get("skip_reason", "")
            log(f"    ✗ Failed: benchmark was skipped ({skip_reason})")
            return {"status": "SKIPPED", "skip_reason": skip_reason}

        params = record.get("params", {})
        counters = record.get("counters", {})
        data = {
            "world_size": record.get("world_size"),
            "operation": record.get("benchmark"),
            "m": int(params.get(axes["m"], m)),
            "n": int(params.get(axes["n"], n)),
            "k": int(params.get(axes["k"], k)),
            "datatype": params.get("dtype", "float16"),
            "total_ms": record.get("gpu_time_ms"),
            "gpu_time_ms": record.get("gpu_time_ms"),
            "all_times_ms": record.get("all_times_ms", []),
            "bandwidth_gbps": record.get("bandwidth_gbps"),
            "tflops": record.get("tflops"),
        }
        data.update(counters)
        log("    ✓ Success: Loaded bench JSON results")
        return data

    except subprocess.TimeoutExpired as timeout_err:
        log(f"    ✗ Timeout after {TIMEOUT_SECONDS}s - killing process group")
        partial_stdout = timeout_err.stdout.decode("utf-8", errors="replace") if timeout_err.stdout else ""
        partial_stderr = timeout_err.stderr.decode("utf-8", errors="replace") if timeout_err.stderr else ""
        if process:
            try:
                os.killpg(os.getpgid(process.pid), signal.SIGTERM)
                try:
                    process.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    log("    Process didn't terminate, force killing...")
                    os.killpg(os.getpgid(process.pid), signal.SIGKILL)
                    process.wait()
            except ProcessLookupError:
                pass
        error_log_file = PROJECT_ROOT / f"benchmark_timeout_{operation}_{benchmark_name}_M{m}_N{n}_K{k}.log"
        with open(error_log_file, "w") as f:
            f.write(f"Operation: {operation}\n")
            f.write(f"Benchmark: {benchmark_name}\n")
            f.write(f"Dimensions: M={m}, N={n}, K={k}\n")
            f.write(f"Command: {' '.join(cmd)}\n")
            f.write(f"Status: TIMEOUT after {TIMEOUT_SECONDS}s\n\n")
            f.write("=" * 80 + "\n")
            f.write("PARTIAL STDOUT (before timeout):\n")
            f.write("=" * 80 + "\n")
            f.write(partial_stdout)
            f.write("\n" + "=" * 80 + "\n")
            f.write("PARTIAL STDERR (before timeout):\n")
            f.write("=" * 80 + "\n")
            f.write(partial_stderr)
        log(f"    Timeout logged to: {error_log_file}")
        return None
    except json.JSONDecodeError as e:
        log(f"    ✗ Error: Failed to parse bench JSON: {e}")
        return None
    except Exception as e:
        log(f"    ✗ Error: {e}")
        return None
    finally:
        try:
            os.remove(benchmark_out)
        except OSError:
            pass


def generate_dimension_configs(
    models: List[str],
    operation: str,
    batch_sizes: List[int],
    tp_degree: int = NUM_GPUS,
) -> List[Dict[str, Any]]:
    """Generate DIMENSION_CONFIGS-compatible dicts from model definitions.

    Args:
        models: List of model names from MODELS registry
        operation: Sweep operation type ("all_gather_matmul", "matmul_all_gather", "matmul_all_reduce")
        batch_sizes: List of batch sizes (token counts) to benchmark
        tp_degree: Tensor parallelism degree (world size)

    Returns:
        List of dicts with keys: m_local, n, k, label, operation_type
        matching the format expected by _run_bench_benchmark()

    Raises:
        ValueError: If unknown model name or TP degree doesn't divide dimensions evenly
    """
    configs = []

    # Convert string operation to SweepOperation enum
    sweep_op = SweepOperation(operation)
    operation_types = OPERATION_TYPE_MAP[sweep_op]

    for model_name in models:
        if model_name not in MODELS:
            log(f"Warning: Unknown model '{model_name}', skipping")
            continue

        model = MODELS[model_name]

        for op_type in operation_types:
            if op_type == OperationType.MLP_DOWN and model.num_dense_layers == 0:
                continue
            if (
                op_type in (OperationType.EXPERT_MLP_DOWN, OperationType.ACTIVE_MOE_MLP_DOWN)
                and model.expert_intermediate_size is None
            ):
                continue
            for batch_size in batch_sizes:
                try:
                    # compute_dimensions now takes sweep_operation and handles dimension mapping
                    dims = compute_dimensions(model, op_type, sweep_op, batch_size, tp_degree)

                    # Convert DimensionSpec to dict and add operation_type enum
                    config = asdict(dims)
                    config["operation_type"] = op_type  # Add enum for _run_bench_benchmark
                    configs.append(config)

                except ValueError as e:
                    log(f"Skipping {model_name} {op_type.value} batch={batch_size}: {e}")

    return configs


def main():
    parser = argparse.ArgumentParser(
        description="Model-centric benchmark sweep",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--operation",
        type=str,
        required=True,
        choices=["matmul_all_gather", "all_gather_matmul", "matmul_all_reduce"],
        help="Operation type to benchmark",
    )
    parser.add_argument(
        "--models",
        type=str,
        default=None,
        help="Comma-separated model names (default: all models). "
        f"Available: {', '.join(MODELS.keys())}",
    )
    parser.add_argument(
        "--batch-sizes",
        type=str,
        default=None,
        help="Comma-separated batch sizes in tokens (default: decode+prefill+training). "
        f"Presets: decode={DECODE_BATCH_SIZES}, prefill={PREFILL_BATCH_SIZES}, "
        f"training={TRAINING_BATCH_SIZES}",
    )
    parser.add_argument(
        "--tp-degree",
        type=int,
        default=NUM_GPUS,
        help=f"Tensor parallelism degree / world size (default: {NUM_GPUS})",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Output JSON file (default: benchmark/ops/model_sweep_results_{operation}.json)",
    )
    args = parser.parse_args()

    # Parse models
    if args.models:
        models = [m.strip() for m in args.models.split(",")]
    else:
        models = list(MODELS.keys())

    # Parse batch sizes
    if args.batch_sizes:
        batch_sizes = [int(b.strip()) for b in args.batch_sizes.split(",")]
    else:
        batch_sizes = ALL_BATCH_SIZES

    # Generate configs
    dimension_configs = generate_dimension_configs(
        models=models,
        operation=args.operation,
        batch_sizes=batch_sizes,
        tp_degree=args.tp_degree,
    )

    if not dimension_configs:
        log("ERROR: No valid dimension configurations generated")
        log("Check that:")
        log("  - Model names are spelled correctly")
        log(f"  - TP degree {args.tp_degree} divides all model dimensions evenly")
        log(f"  - Available models: {', '.join(MODELS.keys())}")
        return 1

    # Determine output file
    if args.output:
        output_file = Path(args.output)
    else:
        output_file = PROJECT_ROOT / f"benchmark/ops/model_sweep_results_{args.operation}.json"

    # Run sweep (reuse sweep_bench infrastructure)
    operation = args.operation
    sweep_op = SweepOperation(operation)  # Convert to enum for OPERATION_TYPE_MAP
    benchmarks = BENCHMARK_CONFIGS[operation]

    log("=" * 80)
    log(f"{operation.upper().replace('_', '-')} Model Sweep")
    log("=" * 80)
    log(f"Models: {', '.join(models)}")
    log(f"Batch sizes: {batch_sizes}")
    log(f"TP degree: {args.tp_degree}")
    log(f"Operation types: {', '.join(ot.value for ot in OPERATION_TYPE_MAP[sweep_op])}")
    log(f"Dimension configurations: {len(dimension_configs)}")
    for config in dimension_configs:
        m, n, k = _dimension_values(config)
        log(f"  - {_dimension_label(config)}: M={m}, N={n}, K={k}")
    log(f"Benchmarks per configuration: {len(benchmarks)}")
    log(f"Total benchmarks: {len(dimension_configs) * len(benchmarks)}")
    log(f"GPUs: {args.tp_degree}")
    log(f"Output file: {output_file}")
    log("=" * 80)
    log("")

    results = []

    log(f"Running {len(dimension_configs)} dimension configurations...\n")

    for idx, config in enumerate(dimension_configs, 1):
        # Extract dimensions based on what each benchmark expects
        # - all_gather_matmul: global M, N, K (does gathering internally)
        # - matmul_all_reduce: global M, N, K_local (expects pre-sharded K input)
        # - matmul_all_gather: M_local, global N, K (expects pre-sharded M input)

        if operation == "matmul_all_reduce":
            # Pass K_local (sharded K dimension)
            m, n, k = config["m"], config["n"], config["k_local"]
        elif operation == "matmul_all_gather":
            # Pass M_local (sharded M dimension)
            m, n, k = config["m_local"], config["n"], config["k"]
        else:
            # all_gather_matmul: use global dimensions
            m, n, k = config["m"], config["n"], config["k"]

        label = config["label"]
        operation_type = config.get("operation_type")

        # Use global dimensions for heap calculation
        heap_size = _calculate_heap_size(config["m"], config["n"], config["k"], operation, args.tp_degree)
        log(f"[{idx}/{len(dimension_configs)}] Testing {label}: M={m}, N={n}, K={k}")
        log(f"  Calculated heap size: {heap_size / (1 << 30):.2f} GB")
        log(f"  Operation type: {operation_type.value if operation_type else 'unknown'}")

        row = {"label": label, "M": m, "N": n, "K": k, "operation": operation, "benchmarks": {}}

        # Run each benchmark variant using our local _run_bench_benchmark
        # Dimensions are already correct from compute_dimensions()
        for bench_key, bench_config in benchmarks.items():
            result = _run_bench_benchmark(
                benchmark_name=bench_key,
                script=bench_config["script"],
                m=m,
                n=n,
                k=k,
                benchmark_filter=bench_config["benchmark_filter"],
                axes=bench_config.get("axes", {"m": "M", "n": "N", "k": "K"}),
                heap_size=heap_size,
                operation=operation,
                variant=bench_config.get("variant"),
                tp_degree=args.tp_degree,
            )

            if result is not None:
                row["benchmarks"][bench_key] = result
            else:
                row["benchmarks"][bench_key] = {"status": "FAILED"}

        results.append(row)
        log("")

    # Write JSON file
    log(f"Writing results to {output_file}...")
    output_file.parent.mkdir(parents=True, exist_ok=True)
    with open(output_file, "w") as f:
        json.dump(results, f, indent=2)

    log(f"✓ Results saved to {output_file}\n")

    log("\n" + "=" * 80)
    log("Model sweep complete!")
    log("=" * 80)

    return 0


if __name__ == "__main__":
    sys.exit(main())
