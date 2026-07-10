#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.

"""
Model-centric validation sweep for the same shapes used by model_sweep.py.

This runs the pytest-based distributed correctness tests under torchrun and
records pass/fail/skip results in a JSON file, separate from performance data.

Usage:
    # Validate all shapes for all_gather_matmul
    python model_validate.py --operation all_gather_matmul

    # Validate specific models only
    python model_validate.py --operation all_gather_matmul --models llama3_8b,gpt_oss_120b

    # Validate specific batch sizes
    python model_validate.py --operation matmul_all_reduce --batch-sizes 16384,32768
"""

import argparse
import json
import os
import shlex
import signal
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict

from model_configs import MODELS, SweepOperation
from model_sweep import generate_dimension_configs, DECODE_BATCH_SIZES, PREFILL_BATCH_SIZES, TRAINING_BATCH_SIZES

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
TIMEOUT_SECONDS = 180
NUM_GPUS = 8
DEFAULT_HEAP_SIZE = 1 << 34
DEFAULT_ELEMENT_SIZE_BYTES = 2
HEAP_HEADROOM_FACTOR = 1.25
HEAP_ALIGNMENT_BYTES = 1 << 30
COPY_ENGINE_HEAP_ESTIMATE_BLOCK_M = 512
COPY_ENGINE_HEAP_ESTIMATE_BLOCK_N = 512


# Validation test configurations (from validate_sweep.py)
VALIDATION_TESTS = {
    "matmul_all_gather": [
        {
            "name": "baseline",
            "path": "tests/ops/test_matmul_all_gather.py",
            "pytest_k": "test_matmul_all_gather and not test_tritonblas",
        },
    ],
    "all_gather_matmul": [
        {
            "name": "baseline",
            "path": "tests/ops/test_all_gather_matmul.py",
            "pytest_k": "test_all_gather_matmul_baseline",
        },
        {
            "name": "tritonblas_rcclbaseline",
            "path": "tests/ops/test_all_gather_matmul.py",
            "pytest_k": "test_tritonblas_rccl_all_gather_matmul",
        },
        {
            "name": "hbm_buffer",
            "path": "tests/ops/test_all_gather_matmul.py",
            "pytest_k": "test_all_gather_matmul_hbm_buffer and not test_all_gather_matmul_hbm_buffer_with_bias",
        },
        {
            "name": "host_copy_engine",
            "path": "tests/ops/test_all_gather_matmul_copy_engine.py",
            "pytest_k": "test_all_gather_matmul_copy_engine",
            "env": {"IRIS_TEST_COPY_ENGINE_MODE": "host"},
        },
        {
            "name": "copy_engine_host_hip_memcpy",
            "path": "tests/ops/test_all_gather_matmul_copy_engine.py",
            "pytest_k": "test_all_gather_matmul_copy_engine",
            "env": {
                "IRIS_TEST_COPY_ENGINE_MODE": "host",
                "IRIS_TEST_HOST_TRANSFER_BACKEND": "hip_memcpy",
            },
        },
        {
            "name": "device_copy_engine",
            "path": "tests/ops/test_all_gather_matmul_copy_engine.py",
            "pytest_k": "test_all_gather_matmul_copy_engine",
            "env": {"IRIS_TEST_COPY_ENGINE_MODE": "device"},
        },
    ],
    "matmul_all_reduce": [
        {
            "name": "one_shot",
            "path": "tests/ops/test_matmul_all_reduce.py",
            "pytest_k": "test_matmul_all_reduce[one_shot",
        },
        {
            "name": "two_shot",
            "path": "tests/ops/test_matmul_all_reduce.py",
            "pytest_k": "test_matmul_all_reduce[two_shot",
        },
        {
            "name": "copy_engine_one_shot",
            "path": "tests/ops/test_matmul_all_reduce_copy_engine.py",
            "pytest_k": "test_matmul_all_reduce_copy_engine[one_shot",
        },
        {
            "name": "copy_engine_two_shot",
            "path": "tests/ops/test_matmul_all_reduce_copy_engine.py",
            "pytest_k": "test_matmul_all_reduce_copy_engine[two_shot",
        },
    ],
}


def log(msg: str) -> None:
    print(msg, file=sys.stderr, flush=True)


def _coerce_subprocess_output(value: str | bytes | None) -> str:
    if value is None:
        return ""
    if isinstance(value, bytes):
        return value.decode("utf-8", errors="replace")
    return value


def _format_repro_command(cmd: list[str], env: dict[str, str]) -> str:
    env_keys = [
        "PYTORCH_ALLOC_CONF",
        "IRIS_TEST_M",
        "IRIS_TEST_N",
        "IRIS_TEST_K",
        "IRIS_TEST_K_LOCAL",
        "IRIS_TEST_HEAP_SIZE",
        "IRIS_TEST_COPY_ENGINE_MODE",
        "IRIS_TEST_HOST_TRANSFER_BACKEND",
    ]
    env_prefix = " ".join(shlex.quote(f"{key}={env[key]}") for key in env_keys if key in env)
    cmd_str = shlex.join(cmd)
    if env_prefix:
        return f"cd {shlex.quote(str(PROJECT_ROOT))} && {env_prefix} {cmd_str}"
    return f"cd {shlex.quote(str(PROJECT_ROOT))} && {cmd_str}"


def _shape_env(config: Dict[str, Any], operation: str) -> Dict[str, str]:
    """Set environment variables for test based on operation and dimensions.

    config contains both global (m, n, k) and local (m_local, n_local, k_local) dimensions.
    """
    m = config["m"]
    n = config["n"]
    k = config["k"]
    m_local = config["m_local"]
    k_local = config["k_local"]

    env = {
        "IRIS_TEST_N": str(n),
    }

    if operation == "matmul_all_gather":
        # Tests expect global M and global K
        env["IRIS_TEST_M"] = str(m)
        env["IRIS_TEST_K"] = str(k)
    elif operation == "matmul_all_reduce":
        # Tests expect global M and K_local (sharded K)
        env["IRIS_TEST_M"] = str(m)
        env["IRIS_TEST_K"] = str(k_local)
    else:  # all_gather_matmul
        # Tests expect global M and K_LOCAL
        env["IRIS_TEST_M"] = str(m)
        env["IRIS_TEST_K_LOCAL"] = str(k_local)

    return env


def _estimate_heap_bytes(operation: str, test_name: str, config: Dict[str, Any]) -> int | None:
    """Estimate heap size needed for validation test.

    Uses global dimensions since that's what heap calculation is based on.
    """
    m = config["m"]
    n = config["n"]
    k = config["k"]
    k_local = config["k_local"]
    elem = DEFAULT_ELEMENT_SIZE_BYTES

    if operation == "matmul_all_gather":
        # Matches the test allocations:
        # A_local (M, K), B (K, N), output (M, N)
        # Output is ALWAYS allocated from heap via shmem.zeros()
        total = (m * k + k * n + m * n) * elem
        if test_name == "tritonblas_rcclbaseline":
            # The direct tritonBLAS+RCCL path also materializes local C_local (M, N).
            total += (m * n) * elem
        return total

    if operation == "matmul_all_reduce":
        # Allocations for matmul_all_reduce:
        # A (M, K_local), B (K_local, N), C (M, N)
        total = (m * k_local + k_local * n + m * n) * elem

        if test_name == "one_shot":
            # Rank-major all-inbox: a_inbox[src_rank * M + row, col].
            total += (NUM_GPUS * m * n) * elem
        elif test_name == "two_shot":
            # Row-shard inbox: a_inbox[src_rank * rows_per_rank + row, col].
            total += (m * n) * elem
        elif test_name == "copy_engine_one_shot":
            # Rank-major all-inbox: a_inbox[src_rank * M + row, col].
            total += (NUM_GPUS * m * n) * elem
        elif test_name == "copy_engine_two_shot":
            # Tile-major flat staging
            block_m = COPY_ENGINE_HEAP_ESTIMATE_BLOCK_M
            block_n = COPY_ENGINE_HEAP_ESTIMATE_BLOCK_N
            partition_rows = (m + NUM_GPUS - 1) // NUM_GPUS
            partition_tiles_m = (partition_rows + block_m - 1) // block_m + 1
            partition_tiles_n = (n + block_n - 1) // block_n
            max_partition_tiles = max(1, partition_tiles_m * partition_tiles_n)
            tile_elements = block_m * block_n
            flat_staging_elements = NUM_GPUS * max_partition_tiles * tile_elements
            total += 2 * flat_staging_elements * elem
        elif "copy_engine" in test_name:
            total += (m * n) * elem

            # Flags for copy engine coordination
            block_m = 64
            block_n = 64
            num_m_tiles = (m + block_m - 1) // block_m
            num_n_tiles = (n + block_n - 1) // block_n
            total_tiles = num_m_tiles * num_n_tiles
            total += total_tiles * 4  # flags array

        # Lock array for all variants that use locks
        if test_name in {"spinlock", "one_shot", "two_shot"} or "copy_engine" in test_name:
            block_m = 64
            block_n = 64
            num_m_tiles = (m + block_m - 1) // block_m
            num_n_tiles = (n + block_n - 1) // block_n
            total_tiles = num_m_tiles * num_n_tiles
            total += total_tiles * 4  # 4 bytes per lock (int32)

        return total

    if operation == "all_gather_matmul":
        # Common allocations across the validation tests:
        # A_sharded (M, K_local), B (K, N), output (M, N)
        total = (m * k_local + k * n + m * n) * elem

        if test_name in {"hbm_buffer", "host_copy_engine", "copy_engine_host_hip_memcpy", "device_copy_engine"}:
            # Both HBM-buffer and copy-engine variants allocate staged_a as (M, K).
            total += (m * k) * elem

        if test_name == "hbm_buffer":
            block_m = 64
            block_k = 32
            k_per_flag = 8
            num_m_tiles = (m + block_m - 1) // block_m
            num_k_blocks_local = k_local // block_k
            num_flag_groups_k = (num_k_blocks_local + k_per_flag - 1) // k_per_flag
            total += num_m_tiles * num_flag_groups_k * 4
        elif test_name in {"host_copy_engine", "copy_engine_host_hip_memcpy", "device_copy_engine"}:
            block_m = 64
            block_n = 64
            num_m_tiles = (m + block_m - 1) // block_m
            num_tiles_n = (n + block_n - 1) // block_n
            total_tiles = num_m_tiles * num_tiles_n
            num_batches = num_m_tiles
            total += total_tiles * 4
            total += num_batches * 4

        return total

    return None


def _round_up(value: int, alignment: int) -> int:
    return ((value + alignment - 1) // alignment) * alignment


def _run_validation_test(operation: str, test_cfg: Dict[str, str], config: Dict[str, Any]) -> Dict[str, Any]:
    label = config["label"]
    env = os.environ.copy()
    env.setdefault("PYTORCH_ALLOC_CONF", "expandable_segments:True")
    env.update(_shape_env(config, operation))
    env.update(test_cfg.get("env", {}))

    estimated_heap_bytes = _estimate_heap_bytes(operation, test_cfg["name"], config)
    requested_heap_size = int(env.get("IRIS_TEST_HEAP_SIZE", DEFAULT_HEAP_SIZE))
    if estimated_heap_bytes is not None:
        requested_heap_size = max(
            requested_heap_size,
            _round_up(int(estimated_heap_bytes * HEAP_HEADROOM_FACTOR), HEAP_ALIGNMENT_BYTES),
        )
    env["IRIS_TEST_HEAP_SIZE"] = str(requested_heap_size)

    cmd = [
        "torchrun",
        f"--nproc_per_node={NUM_GPUS}",
        str(PROJECT_ROOT / "tests/run_tests_distributed.py"),
        "-q",
        str(PROJECT_ROOT / test_cfg["path"]),
    ]
    if test_cfg.get("pytest_k"):
        cmd.extend(["-k", test_cfg["pytest_k"]])
    repro_command = _format_repro_command(cmd, env)

    log(f"  Validating {test_cfg['name']}: {label}")
    log(f"    Heap size: {requested_heap_size / (1024**3):.1f} GiB")
    log(f"    M={config['m']}, N={config['n']}, K={config['k']}, K_local={config['k_local']}")

    process = None
    try:
        process = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            cwd=str(PROJECT_ROOT),
            env=env,
            preexec_fn=os.setsid,
        )
        stdout, stderr = process.communicate(timeout=TIMEOUT_SECONDS)

        if process.returncode == 0:
            log("    ✓ PASSED")
            return {
                "status": "PASSED",
                "heap_size_bytes": requested_heap_size,
            }

        log("    ✗ FAILED")
        error_log_file = PROJECT_ROOT / f"validation_error_{operation}_{test_cfg['name']}_{label}.log"
        with open(error_log_file, "w") as f:
            f.write(f"Command: {repro_command}\n\n")
            f.write(stdout)
            f.write("\n")
            f.write(stderr)
        lines = (stdout + stderr).strip().split("\n")
        for line in lines[-5:]:
            log(f"      {line}")
        return {
            "status": "FAILED",
            "log": str(error_log_file),
            "heap_size_bytes": requested_heap_size,
            "command": repro_command,
        }

    except subprocess.TimeoutExpired as timeout_err:
        log("    ✗ TIMEOUT")
        if process is not None:
            try:
                os.killpg(os.getpgid(process.pid), signal.SIGTERM)
            except ProcessLookupError:
                pass
        partial_stdout = _coerce_subprocess_output(timeout_err.stdout)
        partial_stderr = _coerce_subprocess_output(timeout_err.stderr)
        error_log_file = PROJECT_ROOT / f"validation_timeout_{operation}_{test_cfg['name']}_{label}.log"
        with open(error_log_file, "w") as f:
            f.write(f"Command: {repro_command}\n\n")
            f.write(partial_stdout)
            f.write("\n")
            f.write(partial_stderr)
        lines = (partial_stdout + partial_stderr).strip().split("\n") if (partial_stdout or partial_stderr) else []
        for line in lines[-5:]:
            log(f"      {line}")
        return {
            "status": "TIMEOUT",
            "log": str(error_log_file),
            "heap_size_bytes": requested_heap_size,
            "command": repro_command,
        }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Model-centric validation sweep",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--operation",
        type=str,
        required=True,
        choices=["matmul_all_gather", "all_gather_matmul", "matmul_all_reduce"],
        help="Operation type to validate",
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
        help="Comma-separated batch sizes in tokens (default: all). "
        f"Presets: decode={DECODE_BATCH_SIZES}, prefill={PREFILL_BATCH_SIZES}",
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
        help="Output JSON file (default: benchmark/ops/model_validation_results_{operation}.json)",
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
        batch_sizes = DECODE_BATCH_SIZES + PREFILL_BATCH_SIZES

    # Generate dimension configs (same as model_sweep.py)
    dimension_configs = generate_dimension_configs(
        models=models,
        operation=args.operation,
        batch_sizes=batch_sizes,
        tp_degree=args.tp_degree,
    )

    if not dimension_configs:
        log("ERROR: No valid dimension configurations generated")
        return

    # Determine output file
    if args.output:
        output_file = Path(args.output)
    else:
        output_file = PROJECT_ROOT / f"benchmark/ops/model_validation_results_{args.operation}.json"

    tests = VALIDATION_TESTS[args.operation]

    log("=" * 80)
    log(f"{args.operation.upper().replace('_', '-')} Model Validation Sweep")
    log("=" * 80)
    log(f"Models: {', '.join(models)}")
    log(f"Batch sizes: {batch_sizes}")
    log(f"TP degree: {args.tp_degree}")
    log(f"Dimension configurations: {len(dimension_configs)}")
    log(f"Validation tests per config: {len(tests)}")
    log(f"Total validations: {len(dimension_configs) * len(tests)}")
    log(f"Output file: {output_file}")
    log("=" * 80)
    log("")

    results = []

    for idx, config in enumerate(dimension_configs, 1):
        row = {
            "label": config["label"],
            "model_name": config["model_name"],
            "batch_size": config["batch_size"],
            "M": config["m"],
            "N": config["n"],
            "K": config["k"],
            "operation": args.operation,
            "validations": {},
        }
        log(f"[{idx}/{len(dimension_configs)}] Testing {config['label']}")

        for test_cfg in tests:
            row["validations"][test_cfg["name"]] = _run_validation_test(args.operation, test_cfg, config)

        results.append(row)
        log("")

    output_file.parent.mkdir(parents=True, exist_ok=True)
    with open(output_file, "w") as f:
        json.dump(results, f, indent=2)
    log(f"✓ Validation results saved to {output_file}")


if __name__ == "__main__":
    main()
