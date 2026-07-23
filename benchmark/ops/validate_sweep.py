#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.

"""
Distributed validation sweep for the same shapes used by benchmark sweeps.

This runs the pytest-based distributed correctness tests under torchrun and
records pass/fail/skip results in a JSON file, separate from performance data.
"""

import argparse
import json
import os
import shlex
import signal
import subprocess
import sys
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
TIMEOUT_SECONDS = 180
NUM_GPUS = 8
DEFAULT_HEAP_SIZE = 1 << 34
DEFAULT_ELEMENT_SIZE_BYTES = 2
HEAP_HEADROOM_FACTOR = 1.25
HEAP_ALIGNMENT_BYTES = 1 << 30
COPY_ENGINE_HEAP_ESTIMATE_BLOCK_M = 512
COPY_ENGINE_HEAP_ESTIMATE_BLOCK_N = 512


DIMENSION_CONFIGS = [
    {"m_local": 2048, "n": 2048, "k": 16384, "label": "M2048_N2048_K16384"},
    {"m_local": 2048, "n": 16384, "k": 2048, "label": "M2048_N16384_K2048"},
    {"m_local": 2048, "n": 16384, "k": 16384, "label": "M2048_N16384_K16384"},
    {"m_local": 2048, "n": 16384, "k": 65536, "label": "M2048_N16384_K65536"},
    {"m_local": 2048, "n": 131072, "k": 16384, "label": "M2048_N131072_K16384"},
    {"m_local": 16384, "n": 2048, "k": 2048, "label": "M16384_N2048_K2048"},
    {"m_local": 16384, "n": 2048, "k": 16384, "label": "M16384_N2048_K16384"},
    {"m_local": 16384, "n": 2048, "k": 131072, "label": "M16384_N2048_K131072"},
    {"m_local": 16384, "n": 16384, "k": 2048, "label": "M16384_N16384_K2048"},
    {"m_local": 131072, "n": 2048, "k": 16384, "label": "M131072_N2048_K16384"},
    # {"m_local": 131072, "n": 16384, "k": 16384, "label": "g2"},
    # {"m_local": 147456, "n": 28672, "k": 4096, "label": "g14"},
    # {"m_local": 327680, "n": 28672, "k": 4096, "label": "g15"},
    # {"m_local": 229376, "n": 28672, "k": 4096, "label": "g16"},
    # {"m_local": 8192, "n": 8192, "k": 262144, "label": "g5"},
    # {"m_local": 262144, "n": 8192, "k": 8192, "label": "g6"},
    # {"m_local": 16384, "n": 16384, "k": 131072, "label": "g1"},
    # {"m_local": 262144, "n": 28672, "k": 8192, "label": "g8"},
    # {"m_local": 196608, "n": 18432, "k": 16384, "label": "g9"},
    # {"m_local": 4096, "n": 14336, "k": 4096, "label": "mixtral_gate"},
    # {"m_local": 4096, "n": 11008, "k": 4096, "label": "llama7b_gate"},
    # {"m_local": 4096, "n": 4096, "k": 4096, "label": "pow2_4k"},
    # {"m_local": 1024, "n": 3584, "k": 8192, "label": "M1024_N3584_K8192"},
    # {"m_local": 4096, "n": 3584, "k": 8192, "label": "M4096_N3584_K8192"},
    # {"m_local": 16384, "n": 3584, "k": 8192, "label": "M16384_N3584_K8192"},
]


VALIDATION_TESTS = {
    "matmul_all_gather": [
        {
            "name": "baseline",
            "path": "tests/ops/test_matmul_all_gather.py",
            "pytest_k": "test_matmul_all_gather and not test_tritonblas",
        },
        # {
        #     "name": "tritonblas_rcclbaseline",
        #     "path": "tests/ops/test_matmul_all_gather.py",
        #     "pytest_k": "test_tritonblas_rccl_matmul_all_gather",
        # },
        # {
        #     "name": "host_copy_engine",
        #     "path": "tests/ops/test_matmul_all_gather_copy_engine.py",
        #     "pytest_k": "test_matmul_all_gather_copy_engine",
        #     "env": {"IRIS_TEST_COPY_ENGINE_MODE": "host"},
        # },
        # {
        #     "name": "device_copy_engine",
        #     "path": "tests/ops/test_matmul_all_gather_copy_engine.py",
        #     "pytest_k": "test_matmul_all_gather_copy_engine",
        #     "env": {"IRIS_TEST_COPY_ENGINE_MODE": "device"},
        # },
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
        # Spinlock is currently disabled in tests pending cache-modifiers support
        # {
        #     "name": "spinlock",
        #     "path": "tests/ops/test_matmul_all_reduce.py",
        #     "pytest_k": "test_matmul_all_reduce[spinlock",
        # },
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


def _shape_env(config: dict[str, Any], operation: str) -> dict[str, str]:
    m_local = int(config["m_local"])
    n = int(config["n"])
    k = int(config["k"])

    env = {
        "IRIS_TEST_N": str(n),
    }

    if operation == "matmul_all_gather":
        env["IRIS_TEST_M"] = str(m_local * NUM_GPUS)
        env["IRIS_TEST_K"] = str(k)
    elif operation == "matmul_all_reduce":
        env["IRIS_TEST_M"] = str(m_local)
        env["IRIS_TEST_K"] = str(k)
    else:
        if k % NUM_GPUS != 0:
            env["IRIS_TEST_INVALID"] = "1"
        env["IRIS_TEST_M"] = str(m_local)
        env["IRIS_TEST_K_LOCAL"] = str(k // NUM_GPUS)
    return env


def _estimate_heap_bytes(operation: str, test_name: str, shape_cfg: dict[str, Any]) -> int | None:
    m_local = int(shape_cfg["m_local"])
    n = int(shape_cfg["n"])
    k = int(shape_cfg["k"])
    elem = DEFAULT_ELEMENT_SIZE_BYTES

    if operation == "matmul_all_gather":
        m_total = m_local * NUM_GPUS
        # Matches the test allocations:
        # A_local (M_local, K), B (K, N), output (M_total, N)
        # Output is ALWAYS allocated from heap via shmem.zeros()
        total = (m_local * k + k * n + m_total * n) * elem
        if test_name == "tritonblas_rcclbaseline":
            # The direct tritonBLAS+RCCL path also materializes local C_local (M_local, N).
            total += (m_local * n) * elem
        return total

    if operation == "matmul_all_reduce":
        # Allocations for matmul_all_reduce:
        # A (M, K), B (K, N), C (M, N)
        total = (m_local * k + k * n + m_local * n) * elem

        if test_name == "one_shot":
            # Rank-major all-inbox: a_inbox[src_rank * M + row, col].
            total += (NUM_GPUS * m_local * n) * elem
        elif test_name == "two_shot":
            # Row-shard inbox: a_inbox[src_rank * rows_per_rank + row, col].
            total += (m_local * n) * elem
        elif test_name == "copy_engine_one_shot":
            # Rank-major all-inbox: a_inbox[src_rank * M + row, col].
            total += (NUM_GPUS * m_local * n) * elem
        elif test_name == "copy_engine_two_shot":
            # Tile-major flat staging:
            #   aux_buffer/a_inbox: (world_size * max_partition_tiles, tile_elements)
            # Use a conservative 512x512 envelope because the validation script
            # does not instantiate the Origami selector for each shape.
            block_m = COPY_ENGINE_HEAP_ESTIMATE_BLOCK_M
            block_n = COPY_ENGINE_HEAP_ESTIMATE_BLOCK_N
            partition_rows = (m_local + NUM_GPUS - 1) // NUM_GPUS
            partition_tiles_m = (partition_rows + block_m - 1) // block_m + 1
            partition_tiles_n = (n + block_n - 1) // block_n
            max_partition_tiles = max(1, partition_tiles_m * partition_tiles_n)
            tile_elements = block_m * block_n
            flat_staging_elements = NUM_GPUS * max_partition_tiles * tile_elements
            total += 2 * flat_staging_elements * elem
        elif "copy_engine" in test_name:
            total += (m_local * n) * elem

            # Flags for copy engine coordination
            block_m = 64
            block_n = 64
            num_m_tiles = (m_local + block_m - 1) // block_m
            num_n_tiles = (n + block_n - 1) // block_n
            total_tiles = num_m_tiles * num_n_tiles
            total += total_tiles * 4  # flags array

        # Lock array for all variants that use locks
        if test_name in {"spinlock", "one_shot", "two_shot"} or "copy_engine" in test_name:
            block_m = 64
            block_n = 64
            num_m_tiles = (m_local + block_m - 1) // block_m
            num_n_tiles = (n + block_n - 1) // block_n
            total_tiles = num_m_tiles * num_n_tiles
            total += total_tiles * 4  # 4 bytes per lock (int32)

        return total

    if operation == "all_gather_matmul":
        if k % NUM_GPUS != 0:
            return None

        k_local = k // NUM_GPUS
        # Common allocations across the validation tests:
        # A_sharded (M, K_local), B (K, N), output (M, N)
        total = (m_local * k_local + k * n + m_local * n) * elem

        if test_name in {"hbm_buffer", "host_copy_engine", "copy_engine_host_hip_memcpy", "device_copy_engine"}:
            # Both HBM-buffer and copy-engine variants allocate staged_a as (M, K).
            total += (m_local * k) * elem

        if test_name == "hbm_buffer":
            block_m = 64
            block_k = 32
            k_per_flag = 8
            num_m_tiles = (m_local + block_m - 1) // block_m
            num_k_blocks_local = k_local // block_k
            num_flag_groups_k = (num_k_blocks_local + k_per_flag - 1) // k_per_flag
            total += num_m_tiles * num_flag_groups_k * 4
        elif test_name in {"host_copy_engine", "copy_engine_host_hip_memcpy", "device_copy_engine"}:
            block_m = 64
            block_n = 64
            num_m_tiles = (m_local + block_m - 1) // block_m
            num_tiles_n = (n + block_n - 1) // block_n
            total_tiles = num_m_tiles * num_tiles_n
            num_batches = num_m_tiles
            total += total_tiles * 4
            total += num_batches * 4

        return total

    return None


def _round_up(value: int, alignment: int) -> int:
    return ((value + alignment - 1) // alignment) * alignment


def _run_validation_test(operation: str, test_cfg: dict[str, str], shape_cfg: dict[str, Any]) -> dict[str, Any]:
    label = shape_cfg["label"]
    env = os.environ.copy()
    env.setdefault("PYTORCH_ALLOC_CONF", "expandable_segments:True")
    env.update(_shape_env(shape_cfg, operation))
    env.update(test_cfg.get("env", {}))
    estimated_heap_bytes = _estimate_heap_bytes(operation, test_cfg["name"], shape_cfg)
    requested_heap_size = int(env.get("IRIS_TEST_HEAP_SIZE", DEFAULT_HEAP_SIZE))
    if estimated_heap_bytes is not None:
        requested_heap_size = max(
            requested_heap_size,
            _round_up(int(estimated_heap_bytes * HEAP_HEADROOM_FACTOR), HEAP_ALIGNMENT_BYTES),
        )
    env["IRIS_TEST_HEAP_SIZE"] = str(requested_heap_size)

    if env.get("IRIS_TEST_INVALID") == "1":
        return {
            "status": "SKIPPED",
            "reason": f"K={shape_cfg['k']} not divisible by world_size={NUM_GPUS}",
        }

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
    log(f"    Command: {' '.join(cmd)}")

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
            return {
                "status": "PASSED",
                "heap_size_bytes": requested_heap_size,
            }

        error_log_file = PROJECT_ROOT / f"validation_error_{operation}_{test_cfg['name']}_{label}.log"
        with open(error_log_file, "w") as f:
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
        if process is not None:
            try:
                os.killpg(os.getpgid(process.pid), signal.SIGTERM)
            except ProcessLookupError:
                pass
        partial_stdout = _coerce_subprocess_output(timeout_err.stdout)
        partial_stderr = _coerce_subprocess_output(timeout_err.stderr)
        error_log_file = PROJECT_ROOT / f"validation_timeout_{operation}_{test_cfg['name']}_{label}.log"
        with open(error_log_file, "w") as f:
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
        description="Run distributed validation sweep for benchmark shapes",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--operation",
        type=str,
        required=True,
        choices=["matmul_all_gather", "all_gather_matmul", "matmul_all_reduce"],
        help="Operation family to validate",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Output JSON file path",
    )
    args = parser.parse_args()

    output_file = (
        Path(args.output)
        if args.output
        else PROJECT_ROOT / f"benchmark/ops/validation_sweep_results_{args.operation}.json"
    )

    tests = VALIDATION_TESTS[args.operation]
    results = []

    for cfg in DIMENSION_CONFIGS:
        row = {
            "label": cfg["label"],
            "M": int(cfg["m_local"]) * NUM_GPUS if args.operation == "matmul_all_gather" else int(cfg["m_local"]),
            "N": int(cfg["n"]),
            "K": int(cfg["k"]),
            "operation": args.operation,
            "validations": {},
        }
        log(f"Testing {cfg['label']}")
        for test_cfg in tests:
            row["validations"][test_cfg["name"]] = _run_validation_test(args.operation, test_cfg, cfg)
        results.append(row)

    output_file.parent.mkdir(parents=True, exist_ok=True)
    with open(output_file, "w") as f:
        json.dump(results, f, indent=2)
    log(f"Saved validation results to {output_file}")


if __name__ == "__main__":
    main()
