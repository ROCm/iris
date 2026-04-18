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
import signal
import subprocess
import sys
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
TIMEOUT_SECONDS = 180
NUM_GPUS = 8
DEFAULT_HEAP_SIZE = 1 << 34


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
]


VALIDATION_TESTS = {
    "matmul_all_gather": [
        {
            "name": "baseline",
            "path": "tests/ops/test_matmul_all_gather.py",
            "pytest_k": "test_matmul_all_gather",
        },
        {
            "name": "host_copy_engine",
            "path": "tests/ops/test_matmul_all_gather_copy_engine.py",
            "pytest_k": "test_matmul_all_gather_copy_engine",
            "env": {"IRIS_TEST_COPY_ENGINE_MODE": "host"},
        },
        {
            "name": "device_copy_engine",
            "path": "tests/ops/test_matmul_all_gather_copy_engine.py",
            "pytest_k": "test_matmul_all_gather_copy_engine",
            "env": {"IRIS_TEST_COPY_ENGINE_MODE": "device"},
        },
    ],
    "all_gather_matmul": [
        {
            "name": "baseline",
            "path": "tests/ops/test_all_gather_matmul.py",
            "pytest_k": "test_all_gather_matmul_baseline",
        },
        {
            "name": "hbm_buffer",
            "path": "tests/ops/test_all_gather_matmul.py",
            "pytest_k": "test_all_gather_matmul_hbm_buffer",
        },
        {
            "name": "host_copy_engine",
            "path": "tests/ops/test_all_gather_matmul_copy_engine.py",
            "pytest_k": "test_all_gather_matmul_copy_engine",
            "env": {"IRIS_TEST_COPY_ENGINE_MODE": "host"},
        },
        {
            "name": "device_copy_engine",
            "path": "tests/ops/test_all_gather_matmul_copy_engine.py",
            "pytest_k": "test_all_gather_matmul_copy_engine",
            "env": {"IRIS_TEST_COPY_ENGINE_MODE": "device"},
        },
    ],
}


def log(msg: str) -> None:
    print(msg, file=sys.stderr, flush=True)


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
    else:
        if k % NUM_GPUS != 0:
            env["IRIS_TEST_INVALID"] = "1"
        env["IRIS_TEST_M"] = str(m_local)
        env["IRIS_TEST_K_LOCAL"] = str(k // NUM_GPUS)
    return env


def _run_validation_test(operation: str, test_cfg: dict[str, str], shape_cfg: dict[str, Any]) -> dict[str, Any]:
    label = shape_cfg["label"]
    env = os.environ.copy()
    env.update(_shape_env(shape_cfg, operation))
    env.update(test_cfg.get("env", {}))
    env.setdefault("IRIS_TEST_HEAP_SIZE", str(DEFAULT_HEAP_SIZE))

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

    log(f"  Validating {test_cfg['name']}: {label}")
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
            return {"status": "PASSED"}

        error_log_file = PROJECT_ROOT / f"validation_error_{operation}_{test_cfg['name']}_{label}.log"
        with open(error_log_file, "w") as f:
            f.write(stdout)
            f.write("\n")
            f.write(stderr)
        lines = (stdout + stderr).strip().split("\n")
        for line in lines[-5:]:
            log(f"      {line}")
        return {"status": "FAILED", "log": str(error_log_file)}

    except subprocess.TimeoutExpired as timeout_err:
        if process is not None:
            try:
                os.killpg(os.getpgid(process.pid), signal.SIGTERM)
            except ProcessLookupError:
                pass
        partial_stdout = timeout_err.stdout or ""
        partial_stderr = timeout_err.stderr or ""
        error_log_file = PROJECT_ROOT / f"validation_timeout_{operation}_{test_cfg['name']}_{label}.log"
        with open(error_log_file, "w") as f:
            f.write(partial_stdout)
            f.write("\n")
            f.write(partial_stderr)
        return {"status": "TIMEOUT", "log": str(error_log_file)}


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run distributed validation sweep for benchmark shapes",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--operation",
        type=str,
        required=True,
        choices=["matmul_all_gather", "all_gather_matmul"],
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
        else PROJECT_ROOT / f"benchmark/ops/{args.operation}/validation_sweep_results.json"
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
